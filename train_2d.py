# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training#train-the-model
import diffusers 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, DiffusionPipeline
import numpy as np 
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from PIL import Image, ImageDraw
from dataclasses import dataclass, asdict
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import json
import wandb
import argparse 
import subprocess
import math
from utils.dataset import ImageDataset


# used for argparsing lists
def comma_separated_ints(value):
    return [int(x) for x in value.split(",")]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--image_size', type=int, default=32, help='the height and the width of the images')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8, 
                        help='if sampling, the sample batch size. If training and if val dataset provided, the validation batch size')
    parser.add_argument('--max_val_steps', type=int, default=100, help='max number of batches to run during validation')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='if train=True, total number of epochs to train for')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', '-lr', type=float, default=1e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5, help='Stop if no improvement for 5 epochs')
    parser.add_argument('--min_delta', type=float, default=1e-6)
    parser.add_argument('--save_image_epochs', type=int, default=10, help='obsolete')
    parser.add_argument('--save_model_epochs', type=int, default=10, help='how often to save model during training')
    parser.add_argument('--validation_epochs', type=int, default=10, help='how often to perform validation during training')
    parser.add_argument('--name', type=str, default='debug2d', 
                        help='name of this run. Directory will be checkpoint_dir+name and wandb will use this name')
    parser.add_argument('--checkpoint_dir', type=str, default='/hdd3/sonia/cycloneSVD/checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', '-d', type=str, required=True, help='path to training data, or prompt data for sampling')
    parser.add_argument('--val_dataset', '-v', type=str, required=False, default=None,
                        help='path to validation data (training only -- does nothing for sampling runs)')
    parser.add_argument('--val_flip', action='store_true', default=True,
                        help='if true, flip N/S the val data (useful if validating on different hemisphere than training)')
    parser.add_argument('--continue', type=bool, default=False, 
                        help='if true and training true, attempt to resume training. uses training configs specified here')
    parser.add_argument('--sample', type=int, default=0, help='0 for no sampling, else the number of samples to generate')
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'l1', 'huber'], 
                        help='Loss function to use (mse, l1, or huber)')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='delta value for huber loss (only used if loss_fn is huber)')
    parser.add_argument('--snr_gamma', type=float, default=None, 
                        help='SNR weighting gamma for loss balancing. Recommended value is 5.0.')
    
    parser.add_argument('--unet_layers_per_block', type=int, default=2, help='number of conv layers per unet block')
    parser.add_argument('--unet_block_out_channels', type=comma_separated_ints, default=[32, 64, 128], help='number of channels for unet blocks')
    parser.add_argument('--unet_cross_attention_dim', type=int, default=768, help='dimension for cross attention layers in unet')
    args = parser.parse_args()
    return args
    
    
args = get_args()
config = vars(args)

try:
    with open('wandb.key', 'r') as f:
        key = f.read().strip()
    wandb.login(key=key)
except:
    wandb.login()

output_dir = os.path.join(args.checkpoint_dir, args.name)

    
dataset = ImageDataset(dataset=config['dataset'])
dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=True)

val_dataloader = None
if config['val_dataset'] is not None:
    val_dataset = ImageDataset(dataset=config['val_dataset'], flip=config['val_flip'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], shuffle=True, drop_last=True)


# create the model
if not config.get('continue', False):
    unet = diffusers.UNet2DConditionModel(
        sample_size        = config['image_size'],
        in_channels        = dataset.channels, 
        out_channels       = dataset.channels,
        block_out_channels = config['unet_block_out_channels'],
        layers_per_block   = config['unet_layers_per_block'],
        down_block_types   = ("CrossAttnDownBlock2D",
                            "CrossAttnDownBlock2D",
                            "DownBlock2D"),
        up_block_types     = ("UpBlock2D",
                            "CrossAttnUpBlock2D",
                            "CrossAttnUpBlock2D"),
        cross_attention_dim= config['unet_cross_attention_dim']
    )
    last_epoch=0
else: # resume 
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        output_dir, subfolder="unet", revision="main"
    )
    with open(os.path.join(output_dir, 'train_log.txt'), 'r') as f:
        logs = f.readlines()
    last_epoch = int(logs[-1].split()[1][:-1])
# print(unet)


optimizer = torch.optim.AdamW(unet.parameters(), lr=config['lr'])
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config['lr_warmup_steps'],
    num_training_steps=(len(dataloader) * config['epochs']),
)

zeros = torch.zeros(config['train_batch_size'], 1, config['unet_cross_attention_dim'])

class CondDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.unet = unet 
        self.scheduler = scheduler 
        
    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, encoder_hidden_states=None, **kwargs):
        device = self.unet.device
        sample  = torch.randn(
            batch_size, self.unet.config['out_channels'], 32, 32,
            generator=generator, device=device
        )
        
        for t in self.scheduler.timesteps:
            eps = self.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            sample = self.scheduler.step(eps, t, sample).prev_sample
            
        return {"images": sample.cpu()}
        

# def evaluate(config, epoch, pipeline):
#     # Sample some images from random noise (this is the backward diffusion process).
#     images = pipeline(
#         batch_size=config['eval_batch_size'],
#         generator=torch.Generator(device='cuda').manual_seed(config['seed']), 
#         encoder_hidden_states=zeros.to('cuda'),
#         # Use a separate torch generator to avoid rewinding the random state of the main training loop
#     )['images']
    
#     test_dir = os.path.join(output_dir, "train_samples", str(epoch))
#     os.makedirs(test_dir, exist_ok=True)
    
#     for i in range(config['eval_batch_size']):
#         # output from model is *tensor* [channels, height, width]
#         # on [-1,1]  scale; we will convert to [0,255] for visualization
#         channel_frames = [np.asarray(images[i][c]) for c in range(dataset.channels)]
#         channel_frames = [Image.fromarray(255/2*(frame+1)) for frame in channel_frames]

#         # Make a grid out of the images
#         image_grid = make_image_grid(channel_frames, rows=1, cols=len(channel_frames))
#         image_grid.save(f"{test_dir}/{i}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="wandb",
        project_dir=os.path.join(output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
        
        env_file_path = os.path.join(output_dir, 'environment.yml')
        subprocess.run(f"conda env export > {env_file_path}", shell=True)
        artifact = wandb.Artifact("conda-env", type="environment")
        artifact.add_file(env_file_path)
        wandb.log_artifact(artifact)
        
        with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
            json.dump(config, f, indent=4)
        
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    loss_fn = F.mse_loss
    if config['loss_fn'] == 'l1':
        loss_fn = F.l1_loss
    elif config['loss_fn'] == 'huber':
        delta_val = config.get('huber_delta', 1.0) 
        loss_fn = lambda a, b, **kwargs: F.huber_loss(a, b, delta=delta_val, **kwargs)
    best_loss = float('inf')
    patience_counter = 0
    
    global_step = 0
    for epoch in range(last_epoch, config['epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = []
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"]

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            zeros = torch.zeros((bs, 1, config['unet_cross_attention_dim']), device=clean_images.device, dtype=clean_images.dtype)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros.to(clean_images.device), return_dict=False)[0]
                loss = loss_fn(noise_pred, noise, reduction="none")
                loss = loss.reshape(bs, -1).mean(dim=1) # average over all non-batch dims
                
                # min-snr weighting, if using
                if config.get('snr_gamma') is not None:
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(loss.device)
                    alpha_t = alphas_cumprod[timesteps]
                    
                    # Compute Signal-to-Noise Ratio (SNR) for the sampled timesteps
                    snr = alpha_t / (1.0 - alpha_t)
                    
                    # Calculate Min-SNR weights: min(SNR, gamma) / SNR
                    snr_weight = torch.clamp(snr, max=config['snr_gamma']) / snr
                    
                    # Multiply the per-item loss by its timestep weight
                    loss = loss * snr_weight
                    
                    
                loss = loss.mean() # now full average to get single scalar
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            losses.append(logs['loss'])
            global_step += 1
            
        loss = np.mean(losses)
        
        # early stopping stuff
        if loss < best_loss - config['min_delta']:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if accelerator.is_main_process:
                print(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{config['patience']}")
           
        if accelerator.is_main_process:     
            wandb.log({'epoch_loss': loss, 'epoch': epoch, 'patience': patience_counter}, step=global_step)
            progress_bar.set_postfix(**{"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
            with open(os.path.join(output_dir, 'train_log.txt'), 'a') as f:
                f.write(f"Epoch {epoch}, Step {global_step}, Loss: {loss}, LR: {logs['lr']}\n")
                
        if patience_counter >= config['patience']:
            if accelerator.is_main_process:
                print(f"Early stopping triggered at epoch {epoch}. Best Loss: {best_loss}")
            break
        
        # sample demo images, save model
        if (epoch + 1) % config['save_model_epochs'] == 0 and accelerator.is_main_process:
            pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).cuda(), scheduler=noise_scheduler)
            pipeline.save_pretrained(output_dir)
        if (epoch + 1) % config['validation_epochs'] == 0 and config['val_dataset'] is not None:
            if config['max_val_steps'] is not None:
                progress_bar = tqdm(total=config['max_val_steps'], disable=not accelerator.is_local_main_process)
            else:
                progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Val Epoch {epoch}")
            model.eval() # Switch to evaluation mode
            val_losses = []
            with torch.no_grad():
                for val_step, val_batch in enumerate(val_dataloader):
                    if config['max_val_steps'] is not None and val_step >= config['max_val_steps']:
                        break
                    
                    clean_images = val_batch['pixel_values']
                    bs = clean_images.shape[0]
                    noise = torch.randn(clean_images.shape, device=clean_images.device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                        dtype=torch.int64
                    )
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    zeros = torch.zeros((bs, 1, config['unet_cross_attention_dim']), 
                                                device=clean_images.device, dtype=clean_images.dtype)
                    noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros.to(clean_images.device), return_dict=False)[0]
                    v_loss = loss_fn(noise_pred, noise)
                    
                    gathered_v_loss = accelerator.gather(v_loss.unsqueeze(0)).mean().item()
                    val_losses.append(gathered_v_loss)
                    progress_bar.update(1)
                    progress_bar.set_postfix(**{"loss": gathered_v_loss, "step": val_step})
                    
            val_loss = np.mean(val_losses)
            model.train()
            if accelerator.is_main_process:
                wandb.log({'val_loss': val_loss, 'epoch': epoch}, step=global_step)
                with open(os.path.join(output_dir, 'train_log.txt'), 'a') as f:
                    f.write(f"Epoch {epoch}, Step {global_step}, Val Loss: {val_loss}\n")
                
                
    # final save 
    if accelerator.is_main_process:
        pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).cuda(), scheduler=noise_scheduler)
        pipeline.save_pretrained(output_dir)
            

def sample_loop(config, model, noise_scheduler):
    os.makedirs(os.path.join(config['checkpoint_dir'], config['name'], 'samples'), exist_ok=True)
    pipeline = CondDiffusionPipeline(unet=model.cuda(), scheduler=noise_scheduler)
    n_batches = math.ceil(config['sample'] / config['eval_batch_size'])
    generator = torch.Generator(device='cuda').manual_seed(config['seed'])
    
    batches = []
    for _ in range(n_batches):
        batch_out = pipeline(
            batch_size=config['eval_batch_size'],
            generator=generator, 
            encoder_hidden_states=zeros.to('cuda'),
        )['images']
        batches.append(batch_out)
        
    images = torch.cat(batches, dim=0)[:config['sample']].squeeze() # on [-1, 1] scale
    images = (images + 1)/2 * (dataset.max - dataset.min) + dataset.min # [dataset min, dataset max] scale
    images = images.cpu().numpy() 
    
    for i in range(config['sample']):
        np.save(os.path.join(config['checkpoint_dir'], config['name'], 'samples', f'{i}.npy'), images[i])
    


noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000, clip_sample=False)

if config['train']:
    run = wandb.init(
        project='geodes',
        name=config['name'],
        config=config,
        group = '2d'
    )
    train_loop(config, unet, noise_scheduler, optimizer, dataloader, val_dataloader, lr_scheduler)

if config['sample'] > 0:
    sample_loop(config, unet, noise_scheduler)
