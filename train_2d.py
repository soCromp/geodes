# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training#train-the-model
import diffusers 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, DiffusionPipeline
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
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


def comma_separated_ints(value):
    return [int(x) for x in value.split(",")]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--image_size', type=int, default=32, help='the height and the width of the images')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=250, help='if train=True, total number of epochs to train for')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--save_image_epochs', type=int, default=10, help='how often to sample (eval) during training')
    parser.add_argument('--save_model_epochs', type=int, default=10, help='how often to save model during training')
    parser.add_argument('--name', type=str, default='debug2d', 
                        help='name of this run. Directory will be checkpoint_dir+name and wandb will use this name')
    parser.add_argument('--checkpoint_dir', type=str, default='/hdd3/sonia/cycloneSVD/checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True, help='path to training data, or val data for sampling')
    parser.add_argument('--continue', type=bool, default=False, 
                        help='if true and training true, attempt to resume training. uses training configs specified here')
    parser.add_argument('--sample', type=int, default=0, help='0 for no sampling, else the number of samples to generate')
    
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


class DummyDataset(Dataset):
    def __init__(self, dataset, width=32, height=32, sample_frames=8):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = dataset
        self.folders = [f for f in os.listdir(self.base_folder) if os.path.isdir(os.path.join(self.base_folder, f))]
        self.num_samples = len(self.folders)
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        
        # # get min, max values for normalization
        # self.min = np.inf
        # self.max = -1 * np.inf
        # for folder in self.folders:
        #     for i in range(sample_frames):
        #         frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
        #         self.min = min(self.min, frame.min())
        #         self.max = max(self.max, frame.max())
        
        # Accumulate pixel data to find percentiles
        sampled_pixels = []
        for folder in self.folders:
            for i in range(sample_frames):
                frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
                
                # Flatten to 1D array
                flat = frame.flatten()
                
                # --- MEMORY SAFETY ---
                # If your dataset is huge, you don't need every single pixel.
                # Taking every 100th pixel is statistically sufficient for normalization.
                # Remove '[::100]' if you have infinite RAM.
                sampled_pixels.append(flat[::100])
                
        # 1. Combine into one giant array
        all_data = np.concatenate(sampled_pixels)
        # 2. Apply Log Transform (log(x + 1))
        # We do this BEFORE finding percentiles so self.min/max are in "log space"
        all_data_log = np.log1p(all_data)
        # 3. Calculate 1st and 99th Percentiles
        self.min = np.percentile(all_data_log, 1)
        self.max = np.percentile(all_data_log, 99)
        del sampled_pixels, all_data, all_data_log # free memory
                
        if frame.ndim == 2:
            self.channels = 1
        elif frame.ndim == 3:
            self.channels = frame.shape[2]
        else:
            raise ValueError(f'Frame has invalid number of dimensions, should be 2 or 3 but is {frame.ndim}')
                

    def __len__(self):
        return self.sample_frames * len(self.folders)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        folder_idx = idx // self.sample_frames
        frame_idx = idx % self.sample_frames
        frame_path = os.path.join(self.base_folder, self.folders[folder_idx], f'{frame_idx}.npy')
        
        img = np.load(frame_path)
        img_log = np.log1p(img)
        img_log = torch.from_numpy(img_log).float()
        img_log[img_log.isnan()] = 0.0
        
        img_norm = 2.0 * (img_log - self.min) / (self.max - self.min) - 1.0
        img_norm = torch.clamp(img_norm, -1.0, 1.0)

        # with Image.fromarray(np.load(frame_path)) as img:
        #     # Resize the image and convert it to a tensor
        #     img_resized = img.resize((self.width, self.height))
        #     img_tensor = torch.from_numpy(np.array(img_resized)).float()
        #     img_tensor[img_tensor.isnan()] = 0.0
        #     if img_tensor.isnan().sum()>0:
        #         raise ValueError(
        #             f"{img_tensor.isnan().sum()} NaN values found in the image tensor for frame {frame_name} in folder {chosen_folder}.")
        #     elif img_tensor.isinf().sum()>0:
        #         raise ValueError(
        #             f"Inf values found in the image tensor for frame {frame_name} in folder {chosen_folder}.")

        #     # Normalize the image by scaling pixel values to [-1, 1]
        #     # img_normalized = (img_tensor / img_tensor.max() * 2) -1.0
        #     img_normalized = 2 * (img_tensor-self.min) / (self.max - self.min) - 1.0
        #     img_normalized[img_normalized<-1] = -1 # in case of rounding errors
        #     img_normalized[img_normalized>1] = 1

        if self.channels == 1:
            img_norm = img_norm.unsqueeze(0)
        return {'pixel_values': img_norm}
    
dataset = DummyDataset(dataset=config['dataset'])
dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True)


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
print(unet)


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
        


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config['eval_batch_size'],
        generator=torch.Generator(device='cuda').manual_seed(config['seed']), 
        encoder_hidden_states=zeros.to('cuda'),
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    )['images']
    # print(images.shape, np.array(images[0]).squeeze().shape)
    
    # output from model is on [-1,1]  scale; convert to [0,255] for visualization
    images = [Image.fromarray(255/2*(np.array(images[i]).squeeze() + 1)) for i in range(config['eval_batch_size'])]

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=1, cols=config['eval_batch_size'])

    # Save the images
    test_dir = os.path.join(output_dir, "train_samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
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
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
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
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros.to(clean_images.device), return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
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
        wandb.log({'epoch_loss': loss, 'epoch': epoch}, step=global_step)
        progress_bar.set_postfix(**{"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
        with open(os.path.join(output_dir, 'train_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}, Step {global_step}, Loss: {loss}, LR: {logs['lr']}\n")
        
        # sample demo images, save model
        if accelerator.is_main_process:
            pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).cuda(), scheduler=noise_scheduler)
            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['epochs'] - 1:
                evaluate(config, epoch, pipeline)
            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['epochs'] - 1:
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
    train_loop(config, unet, noise_scheduler, optimizer, dataloader, lr_scheduler)

if config['sample'] > 0:
    sample_loop(config, unet, noise_scheduler)
