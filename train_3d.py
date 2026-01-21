# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training#train-the-model

import diffusers 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, DiffusionPipeline, UNet3DConditionModel
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
import torch.multiprocessing as mp
import os
from PIL import Image, ImageDraw
from dataclasses import dataclass, asdict
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import json
import random
import math
import wandb 
import argparse
import subprocess
from utils.dataset import VideoDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', action="store_true")
    parser.add_argument('--image_size', type=int, default=32, help='the height and the width of the images')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epochs', '-e', type=int, default=250, help='if train=True, total number of epochs to train for')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--save_image_epochs', type=int, default=10, help='how often to sample (eval) during training')
    parser.add_argument('--save_model_epochs', type=int, default=10, help='how often to save model during training')
    parser.add_argument('--name', type=str, default='debug3d', help='name of this run. Directory will be checkpoint_dir+name')
    parser.add_argument('--checkpoint_dir', type=str, default='/mnt/data/sonia/cyclone/checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', '-d', type=str, required=True, help='path to training data, or val data for sampling')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--continue', action='store_true', 
                        help='if true and training true, attempt to resume training. uses training configs specified here')
    parser.add_argument('--img_model', type=str, 
                        default=None, help='if training video from scratch, builds from this image model')
    parser.add_argument('--correlated_noise', type=float, default=0.95, help='0 is iid noise')
    # parser.add_argument('--sample', type=int, default=0, help='0 for no sampling, else the number of samples to generate')
    args = parser.parse_args()
    return args

args = get_args()
config = vars(args)
config['dtype'] = torch.float32

try:
    with open('wandb.key', 'r') as f:
        key = f.read().strip()
    wandb.login(key=key)
except:
    wandb.login()


dataset = VideoDataset(dataset=config['dataset'], sample_frames=config['frames'],
                       width=config['image_size'], height=config['image_size'])
if args.train:
    dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=True,) # otherwise crashes on last batch
else:
    dataloader = DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False,) 

def image_to_video_model(config, time_avg=True):
    # 1) load 2-D UNet and grab its config
    unet2d = diffusers.UNet2DConditionModel.from_pretrained(
        os.path.join(config['checkpoint_dir'], config['img_model']), subfolder='unet', revision='main')
    cfg = dict(unet2d.config)
    cfg['down_block_types'] = ['3D'.join(name.split('2D')) for name in cfg['down_block_types']]  
    cfg['up_block_types'] = ['3D'.join(name.split('2D')) for name in cfg['up_block_types']]  
    cfg['mid_block_type'] = '3D'.join(cfg['mid_block_type'].split('2D'))

    # 2) build a fresh 3-D UNet with matching hyper-params
    unet3d  = UNet3DConditionModel.from_config(cfg)
    print('Video model config', unet3d.config)

    # 3) copy 2-D weights -> 3-D
    sd2 = unet2d.state_dict()
    sd3 = unet3d.state_dict()
    for k, w in sd2.items():
        w3d = sd3.get(k, np.asarray([]))
        if w.ndim == 4 and w3d.ndim==5:                                 # (O, I, H, W)
            w3 = w.unsqueeze(2).repeat(1, 1, config['frames'], 1, 1) # add time dimension
            if time_avg:                                # Ho et al., 2022
                w3 /= config['frames']
            sd3[k] = w3
        elif w.ndim > w3d.ndim:
            sd3[k] = w.squeeze()
        else: # no change
            sd3[k] = w
    unet3d.load_state_dict(sd3, strict=False)
    
    unet2d = None
    return unet3d

if not config.get('continue', False) and config['train']: 
    unet = image_to_video_model(config)
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        os.path.join(config['checkpoint_dir'], config['img_model']), subfolder='scheduler', revision='main')
    start_epoch = 0
else: # sample or resume training
    unet = diffusers.UNet3DConditionModel.from_pretrained(
        os.path.join(config['checkpoint_dir'], config['name']), subfolder="unet", revision="main")
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        os.path.join(config['checkpoint_dir'], config['name']), subfolder='scheduler', revision='main')
    with open(os.path.join(config['checkpoint_dir'], config['name'], 'train_log.txt'), 'r') as f:
        logs = f.readlines()
    start_epoch = int(logs[-1].split()[1][:-1]) + 1
    
unet = unet.to('cuda', dtype=config['dtype'])
cross_attention_dim = unet.config['cross_attention_dim']

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config['lr'],
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config['lr_warmup_steps'],
    num_training_steps=(len(dataloader) * config['epochs']),
)

# generate iid (rho==0) or correlated noise
def noisef(shape, rho=0, device=None, dtype=None, generator=None):
    # shape: (B, C, F, H, W)
    B, C, F, H, W = shape
    n = torch.randn(B, C, F, H, W, device=device, dtype=dtype, generator=generator)
    if F <= 1 or rho <= 0:  # falls back to i.i.d. noise
        return n
    s = math.sqrt(1 - rho * rho)
    n[:, :, 1:] = rho * n[:, :, :-1] + s * torch.randn(n[:, :, 1:].size(), dtype=dtype,
                        layout=n.layout, device=device, generator=generator)
    return n



class CondDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.unet = unet 
        self.scheduler = scheduler 
        
    @torch.no_grad()
    def __call__(self, input, num_frames=8, generator=None, encoder_hidden_states=None, **kwargs):
        """Input should be [batch_size, channels, 1, height, width], where frame=0 is the prompt frame"""
        device = self.unet.device
        batch_size = input.shape[0]
        noise  = noisef(
            (batch_size, self.unet.config['in_channels'], num_frames, config['image_size'], config['image_size']),
            generator=generator, device=device, dtype=config['dtype']
        )
        sample = noise.clone()
        prompt = input.to(device=device, dtype=config['dtype'])
        
        for i, t in enumerate(self.scheduler.timesteps):
            eps = self.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            sample = self.scheduler.step(eps, t, sample, generator=generator).prev_sample
            t_prev = self.scheduler.timesteps[i+1] if i+1 < len(self.scheduler.timesteps) else t
            z = torch.randn((batch_size, self.unet.config['in_channels'], config['image_size'], config['image_size']), 
                            device=device, dtype=config['dtype'], generator=generator)
            sample[:, :, 0, :, :] = self.scheduler.add_noise(prompt, z, t_prev)
            
        return {"images": sample.cpu()}


def evaluate(samples, config, epoch, pipeline, device):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        torch.as_tensor(samples, dtype=config['dtype'], device=device),
        num_frames=config['frames'],
        generator=torch.Generator(device=device).manual_seed(config['seed']), 
        encoder_hidden_states=torch.zeros((config['train_batch_size'], 1, cross_attention_dim),
                                          device=device)
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    )['images']
    
    # output from model is on [-1,1]  scale; convert to [0,255]
    images = 255/2 * ( 1+np.array(images) )
    
    samples_dir = os.path.join(config['checkpoint_dir'], config['name'], "training_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(len(images)):
        for t in range(config['frames']):
            frame = Image.fromarray(images[i, :, t, :, :].squeeze()).convert('P')
            frame.save(os.path.join(samples_dir, f'{epoch:04d}_s{i:02d}_t{t:02d}.png'))
 
    
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="wandb",
        project_dir=os.path.join(config['checkpoint_dir'], config['name'], "logs"),
    )
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(config['checkpoint_dir'], config['name'],), exist_ok=True)
        run = wandb.init(
            project = 'geodes',
            config = config,
            name = config['name'],
            group = '3d'
        )
        accelerator.init_trackers(
            'geodes',
            init_kwargs={"wandb": {"resume": 'allow', 'id': run.id}}
        )
        
        env_file_path = os.path.join(config['checkpoint_dir'], config['name'], 'environment.yml')
        subprocess.run(f"conda env export > {env_file_path}", shell=True)
        artifact = wandb.Artifact("conda-env", type="environment")
        artifact.add_file(env_file_path)
        wandb.log_artifact(artifact)
        
        with open(os.path.join(config['checkpoint_dir'], config['name'], 'config.txt'), 'w') as f:
            str_config = {k:str(v) for k, v in config.items()}
            json.dump(str_config, f, indent=4)
    else: # disable wandb in worker processes
        wandb.init(mode="disabled")
        
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
        
    global_step = 0
    for epoch in range(start_epoch, config['epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = []
        
        for step, batch in enumerate(train_dataloader):
            clean_images = torch.as_tensor(batch["pixel_values"], device=accelerator.device, dtype=config['dtype'])
            zeros = torch.zeros((config['train_batch_size'], 1, cross_attention_dim), 
                                device=accelerator.device, dtype=config['dtype'])
            
            # noise = torch.randn(clean_images.shape, device=clean_images.device, dtype=config['dtype'])
            noise = noisef(clean_images.shape, rho=config['correlated_noise'], device=clean_images.device, dtype=config['dtype'])
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64 # leave as is- don't change to config['dtype']
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros,
                                    return_dict=False)[0]
                loss = F.mse_loss(noise_pred[:,:,1:,:,:], noise[:,:,1:,:,:]) # skip zeroth/prompt frame
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
        with open(os.path.join(config['checkpoint_dir'], config['name'], 'train_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}, Step {global_step}, Loss: {loss}, LR: {logs['lr']}\n")
            
        # sample demo images, save model
        if accelerator.is_main_process:
            pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).to(accelerator.device), 
                                             scheduler=noise_scheduler)
            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['epochs'] - 1: # IMAGE
                # get just the first time step/prompt frame
                evaluate(batch["pixel_values"][:, :, 0, :, :], config, epoch, pipeline, accelerator.device)
            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['epochs'] - 1: # MODEL
                pipeline.save_pretrained(os.path.join(config['checkpoint_dir'], config['name']))
        
        
def sample_loop(config, model, noise_scheduler, dataloader):
    os.makedirs(os.path.join(config['checkpoint_dir'], config['name'], 'samples'), exist_ok=True)
    pipeline = CondDiffusionPipeline(unet=model.cuda(), scheduler=noise_scheduler)
    generator = torch.Generator(device='cuda').manual_seed(config['seed'])
    for batch in tqdm(dataloader):
        prompt = batch['pixel_values'][:, :, 0, :, :]
        prompt_batch_size = prompt.shape[0] # final batch may be smaller
        images = pipeline(
            torch.as_tensor(prompt, dtype=config['dtype'], device='cuda'),
            num_frames=config['frames'],
            generator=generator, 
            encoder_hidden_states=torch.zeros((prompt_batch_size, 1, cross_attention_dim), device='cuda')
        )['images']
        
        # # output from model is on [-1,1] log scale; convert to [dataset min, dataset max] nonlog
        images = (images + 1)/2 * (dataset.max - dataset.min) + dataset.min
        images = np.expm1(images)
        
        for i, name in enumerate(batch['name']):
            os.makedirs(os.path.join(config['checkpoint_dir'], config['name'], 'samples', name), exist_ok=True)
            for t in range(config['frames']):
                np.save(os.path.join(config['checkpoint_dir'], config['name'], 'samples', name, f'{t}.npy'), 
                        images[i, :, t, :, :])
        
        
if config['train']:
    # run = wandb.init(
    #     project='geodes',
    #     name=config['name'],
    #     config=config,
    # )
    print('noise scheduler config:\n', noise_scheduler.config)
    train_loop(config, unet, noise_scheduler, optimizer, dataloader, lr_scheduler)
else:
    sample_loop(config, unet, noise_scheduler, dataloader)
    