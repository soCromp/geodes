# Cyclone Video Diffusion

## Installation

conda create --name geodes python==3.11.11

pip install numpy torch==2.6.0 torchvision torchaudio xarray diffusers==0.32.2 tqdm transformers==4.50.0 pandas matplotlib notebook accelerate==1.5.2 opencv-python==4.11.0.86 einops==0.8.1 wandb scipy scikit-learn

## Training code

**Image pretraining phase** (be sure to edit the config dict inside the code)<br>:
```python train_2d.py --train --epochs 1 --lr 1e-9 --dataset /mnt/data/sonia/cyclone/natlantic2/train```

**Video training phase** (be sure to edit the config dict inside the code)<br>:
```python train_3d.py```

**Stable diffusion (older approach for comparisons):** <br>
```accelerate launch train_svd.py     --dataset /home/cyclone/train/windmag/10m/natlantic  --output_dir /home/sonia/cycloneSVD/debug     --per_gpu_batch_size=16 --gradient_accumulation_steps=1     --max_train_steps=500     --channels=1     --width=32     --height=32     --checkpointing_steps=500 --checkpoints_total_limit=1     --learning_rate=1e-5 --lr_warmup_steps=0     --seed=123      --validation_steps=100     --num_frames=8     --mixed_precision="fp16" ```

**Stable diffusion with sliding probability training datasets:**<br>
```accelerate launch train_svd.py     --dataset /home/cyclone/train/windmag_natlantic --dataset2 /home/cyclone/train/windmag_npacific     --output_dir /home/sonia/cycloneSVD/windmag_atlanticpacific3     --per_gpu_batch_size=16 --gradient_accumulation_steps=1     --max_train_steps=50000     --channels=1     --width=32     --height=32     --checkpointing_steps=1000 --checkpoints_total_limit=3     --learning_rate=1e-5 --lr_warmup_steps=0     --seed=123      --validation_steps=1000     --num_frames=8     --mixed_precision="fp16" --choice_func="linear"```


## Attributions

- SVD code is from https://github.com/wangqiang9/SVD_Xtend/
