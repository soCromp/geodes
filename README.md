# ⛈️ GeoDES (Geospatial Diffusion-Based Evolution Synthesis)

**Generating high-fidelity, physically consistent synthetic weather events for research.**

Welcome to GeoDES! This project uses [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) to create realistic synthetic examples of extratropical cyclones.

**Project goals:**
- *Escape the "Mean State":* Demonstrate that generative models can produce sharp, realistic weather features rather than blurred averages.
- *Tail Convergence:* Accurately reproduce the statistical distribution of wind speeds, especially at the extreme tail (high-impact storms).
- *Physical Consistency:* Ensure that generated variables (Wind, Pressure, Humidity) maintain realistic correlations.

**Current status:**
- Ready to run in the single climate variable case (e.g., generate only SLP or only 500hpa wind magnitude)
    - Validated to match the [Power Spectral Density (PSD)](https://en.wikipedia.org/wiki/Spectral_density) of ERA5 data
    - Outperforms [ClimaX](https://arxiv.org/abs/2301.10343) and Climatology baselines in distributional metrics ([FVD](https://arxiv.org/abs/1812.01717) and [KVD](https://arxiv.org/abs/1801.01401))
- Debugged in the multivariate case, with training and evaluations ongoing

![Example Training Datapoint vs GeoDES Output](https://github.com/soCromp/geodes/blob/main/demo.png?raw=true)

## Installation

```shell
conda create --name geodes python==3.11.11

pip install numpy torch==2.6.0 torchvision torchaudio xarray diffusers==0.32.2 tqdm transformers==4.50.0 pandas matplotlib notebook accelerate==1.5.2 opencv-python==4.11.0.86 einops==0.8.1 wandb scipy scikit-learn
```

## Training code

**Image pretraining phase:**<br>
```shell
python train_2d.py --train --epochs 1 --lr 1e-9 --dataset /mnt/data/sonia/cyclone/natlantic2/train --name debug2d
```

**Video training phase:**<br>
Single-GPU:
```shell
python train_3d.py --train --epochs 1 --lr 1e-9 --dataset <dataset> --img_model debug2d
```
Multi-GPU:
```shell 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes <num_GPUs> train_3d.py --train --epochs 1 --lr 1e-9 --dataset <dataset> --img_model debug2d
```

**Stable diffusion (older approach for baseline):** <br>
```accelerate launch train_svd.py     --dataset /home/cyclone/train/windmag/10m/natlantic  --output_dir /home/sonia/cycloneSVD/debug     --per_gpu_batch_size=16 --gradient_accumulation_steps=1     --max_train_steps=500     --channels=1     --width=32     --height=32     --checkpointing_steps=500 --checkpoints_total_limit=1     --learning_rate=1e-5 --lr_warmup_steps=0     --seed=123      --validation_steps=100     --num_frames=8     --mixed_precision="fp16" ```

## Attributions

- SVD code is from https://github.com/wangqiang9/SVD_Xtend/

## Contact

Please reach out to Sonia Cromp ([cromp@wisc.edu](mailto:cromp@wisc.edu), [socromp.github.io](socromp.github.io)) with any questions or comments (or postdoc opportunities :))!
