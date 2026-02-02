import os 
import numpy as np
import torch
from torch.utils.data import Dataset

class HybridNormalizer:
    def __init__(self, channel_names):
        """
        channel_names: list of strings, e.g., ['u', 'v', 't', 'q', 'slp']
        """
        self.channels = channel_names
        self.stats = {} 


    def fit(self, data):
        """
        Compute per-channel 1st and 99th percentiles.
        """
        # Initialize storage
        # We need separate lists for each channel to avoid memory issues
        pixels = {c: [] for c in self.channels}
        
        print("Computing channel statistics...")
                
        for i, char in enumerate(self.channels):
            data_c = data[:, i, ...].flatten() #[::100] to save memory if desired
            valid_mask = np.isfinite(data_c)
            pixels[char].append(data_c[valid_mask])
        
        # Calculate stats
        for i, char in enumerate(self.channels):
            # Concatenate all pixels for this channel
            arr = np.concatenate(pixels[char])
            
            # --- LOGIC SWITCH ---
            if char in ['q', 'humidity', 'precip', 'windmag', 'wind-mag']:
                # Log Transform Logic
                arr = np.log1p(arr)
                self.stats[char] = {
                    'min': np.percentile(arr, 1), # 1st percentile in log space
                    'max': np.percentile(arr, 99),
                    'median': np.median(arr), # compute median to fill nans with
                    'method': 'log_robust'
                }
            else:
                # Linear Logic (U, V, T, SLP)
                self.stats[char] = {
                    'min': np.percentile(arr, 1),
                    'max': np.percentile(arr, 99),
                    'median': np.median(arr), # compute median to fill nans with
                    'method': 'linear_robust'
                }
            
            print(f"Var {char} ({self.stats[char]['method']}): "
                  f"Min={self.stats[char]['min']:.3f}, Max={self.stats[char]['max']:.3f},",
                  f"#Nans={np.sum(~np.isfinite(arr))}")


    def normalize(self, x):
        """
        x: (B, C, H, W), (B, F, C, H, W) or (C, H, W) input tensor/array
        Returns: Normalized x in [-1, 1]
        """
        # Ensure we can broadcast (C, 1, 1)
        if isinstance(x, torch.Tensor):
            x_out = x.clone().float()
        else:
            x_out = torch.from_numpy(x.copy()).float()
            
        for i, char in enumerate(self.channels):
            stat = self.stats[char]
            
            # 1. Select channel
            val = x_out[..., i, :, :] if x.ndim >= 4 else x_out[i, ...]
            
            # 2. Fill nans with median value
            if isinstance(val, torch.Tensor):
                val = torch.nan_to_num(val, nan=float(stat['median']))
            else:
                val = np.nan_to_num(val, nan=float(stat['median']))
            
            # 3. Apply Log if needed
            if stat['method'] == 'log_robust':
                val[val < 0] = 0.0 # these variables can't be negative but sometimes have rounding errors
                if isinstance(val, torch.Tensor):
                    val = torch.log1p(val) #1p handles zeroes
                else:
                    val = np.log1p(val)
            
            # 4. Robust Scale to [-1, 1]
            # formula: 2 * (x - min) / (max - min) - 1
            scale = stat['max'] - stat['min']
            val = 2.0 * (val - stat['min']) / (scale + 1e-6) - 1.0
            
            # 5. Clamp (Safety)
            if isinstance(val, torch.Tensor):
                val = torch.clamp(val, -1.0, 1.0)
            else:
                val = np.clip(val, -1.0, 1.0)
                
            # 6. Assign back
            if x.ndim == 4:
                x_out[:, i, :, :] = val
            elif x.ndim == 5:
                x_out[:, :, i, :, :] = val
            else:
                x_out[i, ...] = val
                
                
        # x_out = torch.nan_to_num(x_out, nan=0.0)
        return x_out


    def denormalize(self, x):
        """ Reverse operations """
        if isinstance(x, torch.Tensor):
            x_out = x.clone()
        else:
            x_out = x.copy()
            
        for i, char in enumerate(self.channels):
            stat = self.stats[char]
            val = x_out[..., i, :, :] if x.ndim == 4 else x_out[i, ...]
            
            # 1. Inverse Scale
            # x_orig = (x_norm + 1)/2 * scale + min
            scale = stat['max'] - stat['min']
            val = (val + 1.0) / 2.0 * scale + stat['min']
            
            # 2. Inverse Log if needed
            if stat['method'] == 'log_robust':
                if isinstance(val, torch.Tensor):
                    val = torch.expm1(val)
                else:
                    val = np.expm1(val)
                # Physics check: Humidity cannot be negative
                if isinstance(val, torch.Tensor):
                    val = torch.clamp(val, min=0.0)
                else:
                    val = np.maximum(val, 0.0)
            
            if x.ndim == 4:
                x_out[:, i, :, :] = val
            else:
                x_out[i, ...] = val
                
        return x_out
    

class ImageDataset(Dataset):
    def __init__(self, dataset, width=32, height=32, sample_frames=8):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        
        # Define the path to the folder containing video frames
        self.base_folder = dataset
        folders = [f for f in os.listdir(self.base_folder) if os.path.isdir(os.path.join(self.base_folder, f))]
        
        data = [] 
        for folder in folders:
            for i in range(self.sample_frames):
                frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
                data.append(frame)
        self.data = np.stack(data, axis=0)
        if self.data.ndim == 3:
            self.data = np.expand_dims(self.data, axis=1) # add channel dim if missing
        else:
            self.data = np.transpose(self.data, (0, 3, 1, 2))
        assert self.data.shape[2] == self.height and self.data.shape[3] == self.width
        
        with open(os.path.join(self.base_folder, '../channels.txt'), 'r') as f:
            self.channel_names = [line.strip() for line in f.readlines()]
            
        self.channels = len(self.channel_names) # convenience variable
        self.normalizer = HybridNormalizer(self.channel_names)
        self.normalizer.fit(self.data)
        self.data = self.normalizer.normalize(self.data)
 

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        img = self.data[idx] # already normalized!
        return {'pixel_values': img}
    
    
    def denormalize(self, x):
        return self.normalizer.denormalize(x)
    
    
class VideoDataset(Dataset):
    def __init__(self, dataset, 
                 width=32, height=32, sample_frames=8):
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        
        # Define the path to the folder containing video frames
        self.base_folder = dataset
        self.folders = [f for f in os.listdir(self.base_folder) if \
            os.path.isdir(os.path.join(self.base_folder, f))]
        
        data = [] # B C F H W
        for folder in self.folders:
            frames = []
            for i in range(self.sample_frames):
                frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
                frames.append(frame)
            data.append(np.stack(frames, axis=0))
        self.data = np.stack(data, axis=0)
        if self.data.ndim == 4:
            self.data = np.expand_dims(self.data, axis=1) # add channel dim if missing
        else:
            self.data = np.transpose(self.data, (0, 4, 1, 2, 3)) # B C F H W
        assert self.data.shape[3] == self.height and self.data.shape[4] == self.width
        
        with open(os.path.join(self.base_folder, '../../channels.txt'), 'r') as f:
            self.channel_names = [line.strip() for line in f.readlines()]
            
        self.channels = len(self.channel_names) # convenience variable
        self.normalizer = HybridNormalizer(self.channel_names)
        self.normalizer.fit(self.data)
        self.data = self.normalizer.normalize(self.data)
                
        # # Accumulate pixel data to find percentiles
        # sampled_pixels = []
        # for folder in self.folders:
        #     for i in range(sample_frames):
        #         frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
                
        #         # Flatten to 1D array
        #         flat = frame.flatten()
                
        #         # --- MEMORY SAFETY ---
        #         # If your dataset is huge, you don't need every single pixel.
        #         # Taking every 100th pixel is statistically sufficient for normalization.
        #         # Remove '[::100]' if you have infinite RAM.
        #         sampled_pixels.append(flat[::100])
                
        # # 1. Combine into one giant array
        # all_data = np.concatenate(sampled_pixels)
        # # 2. Apply Log Transform (log(x + 1))
        # # We do this BEFORE finding percentiles so self.min/max are in "log space"
        # all_data_log = np.log1p(all_data)
        # # 3. Calculate 1st and 99th Percentiles
        # self.min = np.percentile(all_data_log, 1)
        # self.max = np.percentile(all_data_log, 99)
        # del sampled_pixels, all_data, all_data_log # free memory
                

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of 
            shape (batch, channels, time, H, W).
        """
        return {'pixel_values': self.data[idx], 'name': self.folders[idx]}
        # # Randomly select a folder (representing a video) from the base folder
        # chosen_folder = self.folders[idx] # random.choice(self.folders)
        # folder_path = os.path.join(self.base_folder, chosen_folder)
        # frames = sorted(os.listdir(folder_path))[:self.sample_frames] #paths
        # frames = [np.load(os.path.join(folder_path, f)) for f in frames] #frame matrices

        # # Initialize a tensor to store the pixel values (1 in batch size dimension for dataloader)
        # pixel_values = torch.empty((1, self.sample_frames, self.height, self.width))

        # # Load and process each frame
        # for i, frame in enumerate(frames):
        #     frame_log = np.log1p(frame)
        #     img_tensor = torch.from_numpy(frame_log).float()
        #     img_tensor[img_tensor.isnan()] = 0.0
        #     img_norm = 2.0 * (img_tensor - self.min) / (self.max - self.min) - 1.0
        #     img_norm = torch.clamp(img_norm, -1.0, 1.0)
            
        #     # Normalize the image by scaling pixel values to [-1, 1]
        #     # img_normalized = 2 * (img_tensor - self.min) / (self.max - self.min) - 1
        #     # img_normalized[img_normalized<-1] = -1 # in case of rounding errors
        #     # img_normalized[img_normalized>1] = 1

        #     pixel_values[:, i, :, :] = img_norm
            
        # return {'pixel_values': pixel_values, 'name': chosen_folder}


    def denormalize(self, x):
        return self.normalizer.denormalize(x)
