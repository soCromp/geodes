import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import RmseAccumulator, PsdAccumulator
import random

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 12})

train_path = sys.argv[1] 
synth_path = sys.argv[2]
len_datapoint = 8 # number of frames per datapoint

split = train_path.split('/')[-1]
variable = train_path.split('/')[4]
if variable == 'windmag':
    variable = 'windmag_' + train_path.split('/')[5]
    channel_names = ['windmag']
elif variable == 'multivar':
    channel_names = ['slp', 'u', 'v', 't', 'q']
method = 'unknown_method'
if 'geodes' in synth_path:
    method = 'geodes'
elif 'svd' in synth_path:
    method = 'svd'
elif 'climax' in synth_path:
    method = 'ClimaX'
elif 'clima' in synth_path:
    method = 'climatology'
elif 'codicast' in synth_path:
    method = 'CoDiCast'
elif 'cont' in synth_path:
    method = 'CEF'
elif 'test' in synth_path:
    method = 'Test'
    
print(f'***Detected {method} method, variable(s) {variable} and {split} split***')

def load_data(path):
    data = []
    names = []
    for d in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, d)):
            assert len(os.listdir(os.path.join(path, d))) == len_datapoint, f"Data point {d} in {path} does not have {len_datapoint} frames"
            names.append(d)
            point = [] 
            for i in range(len_datapoint):
                point.append(np.load(os.path.join(path, d, f'{i}.npy')).squeeze())
            data.append(np.stack(point, axis=0))
    return {'names': names, 'data': np.stack(data, axis = 0)}


def get_fvd(train, synth, encoder=None):
    train_nonan = np.nan_to_num(train)[:, 1:, ...] # skip prompt
    synth_nonan = np.nan_to_num(synth)[:, 1:, ...] # skip prompt
    fvd, encoder = compute_fvd(train_nonan, synth_nonan, encoder=encoder, n_components=128)
    return fvd, encoder


def get_kvd(train, synth, encoder=None):
    train_nonan = np.nan_to_num(train)[:, 1:, ...] # skip prompt
    synth_nonan = np.nan_to_num(synth)[:, 1:, ...] # skip prompt
    kvd, _, _, _ = compute_kvd(train_nonan, synth_nonan, encoder=encoder, n_components=128)
    return kvd


def get_rmse(train, synth, batch_size=64):
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
    train = train[:, 1:, ...] # skip prompt
    synth = synth[:, 1:, ...] # skip prompt
    _, T, H, W, V = train.shape
    accumulator = RmseAccumulator(
        T, H, W, V,
        train.mean(), train.std(),
        standardize=False, var_weights=None,
    )
    n_splits = int(np.ceil(train.shape[0] / batch_size))
    for batch_synth, batch_train in zip(np.array_split(synth, n_splits), np.array_split(train, n_splits)):
        accumulator.update(batch_synth, batch_train)
    return accumulator.results()


def mins_maxes(train, synth): # good as a sanity check
    print('train maxes', train.max(axis=(1,2,3)).mean(), 
                            train.max(axis=(1,2,3)).min(), train.max(axis=(1,2,3)).max())
    print('synth maxes', synth.max(axis=(1,2,3)).mean(), 
                            synth.max(axis=(1,2,3)).min(), synth.max(axis=(1,2,3)).max())

    print('train mins', train.min(axis=(1,2,3)).mean(), 
                            train.min(axis=(1,2,3)).min(), train.min(axis=(1,2,3)).max())
    print('synth mins', synth.min(axis=(1,2,3)).mean(), 
                            synth.min(axis=(1,2,3)).min(), synth.min(axis=(1,2,3)).max())


def maxes_histogram(train, synth, ):
    # axis=1,2,3 to average across time, lat and lon (axis 0 is number of samples)
    plt.hist(train.max(axis=(1,2,3)), density=True,bins=30, alpha=0.5, label='Ground Truth',)
    plt.hist(synth.max(axis=(1,2,3)), density=True,bins=30, alpha=0.5, label=f'{method.title()}-Predicted')
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel(f'Maximum Storm Windspeed (m/s)')
    plt.title(f'Maximum Wind Speed: {method.title()} Synthetic vs Real Storms')
    plt.savefig(f'{variable}_{split}_{method}.pdf')
    print('histogram saved to', f'{variable}_{split}_{method}.pdf')


def get_psd(train, synth, channel_names, batch_size=64):
    train = train[:, 1:, ...] # skip prompt
    synth = synth[:, 1:, ...] # skip prompt
    
    # Ensure shape is (B, T, H, W, V)
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
        
    B, T, H, W, V = train.shape
    accumulator = PsdAccumulator(H, W, V)
    n_splits = int(np.ceil(train.shape[0] / batch_size))
    
    for batch_synth, batch_train in zip(np.array_split(synth, n_splits), np.array_split(train, n_splits)):
        accumulator.update(batch_synth, batch_train)
        
    results = accumulator.results()
    k = results['k_wavenumbers']
    
    # Loop through each variable to create separate plots
    for v in range(V):
        var_name = channel_names[v]
        p_pred = results['psd_pred'][v] # Use specific variable index
        p_true = results['psd_true'][v]
        
        plt.figure(figsize=(6, 6))
        plt.loglog(k, p_true, label='Real', color='blue')
        plt.loglog(k, p_pred, label=f'Synth ({method.title()})', color='orange')
        
        plt.xlabel('Wavenumber (Frequency)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'PSD: {var_name.upper()} Sharpness Analysis')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.savefig(f'psd_{variable}_{split}_{var_name}_{method}.png')
        plt.close() # Important to close when looping
        print(f'PSD plot saved for {var_name}:', f'psd_{variable}_{split}_{var_name}_{method}.png')


def qq_plot(train, synth, channel_names):
    # train/synth shape: (B, T, H, W, V)
    V = train.shape[-1]
    
    for v in range(V):
        var_name = channel_names[v]
        
        # Extract maxes for the current variable only
        t_var = train[..., v]
        s_var = np.nan_to_num(synth[..., v], 0.0)
        
        train_maxes = t_var.max(axis=(1,2,3))
        synth_maxes = s_var.max(axis=(1,2,3))
        
        # Dynamic outlier removal: Filter values above the 99.9th percentile of real data
        # This replaces the hardcoded '150' for all variables
        threshold = np.percentile(train_maxes, 99.9)
        train_maxes = train_maxes[train_maxes <= threshold]
        synth_maxes = synth_maxes[synth_maxes <= threshold]
        
        quantiles = np.linspace(0, 100, 1000)
        q_real = np.percentile(train_maxes, quantiles)
        q_synth = np.percentile(synth_maxes, quantiles)
        
        plt.figure(figsize=(7, 7), dpi=120)
        plt.plot(q_real, q_synth, lw=2.5, color='orange', label=f'Synth {method.title()}')
        
        # Dynamic line bounds
        min_val = min(q_real.min(), q_synth.min())
        max_val = max(q_real.max(), q_synth.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                 ls='--', color='blue', alpha=0.6, label='Perfect Calibration')
        
        plt.xlabel(f'Real Max {var_name.upper()}', fontsize=12)
        plt.ylabel(f'{method.title()} Max {var_name.upper()}', fontsize=12)
        plt.title(f'Q-Q Plot: {var_name.upper()} Intensity Distribution')
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'qq_{variable}_{split}_{var_name}_{method}.png')
        plt.close()
        print(f'Q-Q plot saved for {var_name}:', f'qq_{variable}_{split}_{var_name}_{method}.png')
    

train = load_data(train_path)
synth = load_data(synth_path)
# synth['data'] = np.expm1(synth['data'])

print('train', train['data'].shape, 'synth', synth['data'].shape)

####### Noise baseline for FVD and KVD
print('computing FVD, KVD for equal splits of train (noise baseline)')
split1 = random.sample(range(len(train['data'])), len(train['data'])//2)
split2 = [i for i in range(len(train['data'])) if i not in split1]
train1 = train['data'][split1]
train2 = train['data'][split2]
fvdb, encoderb = get_fvd(train1, train2)
kvdb = get_kvd(train1, train2, encoder=encoderb)
print('fvd', fvdb, 'kvd', kvdb)

####### FVD / KVD
print('computing FVD, KVD for train vs synth')
fvd, encoder = get_fvd(train['data'], synth['data'])
kvd = get_kvd(train['data'], synth['data'], encoder=encoder)
print('fvd', fvd, 'kvd', kvd)

###### RMSE (requires at least some data points to match up)
train_match_data = []
for name in synth['names']:
    assert name in train['names'], f"{name} not in training set"
    train_match_data.append(train['data'][train['names'].index(name)])
train_match = {'names': synth['names'], 'data': np.stack(train_match_data, axis=0)}
rmse = get_rmse(train_match['data'], synth['data'])
print('rmse', rmse)


if variable == 'multivar':
    windmag_train = np.sqrt(train['data'][..., 1]**2 + train['data'][..., 2]**2)
    windmag_synth = np.sqrt(synth['data'][..., 1]**2 + synth['data'][..., 2]**2)

####### Histogram
if variable == 'multivar':
    maxes_histogram(windmag_train, windmag_synth)
else:
    maxes_histogram(train['data'], synth['data'])

####### Power spectral density PSD
get_psd(train['data'], synth['data'], channel_names)
if variable == 'multivar':
    get_psd(windmag_train, windmag_synth, ['windmag'])

####### Quantile-quantile plot
qq_plot(train['data'], synth['data'], channel_names)
if variable == 'multivar':
    qq_plot(np.expand_dims(windmag_train, -1), 
            np.expand_dims(windmag_synth, -1), ['windmag'])
