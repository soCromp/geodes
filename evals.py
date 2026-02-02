import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import RmseAccumulator, PsdAccumulator
import random

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 15})

train_path = sys.argv[1] 
synth_path = sys.argv[2]
len_datapoint = 8 # number of frames per datapoint

split = train_path.split('/')[-1]
variable = train_path.split('/')[4]
if variable == 'windmag':
    variable = 'windmag_' + train_path.split('/')[5]
method = 'unknown_method'
if 'geodes' in synth_path:
    method = 'geodes'
elif 'svd' in synth_path:
    method = 'svd'
elif 'climax' in synth_path:
    method = 'climax'
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
    plt.hist(train.max(axis=(1,2,3)), density=True,bins=30, alpha=0.5, label='Real')
    plt.hist(synth.max(axis=(1,2,3)), density=True,bins=30, alpha=0.5, label=f'Synth ({method.title()})')
    plt.legend()
    plt.title(f'Maximum Wind Speed In {method.title()} Synthetic vs Real Storms')
    plt.savefig(f'{variable}_{split}_{method}.pdf')
    print('histogram saved to', f'{variable}_{split}_{method}.pdf')


def get_psd(train, synth, batch_size=64):
    train = train[:, 1:, ...] # skip prompt
    synth = synth[:, 1:, ...] # skip prompt
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
    _, T, H, W, V = train.shape
    accumulator = PsdAccumulator(H, W, V)
    n_splits = int(np.ceil(train.shape[0] / batch_size))
    for batch_synth, batch_train in zip(np.array_split(synth, n_splits), np.array_split(train, n_splits)):
        accumulator.update(batch_synth, batch_train)
        
    results = accumulator.results()
    k = results['k_wavenumbers']
    p_pred = results['psd_pred'][0] # Variable 0
    p_true = results['psd_true'][0] # Variable 0
    
    plt.figure(figsize=(6, 6))
    # Log-Log plot is standard for PSD
    plt.loglog(k, p_true, label='Real', color='blue')
    plt.loglog(k, p_pred, label=f'Synth ({method.title()})', color='orange')
    plt.ylim(bottom=0,)
    
    plt.xlabel('Wavenumber (Frequency)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'PSD: Sharpness Analysis for Real vs {method.title()} Synthetic')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f'psd_{variable}_{split}_{method}.png')
    print(f'PSD plot saved to psd_{variable}_{split}_{method}.png')


def qq_plot(train, synth):
    train_maxes = train.max(axis=(1,2,3,))
    train_maxes = train_maxes[train_maxes < 150] # remove crazy outliers (may need to change for other variables)
    synth = np.nan_to_num(synth, 0.0)
    synth_maxes = synth.max(axis=(1,2,3,))
    synth_maxes = synth_maxes[synth_maxes < 150] # remove crazy outliers (may need to change for other variables)
    quantiles = np.linspace(0, 100, 1000)
    q_real = np.percentile(train_maxes, quantiles)
    q_synth = np.percentile(synth_maxes, quantiles)
    # print(q_real, q_synth )
    
    plt.figure(figsize=(7, 7), dpi=120)
    plt.plot(q_real, q_synth, lw=2.5, color='orange', label=f'Synth {method.title()} Distribution')
    
    min_val = min(q_real.min(), q_synth.min())
    max_val = max(q_real.max(), q_synth.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             ls='--', color='blue', alpha=0.6, label='Perfect Calibration')
    
    plt.xlabel(f'Real Max {variable.title()}', fontsize=12)
    plt.ylabel(f'{method.title()} Max {variable.title()}', fontsize=12)
    plt.title(f'Q-Q plot for Real vs {method.title()} Synthetic', fontsize=14, pad=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'qq_{variable}_{split}_{method}.png')
    print(f'Q-Q plot saved to qq_{variable}_{split}_{method}.png')
    

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

####### RMSE (requires at least some data points to match up)
# train_match_data = []
# for name in synth['names']:
#     assert name in train['names'], f"{name} not in training set"
#     train_match_data.append(train['data'][train['names'].index(name)])
# train_match = {'names': synth['names'], 'data': np.stack(train_match_data, axis=0)}
# rmse = get_rmse(train_match['data'], synth['data'])
# print('rmse', rmse)

####### Histogram
maxes_histogram(train['data'], synth['data'])

####### Power spectral density PSD
get_psd(train['data'], synth['data'])

####### Quantile-quantile plot
qq_plot(train['data'], synth['data'])
