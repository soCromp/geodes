import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import RmseAccumulator
import random
from matplotlib import pyplot as plt

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
        train = np.expand_dims(train, -1)[:, 1:, ...] # skip prompt
        synth = np.expand_dims(synth, -1)[:, 1:, ...] # skip prompt
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


train = load_data(train_path)
synth = load_data(synth_path)
synth['data'] = np.expm1(synth['data'])

train_match_data = []
for name in synth['names']:
    assert name in train['names'], f"{name} not in training set"
    train_match_data.append(train['data'][train['names'].index(name)])
train_match = {'names': synth['names'], 'data': np.stack(train_match_data, axis=0)}

print('train', train['data'].shape, 'synth', synth['data'].shape)

print('computing FVD, KVD for equal splits of train (noise baseline)')
split1 = random.sample(range(len(train['data'])), len(train['data'])//2)
split2 = [i for i in range(len(train['data'])) if i not in split1]
train1 = train['data'][split1]
train2 = train['data'][split2]
fvdb, encoderb = get_fvd(train1, train2)
kvdb = get_kvd(train1, train2, encoder=encoderb)
print('fvd', fvdb, 'kvd', kvdb)

print('computing FVD, KVD for train vs synth')
fvd, encoder = get_fvd(train['data'], synth['data'])
kvd = get_kvd(train['data'], synth['data'], encoder=encoder)
print('fvd', fvd, 'kvd', kvd)

rmse = get_rmse(train_match['data'], synth['data'])
print('rmse', rmse)

####### Mins and Maxes
# train['data'][np.isnan(train['data'])] = 0.0

print('train maxes', train['data'].max(axis=(1,2,3)).mean(), 
                        train['data'].max(axis=(1,2,3)).min(), train['data'].max(axis=(1,2,3)).max())
print('synth maxes', synth['data'].max(axis=(1,2,3)).mean(), 
                        synth['data'].max(axis=(1,2,3)).min(), synth['data'].max(axis=(1,2,3)).max())

print('train mins', train['data'].min(axis=(1,2,3)).mean(), 
                        train['data'].min(axis=(1,2,3)).min(), train['data'].min(axis=(1,2,3)).max())
print('synth mins', synth['data'].min(axis=(1,2,3)).mean(), 
                        synth['data'].min(axis=(1,2,3)).min(), synth['data'].min(axis=(1,2,3)).max())

####### Histogram
plt.hist(train['data'].max(axis=(1,2,3)), bins=30, alpha=0.5, label='real')
plt.hist(synth['data'].max(axis=(1,2,3)), bins=30, alpha=0.5, label='synth')
plt.legend()
plt.title('Maximum Wind Speed In Synthetic vs Real Storms')
plt.savefig(f'{variable}_{split}_{method}.png')
print('histogram saved to', f'{variable}_{split}_{method}.png')
