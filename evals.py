import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import RmseAccumulator, PsdAccumulator
import random
from scipy.ndimage import gaussian_filter

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
    

if len(sys.argv) > 3:
    method = sys.argv[3]
else:
    method = 'unknown_method'
    if 'geodes' in synth_path:
        method = 'geodes'
    elif 'svd' in synth_path:
        method = 'svd'
    elif 'climax' in synth_path:
        method = 'ClimaX'
    elif 'aurora' in synth_path:
        method = 'aurora'
    elif 'clima' in synth_path:
        method = 'climatology'
    elif 'codicast' in synth_path:
        method = 'CoDiCast'
    elif 'cef' in synth_path:
        method = 'CEF'
    elif 'test' in synth_path:
        method = 'Test'
    
print(f'***Detected {method} method, variable(s) {variable} and {split} split***')

def load_data(path):
    data = []
    names = []
    for d in sorted(os.listdir(path), reverse=False):
        if d.startswith('2015'): continue
        if os.path.isdir(os.path.join(path, d)):
            # assert len(os.listdir(os.path.join(path, d))) == len_datapoint, f"Data point {d} in {path} does not have {len_datapoint} frames"
            names.append(d)
            point = [] 
            for i in range(len_datapoint):
                point.append(np.load(os.path.join(path, d, f'{i}.npy')).squeeze())
            data.append(np.stack(point, axis=0))
    return {'names': names, 'data': np.stack(data, axis = 0)}


def get_ike_err(train, synth, threshold=17.0):
    """
    Measures error in total kinetic energy of tropical storm-force winds at T=final.
    """
    # Extract final timestep U and V components
    train_u, train_v = train[:, -1, :, :, 1], train[:, -1, :, :, 2]
    synth_u, synth_v = synth[:, -1, :, :, 1], synth[:, -1, :, :, 2]

    # Calculate wind magnitude
    train_wind = np.sqrt(train_u**2 + train_v**2)
    synth_wind = np.sqrt(synth_u**2 + synth_v**2)

    # Calculate Kinetic Energy (V^2) only where winds exceed the threshold
    train_ike = np.sum(np.where(train_wind > threshold, train_wind**2, 0), axis=(1, 2))
    synth_ike = np.sum(np.where(synth_wind > threshold, synth_wind**2, 0), axis=(1, 2))

    # Return Mean Absolute Error
    return np.abs(train_ike - synth_ike).mean()


def get_tiered_ike_err(train, synth):
    """
    Measures error in total kinetic energy across two critical thresholds:
    10 m/s (Captures broad extratropical circulation)
    17 m/s (Captures tropical storm-force / gale destructive cores)
    """
    # Extract final timestep U and V components
    train_u, train_v = train[:, -1, :, :, 1], train[:, -1, :, :, 2]
    synth_u, synth_v = synth[:, -1, :, :, 1], synth[:, -1, :, :, 2]

    # Calculate wind magnitude
    train_wind = np.sqrt(train_u**2 + train_v**2)
    synth_wind = np.sqrt(synth_u**2 + synth_v**2)

    # --- Threshold 1: Broad Circulation (> 10 m/s) ---
    train_ike_10 = np.sum(np.where(train_wind > 10.0, train_wind**2, 0), axis=(1, 2))
    synth_ike_10 = np.sum(np.where(synth_wind > 10.0, synth_wind**2, 0), axis=(1, 2))
    err_10 = np.abs(train_ike_10 - synth_ike_10).mean()

    # --- Threshold 2: Destructive Core (> 17 m/s) ---
    train_ike_17 = np.sum(np.where(train_wind > 17.0, train_wind**2, 0), axis=(1, 2))
    synth_ike_17 = np.sum(np.where(synth_wind > 17.0, synth_wind**2, 0), axis=(1, 2))
    err_17 = np.abs(train_ike_17 - synth_ike_17).mean()

    return err_10, err_17


def get_vorticity_err(train, synth):
    """
    Calculates Maximum Relative Vorticity error using spatial gradients at T=final.
    ζ = dV/dx - dU/dy
    """
    train_u, train_v = train[:, -1, :, :, 1], train[:, -1, :, :, 2]
    synth_u, synth_v = synth[:, -1, :, :, 1], synth[:, -1, :, :, 2]

    # np.gradient returns gradients along axis=1 (Height/Y) and axis=2 (Width/X)
    train_dv_dy, train_dv_dx = np.gradient(train_v, axis=(1, 2))
    train_du_dy, train_du_dx = np.gradient(train_u, axis=(1, 2))
    
    synth_dv_dy, synth_dv_dx = np.gradient(synth_v, axis=(1, 2))
    synth_du_dy, synth_du_dx = np.gradient(synth_u, axis=(1, 2))

    # Calculate relative vorticity
    train_vort = train_dv_dx - train_du_dy
    synth_vort = synth_dv_dx - synth_du_dy

    # Find the maximum rotational intensity per storm
    train_max_vort = np.max(train_vort, axis=(1, 2))
    synth_max_vort = np.max(synth_vort, axis=(1, 2))

    return np.abs(train_max_vort - synth_max_vort).mean()


def get_rmw_err(train, synth):
    """
    Calculates the spatial distance between the pressure minimum (eye) 
    and the wind speed maximum (eyewall) at T=final.
    """
    # Channel 0 is SLP
    train_slp, synth_slp = train[:, -1, :, :, 0], synth[:, -1, :, :, 0]
    
    # Calculate wind magnitudes
    train_wind = np.sqrt(train[:, -1, :, :, 1]**2 + train[:, -1, :, :, 2]**2)
    synth_wind = np.sqrt(synth[:, -1, :, :, 1]**2 + synth[:, -1, :, :, 2]**2)

    B, H, W = train_slp.shape
    train_rmw_list, synth_rmw_list = [], []

    for b in range(B):
        # Locate the eye (Minimum SLP coordinates)
        t_eye_y, t_eye_x = np.unravel_index(np.argmin(train_slp[b]), (H, W))
        s_eye_y, s_eye_x = np.unravel_index(np.argmin(synth_slp[b]), (H, W))

        # Locate the eyewall peak (Maximum Wind coordinates)
        t_mw_y, t_mw_x = np.unravel_index(np.argmax(train_wind[b]), (H, W))
        s_mw_y, s_mw_x = np.unravel_index(np.argmax(synth_wind[b]), (H, W))

        # Calculate Euclidean distance (in grid cells)
        train_rmw_list.append(np.sqrt((t_eye_y - t_mw_y)**2 + (t_eye_x - t_mw_x)**2))
        synth_rmw_list.append(np.sqrt((s_eye_y - s_mw_y)**2 + (s_eye_x - s_mw_x)**2))

    return np.abs(np.array(train_rmw_list) - np.array(synth_rmw_list)).mean()


def get_hf_spectral_ratio(train, synth, batch_size=64):
    """
    Integrates the high-frequency half of the Power Spectral Density 
    to output a single 'Sharpness Ratio' for the wind fields.
    """
    train_var, synth_var = train[:, 1:, ...], synth[:, 1:, ...]
    B, T, H, W, V = train_var.shape

    accumulator = PsdAccumulator(H, W, V)
    n_splits = int(np.ceil(B / batch_size))
    
    for batch_synth, batch_train in zip(np.array_split(synth_var, n_splits), np.array_split(train_var, n_splits)):
        accumulator.update(batch_synth, batch_train)
        
    results = accumulator.results()
    
    # We only care about the right half of the PSD graph (high frequencies / fine details)
    midpoint = len(results['k_wavenumbers']) // 2
    
    # Analyze U (1) and V (2) wind channels
    hf_ratios = []
    for v in [1, 2]: 
        pred_hf_power = np.sum(results['psd_pred'][v][midpoint:])
        true_hf_power = np.sum(results['psd_true'][v][midpoint:])
        # Ratio of predicted energy to real energy
        hf_ratios.append(pred_hf_power / (true_hf_power + 1e-8))

    return np.mean(hf_ratios)


def get_mslp_err(train, synth):
    true_mslp = np.min(train[..., 0], axis=(2,3))
    pred_mslp = np.min(synth[..., 0], axis=(2,3))
    return np.abs(true_mslp[:,-1] - pred_mslp[:,-1]).mean()


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


def get_synoptic_acc(train, synth, sigma=1.5):
    """
    Calculates ACC after applying a spatial Gaussian filter.
    This removes the 'Double Penalty' for sharp, slightly offset features,
    allowing a fair spatial comparison against blurry autoregressive models.
    """
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
        
    t_true = train[:, 1:, ...]
    t_pred = synth[:, 1:, ...]

    # Apply spatial blur (sigma) over the Height and Width axes (axes 2 and 3)
    # We loop through the batch and time to avoid blurring across independent frames
    B, T, H, W, C = t_true.shape
    t_true_blurred = np.zeros_like(t_true)
    t_pred_blurred = np.zeros_like(t_pred)
    
    for b in range(B):
        for t in range(T):
            for c in range(C):
                t_true_blurred[b, t, :, :, c] = gaussian_filter(t_true[b, t, :, :, c], sigma=sigma)
                t_pred_blurred[b, t, :, :, c] = gaussian_filter(t_pred[b, t, :, :, c], sigma=sigma)

    # 1. Calculate spatial means
    true_mean = np.mean(t_true_blurred, axis=(2, 3), keepdims=True)
    pred_mean = np.mean(t_pred_blurred, axis=(2, 3), keepdims=True)

    # 2. Calculate spatial anomalies
    true_anom = t_true_blurred - true_mean
    pred_anom = t_pred_blurred - pred_mean

    # 3. Calculate Covariance & Variances
    numerator = np.sum(true_anom * pred_anom, axis=(2, 3))
    true_var = np.sum(true_anom**2, axis=(2, 3))
    pred_var = np.sum(pred_anom**2, axis=(2, 3))
    
    denominator = np.sqrt(true_var * pred_var)

    # 4. Calculate ACC 
    acc = numerator / (denominator + 1e-8)

    return np.mean(acc, axis=0)


def get_acc(train, synth):
    """
    Calculates the spatial Pattern Correlation (ACC) between predicted and true grids.
    Returns: Array of shape (Timesteps, Channels) representing mean ACC.
    """
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
        
    # Skip the prompt frame
    t_true = train[:, 1:, ...]
    t_pred = synth[:, 1:, ...]

    # 1. Calculate spatial means across H and W (axes 2 and 3)
    true_mean = np.mean(t_true, axis=(2, 3), keepdims=True)
    pred_mean = np.mean(t_pred, axis=(2, 3), keepdims=True)

    # 2. Calculate spatial anomalies
    true_anom = t_true - true_mean
    pred_anom = t_pred - pred_mean

    # 3. Calculate Covariance (Numerator)
    numerator = np.sum(true_anom * pred_anom, axis=(2, 3))

    # 4. Calculate Variances (Denominator)
    true_var = np.sum(true_anom**2, axis=(2, 3))
    pred_var = np.sum(pred_anom**2, axis=(2, 3))
    denominator = np.sqrt(true_var * pred_var)

    # 5. Calculate ACC (add epsilon to prevent division by zero)
    acc = numerator / (denominator + 1e-8)

    # 6. Average across all storms in the batch
    # Resulting shape: (Timesteps, Channels)
    mean_acc = np.mean(acc, axis=0) 
    
    return mean_acc


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
    plt.savefig(f'plots/{variable}_{split}_{method}.pdf')
    print('histogram saved to', f'plots/{variable}_{split}_{method}.pdf')


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
        
        plt.savefig(f'plots/psd_{variable}_{split}_{var_name}_{method}.png')
        plt.close() # Important to close when looping
        print(f'PSD plot saved for {var_name}:', f'plots/psd_{variable}_{split}_{var_name}_{method}.png')


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
        
        # # Dynamic outlier removal: Filter values above the 99.9th percentile of real data
        # # This replaces the hardcoded '150' for all variables
        # threshold = np.percentile(train_maxes, 99.9)
        # train_maxes = train_maxes[train_maxes <= threshold]
        # synth_maxes = synth_maxes[synth_maxes <= threshold]
        
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
        
        plt.savefig(f'plots/qq_{variable}_{split}_{var_name}_{method}.png')
        plt.close()
        print(f'Q-Q plot saved for {var_name}:', f'plots/qq_{variable}_{split}_{var_name}_{method}.png')
    
    
def plot_temporal_jitter(train_wind, synth_wind):
    # Get max wind speed per patch per timestep -> Shape: (Batch, Timesteps)
    train_vmax = train_wind.max(axis=(2, 3))
    synth_vmax = synth_wind.max(axis=(2, 3))
    
    # Calculate absolute change between consecutive timesteps (dV/dt)
    train_dvdt = np.abs(np.diff(train_vmax, axis=1)).flatten()
    synth_dvdt = np.abs(np.diff(synth_vmax, axis=1)).flatten()
    
    plt.figure(figsize=(7, 5), dpi=120)
    plt.hist(train_dvdt, bins=40, alpha=0.5, density=True, label='Real Storms', color='blue')
    plt.hist(synth_dvdt, bins=40, alpha=0.5, density=True, label=f'Synth ({method})', color='orange')
    
    plt.xlabel('Absolute Change in Max Wind Speed per Timestep (m/s)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Temporal Jitter: Rate of Intensity Change', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/temporal_jitter_{method}.png')
    print("Saved temporal jitter plot:", f'plots/temporal_jitter_{method}.png')
    
    
def plot_wind_pressure_relationship(train_data, synth_data):
    # SLP is channel 0
    train_slp = train_data[:, :, :, :, 0]
    synth_slp = synth_data[:, :, :, :, 0]
    
    train_wind = np.sqrt(train_data[:, :, :, :, 1]**2 + train_data[:, :, :, :, 2]**2)
    synth_wind = np.sqrt(synth_data[:, :, :, :, 1]**2 + synth_data[:, :, :, :, 2]**2)
    
    # Min pressure vs Max wind
    train_min_slp = train_slp.min(axis=(2, 3)).flatten()
    train_max_wind = train_wind.max(axis=(2, 3)).flatten()
    
    synth_min_slp = synth_slp.min(axis=(2, 3)).flatten()
    synth_max_wind = synth_wind.max(axis=(2, 3)).flatten()
    
    plt.figure(figsize=(7, 6), dpi=120)
    plt.scatter(train_min_slp, train_max_wind, alpha=0.3, label='Real', color='blue', s=10)
    plt.scatter(synth_min_slp, synth_max_wind, alpha=0.3, label=f'Synth ({method})', color='orange', s=10)
    
    # Invert X-axis because lower pressure = stronger storm
    plt.gca().invert_xaxis() 
    
    plt.xlabel('Minimum Sea Level Pressure', fontsize=12)
    plt.ylabel('Maximum Wind Speed (m/s)', fontsize=12)
    plt.title('Physical Coherence: Wind-Pressure Relationship', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/wind_pressure_{method}.png')
    print("Saved wind_pressure plot:", f'plots/wind_pressure_{method}.png')
    
    
def plot_eye_wobble(train_data, synth_data):
    def calculate_acceleration(slp_data):
        B, T, H, W = slp_data.shape
        accelerations = []
        
        for b in range(B):
            # Extract the (y, x) coordinates of the min SLP for each timestep
            centers = []
            for t in range(T):
                frame = slp_data[b, t]
                y, x = np.unravel_index(np.argmin(frame), frame.shape)
                centers.append([y, x])
            
            centers = np.array(centers) # Shape: (T, 2)
            
            # Velocity = displacement between frames
            velocity = np.diff(centers, axis=0)
            # Acceleration = change in velocity
            acceleration = np.diff(velocity, axis=0)
            
            # Get the magnitude of the acceleration vector
            accel_mag = np.linalg.norm(acceleration, axis=1)
            accelerations.extend(accel_mag)
            
        return np.array(accelerations)

    train_slp = train_data[:, :, :, :, 0]
    synth_slp = synth_data[:, :, :, :, 0]
    
    train_accel = calculate_acceleration(train_slp)
    synth_accel = calculate_acceleration(synth_slp)
    
    plt.figure(figsize=(7, 5), dpi=120)
    plt.hist(train_accel, bins=np.arange(0, 15, 1), alpha=0.5, 
             density=True, label='Real Storms', color='blue')
    plt.hist(synth_accel, bins=np.arange(0, 15, 1), alpha=0.5, 
             density=True, label=f'Synth ({method})', color='orange')
    
    plt.xlabel('Track Acceleration (Grid Cells / Timestep^2)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Eye Wobble: Track Smoothness & Acceleration', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/eye_wobble_{method}.png')
    print("Saved eye_wobble plot:", f'plots/eye_wobble_{method}.png')
    

train = load_data(train_path)
synth = load_data(synth_path)

if synth['data'].shape[2] == len(channel_names):
    synth['data'] = np.transpose(synth['data'], (0, 1, 3, 4, 2))

print('train', train['data'].shape, 'synth', synth['data'].shape)
print('train min:', train['data'].min(axis=(0,1,2,3)))
print('synth min:', synth['data'].min(axis=(0,1,2,3)))
print('train mean:', train['data'].mean(axis=(0,1,2,3)))
print('synth mean:', synth['data'].mean(axis=(0,1,2,3)))
print('train max:', train['data'].max(axis=(0,1,2,3)))
print('synth max:', synth['data'].max(axis=(0,1,2,3)))

####### Noise baseline for FVD and KVD
# print('computing FVD, KVD for equal splits of train (noise baseline)')
# split1 = random.sample(range(len(train['data'])), len(train['data'])//2)
# split2 = [i for i in range(len(train['data'])) if i not in split1]
# train1 = train['data'][split1]
# train2 = train['data'][split2]
# fvdb, encoderb = get_fvd(train1, train2)
# kvdb = get_kvd(train1, train2, encoder=encoderb)
# print('fvd', fvdb, 'kvd', kvdb)


###### Integrated Kinetic Energy (IKE)
####### Advanced Physical Evaluation
if variable == 'multivar':
    ike_err_mid, ike_err_strong = get_tiered_ike_err(train['data'], synth['data'])
    print(f"Integrated Kinetic Energy Error (Mid): {ike_err_mid:.2f}")
    print(f"Integrated Kinetic Energy Error (Strong): {ike_err_strong:.2f}")
    
    vort_err = get_vorticity_err(train['data'], synth['data'])
    print(f"Peak Relative Vorticity Error: {vort_err:.4f}")
    
    rmw_err = get_rmw_err(train['data'], synth['data'])
    print(f"Radius of Max Winds Error (Grid Cells): {rmw_err:.2f}")
    
    hf_ratio = get_hf_spectral_ratio(train['data'], synth['data'])
    print(f"High-Frequency Spectral Ratio: {hf_ratio:.4f} (1.0 is perfect)")

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


####### MSLP error 
print('computing MSLP error')
mslp_err = get_mslp_err(train_match['data'], synth['data'])
print('mslp error', mslp_err)


####### ACC (Anomaly Correlation Coefficient)
print('computing Synoptic spatial ACC')
acc_results = get_synoptic_acc(train_match['data'], synth['data'])

# Print the final timestep ACC for each variable
print(f"Final Frame (T={acc_results.shape[0]}) ACC Breakdown:")
if variable == 'multivar':
    for v_idx, v_name in enumerate(channel_names):
        print(f"  {v_name.upper()} ACC: {acc_results[-1, v_idx]:.4f}")
else:
    print(f"  {variable.upper()} ACC: {acc_results[-1, 0]:.4f}")


acc_results = get_acc(train_match['data'], synth['data'])

# Print the final timestep ACC for each variable
print(f"Final Frame (T={acc_results.shape[0]}) ACC Breakdown:")
if variable == 'multivar':
    for v_idx, v_name in enumerate(channel_names):
        print(f"  {v_name.upper()} ACC: {acc_results[-1, v_idx]:.4f}")
else:
    print(f"  {variable.upper()} ACC: {acc_results[-1, 0]:.4f}")


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
    
    
####### Temporal jitter plot (dV/dt)
if variable == 'multivar':
    plot_temporal_jitter(windmag_train, windmag_synth)
else:
    plot_temporal_jitter(train['data'], synth['data'])
    
    
####### Wind-SLP relationship and eye wobble plots
if variable == 'multivar':
    plot_wind_pressure_relationship(train['data'], synth['data'])
    plot_eye_wobble(train['data'], synth['data'])
