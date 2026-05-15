import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import xarray as xr

# --- Paths & Setup ---
datapath = '/mnt/data/sonia/codicast-data/multivar/concat_1940_2015_5.625_5var.npy'
outpath = f'/mnt/data/sonia/aurora-data/date/input-{basin}-multivar-fullcontext'
trackspath1 = '/mnt/data/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2 = '/mnt/data/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 

reg_id = 110
hemi = 's'

basin = 'atlantic' if reg_id == 110 else 'pacific' if reg_id == 111 else ''
basin = hemi + basin


train_outpath = os.path.join(outpath, 'train')
os.makedirs(train_outpath, exist_ok=True)

# --- Data Processing ---
col_names = ['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 'z1', 'z2', 'unk7', 'tid', 'sid']

# Tracks 1
tracks1 = pd.read_csv(trackspath1, sep=' ', header=None, names=col_names)
sids1 = tracks1[(tracks1['sid'] == tracks1['tid']) & (tracks1['year'] < joinyear)]['sid'].unique()
tracks1 = tracks1[tracks1['sid'].isin(sids1)]

# Tracks 2 (Fixed logic to exclude exactly Jan 1, 00:00 of joinyear)
tracks2 = pd.read_csv(trackspath2, sep=' ', header=None, names=col_names)
is_init = (tracks2['sid'] == tracks2['tid'])
is_not_overlap_start = ~((tracks2['year'] == joinyear) & (tracks2['month'] == 1) & (tracks2['day'] == 1) & (tracks2['hour'] == 0))
sids2 = tracks2[is_init & is_not_overlap_start]['sid'].unique()
tracks2 = tracks2[tracks2['sid'].isin(sids2)]

# Concatenate and clean
tracks = pd.concat([tracks1, tracks2], ignore_index=True)
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])

tracks['lat'] = 90 - tracks['unk1'].values / 100
tracks['lon'] = tracks['unk2'].values / 100
tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]

# Isolate initial frames and calculate time delta index safely using .copy()
init_frames = tracks[tracks['sid'] == tracks['tid']].copy()

# Efficient Spatial Filtering
print(f"Initial storms before filtering: {len(init_frames)}")

# 1. Poleward Check (vectorized)
valid_lat = init_frames['lat'].abs() <= 70

# 2. Hemisphere Check (vectorized)
if hemi == 'n':
    valid_hemi = init_frames['lat'] >= 0
else:
    valid_hemi = init_frames['lat'] < 0

# 3. Region Mask Check (Vectorized xarray lookup)
regmask = xr.open_dataset('/home/cyclone/regmask_0723_anl.nc')
# Create xarray DataArrays from our Pandas columns to query the mask all at once
target_lons = xr.DataArray(init_frames['lon'].values - 180, dims='storm')
target_lats = xr.DataArray(init_frames['lat'].values, dims='storm')

# Perform a single nearest-neighbor lookup for the entire dataset
extracted_regions = regmask['regmaskoc'].sel(reglev=1).sel(
    lono=target_lons, 
    lato=target_lats, 
    method='nearest'
).values


def is_in_region(val, target_id):
    if isinstance(val, (list, np.ndarray)):
        return target_id in val
    return target_id == val

valid_region = np.array([is_in_region(v, reg_id) for v in extracted_regions])

# Apply all filters simultaneously
init_frames = init_frames[valid_lat & valid_hemi & valid_region].copy()

print(f"Storms remaining after spatial filtering: {len(init_frames)}")

# Now calculate timestamps and deltas only for the valid storms
init_frames['timestamp'] = pd.to_datetime(init_frames[['year', 'month', 'day', 'hour']])
start_date = pd.Timestamp("1940-01-01 00:00:00")
init_frames['delta'] = ((init_frames['timestamp'] - start_date).dt.total_seconds() / (3600 * 6)).astype(int)
init_frames = init_frames[['sid', 'delta']]

# --- High-Performance Array Slicing & Saving ---

# CRITICAL: mmap_mode='r' prevents loading the entire 75-year dataset into RAM
print("Memory-mapping massive dataset...")
data = np.load(datapath, mmap_mode='r')
max_len = data.shape[0]

def save_storm_slice(sid, delta):
    """Helper function to extract and save a single storm slice."""
    # Bounds check: Ensure we have 8 frames available
    if delta + 8 <= max_len:
        # We wrap it in np.array() to pull the specific slice out of the mmap 
        # into active memory before saving it to a new file.
        slice_data = np.array(data[delta:delta+8]) 
        np.save(os.path.join(train_outpath, f'{sid}.npy'), slice_data)
    else:
        # Optional logging: print(f"Skipping {sid}: Array out of bounds.")
        pass

print(f"Extracting and saving {len(init_frames)} storm sequences...")

# Use ThreadPoolExecutor for fast I/O bound saving
# max_workers=os.cpu_count() * 2 is a good rule of thumb for I/O bound disk operations
with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
    # We use .itertuples(index=False) which is significantly faster than .iterrows()
    list(tqdm(executor.map(lambda row: save_storm_slice(row.sid, row.delta), init_frames.itertuples(index=False)), total=len(init_frames)))

print("Processing complete.")
