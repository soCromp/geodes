# %%
# nts: activate svd 
import logging
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import contextlib
import threading
from pyproj import Proj 
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import random 
import itertools

###### settings
trackspath1='/mnt/data/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/mnt/data/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 # overlap for the track data

use_temperature_2m = False
use_geopotential = False #925hpa
use_slp = True # whether to include slp channel
use_windmag = False #include wind magnitude channel # NOTE THIS IS 500hPa
use_winduv = True # include wind u and v components channels # NOTE THIS IS 500hPa
use_temperature = True # temperature at 925hPa 
use_humidity = True # specific humidity at 500hPa
use_topo = False # include topography channel
skip_preexisting = False # skip existing datapoints (ensures they have 8 frames)
threads = 8 # Updated for multi-threading

val_is_in_test = True

# atlantic ocean is regmask['reg_name'].values[109] # so 110 in regmaskoc values
# atlantic: 110
# pacific: 111
reg_id = 110
hemi = 'n' # n or s
###### 

if reg_id == 110:
    basin = 'atlantic'
elif reg_id == 111:
    basin = 'pacific'
basin = hemi + basin

outpath = f'/mnt/data/sonia/aurora-data/date/input-{basin}-multivar-fullcontext'

# %%
regmask = xr.open_dataset('/home/cyclone/regmask_0723_anl.nc')

####### make dataframe of all tracks 
tracks1 = pd.read_csv(trackspath1, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# storms that start before the join year (even if they continue into the join year):
sids1 = tracks1[(tracks1['sid']==tracks1['tid']) & (tracks1['year']<joinyear)]['sid'].unique()
tracks1 = tracks1[tracks1['sid'].isin(sids1)]

tracks2 = pd.read_csv(trackspath2, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# filter out storms that "start" at the beginning of the join year since they probably started before and are 
# included in tracks1
sids2 = tracks2[(tracks2['sid']==tracks2['tid']) & \
        ((tracks2['year']>=joinyear) | (tracks2['month']>1) | (tracks2['day']>1) | (tracks2['hour']>0))]['sid'].unique()
tracks2 = tracks2[tracks2['sid'].isin(sids2)]

tracks = pd.concat([tracks1, tracks2], ignore_index=True)
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])

# conversions from the MCMS lat/lon system, as described in Jimmy's email:
tracks['lat'] = 90-tracks['unk1'].values/100
tracks['lon'] = tracks['unk2'].values/100

tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]

# %%
####### variables prep
grid = 0.25
varnames = [] # list of variables that will be included in this output dataset
varlocs = {'temperature_2m': f'/mnt/data/sonia/cyclone/{grid}/temperature_2m',
           'geopotential': f'/mnt/data/sonia/cyclone/{grid}/geopotential',
           'slp': f'/mnt/data/sonia/codicast-data/5.625/slp', 
           'wind_500hpa': f'/mnt/data/sonia/codicast-data/5.625/wind_500hpa',
           'temperature': f'/mnt/data/sonia/codicast-data/5.625/temperature',
           'humidity': f'/mnt/data/sonia/codicast-data/5.625/humidity',
           'topo': f'/mnt/data/sonia/cyclone/{grid}/slp/topo.nc'}

varfuncs = {}
climatology = {}

if use_temperature_2m:
    varnames.append('temperature_2m')
    def f_temperature_2m(ds, lats, lons, time=None):
        if time is None:
            data = ds.sel(lat=lats, lon=lons)['t2m']
        else:
            data = ds.sel(lat=lats, lon=lons, time=time)['t2m']
        return data 
    varfuncs['temperature_2m'] = f_temperature_2m 

if use_geopotential:
    varnames.append('geopotential')
    def f_geopotential(ds, lats, lons, time=None):
        if time is None:
            data = ds.sel(lat=lats, lon=lons, pressure_level=925)['z']
        else:
            data = ds.sel(lat=lats, lon=lons, time=time, pressure_level=925)['z']
        data = data.drop_vars('pressure_level')
        return data
    varfuncs['geopotential'] = f_geopotential

if use_slp:
    varnames.append('slp')
    def f_slp(ds, lats, lons, time=None):
        if time is None:
            return ds.sel(lat=lats, lon=lons)['slp']
        else:
            return ds.sel(lat=lats, lon=lons, time=time)['slp']
    varfuncs['slp'] = f_slp

if use_windmag:
    varnames.append('wind_500hpa')
    def f_wind(ds, lats, lons, time=None):
        if time is None:
            u = ds.sel(lat=lats, lon=lons, pressure_level=500)['u'] 
            v = ds.sel(lat=lats, lon=lons, pressure_level=500)['v']
        else:
            u = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)['u'] 
            v = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)['v']
        windmag = np.sqrt(u**2 + v**2).drop_vars('pressure_level')
        return windmag
    varfuncs['wind_500hpa'] = f_wind

if use_winduv:
    varnames.append('wind_500hpa')
    def f_winduv(ds, lats, lons, time=None):
        if time is None:
            data = ds.sel(lat=lats, lon=lons, pressure_level=500)[['u', 'v']] 
        else:
            data = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)[['u', 'v']]
        data = data.drop_vars('pressure_level')
        return data
    varfuncs['wind_500hpa'] = f_winduv

if use_temperature:
    varnames.append('temperature')
    def f_temperature(ds, lats, lons, time=None):
        if time is None:
            data = ds.sel(lat=lats, lon=lons, pressure_level=925)['t']
        else:
            data = ds.sel(lat=lats, lon=lons, time=time, pressure_level=925)['t']
        data = data.drop_vars('pressure_level')
        return data
    varfuncs['temperature'] = f_temperature

if use_humidity:
    varnames.append('humidity')
    def f_humidity(ds, lats, lons, time=None):
        if time is None:
            data = ds.sel(lat=lats, lon=lons, pressure_level=500)['q']
        else:
            data = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)['q']
        data = data.drop_vars('pressure_level')
        return data
    varfuncs['humidity'] = f_humidity

topo = None
if use_topo: 
    varnames.append('topo')
    topo = xr.open_dataset(varlocs['topo'], engine='netcdf4')
    def f_topo(ds, lats, lons, time=None):
        return ds.sel(lat=lats, lon=lons)['lsm']

# %%
truetrain = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/train'))
trueval = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/val'))
truetest = set(os.listdir(f'/home/cyclone/train/windmag/500hpa/0.25/date/{basin}/test'))

tracks['split'] = 0
tracks.loc[tracks['sid'].isin(truetest), 'split'] = 2
tracks.loc[tracks['sid'].isin(trueval), 'split'] = 1

# %% [markdown]
# # Forwards
# Create climaX prompts

# %%
# %% [markdown]
# # Forwards
# Create climaX prompts (Optimized NumPy Array Implementation)

# %%
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

init_frames = tracks[(tracks['sid']==tracks['tid']) & (tracks['split']==0)].to_dict('records')

os.makedirs(outpath, exist_ok=True)
os.makedirs(os.path.join(outpath, 'train'), exist_ok=True)
os.makedirs(os.path.join(outpath, 'val'), exist_ok=True)
os.makedirs(os.path.join(outpath, 'test'), exist_ok=True)

try:
    exists_files = os.listdir(os.path.join(outpath, 'val')) + \
                   os.listdir(os.path.join(outpath, 'test')) + \
                   os.listdir(os.path.join(outpath, 'train'))
    exists = set([f.replace('.np', '') for f in exists_files])
except FileNotFoundError:
    exists = set()

print(f"Initial frames to process: {len(init_frames)}")

# Sort frames by year
init_frames = sorted(init_frames, key=lambda x: x['year'])

def save_frame(frame, np_data, time_index):
    """Extremely fast write function using pure NumPy slicing"""
    # np_data shape is expected to be (variables, time, lat, lon)
    result = np_data[:, time_index, :, :]
    
    # Save files based on split
    if frame['split'] == 0:
        np.save(os.path.join(outpath, 'train', f"{frame['sid']}.np"), result)
    elif frame['split'] == 1:
        np.save(os.path.join(outpath, 'val', f"{frame['sid']}.np"), result)
        if val_is_in_test:
            np.save(os.path.join(outpath, 'test', f"{frame['sid']}.np"), result)
    elif frame['split'] == 2:
        np.save(os.path.join(outpath, 'test', f"{frame['sid']}.np"), result)

# Group frames by year
for year, group in itertools.groupby(init_frames, key=lambda x: x['year']):
    frames_this_year = list(group)
    
    if skip_preexisting:
        frames_this_year = [f for f in frames_this_year if str(f['sid']) not in exists]
        
    if not frames_this_year:
        continue
        
    print(f"\nLoading and merging NetCDF data for year {year} into RAM...")
    
    # 1. Load and process all variables ONCE for the entire year
    data_vars_year = []
    for var in varnames:
        ds = xr.open_dataset(f'{varlocs[var]}/{var}.{year}.nc', engine='netcdf4')
        correct_time = ds['time'].values[0] + pd.to_timedelta(np.arange(ds.dims['time']) * 6, unit='h')
        ds = ds.assign_coords(time=correct_time)
        
        # Apply the specific variable logic (pressure levels, magnitude math, etc.) to the ENTIRE year at once
        # Note: We do NOT pass `time` here, so the functions process all times.
        processed_ds = varfuncs[var](ds, slice(90,-90), slice(0,360), time=None)
        data_vars_year.append(processed_ds)
        ds.close()

    # 2. Merge into a single Xarray Dataset and convert to a pure NumPy Array
    # This takes a few seconds but makes the inner loop thousands of times faster
    merged_year_ds = xr.merge(data_vars_year) if len(varnames) > 1 else data_vars_year[0]
    
    # Extract times array and create a lookup dictionary (time string -> integer index)
    time_coords = pd.to_datetime(merged_year_ds['time'].values)
    time_to_idx = {t.strftime('%Y-%m-%dT%H:%M:%S'): i for i, t in enumerate(time_coords)}
    
    # Convert entirely to NumPy. 
    # Shape becomes: (num_variables, num_timesteps, lats, lons)
    print("Converting to pure NumPy array...")
    np_year_data = merged_year_ds.to_array(dim="variable").values
    merged_year_ds.close() # Free memory
    
    # 3. Process all frames for this year
    print(f"Writing {len(frames_this_year)} frames to disk for {year}...")
    
    # Because slicing pure NumPy arrays is instantaneous, thread pooling is now strictly for Disk I/O
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for frame in frames_this_year:
            # Build the time string to match the dictionary
            t_str = f"{frame['year']}-{frame['month']:02d}-{frame['day']:02d}T{frame['hour']:02d}:00:00"
            
            if t_str in time_to_idx:
                idx = time_to_idx[t_str]
                futures.append(executor.submit(save_frame, frame, np_year_data, idx))
            else:
                print(f"Warning: Time {t_str} not found in {year} data.")
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Year {year} Writes"):
            pass
            
    # Explicitly clear the massive numpy array from RAM before the next year starts
    del np_year_data
    
    