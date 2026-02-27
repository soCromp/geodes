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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random 

###### settings
trackspath1='/home/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/home/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
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
threads = 1 # >1 not implemented

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
           'slp': f'/mnt/data/sonia/cyclone/{grid}/slp', #'wind10m': '/home/cyclone/wind',
           'wind_500hpa': f'/mnt/data/sonia/cyclone/{grid}/wind_500hpa',
           'temperature': f'/mnt/data/sonia/cyclone/{grid}/temperature',
           'humidity': f'/mnt/data/sonia/cyclone/{grid}/humidity',
           'topo': f'/mnt/data/sonia/cyclone/{grid}/slp/topo.nc'} # where the source data is stored 
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
    def f_slp(ds, lats, lons, time=None): # function to run when new SLP file is loaded
        if time is None:
            return ds.sel(lat=lats, lon=lons)['slp']
        else:
            return ds.sel(lat=lats, lon=lons, time=time)['slp']
    varfuncs['slp'] = f_slp
if use_windmag:
    varnames.append('wind_500hpa')
    def f_wind(ds, lats, lons, time=None):
        if time is None:
            u = ds.sel(lat=lats, lon=lons, pressure_level=500)['u'] # for 10m: [['u10', 'v10']] 
            v = ds.sel(lat=lats, lon=lons, pressure_level=500)['v']
        else:
            u = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)['u'] # for 10m: [['u10', 'v10']] 
            v = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)['v']
        windmag = np.sqrt(u**2 + v**2).drop_vars('pressure_level')
        return windmag
    varfuncs['wind_500hpa'] = f_wind
if use_winduv:
    varnames.append('wind_500hpa')
    def f_winduv(ds, lats, lons, time=None):
        # print(ds.sel(lat=lats, lon=lons))
        if time is None:
            data = ds.sel(lat=lats, lon=lons, pressure_level=500)[['u', 'v']] # for 10m: [['u10', 'v10']] 
        else:
            data = ds.sel(lat=lats, lon=lons, time=time, pressure_level=500)[['u', 'v']] # for 10m: [['u10', 'v10']] 
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
varnames, varfuncs

resolution = grid # resolution of data in degs (may later get redefined by climax checkpoint reso)
# l = 800 # (half length: l/2 km from center in each direction)
# s = 32 # box will be dimensions s by s (eg 32x32)
# x_lin = np.linspace(-l, l, s)
# y_lin = np.linspace(-l, l, s)
# x_grid, y_grid = np.meshgrid(x_lin, y_lin) # equal-spaced points from -l to l in both x and y dimensions

# %%
file_year = 1940
end_year = 2024
cur_datas = {}
for var in varnames:
    cur_data = xr.open_dataset(f'{varlocs[var]}/{var}.{file_year}.nc', engine='netcdf4')
    correct_time = cur_data['time'].values[0] + pd.to_timedelta(np.arange(cur_data.dims['time']) * 6, unit='h')
    cur_data = cur_data.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
    cur_datas[var] = cur_data

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

init_frames = tracks[(tracks['sid']==tracks['tid']) & (tracks['split']>0)].to_dict('records')
len(init_frames)

try:
    exists = set(os.listdir(os.path.join(outpath, 'val')) + \
        os.listdir(os.path.join(outpath, 'test')))
except:
    exists = []

# %%
len(exists)

# %%
def prep_point_fulldata(frame): # provide actual climate data, not patch plus climatology
    """make one training datapoint. df contains year/../hr, lat, lon of center"""
    if skip_preexisting and frame['sid'] in exists:
        return
    boxes = []
    global file_year
    if frame['year'] != file_year: # starts in next year, so we know no following storm will start in cur year
        file_year = frame['year'] # advance one year (or more if there were no storms in this year / we skip already processed points)
        for var in varnames:
            next_data = xr.open_dataset(f'{varlocs[var]}/{var}.{file_year}.nc', engine='netcdf4')
            correct_time = next_data['time'].values[0] + pd.to_timedelta(np.arange(next_data.dims['time']) * 6, unit='h')
            next_data = next_data.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
            cur_datas[var] = next_data
            
    year, month, day, hour = frame['year'], frame['month'], frame['day'], frame['hour']
    time = f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00'
    
    data_vars = []
    for var in varnames:
        data = varfuncs[var](cur_datas[var], 
                             slice(90,-90), 
                             slice(0,360), time)
        data_vars.append(data)
        
    if len(varnames)>1:
        data = xr.merge(data_vars)
    else:
        data = data_vars[0]

    # new_lat = np.linspace(90,-90, 128) # north to south
    # new_lon = np.linspace(0, 360, 256, endpoint=False)
    # data_resized = data.interp(lat=new_lat, lon=new_lon, method="linear")
        
    # result = data_resized.to_array(dim="variable").values.squeeze()
    result = data.to_array(dim="variable").values.squeeze()
    # print(result.shape)
    
    # print(data_resized, result.shape)
    if frame['split'] == 0:
        np.save(os.path.join(outpath, 'train', f"{frame['sid']}.np"), result)
    elif frame['split'] == 1:
        np.save(os.path.join(outpath, 'val', f"{frame['sid']}.np"), result)
        if val_is_in_test:
            np.save(os.path.join(outpath, 'test', f"{frame['sid']}.np"), result)
    elif frame['split'] == 2:
        np.save(os.path.join(outpath, 'test', f"{frame['sid']}.np"), result)
    else:
        raise ValueError(f"Unexpected split value {frame['split']}")
    
# prompt = prep_point(tracks.iloc[0])

# %%
os.makedirs(outpath, exist_ok=True)
os.makedirs(os.path.join(outpath, 'val'), exist_ok=True)
os.makedirs(os.path.join(outpath, 'test'), exist_ok=True)

for frame in tqdm(init_frames):
    if frame['sid'] not in exists:
        prep_point_fulldata(frame)

