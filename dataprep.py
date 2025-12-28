# nts: activate langchain_env 
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


###### settings
trackspath1='/home/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/home/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 # overlap for the track data

use_slp = False # whether to include slp channel
use_windmag = True #include wind magnitude channel # NOTE THIS IS 500hPa
use_winduv = False # include wind u and v components channels # NOTE THIS IS 500hPa
use_topo = False # include topography channel
skip_preexisting = False # skip existing datapoints (ensures they have 8 frames)
threads = 1
grid = 0.25

readme = """32x32 for 1600x1600km^2 500hpa windmag, 8 frames long, over [1940,2024] in spacific"""

# atlantic ocean is regmask['reg_name'].values[109] # so 110 in regmaskoc values
# atlantic: 110
# pacific: 111
reg_id = 111
hemi = 's' # n or s

if reg_id == 110:
    basin = hemi + 'atlantic'
elif reg_id == 111:
    basin = hemi + 'pacific'

outpath = f'/home/cyclone/train/windmag/500hpa/{grid}/{basin}'
###### 
print(outpath)
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

####### variables prep
varnames = [] # list of variables that will be included in this output dataset
varlocs = {'slp': f'/mnt/data/sonia/cyclone/{grid}/slp', #'wind10m': '/home/cyclone/wind',
           'wind': f'/mnt/data/sonia/cyclone/{grid}/wind_500hpa',
           'topo': f'/mnt/data/sonia/cyclone/{grid}/slp/topo.nc'} # where the source data is stored 
varfuncs = {}
if use_slp:
    varnames.append('slp')
    def f_slp(ds, lats, lons, time): # function to run when new SLP file is loaded
        return ds.sel(lat=lats, lon=lons, time=time)['slp']
    varfuncs['slp'] = f_slp
if use_windmag:
    varnames.append('wind')
    def f_wind(ds, lats, lons, time):
        u = ds.sel(lat=lats, lon=lons, time=time)['u'] # for 10m: [['u10', 'v10']] 
        v = ds.sel(lat=lats, lon=lons, time=time)['v']
        windmag = np.sqrt(u**2 + v**2)
        return windmag
    varfuncs['wind'] = f_wind
if use_winduv:
    varnames.append('wind')
    def f_winduv(ds, lats, lons, time):
        data = ds.sel(lat=lats, lon=lons, time=time)[['u', 'v']] # for 10m: [['u10', 'v10']] 
        return data
    varfuncs['wind'] = f_winduv
topo = None
if use_topo: 
    varnames.append('topo')
    topo = xr.open_dataset(varlocs['topo'], engine='netcdf4')
    def f_topo(ds, lats, lons, time):
        return ds.sel(lat=lats, lon=lons)['lsm']
varnames, varfuncs

resolution = 0.5 # resolution of data in degs
l = 800 # (half length: l/2 km from center in each direction)
s = 32 # box will be dimensions s by s (eg 32x32)
x_lin = np.linspace(-l, l, s)
y_lin = np.linspace(-l, l, s)
x_grid, y_grid = np.meshgrid(x_lin, y_lin) # equal-spaced points from -l to l in both x and y dimensions

file_year = 1940
end_year = 2024
cur_datas = {}
next_datas = {}
for var in varnames:
    cur_data = xr.open_dataset(f'{varlocs[var]}/{var}.{file_year}.nc', engine='netcdf4')
    correct_time = cur_data['time'].values[0] + pd.to_timedelta(np.arange(cur_data.dims['time']) * 6, unit='h')
    cur_data = cur_data.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
    cur_datas[var] = cur_data
    next_data = xr.open_dataset(f'{varlocs[var]}/{var}.{file_year+1}.nc', engine='netcdf4')
    correct_time = next_data['time'].values[0] + pd.to_timedelta(np.arange(next_data.dims['time']) * 6, unit='h')
    next_data = next_data.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
    next_datas[var] = next_data

def prep_point(df, thread=0):
    """make one training datapoint. df contains year/../hr, lat, lon of center"""
    boxes = []
    global file_year
    print(df['year'])
    if df['year'].iloc[0] != file_year: # starts in next year, so we know no following storm will start in cur year
        file_year = df['year'].iloc[0] # advance one year (or more if there were no storms in this year / we skip already processed points)
        for var in varnames:
            cur_datas[var] = next_datas[var]
            if file_year < end_year:
                next_data = xr.open_dataset(f'{varlocs[var]}/{var}.{file_year+1}.nc', engine='netcdf4')
                correct_time = next_data['time'].values[0] + pd.to_timedelta(np.arange(next_data.dims['time']) * 6, unit='h')
                next_data = next_data.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
                next_datas[var] = next_data
            else:
                next_datas[var] = None
            
    for _, frame in df.iterrows():
        year, month, day, hour = frame['year'], frame['month'], frame['day'], frame['hour']
        time = f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00'
        
        lat_center, lon_center = frame['lat'], frame['lon']
        # 'aeqd': https://proj.org/en/stable/operations/projections/aeqd.html
        proj_km = Proj(proj='aeqd', lat_0=lat_center, lon_0=lon_center, units='km')
        # Project to find lat/lon corners of the equal-area box
        lon_grid, lat_grid = proj_km(x_grid, y_grid, inverse=True) #translate km to deg
        lon_grid=(lon_grid+360)%360 # because these datasets have lon as 0 to 360 (lat is still -90 to 90)
        lon_min = lon_grid.min() - resolution # +- reso because otherwise xarray will not include the edge points
        lon_max = lon_grid.max() + resolution
        lat_min = lat_grid.min() - resolution
        lat_max = lat_grid.max() + resolution
        # print(lat_max, lat_min, lon_min, lon_max)
        
        rawslices = []
        for var in varnames:
            if year == file_year:
                data = varfuncs[var](cur_datas[var], slice(lat_max, lat_min), slice(lon_min, lon_max), time)
            else:
                data = varfuncs[var](next_datas[var], slice(lat_max, lat_min), slice(lon_min, lon_max), time)
            rawslices.append(data.sortby(['lat', 'lon']))
        # rawbox = np.stack(slices).squeeze() # squeeze -- only works with 1 channel for now
        # print(rawbox.shape)
        slices = []
        for data in rawslices:
            lats = data.lat.values 
            lons = data.lon.values
            if data.shape[0] > 1: # for instance, wind u and v components (data.shape[0] or data.to_array().shape[0] ??)
                data = data.to_array().squeeze().values
                for i in range(data.shape[0]):
                    sel = data[i]
                    # Build interpolator
                    interp = RegularGridInterpolator(
                        (lats, lons),
                        sel,
                        bounds_error=False,
                        fill_value=np.nan
                    )

                    # Interpolate at new (lat, lon) pairs
                    interp_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
                    interp_values = interp(interp_points).reshape(s, s)
                    slices.append(interp_values)
            else: # just one channel (eg slp)
                data = np.asarray(data).squeeze()
                # Build interpolator
                interp = RegularGridInterpolator(
                    (lats, lons),
                    data,
                    bounds_error=False,
                    fill_value=np.nan
                )

                # Interpolate at new (lat, lon) pairs
                interp_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
                interp_values = interp(interp_points).reshape(s, s)
                slices.append(interp_values)
        # boxes.append(interp_values)
        boxes.append(np.stack(slices, axis=-1))
        
    datapoint = np.stack(boxes, axis=0).squeeze()
    return datapoint

sids = tracks['sid'].unique()
RADIUS=6371 # Earth radius in km

if not os.path.exists(outpath):
    os.makedirs(outpath)

with open(f'{outpath}/README.txt', 'w') as f:
    f.write(readme)

def worker(sids_chunk, thread_id):
    for i, sid in enumerate(sids_chunk):
        if i % 100 == 0:
            print(f'Thread {thread_id}: Processing sid {i}/{len(sids_chunk)}: {i/len(sids_chunk)*100:.2f}% complete')
        sid_df = tracks[(tracks['sid'] == sid)]
        if len(sid_df) < 8: # storm too short
            continue
        start_lat = sid_df['lat'].iloc[0]
        start_lon = sid_df['lon'].iloc[0]

        if skip_preexisting and os.path.exists(f'{outpath}/{sid}') and len(os.listdir(f'{outpath}/{sid}')) == 8:
            continue # skip completed datapoints
        elif sid_df['lat'].abs().iloc[0] > 70:
            continue # starts poleward of 70 degrees
        elif (hemi == 'n' and start_lat < 0) or (hemi == 's' and start_lat >= 0) or \
            (reg_id not in regmask.sel(lono=start_lon-180, lato=start_lat, method='nearest')['regmaskoc'].sel(reglev=1).values):
            continue # only get desired ocean region area
        
        sid_df = sid_df.sort_values(by=['tid'])
        sid_df = sid_df.iloc[:8]  # only take the first 8 frames for debugging
        
        point = prep_point(sid_df)
        os.makedirs(f'{outpath}/{sid}', exist_ok=True)
        for i, frame in enumerate(point):
            np.save(f'{outpath}/{sid}/{i}.npy', frame)
            
            
for i in range(threads):
    start = i * len(sids) // threads
    end = (i + 1) * len(sids) // threads
    sids_chunk = sids[start:end]
    print(start, end, sids_chunk.shape)
    thread = threading.Thread(target=worker, args=(sids_chunk, i))
    thread.start()
    # worker(sids_chunk, i)
    
for i in range(threads):
    thread.join()
print("All threads completed.")
