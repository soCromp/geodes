# nts: activate langchain_env 
import cdsapi
from shutil import move
from tqdm import tqdm
import xarray as xr
import numpy as np
import zipfile
import os

grid='0.25' #0.5 works 0.25 OOM
client = cdsapi.Client()

# # topo
# print('Downloading topo...')
# fname = client.retrieve(
#     'reanalysis-era5-single-levels',
#     {
#         'product_type': 'reanalysis',
#         'variable': ['z', 'land_sea_mask'],
#         'year': '2000',
#         'month': '01',
#         'day': '01',
#         'time': '00:00',
#         'format': 'netcdf',
#         "download_format": "zip",
#         'grid': f'{grid}/{grid}', 
#     }, 'temp.nc.zip')

# with zipfile.ZipFile('temp.nc.zip', 'r') as zip_ref:
#     fname = zip_ref.namelist()[0]
#     zip_ref.extractall('.')
# ds = xr.open_dataset(fname)
# ds = ds.rename({'z': 'hgt', 'latitude': 'lat', 'longitude': 'lon'})
# ds['hgt'] = ds['hgt']/9.8
# ds.hgt.attrs['units'] = 'm'
# ds.to_netcdf('topo.nc')
# os.remove(fname)

# slp data
dataset = "reanalysis-era5-single-levels"

for year in range(1940, 2025):
    print(f"Downloading data for year {year}...")
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind","10m_v_component_of_wind",
                    #  #"mean_sea_level_pressure",
                    ],
        "year": year,
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", #"01:00", "02:00",
        #"03:00", "04:00", "05:00",
        "06:00", #"07:00", "08:00",
        #"09:00", "10:00", "11:00",
        "12:00", #"13:00", "14:00",
        #"15:00", "16:00", "17:00",
        "18:00", #"19:00", "20:00",
        #"21:00", "22:00", "23:00"
    ],
        "data_format": "netcdf",
        "download_format": "zip",
        'grid': f'{grid}/{grid}', 
    }

    fname = client.retrieve(dataset, request, 'temp.nc.zip')
    print('retrieval complete')

    with zipfile.ZipFile('temp.nc.zip', 'r') as zip_ref:
        fname = zip_ref.namelist()[0]
        zip_ref.extractall('.')
    print('unzipped')
        
    ds = xr.open_dataset(fname)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time', #'msl': 'slp', 
                    })
    print('opened and renamed variables')

    len_time = len(ds.time)
    ds['time'] = np.asarray(np.arange(0, len_time), dtype=int)
    ds.time.attrs['delta_t'] = '0000-00-00 06:00:00'
    ds.time.attrs['units'] = 'hours since %d-01-01 00:00:00'%(year)
    print('adjusted time')

    # # converting slp values to mb from Pa
    # print(ds)
    # ds['slp'] = ds.slp/100.
    # ds.slp.attrs['units'] = 'mb'
    # print('adjusted slp')

    # saving data to output file 
    ds.to_netcdf(f'wind.{year}.nc')
    print('saved out')
    os.remove(fname)
    print('removed temp file')
