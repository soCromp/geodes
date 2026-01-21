# # climatology
import xarray as xr
import glob
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Progress bar

# --- Configuration ---
INPUT_GLOB = "/mnt/data/sonia/cyclone/0.25/temperature/temperature.*.nc"
OUT_DIR = "/mnt/data/sonia/cyclone/0.25/temperature/t_monthly_means"
FINAL_OUT = "/mnt/data/sonia/cyclone/0.25/temperature/t_monthly_climatology.nc"
WORKERS = 10  # Process 10 files at once
skip_preexisting = True

os.makedirs(OUT_DIR, exist_ok=True)

# --- Worker Function (Runs in parallel) ---
def process_year(f):
    """
    Worker function to process a single NetCDF file.
    Calculates monthly means and writes to disk.
    """
    try:
        filename = os.path.basename(f)
        year = filename.split(".")[1]  # Assumes format wind.2020.nc
        
        out_file = os.path.join(OUT_DIR, f"wind.monthly.{year}.nc")
        if skip_preexisting and os.path.exists(out_file):
            print(f"Skipped {year} (Exists)")
            return f"Skipped {year} (Exists)"
        
        # Optional: Skip if already exists to allow resuming
        # if os.path.exists(out_file):
        #     return f"Skipped {year} (Exists)"

        # Open individual file
        # Using a context manager ensures file handles are closed properly
        with xr.open_dataset(f, engine="netcdf4", chunks={"time": 31, "lat": 90, "lon": 90}) as ds:
            
            # 1. Correct Time Index
            # Ensure we calculate timedelta correctly for the specific file length
            correct_time = ds['time'].values[0] + pd.to_timedelta(np.arange(ds.dims['time']) * 6, unit='h')
            ds = ds.assign_coords(time=correct_time)

            # 2. Calculate Monthly Means
            # .compute() ensures the work is done inside the worker process
            monthly = ds.groupby("time.month").mean("time").compute()

            # 3. Write to Disk
            monthly.to_netcdf(out_file)
            
        return f"Completed {year}"

    except Exception as e:
        return f"Error processing {f}: {e}"

# --- Main Execution ---
# Essential for multiprocessing to work safely on Windows/Linux
if __name__ == "__main__":
    files = sorted(glob.glob(INPUT_GLOB))
    print(f"Found {len(files)} source files.")

    # 1. Parallel Processing
    # This replaces the first 'for' loop
    print(f"Starting parallel processing with {WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # map() applies the function to the file list in parallel
        # list() triggers the execution
        # tqdm() adds a progress bar
        results = list(tqdm(executor.map(process_year, files), total=len(files)))
    
    # 2. Aggregation
    # This part remains mostly serial as it operates on small summary files
    MONTHLY_GLOB = os.path.join(OUT_DIR, "wind.monthly.*.nc")
    processed_files = sorted(glob.glob(MONTHLY_GLOB))
    
    if not processed_files:
        print("No processed files found. Exiting.")
        exit()

    print(f"Aggregating {len(processed_files)} yearly monthly means...")
    
    # Optimized loading: open_mfdataset handles the list automatically
    # We combine by 'nested' to create the 'year' dimension implicitly if needed, 
    # or just concat if they share coords.
    with xr.open_mfdataset(processed_files, combine='nested', concat_dim='year') as monthly_stack:
        
        # 3. Compute Final Climatology
        monthly_climatology = monthly_stack.mean("year")
        
        monthly_climatology.to_netcdf(FINAL_OUT)
        print(f"Climatology saved to {FINAL_OUT}")
        
        

# import xarray as xr
# import glob
# import os
# import pandas as pd
# import numpy as np

# INPUT_GLOB = "/mnt/data/sonia/cyclone/0.25/wind_500hpa/wind.*.nc"
# OUT_DIR = "/mnt/data/sonia/cyclone/0.25/wind_500hpa/monthly_means"

# os.makedirs(OUT_DIR, exist_ok=True)

# files = sorted(glob.glob(INPUT_GLOB))
# print(f"Found {len(files)} files")

# for f in files:
#     year = os.path.basename(f).split(".")[1]  # wind.2020.nc → 2020
#     print(f"Processing year {year}")

#     ds = xr.open_dataset(
#         f,
#         engine="netcdf4",
#         chunks={"time": 31, "lat": 90, "lon": 90}
#     )

#     # make sure time is interpreted as 6-hourly
#     correct_time = ds['time'].values[0] + pd.to_timedelta(np.arange(ds.dims['time']) * 6, unit='h')
#     ds = ds.assign_coords(time=correct_time)

#     # Monthly means for this year
#     monthly = ds.groupby("time.month").mean("time")

#     out_file = os.path.join(OUT_DIR, f"wind.monthly.{year}.nc")
#     monthly.to_netcdf(out_file)

#     ds.close()

# MONTHLY_GLOB = "/mnt/data/sonia/cyclone/0.25/wind_500hpa/monthly_means/wind.monthly.*.nc"

# files = sorted(glob.glob(MONTHLY_GLOB))
# print(f"Aggregating {len(files)} yearly monthly means")

# datasets = []
# for f in files:
#     ds = xr.open_dataset(f)
#     datasets.append(ds)

# # Stack along synthetic "year" dimension
# monthly_stack = xr.concat(datasets, dim="year")

# # Multi-year monthly climatology
# monthly_climatology = monthly_stack.mean("year")

# monthly_climatology.to_netcdf(
#     "/mnt/data/sonia/cyclone/0.25/wind_500hpa/monthly_climatology.nc"
# )
