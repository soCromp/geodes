import os
import numpy as np 

clima_dir = '/mnt/data/sonia/clima_patches/date/'
outdir = os.path.join(clima_dir, 'natlantic-multivar-0.25')
timesteps = 8
varnames = []
for var in ['slp', 'wind_500hpa', 'temperature', 'humidity']:
    varnames.append(f'natlantic-{var}-0.25')
    
for sid in os.listdir(os.path.join(clima_dir, varnames[0], 'test')):
    siddir = os.path.join(outdir, 'test', sid)
    os.makedirs(siddir, exist_ok=True)
    
    for t in range(timesteps):
        datas = [np.load(os.path.join(clima_dir, varname, 'test', sid, f'{t}.npy')) for varname in varnames]
        datas = [np.expand_dims(data, -1) if data.ndim == 2 else data for data in datas]
        data = np.concatenate(datas, axis=-1)
        # print(data.shape)
        np.save(os.path.join(siddir, f'{t}.npy'), data)
    

