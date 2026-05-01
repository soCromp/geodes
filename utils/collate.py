import os 
import tarfile
import shutil

work_dir = '/mnt/data/sonia/geodes-samples/multivar/date'
out_name = '3d_h25.2d_h1_6008421_sample_natlantic_test'
in_names = sorted([f for f in os.listdir(work_dir) if f.startswith(out_name) and f.endswith('.tar.gz')])
print('Found', len(in_names), 'files:\n', in_names)
# in_names = [f'3d_l1.h25_6008482_sample_eta1.0_natlantic_test_6045368_{i}.tar.gz' for i in range(15)]

os.makedirs(os.path.join(work_dir, out_name), exist_ok=True)

for in_name in in_names:
    print(in_name)
    with tarfile.open(os.path.join(work_dir, in_name), "r:gz") as tar:
        names = tar.getnames()
        print(os.path.commonprefix(names))
        name = os.path.commonprefix(names).split('/')[1]
        tar.extractall(path=os.path.join(work_dir, 'tmp'))
        
    for sample in os.listdir(os.path.join(work_dir, 'tmp', name, 'samples')):
        try:
            shutil.move(os.path.join(work_dir, 'tmp', name, 'samples', sample), os.path.join(work_dir, out_name), )
        except Exception as e:
            continue

shutil.rmtree(os.path.join(work_dir, 'tmp'))

print(os.path.join(work_dir, out_name), 'now contains', len(os.listdir(os.path.join(work_dir, out_name))), 'samples')
