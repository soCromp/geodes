import shutil 
import sys 
import os

from_rand = False 


# assumption: path goes to some directory where 'rand' is a subdirectory if from_rand. 
# else, path goes to a directory where all contents should get reorganized and moved into a 'date' subdirectory.
path = sys.argv[-1]

os.makedirs(os.path.join(path, 'date'), exist_ok=True)



if from_rand:
    for d in os.listdir(os.path.join(path, 'rand')):
        if not os.path.isdir(os.path.join(path, 'rand', d)): # e.g. channels.txt
            shutil.copyfile(os.path.join(path, 'rand', d), os.path.join(path, 'date', d))
        else: # a basin
            os.makedirs(os.path.join(path, 'date', d)) # make corresponding basin directory
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(path, 'date', d, split)) # make corresponding split directory
                
            for split in os.listdir(os.path.join(path, 'rand', d)):
                print(split)
                if not os.path.isdir(os.path.join(path, 'rand', d, split)): # readme or something
                    shutil.copyfile(os.path.join(path, 'rand', d, split),
                                    os.path.join(path, 'date', d, split))
                    continue
                
                for storm in os.listdir(os.path.join(path, 'rand', d, split)):
                    year = int(storm[:4])
                    trimester = int(storm[4:6])
                    if year <=2014 or (year == 2015 and trimester < 3):
                        shutil.copytree(os.path.join(path, 'rand', d, split, storm),
                                        os.path.join(path, 'date', d, 'train', storm))
                    else:
                        shutil.copytree(os.path.join(path, 'rand', d, split, storm),
                                        os.path.join(path, 'date', d, 'test', storm))
                        if year == 2015 and trimester == 3:
                            shutil.copytree(os.path.join(path, 'rand', d, split, storm),
                                            os.path.join(path, 'date', d, 'val', storm))

else:
    items = os.listdir(path)
    for item in items:
        if item == 'date': continue # empty
        if not os.path.isdir(os.path.join(path, item)): # e.g. channels.txt
            shutil.move(os.path.join(path, item), os.path.join(path, 'date', item))
        else: # a basin
            shutil.move(os.path.join(path, item), os.path.join(path, 'date', item))
            storms = os.listdir(os.path.join(path, 'date', item))
            splits = ['train', 'val', 'test']
            for split in splits:
                os.makedirs(os.path.join(path, 'date', item, split), exist_ok=True)
            for storm in storms:
                if not os.path.isdir(os.path.join(path, 'date', item, storm)): # readme or something
                    continue
                year = int(storm[:4])
                trimester = int(storm[4:6])
                if year <=2014 or (year == 2015 and trimester < 3):
                    shutil.move(os.path.join(path, 'date', item, storm),
                                os.path.join(path, 'date', item, 'train', storm))
                else:
                    shutil.move(os.path.join(path, 'date', item, storm),
                                os.path.join(path, 'date', item, 'test', storm))
                    if year == 2015 and trimester == 3:
                        shutil.copytree(os.path.join(path, 'date', item, 'test', storm),
                                    os.path.join(path, 'date', item, 'val', storm))
                        