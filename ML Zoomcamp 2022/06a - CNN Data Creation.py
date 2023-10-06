import numpy as np
import os
import pandas as pd
import pathlib
import shutil

# read in the data
root_dir = r'F:\Data_Files_Scripts\PycharmProjects\ml_zoomcamp_2022\Data'
data_dir = pathlib.Path(os.path.join(root_dir, 'all_animal_images')).with_suffix('')

# rename the folders
folder_names = os.listdir(data_dir)
new_folder_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
for of, nf in zip(folder_names, new_folder_names):
    if not os.path.exists(os.path.join(data_dir, nf)):
        os.rename(os.path.join(data_dir, of), os.path.join(data_dir, nf))
    else:
        pass

# count the number of images
all_image_count = len(list(data_dir.glob('*/*.jpeg')))
print(all_image_count)

count_list = []
for folder in os.listdir(data_dir):
    num_images = len(list(data_dir.glob(folder+'/*')))
    print('{name}: {img_count}'.format(name=folder, img_count=num_images))
    count_list.append((folder, num_images))

# make a new folder with an equal number of images of each class
new_dir = os.path.join(root_dir, 'equal_animal_images')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
else:
    pass

for folder in new_folder_names:
    if not os.path.exists(os.path.join(new_dir, folder)):
        os.makedirs(os.path.join(new_dir, folder))
    else:
        pass

count_df = pd.DataFrame(count_list, columns=['animal', 'count'])
num_images = count_df['count'].min()

for folder in new_folder_names:
    src_dir = os.path.join(data_dir, folder)
    all_images = os.listdir(src_dir)
    # shuffle just in case
    np.random.shuffle(all_images)
    # take the first num_files images
    images = os.listdir(src_dir)[:num_images]
    dest_dir = os.path.join(new_dir, folder)
    for i in images:
        shutil.copy(src_dir+'\\'+i, dest_dir)

# create a third folder with only a subset of the balanced image dataset
new_dir = os.path.join(root_dir, 'animal_images_small')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
else:
    pass

data_sets = ['train', 'val', 'test']
for d in data_sets:
    if not os.path.exists(os.path.join(new_dir, d)):
        os.makedirs(os.path.join(new_dir, d))
        new_ddir = os.path.join(new_dir, d)
        for folder in new_folder_names:
            os.makedirs(os.path.join(new_ddir, folder))
    else:
        pass

for folder in new_folder_names:
    src_dir = os.path.join(root_dir, 'equal_animal_images', folder)
    src_images = os.listdir(src_dir)
    train_images, val_images, test_images, _ = np.split(np.array(src_images),
                                                        [int(len(src_images)*0.21), int(len(src_images)*0.234),
                                                         int(len(src_images)*0.258)])
    for image in train_images:
        dest_dir = os.path.join(new_dir, 'train', folder)
        shutil.copy(src_dir+'\\'+image, dest_dir)
    for image in val_images:
        dest_dir = os.path.join(new_dir, 'val', folder)
        shutil.copy(src_dir+'\\'+image, dest_dir)
    for image in test_images:
        dest_dir = os.path.join(new_dir, 'test', folder)
        shutil.copy(src_dir+'\\'+image, dest_dir)
