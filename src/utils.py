import os
import numpy as np
import nibabel as nib
import torch
import cv2
from src.enums import DataDict
import logging


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def normalize_img_intensity_range(img):
    min_val, max_val = np.min(img), np.max(img)
    range = max_val - min_val
    return (img - min_val) / range

def read_image_volume(img_path, normalize=False):
    img = nib.load(img_path).get_fdata()
    if normalize:
        return normalize_img_intensity_range(img)
    else:
        return img

def image_directories_handler(population_folder, dir_type):
    image_dict = []
    nifti_extension = '.nii.gz'
    for subject in population_folder:
        sub_id = int(os.path.basename(os.path.normpath(subject)))
        flair_img = f'{subject}/pre/FLAIR{nifti_extension}'
        t1_img = f'{subject}/pre/T1{nifti_extension}'
        label = f'{subject}/wmh{nifti_extension}'
        
        subject_dict = { 
            DataDict.Id: sub_id,
            DataDict.ImageT1 : t1_img,
            DataDict.ImageFlair : flair_img,
            DataDict.Label : label,
            DataDict.CountryDirType : dir_type
        }

        image_dict.append(subject_dict)
      
    return image_dict

def save_slice(img, fname, path, to_nifti):
    if to_nifti:
        img = nib.Nifti1Image(img, affine=np.eye(4))
    else:
        img = np.uint8(img * 255)

    ext = 'nii.gz' if to_nifti else 'png'
    fout = os.path.join(path, f'{fname}.{ext}')

    if to_nifti:
        nib.save(img, fout)  
    else:
        cv2.imwrite(fout, img)
    logging.info(f'Slice saved: {fout}')

def slice_and_save_volume_image(vol, fname, path, to_nifti=True):
    (dimx, dimy, dimz) = vol.shape
    count = 0
    SLICE_X = False
    SLICE_Y = False
    SLICE_Z = True

    SLICE_DECIMATE_IDENTIFIER = 3
    if SLICE_X:
        count += dimx
        logging.info('Slicing X: ')
        for i in range(dimx):
            save_slice(vol[i, :, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path, to_nifti)
    if SLICE_Y:
        count += dimx
        logging.info('Slicing Y: ')
        for i in range(dimy):
            save_slice(vol[:, i, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path, to_nifti)
    if SLICE_Z:
        count += dimx
        logging.info('Slicing Z: ')
        for i in range(dimz):
            save_slice(vol[:, :, i], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path, to_nifti)

    return count



# Get dictionarys from a list_dict filtered by a spcific key value
def get_dicts_from_dicts(dicts, key, value):
    if isinstance(dicts[0][key], int):
        x = [item for item in dicts if item[key] in value]
    # elif isinstance(dicts[0][key], str):
    #     x = [item for item in dicts if item[key] == value]
    else:
        x = [item for item in dicts if item[key] in value]
        # x = [item for item in dicts if (any(map(item[key].__contains__, value)))]
    return x

# Get a dictionary from a list_dict filtered by a spcific key value
def get_dict_from_dicts(dicts, key, value):
    x = next(item for item in dicts if item[key] == value)
    return x

# Convert torch tensor to numpy array
def numpy_from_tensor(x):
    return x.detach().cpu().numpy()


def dict_tensor_to_value(dicts): 
    if next(iter(dicts)) is not type(torch.tensor):
        return dicts

    for keys in dicts:
        dicts[keys] = dicts[keys].item()

    return dicts

def one_hot(torch_tensor):
    return torch.where(torch_tensor > 0.5, 1, 0)

import glob

def get_interim_data_path(data_paths):
    image_dict = []
    ext_type = 'nii.gz'

    for subject in data_paths:
        subj_id = int(os.path.basename(os.path.normpath(subject)))
        
        # # Temporary fix due to dim mismatch
        # if subj_id in [50, 51]:
        #     continue

        flair_imgs = sorted(glob.glob(f'{subject}/Image/Flair/*.{ext_type}'))
        t1_imgs = sorted(glob.glob(f'{subject}/Image/T1/*.{ext_type}'))
        labels = sorted(glob.glob(f'{subject}/Label/*.{ext_type}'))

        if len(flair_imgs) != len(t1_imgs):
            assert ValueError("Flair and T1 size mismatch")
        
        for i in range(len(flair_imgs)):
            slice_dict = {
                DataDict.Id: subj_id,
                DataDict.ImageFlair: flair_imgs[i],
                DataDict.ImageT1: t1_imgs[i],
                DataDict.Label: labels[i],
                DataDict.DepthZ: i
            }

            image_dict.append(slice_dict)

    return image_dict