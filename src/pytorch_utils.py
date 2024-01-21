import os
import numpy as np
import nibabel as nib
import torch
from torch import Tensor
import cv2
from src.enums import DataDict
import glob

from loguru import logger


def normalize_img_intensity_range(img: Tensor):
    """Normalizes the intensity range of an image.

    Args:
        img (Tensor): Input image of shape (B, C, H, W).

    Returns:
        Tensor: Normalized image of shape (B, C, H, W).
    """

    min_val, max_val = np.min(img), np.max(img)
    range = max_val - min_val
    return (img - min_val) / range


def read_image_volume(img_path, normalize=False):
    """Reads a 3D image volume from a file.

    Args:
        img_path (str): Path to the image file.
        normalize (bool): Whether to normalize the image intensity range.

    Returns:
        Tensor: Image volume of shape (C, H, W).
    """

    img = nib.load(img_path).get_fdata()
    
    if normalize:
        return normalize_img_intensity_range(img)
    else:
        return img


def image_directories_handler(population_folder, dir_type):
    """Creates a list of dictionaries containing the paths to the images in a given directory.

    Args:
        population_folder (str): Path to the population folder.
        dir_type (str): Type of directory (e.g., "training", "testing").

    Returns:
        list: List of dictionaries containing the paths to the images.
    """

    image_dict = []
    nifti_extension = ".nii.gz"
    for subject in population_folder:
        sub_id = int(os.path.basename(os.path.normpath(subject)))
        flair_img = f"{subject}/pre/FLAIR{nifti_extension}"
        t1_img = f"{subject}/pre/T1{nifti_extension}"
        label = f"{subject}/wmh{nifti_extension}"

        subject_dict = {
            DataDict.Id: sub_id,
            DataDict.ImageT1: t1_img,
            DataDict.ImageFlair: flair_img,
            DataDict.Label: label,
            DataDict.CountryDirType: dir_type,
        }

        image_dict.append(subject_dict)

    return image_dict


def save_slice(img: Tensor, fname: str, path: str, to_nifti: bool) -> None:
    """Saves a slice of an image to a file.

    Args:
        img (Tensor): Image slice of shape (C, H, W).
        fname (str): Filename for the saved image.
        path (str): Path to the directory where the image will be saved.
        to_nifti (bool): Whether to save the image as a NIFTI file.
    """

    if to_nifti:
        img = nib.Nifti1Image(img, affine=np.eye(4))
    else:
        img = np.uint8(img * 255)

    ext = "nii.gz" if to_nifti else "png"
    fout = os.path.join(path, f"{fname}.{ext}")

    if to_nifti:
        nib.save(img, fout)
    else:
        cv2.imwrite(fout, img)
    logger.info(f"Slice saved: {fout}")


def slice_tensor_volume(vol: Tensor, axis: 0 | 1 | 2 = 2) -> dict | None:
    """Slice tensor volume

    Args:
        vol (Tensor): Image volume of shape (C, H, W).
        axis: Axis to slice volume image by.
    """
    slice_length = vol.shape[axis]

    result = {}
    for slice_idx in range(slice_length):
        match axis:
            case 0:
                result[slice_idx] = vol[slice_idx, ...]
            case 1:
                result[slice_idx] = vol[:, slice_idx, :]
            case 2:
                result[slice_idx] = vol[..., slice_idx]

    return result or None


def slice_volume_handler(
    vol: Tensor,
    fname: str,
    axis_to_slice: list[int] = [2],
    decimate: int = 3,
):
    """Slices a 3D image volume and saves each slice to a file.

    Args:
        vol (Tensor): Image volume of shape (C, H, W).
        fname (str): Filename for the saved slices.
        path (str): Path to the directory where the slices will be saved.
        to_nifti (bool): Whether to save the slices as NIFTI files.
    """

    image_slices = {}
    for axis in axis_to_slice:
        logger.info(f"Slicing along axis {axis}")
        cur_axis_slices = slice_tensor_volume(vol, axis)

        for slice_idx, slice_image in cur_axis_slices.items():
            slice_fname = f"{fname}-slice{str(slice_idx).zfill(decimate)}_axis{str(axis)}"
            image_slices[slice_fname] = slice_image

    return image_slices


def slice_and_save_volume_image_handler(
        vol: Tensor, fname: str, path: str,
        to_nifti: bool = True):
    """Slices a 3D image volume and saves each slice to a file.

    Args:
        vol (Tensor): Image volume of shape (C, H, W).
        fname (str): Filename for the saved slices.
        path (str): Path to the directory where the slices will be saved.
        to_nifti (bool): Whether to save the slices as NIFTI files.
    """
    logger.info("Slicing volume tensors")
    image_slices = slice_volume_handler(vol, fname)
    slices_count = 0

    for slice_fname, slice_image in image_slices.items():
        logger.info(f"Saving slice: {slice_fname}")
        save_slice(
                slice_image,
                slice_fname,
                path,
                to_nifti,
            )
        slices_count += 1

    return slices_count


def slice_and_save_volume_image(
    vol: Tensor, fname: str, path: str, to_nifti: bool = True
):
    """Slices a 3D image volume and saves each slice to a file.

    Args:
        vol (Tensor): Image volume of shape (C, H, W).
        fname (str): Filename for the saved slices.
        path (str): Path to the directory where the slices will be saved.
        to_nifti (bool): Whether to save the slices as NIFTI files.
    """

    (dimx, dimy, dimz) = vol.shape
    count = 0
    SLICE_X = False
    SLICE_Y = False
    SLICE_Z = True

    SLICE_DECIMATE_IDENTIFIER = 3
    if SLICE_X:
        count += dimx
        logger.info("Slicing X: ")
        for i in range(dimx):
            save_slice(
                vol[i, :, :],
                fname + f"-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x",
                path,
                to_nifti,
            )
    if SLICE_Y:
        count += dimx
        logger.info("Slicing Y: ")
        for i in range(dimy):
            save_slice(
                vol[:, i, :],
                fname + f"-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y",
                path,
                to_nifti,
            )
    if SLICE_Z:
        count += dimx
        logger.info("Slicing Z: ")
        for i in range(dimz):
            save_slice(
                vol[:, :, i],
                fname + f"-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z",
                path,
                to_nifti,
            )

    return count


# Get dictionarys from a list_dict filtered by a spcific key value
def get_dicts_from_dicts(dicts, key, value):
    """Gets a list of dictionaries from a list of dictionaries filtered by a specific key value.

    Args:
        dicts (list): List of dictionaries.
        key (str): Key to filter by.
        value (str or int): Value to filter by.

    Returns:
        list: List of dictionaries filtered by the specified key value.
    """

    if isinstance(dicts[0][key], int):
        x = [item for item in dicts if item[key] in value]
    # elif isinstance(dicts[0][key], str):
    #     x = [item for item in dicts if item[key] == value]
    else:
        x = [item for item in dicts if (any(map(item[key].__contains__, value)))]
    return x


# Get a dictionary from a list_dict filtered by a spcific key value
def get_dict_from_dicts(dicts, key, value):
    """Gets a dictionary from a list of dictionaries filtered by a specific key value.

    Args:
        dicts (list): List of dictionaries.
        key (str): Key to filter by.
        value (str or int): Value to filter by.

    Returns:
        dict: Dictionary filtered by the specified key value.
    """

    x = next(item for item in dicts if item[key] == value)
    return x


# Convert torch tensor to numpy array
def numpy_from_tensor(x):
    """Converts a torch tensor to a numpy array.

    Args:
        x (Tensor): Torch tensor.

    Returns:
        numpy.ndarray: Numpy array.
    """

    return x.detach().cpu().numpy()


def dict_tensor_to_value(dicts):
    """Converts a dictionary of tensors to a dictionary of values.

    Args:
        dicts (dict): Dictionary of tensors.

    Returns:
        dict: Dictionary of values.
    """

    if next(iter(dicts)) is not type(torch.tensor):
        return dicts

    for keys in dicts:
        dicts[keys] = dicts[keys].item()

    return dicts


def one_hot(torch_tensor):
    """Converts a torch tensor to a one-hot encoded tensor.

    Args:
        torch_tensor (Tensor): Torch tensor.

    Returns:
        Tensor: One-hot encoded tensor.
    """

    return torch.where(torch_tensor > 0.5, 1, 0)


def get_interim_data_path(data_paths):
    """Gets the paths to the interim data.

    Args:
        data_paths (list): List of paths to the data.

    Returns:
        list: List of paths to the interim data.
    """

    image_dict = []
    ext_type = "nii.gz"

    for subject in data_paths:
        subj_id = int(os.path.basename(os.path.normpath(subject)))

        # # Temporary fix due to dim mismatch
        # if subj_id in [50, 51]:
        #     continue

        flair_imgs = sorted(glob.glob(f"{subject}/Image/Flair/*.{ext_type}"))
        t1_imgs = sorted(glob.glob(f"{subject}/Image/T1/*.{ext_type}"))
        labels = sorted(glob.glob(f"{subject}/Label/*.{ext_type}"))

        if len(flair_imgs) != len(t1_imgs):
            assert ValueError("Flair and T1 size mismatch")

        for i in range(len(flair_imgs)):
            slice_dict = {
                DataDict.Id: subj_id,
                DataDict.Image: t1_imgs[i],
                DataDict.ImageFlair: flair_imgs[i],
                DataDict.ImageT1: t1_imgs[i],
                DataDict.Label: labels[i],
                DataDict.DepthZ: i,
            }

            image_dict.append(slice_dict)

    return image_dict
