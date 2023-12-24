from monai.transforms import MapTransform
from monai.transforms.utils import resize_center
import numpy as np
import torch
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    MapTransform,
    Resized,
    Spacingd,
    ToTensord,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Rand2DElasticd,
    RandZoomd,
)

from src.enums import DataDict
from src.pytorch_utils import normalize_img_intensity_range


class ImagesToMultiChannel(MapTransform):
    def __call__(self, data):
        to_concat_list = []
        for key in self.keys:
            to_concat_list.append(data[key])

        data[DataDict.Image] = torch.cat(to_concat_list, dim=0)
        return data


class Unsqueeze(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.unsqueeze(data[key], dim=0)

        return data


class TestShape(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            print(f"{key}: \t{data[key].shape}")

        return data


class ResizeCentre(MapTransform):
    def __init__(self, keys, dims):
        super().__init__(keys=keys)
        self.dims = dims

    def __call__(self, data):
        print(f"dimensions are {self.dims}")
        for key in self.keys:
            print(f"before {data[key].shape}")
            data[key] = resize_center(data[key], *self.dims)
            print(f"after {data[key].shape}")

        return data


# normalize_img_intensity_range
class NormalizeIntensitydCustom(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key] = normalize_img_intensity_range(data[key])

        return data


class ConvertToMultiChannelBasedOnLabelsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the WMH

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []

            d[key] = np.squeeze(d[key])

            result.append(np.logical_or(d[key] == 0, d[key] == 2))
            # Other labels and background merge as backgrounds
            result.append(d[key] == 1)

            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


train_transform = Compose(
    [
        LoadImaged(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        EnsureChannelFirstd(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=[DataDict.Label]),
        Spacingd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Resized(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            spatial_size=[256, 256],
        ),
        Orientationd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label], axcodes="RAS"
        ),
        RandFlipd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            prob=0.5,
            spatial_axis=0,
        ),
        Rand2DElasticd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            spacing=(30, 40),
            magnitude_range=(0.8, 1.2),
            prob=0.3,
        ),
        RandScaleIntensityd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair], factors=0.1, prob=0.5
        ),
        RandShiftIntensityd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair], offsets=0.1, prob=0.5
        ),
        RandZoomd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            prob=1.0,
            min_zoom=0.9,
            max_zoom=1.1,
        ),
        ToTensord(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ImagesToMultiChannel(keys=[DataDict.ImageT1, DataDict.ImageFlair]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        EnsureChannelFirstd(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=[DataDict.Label]),
        Spacingd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Resized(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            spatial_size=[256, 256],
        ),
        Orientationd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label], axcodes="RAS"
        ),
        ToTensord(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ImagesToMultiChannel(keys=[DataDict.ImageT1, DataDict.ImageFlair]),
    ]
)

test_transform = Compose(
    [
        LoadImaged(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        EnsureChannelFirstd(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=[DataDict.Label]),
        Spacingd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Resized(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label],
            spatial_size=[256, 256],
        ),
        Orientationd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label], axcodes="RAS"
        ),
        ToTensord(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        ImagesToMultiChannel(keys=[DataDict.ImageT1, DataDict.ImageFlair]),
    ]
)
