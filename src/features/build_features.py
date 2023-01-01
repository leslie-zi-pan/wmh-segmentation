
from monai.transforms import (
    MapTransform
)
from monai.transforms.utils import resize_center
import numpy as np 
import torch

from src.enums import DataDict
from src.utils import normalize_img_intensity_range

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
            print(f'{key}: \t{data[key].shape}')
            
        return data

class ResizeCentre(MapTransform):
    def __init__(self, keys, dims):
        super().__init__(keys=keys)
        self.dims = dims

    def __call__(self, data):
        print(f'dimensions are {self.dims}')
        for key in self.keys:
            print(f'before {data[key].shape}')
            data[key] = resize_center(data[key], *self.dims)
            print(f'after {data[key].shape}')
            
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

            # Other labels and background merge as backgrounds
            result.append(np.logical_or(d[key] == 0, d[key] == 2))
            # result.append(d[key] == 0)
            result.append(d[key] == 1)

            d[key] = np.stack(result, axis=0).astype(np.float32)
            
        return d