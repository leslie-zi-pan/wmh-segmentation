from monai.transforms import MapTransform
from monai.transforms.utils import resize_center
import numpy as np
import torch

from src.enums import DataDict
from src.pytorch_utils import normalize_img_intensity_range


class ImagesToMultiChannel(MapTransform):
    """Concatenate all images in the data dict into a single multi-channel image.

    Args:
        keys (str or list of str): Keys of the images to concatenate.
    """

    def __init__(self, keys: str | list[str]):
        super().__init__(keys=keys)

    def __call__(self, data: dict) -> dict:
        """Concatenate all images in the data dict into a single multi-channel image.

        Args:
            data (dict): The data dict containing the images to concatenate.

        Returns:
            dict: The data dict with the concatenated image added.
        """
        to_concat_list = []
        for key in self.keys:
            to_concat_list.append(data[key])

        data[DataDict.Image] = torch.cat(to_concat_list, dim=0)
        return data


class Unsqueeze(MapTransform):
    """Unsqueeze a dimension of a tensor.

    Args:
        keys (str or list of str): Keys of the tensors to unsqueeze.
        dim (int): The dimension to unsqueeze.
    """

    def __init__(self, keys: str | list[str], dim: int):
        super().__init__(keys=keys)
        self.dim = dim

    def __call__(self, data: dict) -> dict:
        """Unsqueeze a dimension of a tensor.

        Args:
            data (dict): The data dict containing the tensors to unsqueeze.

        Returns:
            dict: The data dict with the tensors unsqueezed.
        """
        for key in self.keys:
            data[key] = torch.unsqueeze(data[key], dim=self.dim)

        return data


class TestShape(MapTransform):
    """Print the shape of a tensor.

    Args:
        keys (str or list of str): Keys of the tensors to print the shape of.
    """

    def __init__(self, keys: str | list[str]):
        super().__init__(keys=keys)

    def __call__(self, data: dict) -> dict:
        """Print the shape of a tensor.

        Args:
            data (dict): The data dict containing the tensors to print the shape of.

        Returns:
            dict: The data dict.
        """
        for key in self.keys:
            print(f"{key}: \t{data[key].shape}")

        return data


class ResizeCentre(MapTransform):
    """Resize an image to a specified size, keeping the center of the image intact.

    Args:
        keys (str or list of str): Keys of the images to resize.
        dims (tuple): The size to resize the images to.
    """

    def __init__(self, keys: str | list[str], dims: tuple[int, int]):
        super().__init__(keys=keys)
        self.dims = dims

    def __call__(self, data: dict) -> dict:
        """Resize an image to a specified size, keeping the center of the image intact.

        Args:
            data (dict): The data dict containing the images to resize.

        Returns:
            dict: The data dict with the resized images.
        """
        print(f"dimensions are {self.dims}")
        for key in self.keys:
            print(f"before {data[key].shape}")
            data[key] = resize_center(data[key], *self.dims)
            print(f"after {data[key].shape}")

        return data


# normalize_img_intensity_range
class NormalizeIntensitydCustom(MapTransform):
    """Normalize the intensity range of an image.

    Args:
        keys (str or list of str): Keys of the images to normalize.
    """

    def __init__(self, keys: str | list[str]):
        super().__init__(keys=keys)

    def __call__(self, data: dict) -> dict:
        """Normalize the intensity range of an image.

        Args:
            data (dict): The data dict containing the images to normalize.

        Returns:
            dict: The data dict with the normalized images.
        """
        for key in self.keys:
            data[key] = normalize_img_intensity_range(data[key])

        return data


class ConvertToMultiChannelBasedOnLabelsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the WMH

    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        for key in self.keys:
            result = []

            d[key] = np.squeeze(d[key])

            result.append(np.logical_or(d[key] == 0, d[key] == 2))
            # Other labels and background merge as backgrounds
            result.append(d[key] == 1)

            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d

from monai.transforms import ToTensor


class CustomTransform(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the WMH

    """

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            data[key] = ToTensor(data[key])

        return data