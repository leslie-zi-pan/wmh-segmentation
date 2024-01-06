from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
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
from src.features.transforms import (
    ConvertToMultiChannelBasedOnLabelsClassesd,
    ImagesToMultiChannel,
)


train_transform = Compose(
    [
        LoadImaged(keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]),
        EnsureChannelFirstd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]
        ),
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
        EnsureChannelFirstd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]
        ),
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
        EnsureChannelFirstd(
            keys=[DataDict.ImageT1, DataDict.ImageFlair, DataDict.Label]
        ),
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
