from ast import Str
from fastapi import APIRouter
from pydantic import BaseModel
from monai.data import Dataset
import torch
from src.models.train_model import model
from src.models.predict_model import ImagePredictor
from src.features.build_features import test_transform

from src.settings.config import get_app_settings

from src.utils import create_temp_file

router = APIRouter()


class MRIBrainImage(BaseModel):
    """
    This class represents the MRI Cerebral Images to be segmented

    Args:
        ImageT1: str image of the T1 Image
        ImageFlair: str image of the FLAIR Image
    """

    ImageT1: str
    ImageFlair: str


async def infer(brain_image: MRIBrainImage):
    image_t1 = brain_image.ImageT1
    image_flair = brain_image.ImageFlair

    # Decode them and create temp file location
    image_t1 = str.decode(image_t1)
    image_flair = str.decode(image_flair)

    image_t1_fp = create_temp_file(image_t1)
    image_flair_fp = create_temp_file(image_flair)

    pred_network = model
    checkpoint = torch.load(get_app_settings().inference_model_fp, map_location=torch.device(get_app_settings().device))
    pred_network.load_state_dict(checkpoint['model_state_dict'])

    dataset = Dataset(val_paths, test_transform)
    img_predictor = ImagePredictor(pred_network, dataset)
    test_predictions = img_predictor.predict_handler()
    print('DONE')

    return image_t1, image_flair


async def segment(brain_image: MRIBrainImage):
    return "hello"


@router.post("/segment/")
async def extract_wmh(mri_brain_image: MRIBrainImage):
    """
    This function returns the segmented image of the brain image.

    Args:
        mri_brain_image: The MRI Brain Image to be segmented.

    Returns:
        The segmented image of the brain image.
    """

    return await segment(mri_brain_image)