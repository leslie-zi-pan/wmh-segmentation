from fastapi import APIRouter, UploadFile
from nibabel import Nifti1Image
from src.models.train_model import model

from src.models.predict_model import ImagePredictor

from monai.data import Dataset, DataLoader
from src.features.build_features import test_transform

router = APIRouter()


async def segment(form_data: UploadFile):
    form_data = await form_data.read()

    img = Nifti1Image.from_bytes(form_data)

    test_dataset = Dataset(val_paths, test_transform)
    img_predictor = ImagePredictor(model, test_dataset)
    test_predictions = img_predictor.predict_handler()
    print('DONE')
    return img.get_fdata().shape


@router.post("/segment/")
async def extract_wmh(file: UploadFile):
    """
    This function returns the segmented image of the brain image.

    Args:
        mri_brain_image: The MRI Brain Image to be segmented.

    Returns:
        The segmented image of the brain image.
    """

    return await segment(file)
