from typing import Annotated
from fastapi import APIRouter, Depends
from src.dependencies.segment import Segment
from src.models.train_model import model

router = APIRouter()


segment_dependency = Segment(model)
@router.post("/segment/")
async def extract_wmh(image_shape: Annotated[dict, Depends(segment_dependency)]):
    """
    This function returns the segmented image of the brain image.

    Args:
        mri_brain_image: The MRI Brain Image to be segmented.

    Returns:
        The segmented image of the brain image.
    """

    return image_shape
