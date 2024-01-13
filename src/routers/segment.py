from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
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
    if not image_shape:
        raise HTTPException(status_code=400, detail=[
            "There was an issue reading and processing the data provided", 
            "You must send through .nii or .png file only"])

    return image_shape
