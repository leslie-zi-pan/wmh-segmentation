from fastapi import APIRouter
from pydantic import BaseModel

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


async def segment(raw_text):
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