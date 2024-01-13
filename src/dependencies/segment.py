from typing import Any
from fastapi import UploadFile
from pydantic import BaseModel
import torch
from nibabel import Nifti1Image
from src.settings.config import get_app_settings


class TestMode(BaseModel):
    shape: list[int]
    test: str


class Segment:
    def __init__(self, model) -> None:
        self.model = model
        checkpoint_path = get_app_settings().inference_model_fp
        device = get_app_settings().device
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    async def __call__(self, brain_mri_image: UploadFile) -> TestMode | None:
        try:
            form_data = await brain_mri_image.read()

            img = Nifti1Image.from_bytes(form_data)

            return {
                "shape": img.get_fdata().shape,
                "test": "hello leslie youre brilliant NOT"
            }
        except Exception as e:
            return None
