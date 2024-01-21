from typing import Any
from fastapi import UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from nibabel import Nifti1Image
from src.enums import DataDict
from src.models.predict_model import ImagePredictor
from src.pytorch_utils import normalize_img_intensity_range, slice_tensor_volume
from src.settings.config import get_app_settings
from src.models.train_model import model
from src.features.build_features import serving_transform
from monai.data import Dataset
from pathlib import Path


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

    def predict(self, t1, flair):
        t1 = normalize_img_intensity_range(t1)

        flair = normalize_img_intensity_range(flair)

        t1_slices = slice_tensor_volume(t1)
        flair_slices = slice_tensor_volume(flair)
        inference_dict = [
            {
                DataDict.Id: -1,
                DataDict.ImageT1: v,
                DataDict.ImageFlair: flair_slices.get(idx),
                DataDict.DepthZ: idx,
            }
            for idx, v in t1_slices.items()
        ]

        test_dataset = Dataset(inference_dict, serving_transform)
        img_predictor = ImagePredictor(self.model, test_dataset)
        test_predictions = img_predictor.predict_handler()
        return {
            DataDict.Prediction: test_predictions[0][DataDict.Prediction],
        }

    async def __call__(self, brain_mri_image: list[UploadFile]) -> TestMode | None:
        # async def __call__(self, brain_mri_image: UploadFile) -> TestMode | None:
        try:
            form_data_t1 = await brain_mri_image[0].read()
            form_data_flair = await brain_mri_image[1].read()

            img_t1 = Nifti1Image.from_bytes(form_data_t1)
            img_flair = Nifti1Image.from_bytes(form_data_flair)

            image_path = Path("reports/figures/test_image.png") 
            # return FileResponse(image_path, media_type="image/png")

            return {
                "t1_shape": img_t1.get_fdata().shape,
                "flair_shape": img_flair.get_fdata().shape,
                "test": "hello leslie youre brilliant NOT",
                "pic": FileResponse(image_path, media_type="image/png")
            }
        except Exception as e:
            return None
