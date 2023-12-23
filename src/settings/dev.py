import logging

from src.settings.app import AppSettings


class DevAppSettings(AppSettings):
    debug: bool = True

    title: str = "MRI White Matter Hyperintensity Segmentation Dev application"

    logging_level: int = logging.DEBUG