import logging

from src.settings.app import AppSettings


class TestAppSettings(AppSettings):
    debug: bool = True

    title: str = "MRI White Matter Hyperintensity Segmentation Test Application"

    logging_level: int = logging.DEBUG