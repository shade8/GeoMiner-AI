"""
Image augmentations for GeoMiner AI.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import settings
from src.logging_config import setup_logger

logger = setup_logger()


def get_train_transforms():
    """
    Returns training augmentations.
    """

    logger.info("Creating training transforms.")

    return A.Compose(
        [
            A.Resize(
                settings.IMAGE_SIZE,
                settings.IMAGE_SIZE,
            ),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),

            A.Normalize(),

            ToTensorV2(),
        ]
    )


def get_validation_transforms():
    """
    Validation/Test transforms.
    """

    logger.info("Creating validation transforms.")

    return A.Compose(
        [
            A.Resize(
                settings.IMAGE_SIZE,
                settings.IMAGE_SIZE,
            ),

            A.Normalize(),

            ToTensorV2(),
        ]
    )