"""
Creates PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

from src.config import settings
from src.data.dataset import MineSegmentationDataset
from src.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)

from src.logging_config import setup_logger

logger = setup_logger()


def create_dataloaders():

    logger.info("Creating datasets...")

    train_dataset = MineSegmentationDataset(
        settings.TRAIN_DATASET,
        get_train_transforms(),
    )

    val_dataset = MineSegmentationDataset(
        settings.VAL_DATASET,
        get_validation_transforms(),
    )

    test_dataset = MineSegmentationDataset(
        settings.TEST_DATASET,
        get_validation_transforms(),
    )

    logger.info("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.DEVICE == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.DEVICE == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.DEVICE == "cuda",
    )

    logger.info("Dataloaders created successfully.")

    return (
        train_loader,
        val_loader,
        test_loader,
    )