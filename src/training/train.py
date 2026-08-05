"""
Main training entry point.
"""

from src.config import settings
from src.data.dataloader import create_dataloaders
from src.logging_config import setup_logger
from src.models.model_factory import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed

logger = setup_logger()


def main():

    logger.info("=" * 80)
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info("=" * 80)

    logger.info(
        f"Setting random seed to {settings.RANDOM_SEED}"
    )

    set_seed(settings.RANDOM_SEED)

    logger.info("Creating dataloaders...")

    train_loader, val_loader, _ = create_dataloaders()

    logger.info("Building model...")

    model = build_model()

    logger.info("Creating trainer...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.fit()

    logger.info("=" * 80)
    logger.info("Training finished successfully.")
    logger.info("=" * 80)


if __name__ == "__main__":

    main()