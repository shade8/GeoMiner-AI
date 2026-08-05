"""
Model Factory.
"""

from src.config import settings
from src.logging_config import setup_logger
from src.models.segformer import GeoMinerSegFormer

logger = setup_logger()


def build_model():

    logger.info(
        f"Building model : {settings.MODEL_NAME}"
    )

    model = GeoMinerSegFormer()

    logger.info(
        f"Total Parameters : {model.num_parameters():,}"
    )

    logger.info(
        f"Trainable Parameters : {model.trainable_parameters():,}"
    )

    return model