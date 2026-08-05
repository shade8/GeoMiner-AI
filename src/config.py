from functools import lru_cache
from pathlib import Path
import torch
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    GeoMiner AI configuration.
    """

    # -------------------------------------------------
    # Application
    # -------------------------------------------------

    APP_NAME: str = Field(default="GeoMiner AI")

    APP_VERSION: str = Field(default="1.0.0")

    LOG_LEVEL: str = Field(default="INFO")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    TRAIN_DATASET: str = Field(...)

    VAL_DATASET: str = Field(...)

    TEST_DATASET: str = Field(...)

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    RANDOM_SEED: int = Field(default=42)

    IMAGE_SIZE: int = Field(default=512)

    NUM_CLASSES: int = Field(default=2)

    BATCH_SIZE: int = Field(default=4)

    NUM_WORKERS: int = Field(default=0)

    EPOCHS: int = Field(default=30)

    EARLY_STOPPING_PATIENCE: int = Field(default=5)

    LEARNING_RATE: float = Field(default=1e-4)

    WEIGHT_DECAY: float = Field(default=1e-5)

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME: str = Field(
        default="nvidia/segformer-b0-finetuned-ade-512-512"
    )

    # -------------------------------------------------
    # Project Directories
    # -------------------------------------------------

    CHECKPOINT_DIR: str = str(PROJECT_ROOT / "checkpoints")

    OUTPUT_DIR: str = str(PROJECT_ROOT / "outputs")

    LOG_DIR: str = str(PROJECT_ROOT / "logs")

    class Config:

        env_file = ".env"

        extra = "ignore"


@lru_cache
def get_settings():

    return Settings()


settings = get_settings()