import os
import random

import numpy as np
import torch

from src.logging_config import setup_logger

logger = setup_logger()


def set_seed(seed: int):

    """
    Set all random seeds for reproducibility.
    """

    try:

        logger.info(f"Setting random seed : {seed}")

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        logger.info("Random seed initialized successfully.")

    except Exception:

        logger.exception("Failed to initialize random seed.")

        raise