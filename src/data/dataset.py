"""
PyTorch Dataset for GeoMiner AI.
"""

from pathlib import Path
import time
import numpy as np
import rasterio
from torch.utils.data import Dataset

from src.logging_config import setup_logger

logger = setup_logger()


class MineSegmentationDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        transforms=None,
    ):

        self.dataset_path = Path(dataset_path)

        self.transforms = transforms

        self.samples = self._discover_samples()

        logger.info(
            f"Loaded {len(self.samples)} samples from {self.dataset_path}"
        )

    def _discover_samples(self):

        samples = []

        image_files = sorted(
            self.dataset_path.glob("*_img.tif")
        )

        logger.info(
            f"Searching dataset at {self.dataset_path}"
        )

        for image_file in image_files:

            mask_file = Path(
                str(image_file).replace(
                    "_img.tif",
                    "_mask.tif",
                )
            )

            if not mask_file.exists():

                logger.warning(
                    f"Mask missing for {image_file.name}"
                )

                continue

            samples.append(
                (
                    image_file,
                    mask_file,
                )
            )

        logger.info(
            f"Discovered {len(samples)} valid samples."
        )

        return samples

    def __len__(self):

        return len(self.samples)

    def _load_image(self, image_path):

        try:

            with rasterio.open(image_path) as src:

                image = src.read()

            image = image.transpose(
                1,
                2,
                0,
            ).astype(np.float32)

            image /= 10000.0

            image = np.clip(
                image,
                0,
                1,
            )

            return image

        except Exception:

            logger.exception(
                f"Unable to load image {image_path}"
            )

            raise

    def _load_mask(self, mask_path):

        try:


            for attempt in range(3):
                try:
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                    break

                except Exception as e:

                    if attempt == 2:
                        logger.exception(
                            f"Unable to load mask {mask_path}"
                        )
                        raise

                    logger.warning(
                        f"Retry {attempt+1}/3 reading {mask_path}"
                    )

                    time.sleep(1)

            return mask.astype(np.uint8)

        except Exception:

            logger.exception(
                f"Unable to load mask {mask_path}"
            )

            raise

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = self._load_image(
            image_path
        )

        mask = self._load_mask(
            mask_path
        )

        if self.transforms:

            transformed = self.transforms(
                image=image,
                mask=mask,
            )

            image = transformed["image"]

            mask = transformed["mask"]

        return image, mask