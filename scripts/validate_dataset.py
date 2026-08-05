"""
Validate the GeoMiner dataset.

Usage
-----

Validate only:

    python scripts/validate_dataset.py

Validate and automatically fix problems:

    python scripts/validate_dataset.py --fix
"""

from pathlib import Path
import argparse
import pandas as pd
import rasterio
from src.logging_config import setup_logger

logger = setup_logger()
DATASET_ROOT = Path(
    r"C:\Users\akhil\Downloads\mine-segmentation-main\mine-segmentation-main\data\processed\files"
)

report = []


def remove_file(path: Path):

    if path.exists():

        path.unlink()

        logger.info(f"Deleted file: {path}")

        print(f"Deleted: {path.name}")


def validate_split(split: str, fix: bool):

    split_path = DATASET_ROOT / split
    logger.info("=" * 80)
    logger.info(f"Validating dataset split: {split}")
    logger.info("=" * 80)

    image_files = sorted(split_path.glob("*_img.tif"))
    mask_files = sorted(split_path.glob("*_mask.tif"))
    logger.info(f"Found {len(image_files)} images.")
    logger.info(f"Found {len(mask_files)} masks.")
    print("\n" + "=" * 70)
    print(f"Checking {split.upper()}")
    print("=" * 70)

    image_names = {
        img.stem.replace("_img", "")
        for img in image_files
    }

    mask_names = {
        mask.stem.replace("_mask", "")
        for mask in mask_files
    }

    samples = sorted(image_names | mask_names)

    for sample in samples:

        image = split_path / f"{sample}_img.tif"
        mask = split_path / f"{sample}_mask.tif"

        status = "OK"
        reason = ""

        if not image.exists():

            status = "FAILED"
            reason = "Missing image"

            logger.warning(f"Missing image for sample: {sample}")

            print(f"[Missing Image] {sample}")

            if fix:
                remove_file(mask)

        elif not mask.exists():

            status = "FAILED"
            reason = "Missing mask"

            logger.warning(f"Missing mask for sample: {sample}")

            print(f"[Missing Mask] {sample}")

            if fix:
                remove_file(image)

        else:

            try:

                with rasterio.open(image) as img:

                    img_width = img.width
                    img_height = img.height
                    img_bands = img.count

                with rasterio.open(mask) as msk:

                    mask_width = msk.width
                    mask_height = msk.height
                    mask_bands = msk.count
                    mask_values = set(msk.read(1).flatten())

                if img_bands != 3:

                    status = "FAILED"
                    reason = "Image does not contain 3 bands"

                elif mask_bands != 1:

                    status = "FAILED"
                    reason = "Mask does not contain 1 band"

                elif img_width != mask_width or img_height != mask_height:

                    status = "FAILED"
                    reason = "Image/Mask size mismatch"

                elif not mask_values.issubset({0, 1}):

                    status = "FAILED"
                    reason = "Mask contains values other than 0/1"

            except Exception as e:

                status = "FAILED"
                reason = str(e)

                logger.exception(f"Corrupted sample detected: {sample}")

                print(f"[Corrupted] {sample}")

                if fix:

                    remove_file(image)
                    remove_file(mask)

        if status == "OK":

            logger.info(f"Validated sample: {sample}")

        report.append(
            {
                "split": split,
                "sample": sample,
                "status": status,
                "reason": reason,
            }
        )


def main():

    logger.info("Starting dataset validation.")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete orphan/corrupted files."
    )

    args = parser.parse_args()

    for split in ["train", "val", "test"]:

        validate_split(split, args.fix)

    df = pd.DataFrame(report)

    logger.info("Writing validation report to dataset_validation_report.csv")
    df.to_csv(
        "dataset_validation_report.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split in ["train", "val", "test"]:

        subset = df[df["split"] == split]

        passed = (subset["status"] == "OK").sum()
        failed = (subset["status"] == "FAILED").sum()

        print(f"\n{split.upper()}")

        print(f"Valid Samples : {passed}")
        print(f"Failed Samples: {failed}")

    total_valid = (df["status"] == "OK").sum()
    total_failed = (df["status"] == "FAILED").sum()

    print("\nTOTAL")

    print(f"Valid : {total_valid}")
    print(f"Failed: {total_failed}")

    print("\nValidation report written to dataset_validation_report.csv")
    logger.info("Dataset validation completed successfully.")

if __name__ == "__main__":

    main()