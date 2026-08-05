"""
Training history utilities.
"""

from pathlib import Path
import csv

from src.config import settings
from src.logging_config import setup_logger

logger = setup_logger()


def save_history(
    epoch,
    train_loss,
    val_loss,
    train_iou,
    val_iou,
    learning_rate,
):

    metrics_dir = (
        Path(settings.OUTPUT_DIR)
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_file = metrics_dir / "history.csv"

    file_exists = history_file.exists()

    with open(
        history_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_iou",
                "val_iou",
                "learning_rate",
            ])

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            train_iou,
            val_iou,
            learning_rate,
        ])

    logger.info(
        f"Metrics saved to {history_file}"
    )