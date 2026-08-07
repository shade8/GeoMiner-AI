"""
Training pipeline for GeoMiner AI.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.utils.history import save_history
from src.config import settings
from src.logging_config import setup_logger
from src.training.loss import SegmentationLoss
from src.training.metrics import SegmentationMetrics
from torch.utils.tensorboard import SummaryWriter

logger = setup_logger()

class Trainer:
    """
    Handles training and validation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
    ):

        logger.info("Initializing trainer...")

        self.device = settings.DEVICE

        self.model = model.to(self.device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = SegmentationLoss()

        self.metrics = SegmentationMetrics()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=settings.EPOCHS,
        )

        self.best_iou = 0.0

        self.epochs_without_improvement = 0

        self.writer = SummaryWriter(
            log_dir="runs/geominer_ai"
        )

        self.start_epoch = 0

        checkpoint = (
            Path(settings.CHECKPOINT_DIR)
            / "last_model.pth"
        )

        if checkpoint.exists():

            logger.info(
                "Previous checkpoint found. Resuming training..."
            )

            self.start_epoch = (
                self.load_checkpoint(checkpoint) + 1
            )
        logger.info(
            f"Training on {self.device.upper()}"
        )

        logger.info(
            "Trainer initialized successfully."
        )


    def save_checkpoint(
        self,
        epoch,
        iou,
        best=False,
    ):

        Path(settings.CHECKPOINT_DIR).mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_iou": iou,
        }

        latest_path = (
            Path(settings.CHECKPOINT_DIR)
            / "last_model.pth"
        )

        torch.save(
            checkpoint,
            latest_path,
        )

        logger.info(
            f"Latest checkpoint saved : {latest_path}"
        )

        if best:

            best_path = (
                Path(settings.CHECKPOINT_DIR)
                / "best_model.pth"
            )

            torch.save(
                checkpoint,
                best_path,
            )

            logger.info(
                f"Best checkpoint saved : {best_path}"
            )

    def load_checkpoint(
        self,
        checkpoint_path,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.best_iou = checkpoint["best_iou"]

        logger.info(
            f"Loaded checkpoint : {checkpoint_path}"
        )

        return checkpoint["epoch"]

    def train_one_epoch(
        self,
        epoch: int,
    ):

        logger.info(
            f"Starting training epoch {epoch + 1}/{settings.EPOCHS}"
        )

        self.model.train()

        running_loss = 0.0

        running_metrics = {
            "pixel_accuracy": 0.0,
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}",
        )

        for images, masks in progress_bar:

            images = images.to(self.device)

            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            # Resize logits to match mask resolution
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = self.criterion(
                logits,
                masks,
            )

            if torch.isnan(loss):
                print("\n===== DEBUG =====")
                print("Loss is NaN")

                print(
                    "Images:",
                    images.min().item(),
                    images.max().item(),
                )

                print(
                    "Logits:",
                    logits.min().item(),
                    logits.max().item(),
                )

                print(
                    "Masks:",
                    torch.unique(masks),
                )

                raise RuntimeError("NaN loss detected")

            loss.backward()

            self.optimizer.step()

            batch_metrics = self.metrics.calculate(
                logits,
                masks,
            )

            running_loss += loss.item()

            for key in running_metrics:

                running_metrics[key] += batch_metrics[key]

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_metrics['iou']:.4f}",
            )

        num_batches = len(self.train_loader)

        epoch_loss = running_loss / num_batches

        epoch_metrics = {}

        for key in running_metrics:

            epoch_metrics[key] = (
                running_metrics[key] / num_batches
            )

        logger.info(
            f"Training Loss : {epoch_loss:.4f}"
        )

        logger.info(
            f"Training IoU  : {epoch_metrics['iou']:.4f}"
        )

        return (
            epoch_loss,
            epoch_metrics,
        )

    def validate(
        self,
        epoch: int,
    ):

        logger.info(
            f"Starting validation epoch {epoch + 1}/{settings.EPOCHS}"
        )

        self.model.eval()

        running_loss = 0.0

        running_metrics = {
            "pixel_accuracy": 0.0,
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch + 1}",
        )

        with torch.no_grad():

            for images, masks in progress_bar:

                images = images.to(self.device)

                masks = masks.to(self.device)

                logits = self.model(images)

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = self.criterion(
                    logits,
                    masks,
                )

                batch_metrics = self.metrics.calculate(
                    logits,
                    masks,
                )

                running_loss += loss.item()

                for key in running_metrics:

                    running_metrics[key] += batch_metrics[key]

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    iou=f"{batch_metrics['iou']:.4f}",
                )

        num_batches = len(self.val_loader)

        epoch_loss = running_loss / num_batches

        epoch_metrics = {}

        for key in running_metrics:

            epoch_metrics[key] = (
                running_metrics[key] / num_batches
            )

        logger.info(
            f"Validation Loss : {epoch_loss:.4f}"
        )

        logger.info(
            f"Validation IoU  : {epoch_metrics['iou']:.4f}"
        )

        return (
            epoch_loss,
            epoch_metrics,
        )

    def fit(self):

        logger.info(
            "Starting model training..."
        )

        for epoch in range(
            self.start_epoch,
            settings.EPOCHS,
        ):

            train_loss, train_metrics = self.train_one_epoch(
                epoch,
            )

            val_loss, val_metrics = self.validate(
                epoch,
            )

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            self.scheduler.step()

            logger.info(
                f"Learning Rate : {current_lr:.8f}"
            )

            save_history(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_iou=train_metrics["iou"],
                val_iou=val_metrics["iou"],
                learning_rate=current_lr,
            )
            
            is_best = (
                val_metrics["iou"] > self.best_iou
            )

            if is_best:

                self.best_iou = val_metrics["iou"]

                self.epochs_without_improvement = 0

            else:

                self.epochs_without_improvement += 1

            self.save_checkpoint(
                epoch=epoch,
                iou=self.best_iou,
                best=is_best,
            )

            logger.info("-" * 80)

            logger.info(
                f"Epoch {epoch + 1}/{settings.EPOCHS}"
            )

            logger.info(
                f"Train Loss : {train_loss:.4f}"
            )

            logger.info(
                f"Validation Loss : {val_loss:.4f}"
            )

            logger.info(
                f"Train IoU : {train_metrics['iou']:.4f}"
            )

            logger.info(
                f"Validation IoU : {val_metrics['iou']:.4f}"
            )

            logger.info(
                f"Best IoU : {self.best_iou:.4f}"
            )

            self.writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch,
            )

            self.writer.add_scalar(
                "IoU/Train",
                train_metrics["iou"],
                epoch,
            )

            self.writer.add_scalar(
                "IoU/Validation",
                val_metrics["iou"],
                epoch,
            )

            self.writer.add_scalar(
                "LearningRate",
                current_lr,
                epoch,
            )

            logger.info("-" * 80)

            if (
                self.epochs_without_improvement
                >= settings.EARLY_STOPPING_PATIENCE
            ):

                logger.info(
                    "Early stopping triggered."
                )

                break

        logger.info(
            "Training completed successfully."
        )

        self.writer.close()