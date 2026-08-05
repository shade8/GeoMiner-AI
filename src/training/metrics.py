"""
Evaluation metrics for semantic segmentation.
"""

import torch

from src.logging_config import setup_logger

logger = setup_logger()


class SegmentationMetrics:
    """
    Computes segmentation metrics.
    """

    def __init__(
        self,
        num_classes: int = 2,
        smooth: float = 1e-6,
    ):

        self.num_classes = num_classes
        self.smooth = smooth

        logger.info(
            "Segmentation metrics initialized."
        )

    def _prediction_mask(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:

        return torch.argmax(
            logits,
            dim=1,
        )

    def pixel_accuracy(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        prediction = self._prediction_mask(logits)

        correct = (prediction == masks).float().sum()

        total = masks.numel()

        return (correct / total).item()

    def iou(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        prediction = self._prediction_mask(logits)

        prediction = prediction.bool()

        masks = masks.bool()

        intersection = (
            prediction & masks
        ).float().sum()

        union = (
            prediction | masks
        ).float().sum()

        return (
            (intersection + self.smooth)
            / (union + self.smooth)
        ).item()

    def dice_score(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        prediction = self._prediction_mask(logits)

        prediction = prediction.bool()

        masks = masks.bool()

        intersection = (
            prediction & masks
        ).float().sum()

        dice = (
            2 * intersection + self.smooth
        ) / (
            prediction.float().sum()
            + masks.float().sum()
            + self.smooth
        )

        return dice.item()

    def precision(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        prediction = self._prediction_mask(logits)

        tp = (
            (prediction == 1)
            & (masks == 1)
        ).float().sum()

        fp = (
            (prediction == 1)
            & (masks == 0)
        ).float().sum()

        return (
            (tp + self.smooth)
            / (tp + fp + self.smooth)
        ).item()

    def recall(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        prediction = self._prediction_mask(logits)

        tp = (
            (prediction == 1)
            & (masks == 1)
        ).float().sum()

        fn = (
            (prediction == 0)
            & (masks == 1)
        ).float().sum()

        return (
            (tp + self.smooth)
            / (tp + fn + self.smooth)
        ).item()

    def f1_score(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:

        precision = self.precision(
            logits,
            masks,
        )

        recall = self.recall(
            logits,
            masks,
        )

        return (
            2
            * precision
            * recall
            / (
                precision
                + recall
                + self.smooth
            )
        )

    def calculate(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict:

        return {

            "pixel_accuracy":
                self.pixel_accuracy(
                    logits,
                    masks,
                ),

            "iou":
                self.iou(
                    logits,
                    masks,
                ),

            "dice":
                self.dice_score(
                    logits,
                    masks,
                ),

            "precision":
                self.precision(
                    logits,
                    masks,
                ),

            "recall":
                self.recall(
                    logits,
                    masks,
                ),

            "f1":
                self.f1_score(
                    logits,
                    masks,
                ),
        }