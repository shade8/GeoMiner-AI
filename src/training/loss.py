"""
Loss functions for GeoMiner AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logging_config import setup_logger

logger = setup_logger()


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ):

        probs = torch.softmax(
            logits,
            dim=1,
        )

        probs = probs[:, 1]

        masks = masks.float()

        intersection = (
            probs * masks
        ).sum(dim=(1, 2))

        union = (
            probs.sum(dim=(1, 2))
            + masks.sum(dim=(1, 2))
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()

        logger.info(
            "Initializing segmentation loss."
        )

        self.ce = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

        self.ce_weight = ce_weight

        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ):

        ce_loss = self.ce(
            logits,
            masks.long(),
        )

        dice_loss = self.dice(
            logits,
            masks,
        )

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss