"""
SegFormer model for mine segmentation.
"""

import torch
import torch.nn as nn

from transformers import (
    SegformerForSemanticSegmentation,
)

from src.config import settings
from src.logging_config import setup_logger

logger = setup_logger()


class GeoMinerSegFormer(nn.Module):

    """
    SegFormer wrapper.
    """

    def __init__(self):

        super().__init__()

        logger.info(
            "Loading pretrained SegFormer..."
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                settings.MODEL_NAME,
                num_labels=settings.NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        )

        logger.info(
            "SegFormer loaded successfully."
        )

    def forward(
        self,
        images,
    ):

        outputs = self.model(
            pixel_values=images
        )

        return outputs.logits

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )

    def trainable_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )