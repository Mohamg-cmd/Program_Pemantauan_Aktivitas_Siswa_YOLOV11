"""
Custom YOLO11 detection trainer with element-wise Focal Loss.

Target: Ultralytics YOLO11-M detection, single-stage.
Focal Loss is applied to the classification BCE term only.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss


class ElementwiseFocalLoss(nn.Module):
    """Element-wise binary focal loss so v8DetectionLoss can still sum it."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        prob = pred.sigmoid()
        p_t = target * prob + (1.0 - target) * (1.0 - prob)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        modulating = (1.0 - p_t).pow(self.gamma)
        return alpha_t * modulating * bce


class FocalDetectionLoss(v8DetectionLoss):
    def __init__(self, model, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(model)
        self.bce = ElementwiseFocalLoss(gamma=gamma, alpha=alpha)


class FocalDetectionModel(DetectionModel):
    def init_criterion(self):
        gamma = float(getattr(self.args, "focal_gamma", 2.0))
        alpha = float(getattr(self.args, "focal_alpha", 0.25))
        return FocalDetectionLoss(self, gamma=gamma, alpha=alpha)


class FocalTrainer(DetectionTrainer):
    """DetectionTrainer using FocalDetectionModel."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = FocalDetectionModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
