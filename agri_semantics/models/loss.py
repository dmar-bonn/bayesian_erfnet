""" Specify different loss functions for the domain of semantic segmentation.
"""
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100, weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss.

        Args:
            inputs (torch.Tensor): unnormalized input tensor of shape [B x C x H x W]
            target (torch.Tensor): ground-truth target tensor of shape [B x H x W]

        Returns:
              torch.Tensor: weighted mean of the output losses.
        """

        loss = self.criterion(inputs, target)

        return loss
