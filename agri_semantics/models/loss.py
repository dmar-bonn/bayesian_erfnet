import torch
import torch.nn as nn
import torch.nn.functional as F


def get_kl_divergence_prior_loss(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    alpha_kl = targets + (1 - targets) * alpha
    alpha_kl_sum = torch.sum(alpha_kl, dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    kl_log_term = (
        torch.lgamma(alpha_kl_sum)
        - torch.lgamma(torch.sum(ones, dim=1, keepdim=True))
        - torch.sum(torch.lgamma(alpha_kl), dim=1, keepdim=True)
    )
    kl_digamma_term = torch.sum(
        (alpha_kl - 1) * (torch.digamma(alpha_kl) - torch.digamma(alpha_kl_sum)), dim=1, keepdim=True
    )
    return (kl_log_term + kl_digamma_term).squeeze(dim=1)


class PACType2MLELoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(PACType2MLELoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        pac_type2_mle_loss = torch.sum(targets * (torch.log(S) - torch.log(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (pac_type2_mle_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class CrossEntropyBayesRiskLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(CrossEntropyBayesRiskLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        xentropy_bayes_risk_loss = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (xentropy_bayes_risk_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class MSEBayesRiskLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(MSEBayesRiskLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_prob = alpha / S
        error_term = torch.square(targets - pred_prob)
        variance_term = pred_prob * (1 - pred_prob) / (S + 1)
        mse_bayes_risk_loss = torch.sum(error_term + variance_term, dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (mse_bayes_risk_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weight: torch.Tensor = None,
    ):
        super(CrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(reduction="none", weight=weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss.

        Args:
            inputs (torch.Tensor): unnormalized input tensor of shape [B x C x H x W]
            targets (torch.Tensor): ground-truth targets tensor of shape [B x C x H x W]

        Returns:
              torch.Tensor: weighted output losses.
        """
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        loss = self.criterion(inputs, targets) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class SoftIoULoss(nn.Module):
    def __init__(self, num_classes: int, device: torch.device, ignore_index: int = -100):
        super(SoftIoULoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        msk = (torch.arange(self.num_classes, device=self.device) != self.ignore_index).float()
        batch_size = logits.size()[0]
        probabilities = F.softmax(logits, dim=1)

        intersection = probabilities * targets
        intersection_flattened = intersection.view(batch_size, self.num_classes, -1).sum(2)

        union = probabilities + targets - intersection
        union_flattened = union.view(batch_size, self.num_classes, -1).sum(2)

        loss = intersection_flattened * msk / (union_flattened + 1e-8)

        return -loss.sum() / msk.sum()
