"""
loss.py — Focal Loss + InfoNCE Contrastive Loss
================================================
FocalLoss:
    Addresses the severe class imbalance (1 positive : 3 negatives
    in our sampling, but at inference the real ratio is 1:400+).
    Down-weights easy negatives that the model already handles confidently,
    forcing gradient budget toward hard negatives and the rare positives.

    FL(p, y) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1-p)

InfoNCE (auxiliary contrastive loss):
    Within a batch, pairs (feature_i, class_j) where i==j and class_j is
    a positive (label=1) are treated as positives; all other cross-sample
    pairs are negatives.

    This trains the projection space to bring feature embeddings close to
    the class embeddings of the files they touch, independently of the
    interaction head. Acts as a useful auxiliary signal for generalization,
    especially given the limited dataset size.

    Note: InfoNCE is only meaningful when a batch contains multiple positive
    pairs. It is silently skipped for batches with fewer than 2 positives.

CombinedLoss:
    L = L_focal + lambda_contrastive * L_infonce
    lambda_contrastive defaults to 0.1; set to 0 to disable InfoNCE.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for class-imbalanced binary classification.

    Args:
        alpha : weight for the positive class. Set to (1 - pos_rate) in
                practice. E.g. if ~25% of samples are positive (after
                hard negative sampling), alpha=0.75.
        gamma : focusing parameter. gamma=0 reduces to weighted BCE.
                gamma=2 is the standard value from the original paper.
        reduction : 'mean' (default) | 'sum' | 'none'
    """
    def __init__(
        self,
        alpha:     float = 0.75,
        gamma:     float = 2.0,
        reduction: str   = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (batch,) raw logits
        labels: torch.Tensor,   # (batch,) float in {0, 1} or label-smoothed
    ) -> torch.Tensor:
        # Compute BCE without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )

        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t   = labels * probs + (1 - labels) * (1 - probs)

        # alpha_t: class weight
        alpha_t = labels * self.alpha + (1 - labels) * (1 - self.alpha)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class InfoNCELoss(nn.Module):
    """
    In-batch contrastive loss over projected feature and class embeddings.

    Treats (f_i, c_j) as a positive pair if i == j and label_i == 1.
    All other (f_i, c_k) pairs where k != j are negatives.

    When a batch has no positive pairs or only one, the loss is returned
    as 0.0 (no gradient) since InfoNCE requires at least 2 positive pairs
    to be meaningful.

    Args:
        temperature : softmax temperature tau (default 0.07 from SimCLR)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        f_proj:  torch.Tensor,   # (batch, proj_dim) feature embeddings
        c_proj:  torch.Tensor,   # (batch, proj_dim) class embeddings
        labels:  torch.Tensor,   # (batch,) float — 1 for positive pairs
    ) -> torch.Tensor:
        device = f_proj.device

        # Only compute the loss over positive pairs
        pos_mask = (labels > 0.5)
        n_pos    = pos_mask.sum().item()

        if n_pos < 1:
            # Not enough positives in this batch for a meaningful contrastive loss
            return torch.tensor(0.0, device=device, requires_grad=False)

        # L2-normalise both sides (cosine similarity)
        f_norm = F.normalize(f_proj, dim=-1)   # (batch, dim)
        c_norm = F.normalize(c_proj, dim=-1)   # (batch, dim)

        # Similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(f_norm, c_norm.T) / self.temperature

        # For each positive pair (i, j=i), the loss is:
        #   -log( exp(sim(f_i, c_i)) / sum_k exp(sim(f_i, c_k)) )
        # We compute this only for rows i where label_i == 1
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]

        # Row-wise log-softmax over all class embeddings in the batch
        log_probs = F.log_softmax(sim_matrix, dim=-1)  # (batch, batch)

        # Extract diagonal entries for positive rows: log P(c_i | f_i)
        pos_log_probs = log_probs[pos_indices, pos_indices]   # (n_pos,)

        return -pos_log_probs.mean()


class CombinedLoss(nn.Module):
    """
    L = L_focal + lambda_contrastive * L_infonce

    Args:
        focal_alpha          : positive class weight for focal loss
        focal_gamma          : focusing parameter
        contrastive_temp     : InfoNCE temperature
        lambda_contrastive   : weight of the InfoNCE term (0 to disable)
    """
    def __init__(
        self,
        focal_alpha:        float = 0.75,
        focal_gamma:        float = 2.0,
        contrastive_temp:   float = 0.07,
        lambda_contrastive: float = 0.1,
    ):
        super().__init__()
        self.focal     = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.infonce   = InfoNCELoss(temperature=contrastive_temp)
        self.lambda_c  = lambda_contrastive

    def forward(
        self,
        logits:  torch.Tensor,   # (batch,)
        f_proj:  torch.Tensor,   # (batch, proj_dim)
        c_proj:  torch.Tensor,   # (batch, proj_dim)
        labels:  torch.Tensor,   # (batch,)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss : scalar tensor
            breakdown  : dict with individual loss values for logging
        """
        l_focal = self.focal(logits, labels)

        if self.lambda_c > 0:
            l_contrastive = self.infonce(f_proj, c_proj, labels)
            total = l_focal + self.lambda_c * l_contrastive
        else:
            l_contrastive = torch.tensor(0.0, device=logits.device)
            total         = l_focal

        breakdown = {
            "loss_focal":       l_focal.item(),
            "loss_contrastive": l_contrastive.item(),
            "loss_total":       total.item(),
        }
        return total, breakdown


def build_loss(config: dict) -> CombinedLoss:
    """
    Build loss from config dict.
    Typical config:
        {
          "focal_alpha":        0.75,
          "focal_gamma":        2.0,
          "contrastive_temp":   0.07,
          "lambda_contrastive": 0.1
        }
    """
    return CombinedLoss(
        focal_alpha        = config.get("focal_alpha",        0.75),
        focal_gamma        = config.get("focal_gamma",        2.0),
        contrastive_temp   = config.get("contrastive_temp",   0.07),
        lambda_contrastive = config.get("lambda_contrastive", 0.1),
    )
