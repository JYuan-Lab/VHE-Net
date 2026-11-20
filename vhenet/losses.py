"""Loss functions for VHE-Net.

The main ranking objective applies a softmax over host logits within one virus.
Positive-host targets are weighted by the importance score, which forces the
confirmed hosts of each virus to rank above its negatives. A host-only bias
cannot satisfy the conflicting host preferences of different viruses, so the
model must use viral sequence information.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def per_virus_softmax_loss(logits: torch.Tensor, labels: torch.Tensor,
                           weights: torch.Tensor, pos_clip: float = 20.0):
    """logits/labels/weights: [K] for K host rows of one virus.

    Soft targets normalize the positive importance weights; negative targets
    are zero. Returns -sum(t * log_softmax(z)).
    """
    pos_mask = labels > 0.5
    if pos_mask.sum() == 0:
        return torch.zeros((), device=logits.device)
    t = torch.zeros_like(logits)
    w_pos = weights[pos_mask].clamp(min=1.0, max=pos_clip)
    t[pos_mask] = w_pos / w_pos.sum()
    logp = F.log_softmax(logits, dim=0)
    return -(t * logp).sum()


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor,
                 weights: torch.Tensor, clip: float = 5.0):
    """Sample-weighted BCE used as an auxiliary calibration loss."""
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none").squeeze(-1)
    w = weights.clamp(min=0.0, max=clip)
    return (bce * w).sum() / w.sum().clamp_min(1e-6)
