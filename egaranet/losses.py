"""
EgaraNet — Loss functions for training.

Currently provides a TripletMarginLoss wrapper with convenient defaults.
Additional losses (InfoNCE, ArcFace, etc.) can be added here as needed.
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet Margin Loss for style embedding training.

    Wraps nn.TripletMarginLoss with sensible defaults for L2-normalized
    embeddings.

    Args:
        margin: Margin for the triplet loss. Default: 0.2.
        p: Norm degree for pairwise distance. Default: 2 (Euclidean).
    """

    def __init__(self, margin: float = 0.2, p: int = 2):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [B, D].
            positive: Positive embeddings [B, D].
            negative: Negative embeddings [B, D].

        Returns:
            Scalar loss value.
        """
        return self.loss_fn(anchor, positive, negative)
