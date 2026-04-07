"""
EgaraNet — Illustration Style Embedding Model.

A deep learning model that encodes illustration art styles into
high-dimensional vectors for style-based search, classification,
and clustering.

Example:
    >>> from egaranet import EgaraNet, cosine_similarity
    >>> model = EgaraNet.from_checkpoint("checkpoints/epoch_10.pth")
    >>> vec_a = model.extract_style_vector("image_a.png")
    >>> vec_b = model.extract_style_vector("image_b.png")
    >>> sim = cosine_similarity(vec_a, vec_b)
    >>> print(f"Style similarity: {sim:.4f}")
"""

from .model import EgaraNet, StyleNet, cosine_similarity
from .layers import (
    AttentionPooling,
    RMSNorm,
    SwiGLU,
    TransposedTransformerBlock,
)
from .preprocessing import MaxResizeMod16, preprocess_image, build_transform
from .losses import TripletLoss

__version__ = "1.0.0"

__all__ = [
    # Model
    "EgaraNet",
    "StyleNet",
    "cosine_similarity",
    # Layers
    "TransposedTransformerBlock",
    "AttentionPooling",
    "RMSNorm",
    "SwiGLU",
    # Preprocessing
    "MaxResizeMod16",
    "preprocess_image",
    "build_transform",
    # Losses
    "TripletLoss",
]
