"""
EgaraNet — Model definition.

Provides the full EgaraNet model (DINOv3 backbone + StyleNet head) with
methods for loading from checkpoints and HuggingFace Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .layers import (
    AttentionPooling,
    TransposedAttentionTransformer,
)
from .preprocessing import MaxResizeMod16, build_transform


class StyleNet(nn.Module):
    """StyleNet head: TAT layers → AttentionPooling → Projection → L2 norm.

    Args:
        input_dim: Input channel dimension from the backbone.
        hidden_dim: Internal channel width of TAT layers.
        num_tat_layers: Number of stacked TransposedAttentionTransformer layers.
        num_heads: Number of attention heads in each TAT.
        output_dim: Dimension of the final L2-normalized style vector.
        attn_pool_heads: Number of heads in AttentionPooling.
        rms_norm_eps: Epsilon for RMSNorm.
        swiglu_multiple: SwiGLU hidden dim alignment.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        num_tat_layers: int = 3,
        num_heads: int = 16,
        output_dim: int = 1024,
        attn_pool_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        swiglu_multiple: int = 64,
    ):
        super().__init__()
        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

        self.tat_layers = nn.ModuleList([
            TransposedAttentionTransformer(
                dim=hidden_dim,
                num_heads=num_heads,
                eps=rms_norm_eps,
                swiglu_multiple=swiglu_multiple,
            )
            for _ in range(num_tat_layers)
        ])

        self.attn_pool = AttentionPooling(dim=hidden_dim, num_heads=attn_pool_heads)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone features [B, N, C].

        Returns:
            L2-normalized style embeddings [B, output_dim].
        """
        x = self.input_proj(x)
        for layer in self.tat_layers:
            x = layer(x)
        x = self.attn_pool(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=-1)


class EgaraNet(nn.Module):
    """EgaraNet: DINOv3 ViT backbone + StyleNet head.

    A composite model that extracts illustration style embeddings.
    The backbone (DINOv3 ViT) extracts visual features, and StyleNet
    transforms them into L2-normalized style vectors using Transposed
    Attention Transformer (TAT).

    Example:
        >>> model = EgaraNet.from_checkpoint("checkpoints/epoch_10.pth")
        >>> vec = model.extract_style_vector("illustration.png")
        >>> print(vec.shape)  # (1024,)
    """

    # Default backbone model ID on HuggingFace Hub
    DEFAULT_BACKBONE_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(
        self,
        backbone: nn.Module,
        style_net: StyleNet,
        image_mean: list[float] = None,
        image_std: list[float] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.style_net = style_net
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Union[str, torch.device] = "auto",
    ) -> "EgaraNet":
        """Load EgaraNet from a training checkpoint (.pth file).

        The checkpoint should contain:
            - ``model_state_dict``: StyleNet weights
            - ``config`` (optional): Architecture configuration dict

        The DINOv3 backbone is downloaded from HuggingFace Hub.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            device: Device to load the model on. "auto" selects CUDA if available.

        Returns:
            An EgaraNet model in eval mode.
        """
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Defaults
        backbone_id = cls.DEFAULT_BACKBONE_ID
        hidden_dim = 1024
        output_dim = 1024
        num_tat_layers = 3
        num_heads = 16

        # Load checkpoint and override defaults from config
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            backbone_id = cfg.get("dino_model_id", backbone_id)
            hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
            output_dim = int(cfg.get("output_dim", output_dim))
            num_tat_layers = int(cfg.get("num_tat_layers", num_tat_layers))
            num_heads = int(cfg.get("num_heads", num_heads))

        # Load backbone
        processor = AutoImageProcessor.from_pretrained(backbone_id)
        backbone = AutoModel.from_pretrained(backbone_id).to(device)
        backbone.eval()

        # Build StyleNet
        backbone_hidden_size = backbone.config.hidden_size
        style_net = StyleNet(
            input_dim=backbone_hidden_size,
            hidden_dim=hidden_dim,
            num_tat_layers=num_tat_layers,
            num_heads=num_heads,
            output_dim=output_dim,
        ).to(device)

        weights = checkpoint.get("model_state_dict", checkpoint)
        style_net.load_state_dict(weights)
        style_net.eval()

        model = cls(
            backbone=backbone,
            style_net=style_net,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
        )
        model.to(device)
        model.eval()
        return model

    @classmethod
    def from_huggingface(
        cls,
        model_id: str = "Columba1198/EgaraNet",
        device: Union[str, torch.device] = "auto",
    ) -> "EgaraNet":
        """Load EgaraNet from HuggingFace Hub.

        Requires the model to be stored in the HuggingFace Hub format
        with ``trust_remote_code=True``.

        Args:
            model_id: HuggingFace Hub model ID.
            device: Device to load the model on.

        Returns:
            An EgaraNet model in eval mode.
        """
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        hf_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hf_model = hf_model.to(device)
        hf_model.eval()

        # Wrap the HF model so it provides a consistent API
        model = _HuggingFaceEgaraNetWrapper(hf_model)
        model.to(device)
        model.eval()
        return model

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def preprocess(
        self,
        img: Image.Image,
        max_size: int = 512,
        keep_aspect_ratio: bool = True,
    ) -> torch.Tensor:
        """Preprocess a PIL image to a model input tensor.

        Args:
            img: PIL Image (RGB).
            max_size: Maximum image size.
            keep_aspect_ratio: If True, preserve aspect ratio.

        Returns:
            Tensor of shape [1, 3, H, W].
        """
        transform = build_transform(
            max_size, keep_aspect_ratio,
            mean=self.image_mean, std=self.image_std,
        )
        return transform(img).unsqueeze(0)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        use_bf16: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W].
            use_bf16: If True, use BF16 mixed precision (CUDA only).

        Returns:
            L2-normalized style embeddings [B, output_dim].
        """
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(use_bf16 and self.device.type == "cuda"),
        ):
            features = self.backbone(pixel_values=x).last_hidden_state
            return self.style_net(features)

    def extract_style_vector(
        self,
        image_path: str,
        max_size: int = 512,
        keep_aspect_ratio: bool = True,
        use_bf16: bool = False,
    ) -> np.ndarray:
        """Extract a style vector from a single image.

        Args:
            image_path: Path to the image file.
            max_size: Image size for preprocessing.
            keep_aspect_ratio: If True, preserve aspect ratio.
            use_bf16: If True, use BF16 mixed precision.

        Returns:
            Numpy array of shape [D] (L2-normalized).
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img, max_size, keep_aspect_ratio).to(self.device)
        embedding = self(tensor, use_bf16=use_bf16)
        return embedding.cpu().float().numpy().flatten()

    def extract_style_vectors(
        self,
        image_paths: list[str],
        max_size: int = 512,
        keep_aspect_ratio: bool = True,
        use_bf16: bool = False,
    ) -> np.ndarray:
        """Extract style vectors from multiple images.

        Args:
            image_paths: List of image file paths.
            max_size: Image size for preprocessing.
            keep_aspect_ratio: If True, preserve aspect ratio.
            use_bf16: If True, use BF16 mixed precision.

        Returns:
            Numpy array of shape [N, D].
        """
        vectors = []
        for path in image_paths:
            try:
                vec = self.extract_style_vector(path, max_size, keep_aspect_ratio, use_bf16)
                vectors.append(vec)
            except Exception as e:
                print(f"[WARN] Skipped {path}: {e}", file=sys.stderr)
        return np.stack(vectors) if vectors else np.array([])


class _HuggingFaceEgaraNetWrapper(nn.Module):
    """Internal wrapper to provide consistent API for HuggingFace-loaded models."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    def preprocess(self, img, max_size=512, keep_aspect_ratio=True):
        transform = build_transform(max_size, keep_aspect_ratio,
                                    mean=self.image_mean, std=self.image_std)
        return transform(img).unsqueeze(0)

    @torch.no_grad()
    def forward(self, x, use_bf16=False):
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=(use_bf16 and self.device.type == "cuda"),
        ):
            output = self.hf_model(pixel_values=x)
            return output.style_embedding

    def extract_style_vector(self, image_path, max_size=512,
                             keep_aspect_ratio=True, use_bf16=False):
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img, max_size, keep_aspect_ratio).to(self.device)
        embedding = self(tensor, use_bf16=use_bf16)
        return embedding.cpu().float().numpy().flatten()

    def extract_style_vectors(self, image_paths, max_size=512,
                              keep_aspect_ratio=True, use_bf16=False):
        vectors = []
        for path in image_paths:
            try:
                vec = self.extract_style_vector(path, max_size, keep_aspect_ratio, use_bf16)
                vectors.append(vec)
            except Exception as e:
                print(f"[WARN] Skipped {path}: {e}", file=sys.stderr)
        return np.stack(vectors) if vectors else np.array([])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Since EgaraNet outputs are L2-normalized, this is equivalent
    to the dot product.

    Args:
        vec_a: First vector of shape [D].
        vec_b: Second vector of shape [D].

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    a = vec_a / (np.linalg.norm(vec_a) + 1e-8)
    b = vec_b / (np.linalg.norm(vec_b) + 1e-8)
    return float(np.dot(a, b))
