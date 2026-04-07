"""
EgaraNet — Image preprocessing utilities.

Provides the MaxResizeMod16 transform and helper functions for preparing
images as model input tensors.
"""

from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as T
from PIL import Image


# ImageNet normalization statistics (must match the DINOv3 backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MaxResizeMod16:
    """Resize the long edge to max_size, preserving aspect ratio.

    Both height and width are snapped to the nearest multiple of 16,
    which is required for ViT models with patch_size=16.

    Args:
        max_size: Maximum size for the long edge.
        interpolation: PIL interpolation mode.
    """

    def __init__(self, max_size: int, interpolation=Image.Resampling.BICUBIC):
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w = max(16, round(int(w * scale) / 16) * 16)
        new_h = max(16, round(int(h * scale) / 16) * 16)
        if (new_w, new_h) != (w, h):
            return img.resize((new_w, new_h), self.interpolation)
        return img

    def __repr__(self) -> str:
        return f"MaxResizeMod16(max_size={self.max_size})"


def build_transform(
    max_size: int = 512,
    keep_aspect_ratio: bool = True,
    mean: list[float] = None,
    std: list[float] = None,
) -> T.Compose:
    """Build a torchvision transform pipeline for EgaraNet input.

    Args:
        max_size: Image size (long edge if keep_aspect_ratio, else square).
        keep_aspect_ratio: If True, use MaxResizeMod16; else square resize+crop.
        mean: Normalization mean (default: ImageNet).
        std: Normalization std (default: ImageNet).

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    if keep_aspect_ratio:
        resize_op = MaxResizeMod16(max_size)
    else:
        resize_op = T.Compose([
            T.Resize(max_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(max_size),
        ])

    return T.Compose([resize_op, T.ToTensor(), T.Normalize(mean, std)])


def preprocess_image(
    image_path: Union[str, Path],
    max_size: int = 512,
    keep_aspect_ratio: bool = True,
) -> torch.Tensor:
    """Load and preprocess a single image.

    Args:
        image_path: Path to the image file.
        max_size: Image size for preprocessing.
        keep_aspect_ratio: If True, preserve aspect ratio.

    Returns:
        Tensor of shape [1, 3, H, W] ready for model input.
    """
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(max_size, keep_aspect_ratio)
    return transform(img).unsqueeze(0)


def collect_image_paths(
    path: Union[str, Path],
    recursive: bool = True,
) -> list[str]:
    """Collect image file paths from a file or directory.

    Args:
        path: A single image file or a directory to scan.
        recursive: If True, scan subdirectories recursively.

    Returns:
        Sorted list of image file paths.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    p = Path(path)
    if p.is_file():
        return [str(p)]
    glob_fn = p.rglob if recursive else p.glob
    return sorted(str(f) for f in glob_fn("*") if f.suffix.lower() in exts)
