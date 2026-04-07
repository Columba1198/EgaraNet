"""
EgaraNet — Dataset for triplet-based style training.

Expects the following directory structure:

    dataset_root/
    ├── artist_A/
    │   ├── image_001.pt   (cached backbone features)
    │   ├── image_002.pt
    │   └── ...
    ├── artist_B/
    │   ├── image_010.pt
    │   └── ...
    └── ...

Each subdirectory represents one artist (label). The dataset samples
(anchor, positive, negative) triplets where anchor and positive share
the same artist, and negative comes from a different artist.
"""

import os
import glob
import random

import torch
from torch.utils.data import Dataset


class StyleTripletDataset(Dataset):
    """Triplet dataset for style embedding training.

    Each sample returns (anchor, positive, negative) tensors loaded from
    pre-cached .pt files. Anchor and positive are from the same artist
    directory; negative is from a different artist.

    Args:
        root_dir: Root directory containing artist subdirectories.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data_info: list[tuple[str, int]] = []      # (file_path, label_idx)
        self.label_to_files: dict[int, list[str]] = {}   # label_idx -> [file_paths]

        subdirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for label_idx, subdir in enumerate(subdirs):
            folder_path = os.path.join(root_dir, subdir)
            files = glob.glob(os.path.join(folder_path, "*.pt"))

            if files:
                self.label_to_files[label_idx] = files
                for f in files:
                    self.data_info.append((f, label_idx))

        self.num_classes = len(self.label_to_files)
        self._label_keys = list(self.label_to_files.keys())

        if self.num_classes < 2:
            raise ValueError(
                f"At least 2 artist subdirectories with .pt files are required, "
                f"but found {self.num_classes} in '{root_dir}'."
            )

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, anchor_label = self.data_info[idx]

        # Positive: same artist, different image
        pos_candidates = [
            p for p in self.label_to_files[anchor_label]
            if p != anchor_path
        ]
        if pos_candidates:
            pos_path = random.choice(pos_candidates)
        else:
            # Only one image for this artist — use self as positive (fallback)
            pos_path = anchor_path

        # Negative: different artist
        neg_candidates = [k for k in self._label_keys if k != anchor_label]
        neg_label = random.choice(neg_candidates)
        neg_path = random.choice(self.label_to_files[neg_label])

        # Load cached tensors
        anchor_t = torch.load(anchor_path, map_location="cpu", weights_only=True)
        pos_t = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_t = torch.load(neg_path, map_location="cpu", weights_only=True)

        return anchor_t, pos_t, neg_t
