<div align="center">

[English](README.md) | [日本語](README.ja.md)

# EgaraNet

**Illustration Style Embedding Model — Training & Inference**

Train and run inference with EgaraNet, a model that encodes illustration art styles into high-dimensional vectors.

[![HuggingFace](https://img.shields.io/badge/🤗_Pretrained_Model-Columba1198%2FEgaraNet-yellow)](https://huggingface.co/Columba1198/EgaraNet)
[![Demo](https://img.shields.io/badge/🌐_Demo-egara--net.vercel.app-6c5ce7)](https://egara-net.vercel.app/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

---

## Overview

**EgaraNet** is a deep learning model that encodes the artistic style of illustrations into 1024-dimensional embedding vectors. Trained on approximately 1.2 million illustrations from around 12,000 artists, the model produces embeddings where illustrations by the same artist are close together in the vector space.

This enables applications such as:
- 🎨 **Style Similarity** — Compare how similar two illustrations' styles are
- 🔍 **Style-based Search** — Find illustrations with similar art styles
- 📊 **Style Clustering** — Group illustrations by their visual style

This repository provides the **training and inference code** for EgaraNet. For the pretrained model, see [🤗 Columba1198/EgaraNet](https://huggingface.co/Columba1198/EgaraNet).

> Produced under Article 30-4 of the Japanese Copyright Act.

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Architecture"  style="width: 70%;">
</p>

EgaraNet is a composite model with two main components:

| Component | Description |
|-----------|-------------|
| **Backbone** | DINOv3 ViT-L/16 — A Vision Transformer pretrained with DINOv3 self-supervised learning |
| **Head** | StyleNet — Custom decoder with Transposed Attention Transformer (TAT) |

### Transposed Attention Transformer (TAT)

The **Transposed Attention Transformer** is a Transformer designed to extract style information by operating in channel space rather than spatial space. Inspired by the observation from Gatys et al. ([CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)) that Gram matrices of feature maps encode style independent of content, the TAT computes cross-covariance attention where:

1. Query is transposed: `Q.T` → shape `(HeadDim, N)` instead of `(N, HeadDim)`
2. Attention is computed as `(Q.T @ K)` → shape `(HeadDim, HeadDim)` — a C×C channel correlation matrix
3. This replaces the standard `(N, N)` spatial attention with a channel-to-channel attention map

This discards positional/spatial information, preserving only how channels (features) correlate with each other — exactly the style signature.

<p align="center">
  <img src="assets/tat.png" alt="TAT"  style="width: 75%;">
</p>

Channel attention mechanisms are not new, with several prior studies having explored similar approaches:
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)
- [Multi-Head Transposed Attention Transformer for Sea Ice Segmentation in Sar Imagery](https://ieeexplore.ieee.org/document/10640437)
- [MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://arxiv.org/abs/2204.08958)

### Technical Specifications

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 1024 |
| Backbone | DINOv3 ViT-L/16 (hidden_size=1024) |
| TAT layers | 3 |
| Attention heads | 16 (head_dim=64) |
| Input resolution | Dynamic (any multiple of 16) |
| Training images | ~1.2M from ~12K artists |

## Installation

```bash
git clone https://github.com/Columba1198/EgaraNet.git
cd EgaraNet
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU recommended (CPU is supported but slow)

## Dataset Preparation

Organize your dataset with one subdirectory per artist:

```
dataset/
├── artist_A/
│   ├── image_001.png
│   ├── image_002.jpg
│   └── ...
├── artist_B/
│   ├── image_010.png
│   └── ...
└── ...
```

- Each subdirectory represents one artist (label)
- At least 2 artist subdirectories are required
- At least two images per artist are required
- Supported image formats: PNG, JPEG, WebP, BMP

## Training

### Quick Start

```bash
python train.py --data_dir ./dataset
```

### With Configuration File

```bash
python train.py --data_dir ./dataset --config configs/default.yaml
```

### CLI Arguments

```bash
python train.py --data_dir ./dataset \
    --epochs 20 \
    --lr 1e-5 \
    --accum_steps 64 \
    --margin 0.3 \
    --bf16 true
```

### Resuming Training

```bash
python train.py --data_dir ./dataset --resume checkpoints/epoch_5.pth
```

### Training Process

Training proceeds in two phases:

1. **Feature Caching**: DINOv3 backbone features are extracted once and saved as `.pt` files alongside each image. This avoids re-computing backbone features every epoch.

2. **StyleNet Training**: The StyleNet head is trained on cached features using triplet margin loss, where:
   - **Anchor** and **Positive** are from the same artist
   - **Negative** is from a different artist

> **Note**: When `keep_aspect_ratio=true` (default), batch size is forced to 1 because images have variable dimensions. Gradient accumulation is used to simulate larger effective batch sizes.

## Inference

### CLI Usage

```bash
# Single image — prints vector to stdout
python inference.py --model checkpoints/epoch_10.pth --input image.png

# Directory of images — outputs CSV
python inference.py --model checkpoints/epoch_10.pth --input ./images/ --output vectors.csv

# Compare two images
python inference.py --model checkpoints/epoch_10.pth --compare a.png b.png

# Using pretrained model from HuggingFace
python inference.py --hf Columba1198/EgaraNet --input image.png
```

### Python API

```python
from egaranet import EgaraNet, cosine_similarity

# Load from checkpoint
model = EgaraNet.from_checkpoint("checkpoints/epoch_10.pth")

# Or load from HuggingFace Hub
model = EgaraNet.from_huggingface("Columba1198/EgaraNet")

# Extract style vectors
vec_a = model.extract_style_vector("image_a.png")  # numpy [1024]
vec_b = model.extract_style_vector("image_b.png")

# Compare styles (vectors are L2-normalized, so dot product = cosine sim)
sim = cosine_similarity(vec_a, vec_b)
print(f"Style similarity: {sim:.4f} ({sim * 100:.1f}%)")
```

### Batch Extraction

```python
from egaranet import EgaraNet

model = EgaraNet.from_checkpoint("checkpoints/epoch_10.pth")
vectors = model.extract_style_vectors(["img1.png", "img2.png", "img3.png"])
print(vectors.shape)  # (3, 1024)
```

## Configuration

The default configuration is in `configs/default.yaml`. All settings can be overridden via CLI arguments.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| model | backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 backbone model ID |
| model | tat_layers | 3 | Number of TAT layers |
| model | tat_heads | 16 | Attention heads per TAT |
| model | hidden_dim | 1024 | Internal TAT dimension |
| model | output_dim | 1024 | Style vector dimension |
| preprocessing | max_size | 512 | Max image size (long edge) |
| preprocessing | keep_aspect_ratio | true | Preserve aspect ratio |
| preprocessing | mean | `[0.485, 0.456, 0.406]` | Image normalization mean |
| preprocessing | std | `[0.229, 0.224, 0.225]` | Image normalization standard deviation |
| training | epochs | 10 | Number of training epochs |
| training | batch_size | 1 | Batch size per worker |
| training | accumulation_steps | 128 | Gradient accumulation steps |
| training | learning_rate | 5.0e-6 | AdamW learning rate |
| training | weight_decay | 1.0e-4 | AdamW weight decay |
| training | triplet_margin | 0.2 | Triplet loss margin |
| training | bf16 | true | BF16 Mixed Precision Learning |
| training | checkpoint_dir | `"./checkpoints"` | Directory to save checkpoints |
| training | num_workers | 4 | DataLoader workers |
| inference | bf16 | true | BF16 Mixed Precision Inference |

## Project Structure

```
.
├── egaranet/                  # Python package
│   ├── __init__.py            # Package exports
│   ├── model.py               # EgaraNet model (backbone + StyleNet)
│   ├── layers.py              # Custom layers (RMSNorm, SwiGLU, TAT, AttentionPooling)
│   ├── losses.py              # Loss functions (TripletLoss)
│   ├── preprocessing.py       # Image preprocessing (MaxResizeMod16)
│   └── dataset.py             # Dataset loader (StyleTripletDataset)
├── configs/
│   └── default.yaml           # Default training/inference configuration
├── train.py                   # Training CLI
├── inference.py               # Inference CLI
├── requirements.txt           # Python dependencies
├── LICENSE                    # Apache 2.0
├── README.md                  # English documentation
└── README.ja.md               # Japanese documentation
```

## Input Requirements

- **Image format**: RGB images (PNG, JPEG, WebP, BMP)
- **Resolution**: Dynamic — the model accepts images of any resolution where height and width are multiples of 16. The default preprocessing `MaxResizeMod16(512)` scales the long edge to 512px while preserving aspect ratio and snapping both dimensions to multiples of 16.
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## References

- **DINOv3**: [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- **Style Transfer**: "Image Style Transfer Using Convolutional Neural Networks" [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
- **Restormer**: "Restormer: Efficient Transformer for High-Resolution Image Restoration" [arXiv:2111.09881](https://arxiv.org/abs/2111.09881)
- **MHTA**: "Multi-Head Transposed Attention Transformer for Sea Ice Segmentation in Sar Imagery" [IGARSS 2024](https://ieeexplore.ieee.org/document/10640437)
- **MANIQA**: "MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment" [arXiv:2204.08958](https://arxiv.org/abs/2204.08958)

## Links

- 🌐 **Demo**: [egara-net.vercel.app](https://egara-net.vercel.app/)
- 🤗 **Pretrained Model**: [huggingface.co/Columba1198/EgaraNet](https://huggingface.co/Columba1198/EgaraNet)
- 📖 **GitHub**: [github.com/Columba1198/EgaraNet](https://github.com/Columba1198/EgaraNet)

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
