"""
EgaraNet — Training CLI.

Trains the StyleNet head on top of cached DINOv3 backbone features
using triplet margin loss.

Usage:
    # Basic training
    python train.py --data_dir ./dataset

    # With config file
    python train.py --data_dir ./dataset --config configs/default.yaml

    # Override settings
    python train.py --data_dir ./dataset --epochs 20 --lr 1e-5

    # Resume from checkpoint
    python train.py --data_dir ./dataset --resume checkpoints/epoch_5.pth

The dataset directory should have the following structure:

    dataset/
    ├── artist_A/
    │   ├── image_001.png
    │   ├── image_002.jpg
    │   └── ...
    ├── artist_B/
    │   └── ...
    └── ...

Training proceeds in two phases:
    1. Caching: DINOv3 backbone features are extracted and saved as .pt files
    2. Training: StyleNet is trained on cached features with triplet loss
"""

import argparse
import datetime
import os
import sys
import time

import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add parent to path for egaranet package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egaranet.model import StyleNet
from egaranet.preprocessing import MaxResizeMod16
from egaranet.dataset import StyleTripletDataset
from egaranet.losses import TripletLoss


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not YAML_AVAILABLE:
        print("[WARN] PyYAML not installed. Using default configuration.", file=sys.stderr)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(args: argparse.Namespace) -> dict:
    """Merge config file with CLI arguments (CLI takes priority)."""
    config = {}

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    elif os.path.exists("configs/default.yaml"):
        config = load_config("configs/default.yaml")

    model_cfg = config.get("model", {})
    preproc_cfg = config.get("preprocessing", {})
    train_cfg = config.get("training", {})

    if args.fp32:
        use_bf16 = False
    elif args.bf16:
        use_bf16 = True
    else:
        use_bf16 = train_cfg.get("bf16", True)

    return {
        # Model
        "backbone": args.backbone or model_cfg.get("backbone", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        "tat_layers": args.tat_layers or model_cfg.get("tat_layers", 3),
        "tat_heads": args.tat_heads or model_cfg.get("tat_heads", 16),
        "hidden_dim": args.hidden_dim or model_cfg.get("hidden_dim", 1024),
        "output_dim": args.output_dim or model_cfg.get("output_dim", 1024),
        # Preprocessing
        "max_size": args.max_size or preproc_cfg.get("max_size", 512),
        "keep_aspect_ratio": preproc_cfg.get("keep_aspect_ratio", True),
        # Training
        "data_dir": args.data_dir,
        "epochs": args.epochs or train_cfg.get("epochs", 10),
        "batch_size": args.batch_size or train_cfg.get("batch_size", 1),
        "accumulation_steps": args.accum_steps or train_cfg.get("accumulation_steps", 128),
        "lr": args.lr or train_cfg.get("learning_rate", 5e-6),
        "weight_decay": train_cfg.get("weight_decay", 1e-4),
        "margin": args.margin or train_cfg.get("triplet_margin", 0.2),
        "bf16": use_bf16,
        "checkpoint_dir": args.checkpoint_dir or train_cfg.get("checkpoint_dir", "./checkpoints"),
        "num_workers": train_cfg.get("num_workers", 4),
        # Resume
        "resume": args.resume,
    }


def cache_features(
    data_dir: str,
    backbone_id: str,
    max_size: int,
    keep_aspect_ratio: bool,
    device: torch.device,
    use_bf16: bool,
):
    """Extract and cache DINOv3 backbone features as .pt files.

    For each image file, a corresponding .pt file is saved in the same
    directory containing the backbone output features.
    """
    from transformers import AutoImageProcessor, AutoModel

    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    # Collect uncached images
    uncached = []
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                img_path = os.path.join(root, fname)
                cache_path = os.path.splitext(img_path)[0] + ".pt"
                if not os.path.exists(cache_path):
                    uncached.append(img_path)

    if not uncached:
        print("All images already cached.")
        return

    print(f"Found {len(uncached)} images to cache. Loading DINOv3 ({backbone_id})...")

    processor = AutoImageProcessor.from_pretrained(backbone_id)
    backbone = AutoModel.from_pretrained(backbone_id).to(device)
    backbone.eval()

    # Build transform
    if keep_aspect_ratio:
        transform = T.Compose([
            MaxResizeMod16(max_size),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
    else:
        transform = T.Compose([
            T.Resize(max_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(max_size),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

    print("Caching backbone features...")
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=(use_bf16 and device.type == "cuda"),
    ):
        for img_path in tqdm(uncached, desc="Caching"):
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)
                output = backbone(pixel_values=input_tensor)

                # Save as BF16 to halve cache file size
                features = output.last_hidden_state.bfloat16().cpu().detach().squeeze(0)
                save_path = os.path.splitext(img_path)[0] + ".pt"
                torch.save(features, save_path)
            except Exception as e:
                print(f"[WARN] Failed to cache {img_path}: {e}", file=sys.stderr)

    # Free backbone memory
    del backbone, processor
    torch.cuda.empty_cache()
    print("Caching completed.")


def train(config: dict):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # BF16 check
    use_bf16 = config["bf16"]
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[WARN] BF16 not supported. Falling back to FP32.", file=sys.stderr)
        use_bf16 = False

    # Phase 1: Cache features
    cache_features(
        data_dir=config["data_dir"],
        backbone_id=config["backbone"],
        max_size=config["max_size"],
        keep_aspect_ratio=config["keep_aspect_ratio"],
        device=device,
        use_bf16=use_bf16,
    )

    # Phase 2: Train StyleNet
    print("\n--- Training Phase ---")

    # Force batch_size=1 when using variable aspect ratios
    batch_size = config["batch_size"]
    if config["keep_aspect_ratio"]:
        batch_size = 1
        print("Keep aspect ratio is ON → forcing batch_size=1")

    dataset = StyleTripletDataset(config["data_dir"])
    print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} artists")

    num_workers = config["num_workers"]
    if os.name == "nt":
        num_workers = min(num_workers, 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(not config["keep_aspect_ratio"]),
    )

    # Build model
    model = StyleNet(
        input_dim=1024,
        hidden_dim=config["hidden_dim"],
        num_tat_layers=config["tat_layers"],
        num_heads=config["tat_heads"],
        output_dim=config["output_dim"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = TripletLoss(margin=config["margin"])

    # Resume
    start_epoch = 0
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config["resume"]:
        print(f"Resuming from {config['resume']}...")
        ckpt = torch.load(config["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed at epoch {start_epoch}")

    accum_steps = config["accumulation_steps"]
    target_epochs = config["epochs"]
    model.train()

    total_steps = len(dataloader) * (target_epochs - start_epoch)
    global_step = 0
    start_time = time.time()

    for epoch in range(start_epoch, target_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{target_epochs}",
        )

        for i, (anchor, positive, negative) in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(use_bf16 and device.type == "cuda"),
            ):
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = criterion(emb_a, emb_p, emb_n) / accum_steps

            loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * accum_steps
            epoch_loss += loss_val
            global_step += 1

            # Progress bar update
            elapsed = time.time() - start_time
            avg_time = elapsed / max(global_step, 1)
            remaining = total_steps - global_step
            etr = datetime.timedelta(seconds=int(remaining * avg_time))
            pbar.set_postfix(loss=f"{loss_val:.4f}", etr=str(etr))

        # Flush remaining accumulated gradients at epoch end
        if (i + 1) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} — Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": {
                "dino_model_id": config["backbone"],
                "hidden_dim": config["hidden_dim"],
                "output_dim": config["output_dim"],
                "num_tat_layers": config["tat_layers"],
                "num_heads": config["tat_heads"],
            },
        }, save_path)
        print(f"Checkpoint saved: {save_path}")

    print("\nTraining finished.")


def main():
    parser = argparse.ArgumentParser(
        description="EgaraNet — Train the StyleNet head with triplet loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", required=True,
                        help="Root directory of the dataset (root/artist/images)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: configs/default.yaml)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--margin", type=float, default=None, help="Triplet margin")
    
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--bf16", action="store_true", default=False,
                                 help="Enable BF16 mixed precision")
    precision_group.add_argument("--fp32", action="store_true", default=False,
                                 help="Force FP32 precision (disables BF16 to override config)")
                                 
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory to save checkpoints")

    # Model overrides
    parser.add_argument("--backbone", default=None,
                        help="Backbone model HuggingFace ID")
    parser.add_argument("--tat_layers", type=int, default=None,
                        help="Number of TAT layers")
    parser.add_argument("--tat_heads", type=int, default=None,
                        help="Number of attention heads in TAT")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=None)

    # Preprocessing overrides
    parser.add_argument("--max_size", type=int, default=None,
                        help="Image max size for preprocessing")

    args = parser.parse_args()
    config = get_config(args)
    train(config)


if __name__ == "__main__":
    main()
