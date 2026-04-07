"""
EgaraNet — Inference CLI.

Extract style vectors from illustrations using a trained EgaraNet model.

Usage:
    # Single image
    python inference.py --model checkpoints/epoch_10.pth --input image.png

    # Directory of images (outputs CSV)
    python inference.py --model checkpoints/epoch_10.pth --input ./images/ --output vectors.csv

    # Compare two images
    python inference.py --model checkpoints/epoch_10.pth --compare a.png b.png

    # Using HuggingFace Hub model
    python inference.py --hf Columba1198/EgaraNet --input image.png

    # BF16 mixed precision inference
    python inference.py --model checkpoints/epoch_10.pth --input image.png --bf16

    # With config file
    python inference.py --model checkpoints/epoch_10.pth --config configs/default.yaml --input image.png
"""

import argparse
import base64
import csv
import os
import sys

import numpy as np
import torch

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egaranet.model import EgaraNet, cosine_similarity
from egaranet.preprocessing import collect_image_paths


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not YAML_AVAILABLE:
        print("[WARN] PyYAML not installed. Using default configuration.", file=sys.stderr)
        return {}
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_inference_config(args: argparse.Namespace) -> dict:
    """Merge config file with CLI arguments (CLI takes priority)."""
    config = {}

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    elif os.path.exists("configs/default.yaml"):
        config = load_config("configs/default.yaml")

    preproc_cfg = config.get("preprocessing", {})
    infer_cfg = config.get("inference", {})

    if args.fp32:
        use_bf16 = False
    elif args.bf16:
        use_bf16 = True
    else:
        use_bf16 = infer_cfg.get("bf16", True)

    return {
        "size": args.size or preproc_cfg.get("max_size", 512),
        "keep_aspect_ratio": (not args.no_keep_ratio) if args.no_keep_ratio else preproc_cfg.get("keep_aspect_ratio", True),
        "bf16": use_bf16,
    }


def main():
    parser = argparse.ArgumentParser(
        description="EgaraNet — Style Vector Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Path to .pth checkpoint")
    model_group.add_argument("--hf", metavar="MODEL_ID",
                             help="HuggingFace Hub model ID (e.g., Columba1198/EgaraNet)")

    # Input mode
    parser.add_argument("--input", help="Image file or directory")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--compare", nargs=2, metavar=("IMG_A", "IMG_B"),
                        help="Compare two images")

    # Config
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: configs/default.yaml)")

    # Options (override config values)
    parser.add_argument("--size", type=int, default=None,
                        help="Image size for preprocessing (default: from config or 512)")
                        
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--bf16", action="store_true", default=False,
                                 help="Use BF16 mixed precision inference")
    precision_group.add_argument("--fp32", action="store_true", default=False,
                                 help="Force FP32 precision (disables BF16 to override config)")
                                 
    parser.add_argument("--no-keep-ratio", action="store_true", default=False,
                        help="Disable aspect ratio preservation (square resize)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Disable recursive directory scan")

    args = parser.parse_args()

    if not args.input and not args.compare:
        parser.error("--input or --compare is required")

    # Merge config with CLI args
    cfg = get_inference_config(args)
    use_bf16 = cfg["bf16"]
    keep_ratio = cfg["keep_aspect_ratio"]
    size = cfg["size"]

    # BF16 availability check
    if use_bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        print("[WARN] BF16 not supported on this device. Falling back to FP32.",
              file=sys.stderr)
        use_bf16 = False

    # Load model
    print("Loading model...", file=sys.stderr)
    if args.model:
        model = EgaraNet.from_checkpoint(args.model)
    else:
        model = EgaraNet.from_huggingface(args.hf)
    print(f"Model loaded on {model.device}" + (" (BF16)" if use_bf16 else ""),
          file=sys.stderr)

    # --- Compare mode ---
    if args.compare:
        vec_a = model.extract_style_vector(
            args.compare[0], size, keep_ratio, use_bf16,
        )
        vec_b = model.extract_style_vector(
            args.compare[1], size, keep_ratio, use_bf16,
        )
        sim = cosine_similarity(vec_a, vec_b)
        print(f"Cosine Similarity: {sim:.6f} ({sim * 100:.2f}%)")
        return

    # --- Extract mode ---
    files = collect_image_paths(args.input, recursive=not args.no_recursive)
    if not files:
        print("No images found.", file=sys.stderr)
        return

    print(f"Processing {len(files)} image(s)...", file=sys.stderr)

    rows = []
    for i, fp in enumerate(files):
        try:
            vec = model.extract_style_vector(fp, size, keep_ratio, use_bf16)

            # Encode as base64 for compact CSV storage
            b64 = base64.b64encode(
                vec.astype(np.float32).tobytes()
            ).decode("ascii")
            rows.append({"path": fp, "embedding": b64})

            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print(f"  [{i + 1}/{len(files)}]", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Skipped {fp}: {e}", file=sys.stderr)

    if not rows:
        print("No vectors extracted.", file=sys.stderr)
        return

    # Single image without --output: print to stdout
    if len(rows) == 1 and not args.output:
        vec = np.frombuffer(
            base64.b64decode(rows[0]["embedding"]), dtype=np.float32
        )
        print(f"Path: {rows[0]['path']}")
        print(f"Dim:  {len(vec)}")
        print(f"Vec:  {vec.tolist()}")
        return

    # Multiple images or explicit --output: write CSV
    out_path = args.output or "style_vectors.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "embedding"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} vectors to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
