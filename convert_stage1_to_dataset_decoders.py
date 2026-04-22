#!/usr/bin/env python3
"""Convert stage-1 checkpoints to per-dataset decoder weights."""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone stage-1 encoder weights into a per-dataset decoder model.")
    parser.add_argument("--input", required=True, help="Path to the original stage-1 checkpoint (state_dict or training bundle).")
    parser.add_argument("--output", required=True, help="Destination path for the converted checkpoint.")
    parser.add_argument("--encoder-name", default="R50-ViT-B_16", help="Backbone identifier used during stage-1 training.")
    parser.add_argument("--encoder-weights", default=None, help="Optional pretrained weights file for the encoder backbone.")
    parser.add_argument("--image-size", type=int, default=256, help="Input resolution used to instantiate the decoder blocks.")
    parser.add_argument("--regression-heatmap-size", type=int, default=64, help="Heatmap size for regression heads.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def resolve_state_dict(raw_checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(raw_checkpoint, dict):
        if "model_state" in raw_checkpoint:
            return raw_checkpoint["model_state"]
        if "state_dict" in raw_checkpoint:
            return raw_checkpoint["state_dict"]
    return raw_checkpoint


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("Building per-dataset decoder model...")
    model = MultiTaskModelFactory(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        task_configs=TASK_CONFIGURATIONS,
        image_size=args.image_size,
        regression_heatmap_size=args.regression_heatmap_size,
        per_dataset_decoders=True,
    ).to(device)

    checkpoint_path = Path(args.input)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading source checkpoint: {checkpoint_path}")
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = resolve_state_dict(raw_checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    print("Encoder weights loaded. Summary of key mismatches:")
    print(f"  Missing keys   : {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Converted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
