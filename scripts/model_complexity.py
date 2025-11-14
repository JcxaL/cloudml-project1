#!/usr/bin/env python3
"""Dump model parameter + activation summaries for report complexity section."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
import torchvision as tv
from torchinfo import summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "vgg16", "mobilenet_v2"],
        help="torchvision model names to summarize",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[32],
        help="batch sizes used to estimate activation footprint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/model_summaries"),
        help="where to write summary text files",
    )
    parser.add_argument("--device", default="cpu", help="device for the dry-run (cpu is fine)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    aggregate = {}

    for arch in args.models:
        if not hasattr(tv.models, arch):
            raise SystemExit(f"torchvision.models has no arch named {arch}")
        model_fn = getattr(tv.models, arch)
        model = model_fn(weights=None).to(device)
        model.eval()
        for bs in args.batch_sizes:
            info = summary(
                model,
                input_size=(bs, 3, 224, 224),
                verbose=0,
                col_names=("input_size", "output_size", "num_params", "mult_adds"),
            )
            out_path = args.output_dir / f"{arch}_bs{bs}.txt"
            out_path.write_text(str(info))
            print(f"Wrote {out_path}")

            total_params = int(info.total_params)
            total_mult_adds = float(info.total_mult_adds or 0.0)
            per_sample_mult_adds = total_mult_adds / bs if bs else total_mult_adds

            aggregate.setdefault(arch, {})[str(bs)] = {
                "params": total_params,
                "mult_adds_per_batch_forward": total_mult_adds,
                "mult_adds_per_sample_forward": per_sample_mult_adds,
            }

    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
