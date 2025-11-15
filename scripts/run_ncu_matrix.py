#!/usr/bin/env python3
"""Batch Nsight Compute runs for the RTX 4090 experiment matrix.

Generates one capture per (arch, precision, batch size) combo so that
analysis/roofline.py can find dram bytes + flop counts.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ARCH_BATCHES = {
    "resnet50": (64, 128, 256),
    "vgg16": (64, 128, 256),
    "mobilenet_v2": (64, 128, 256),
}
PRECISIONS = ("fp32", "amp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/imagenet-mini", help="ImageNet subset root")
    parser.add_argument("--label-prefix", default="rtx4090", help="Label prefix used in log/metric names")
    parser.add_argument("--warmup-iters", default=5, type=int, help="Warmup iterations before profiling")
    parser.add_argument("--iters", default=10, type=int, help="Measured iterations under Nsight")
    parser.add_argument("--workers", default=8, type=int, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned memory when targeting CUDA")
    parser.add_argument(
        "--arches",
        nargs="*",
        choices=sorted(ARCH_BATCHES.keys()),
        default=sorted(ARCH_BATCHES.keys()),
        help="Subset of architectures to profile",
    )
    parser.add_argument("--ncu-dir", default="logs/ncu", help="Where to write Nsight CSV captures")
    parser.add_argument("--metrics-file", default="code/metric_names_ncu.txt", help="Metric list file")
    parser.add_argument("--backend", default="cuda", help="torch backend passed to run_train.py")
    parser.add_argument("--extra-args", nargs="*", default=[], help="Extra args forwarded to run_train.py")
    parser.add_argument(
        "--nvtx",
        dest="nvtx",
        action="store_true",
        default=True,
        help="Emit NVTX ranges in run_train.py (needed when --profile-from-start off)",
    )
    parser.add_argument("--no-nvtx", dest="nvtx", action="store_false", help="Disable NVTX range emission")
    parser.add_argument("--force", action="store_true", help="Re-run even if CSV already exists")
    return parser.parse_args()


def load_metric_flag(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Nsight metric list: {path}")
    return path.read_text().strip().replace("\n", "")


def main() -> None:
    args = parse_args()
    metrics_flag = load_metric_flag(Path(args.metrics_file))
    ncu_dir = Path(args.ncu_dir)
    ncu_dir.mkdir(parents=True, exist_ok=True)

    for arch in args.arches:
        batches = ARCH_BATCHES[arch]
        for precision in PRECISIONS:
            for batch in batches:
                label = f"{args.label_prefix}_{arch}_{precision}_bs{batch}_ncu"
                log_file = ncu_dir / f"{label}.csv"
                if log_file.exists() and not args.force:
                    if log_file.stat().st_size > 0:
                        print(f"[skip] {label} (capture exists)")
                        continue
                    log_file.unlink()

                base_cmd = [
                    "ncu",
                    "--profile-from-start",
                    "off",
                    "--target-processes",
                    "all",
                    "--metrics",
                    metrics_flag,
                    "--csv",
                    "--page",
                    "raw",
                    "--log-file",
                    str(log_file),
                    "python",
                    "code/run_train.py",
                    "--data",
                    args.data,
                    "--arch",
                    arch,
                    "--batch-size",
                    str(batch),
                    "--warmup-iters",
                    str(args.warmup_iters),
                    "--iters",
                    str(args.iters),
                    "--workers",
                    str(args.workers),
                    "--precision",
                    precision,
                    "--backend",
                    args.backend,
                    "--label",
                    label,
                ]
                if args.pin_memory:
                    base_cmd.append("--pin-memory")
                if args.nvtx:
                    base_cmd.append("--nvtx")
                base_cmd.extend(args.extra_args)

                print(f"[ncu] Profiling {label}")
                subprocess.run(base_cmd, check=True)

    print("Nsight Compute batch complete.")


if __name__ == "__main__":
    main()
