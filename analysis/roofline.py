#!/usr/bin/env python3
"""Aggregate metrics + profiler logs into roofline-ready points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

METRIC_FILE = Path("code/metric_names_ncu.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("logs/metrics.csv"))
    parser.add_argument("--ncu-dir", type=Path, default=Path("logs/ncu"))
    parser.add_argument("--peaks", type=Path, default=Path("analysis/peaks.json"))
    parser.add_argument(
        "--complexity-json",
        type=Path,
        default=Path("logs/model_summaries/summary.json"),
        help="Output from scripts/model_complexity.py (optional but needed for non-CUDA points)",
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/roofline_points.csv"))
    return parser.parse_args()


def load_metric_names() -> Iterable[str]:
    text = METRIC_FILE.read_text().strip()
    return [m.strip() for m in text.split(",") if m.strip()]


def load_metrics_csv(path: Path) -> Iterable[dict]:
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["batch_size"] = int(row["batch_size"])
            row["warmup_iters"] = int(row["warmup_iters"])
            row["measured_iters"] = int(row["measured_iters"])
            row["elapsed_sec"] = float(row["elapsed_sec"])
            row["images_per_sec"] = float(row["images_per_sec"])
            rows.append(row)
    return rows


def load_peaks(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    for key, spec in data.items():
        if spec.get("peak_gflops") in (None, 0) or spec.get("peak_gbps") in (None, 0):
            raise ValueError(f"Peak specs missing for '{key}' in {path}; fill in real values")
    return data


def load_complexity(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_ncu_csv(path: Path, metric_names: Iterable[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if not path.exists():
        return values
    wanted = set(metric_names)
    with path.open() as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            for idx, cell in enumerate(row):
                name = cell.strip().strip('"')
                if name in wanted:
                    val: Optional[float] = None
                    for nxt in row[idx + 1 :]:
                        try:
                            val = float(nxt)
                            break
                        except ValueError:
                            continue
                    if val is not None:
                        values[name] = val
    return values


def main() -> None:
    args = parse_args()
    metric_names = list(load_metric_names())
    rows = load_metrics_csv(args.metrics)
    peaks = load_peaks(args.peaks)
    complexity = load_complexity(args.complexity_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for row in rows:
        label = row.get("label") or f"{row['backend']}_{row['arch']}_bs{row['batch_size']}"
        env_key = label.split("_")[0]
        env_spec = peaks.get(env_key)
        if env_spec is None:
            # fallback to backend-based key
            env_spec = peaks.get(row["backend"].lower())
            env_key = row["backend"].lower()
        if env_spec is None:
            print(f"[warn] No peak spec for label '{label}', skipping")
            continue

        flops = None
        dram_bytes = None
        if row["backend"] == "cuda":
            ncu_path = args.ncu_dir / f"{label}.csv"
            ncu_vals = parse_ncu_csv(ncu_path, metric_names)
            if ncu_vals:
                flop_sp = ncu_vals.get("flop_count_sp", 0.0)
                flop_hp = ncu_vals.get("flop_count_hp", 0.0)
                flops = flop_sp + flop_hp
                dram_bytes = ncu_vals.get("dram__bytes_read.sum", 0.0) + ncu_vals.get("dram__bytes_write.sum", 0.0)
        else:
            model = row["arch"]
            batch_key = str(row["batch_size"])
            model_entry = complexity.get(model, {})
            batch_entry = model_entry.get(batch_key)
            if batch_entry is not None:
                flops = batch_entry["mult_adds"] * row["measured_iters"]

        attained = None
        ai = None
        if flops is not None and row["elapsed_sec"] > 0:
            attained = flops / row["elapsed_sec"] / 1e9
        if flops is not None and dram_bytes:
            ai = flops / dram_bytes

        out_rows.append(
            {
                "label": label,
                "env": env_key,
                "arch": row["arch"],
                "backend": row["backend"],
                "batch_size": row["batch_size"],
                "precision": row["precision"],
                "elapsed_sec": row["elapsed_sec"],
                "images_per_sec": row["images_per_sec"],
                "ai": ai,
                "attained_gflops": attained,
                "peak_gflops": env_spec["peak_gflops"],
                "peak_gbps": env_spec["peak_gbps"],
            }
        )

    fieldnames = [
        "label",
        "env",
        "arch",
        "backend",
        "batch_size",
        "precision",
        "elapsed_sec",
        "images_per_sec",
        "ai",
        "attained_gflops",
        "peak_gflops",
        "peak_gbps",
    ]
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print(f"Wrote {args.output} with {len(out_rows)} points")


if __name__ == "__main__":
    main()
