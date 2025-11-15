#!/usr/bin/env python3
"""Aggregate metrics + profiler logs into roofline-ready points."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

METRIC_FILE = Path("code/metric_names_ncu.txt")
TRAIN_FACTOR = 2.0  # approximate forward+backward cost multiplier for training
PRECISION_SYNONYMS = {
    "amp": ["amp", "mixed", "fp16", "bf16", "tensorcore"],
    "fp32": ["fp32", "float32"],
    "fp16": ["fp16", "half"],
    "bf16": ["bf16", "bfloat16"],
}


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
    if not METRIC_FILE.exists():
        return [
            "flop_count_sp",
            "flop_count_hp",
            "flop_count_dp",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "pcie__read_bytes.sum",
            "pcie__write_bytes.sum",
        ]
    text = METRIC_FILE.read_text().strip()
    return [m.strip() for m in text.split(",") if m.strip()]


def load_metrics_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
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
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Empty or invalid peaks file: {path}")
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
    with path.open(newline="") as fh:
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


def estimate_training_flops_per_iter(entry: dict, batch_size: int, train_factor: float) -> Optional[float]:
    if "flops_per_batch_forward" in entry:
        return entry["flops_per_batch_forward"] * train_factor
    if "flops_per_sample_forward" in entry:
        return entry["flops_per_sample_forward"] * batch_size * train_factor
    macs = entry.get("mult_adds_per_sample_forward") or entry.get("mult_adds")
    if macs is not None:
        return 2.0 * macs * batch_size * train_factor
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_env_spec(label: str, row: dict, peaks: Dict[str, dict]) -> Optional[tuple[str, dict]]:
    candidates = [label.split("_")[0], row["backend"].lower()]
    device_name = _normalize(row.get("device_name") or "")
    for key in peaks.keys():
        norm_key = _normalize(key)
        if norm_key and (norm_key in device_name or device_name in norm_key):
            candidates.append(key)
    for candidate in candidates:
        spec = peaks.get(candidate)
        if spec:
            return candidate, spec
    return None


def _resolve_precision_key(value_dict: dict, precision: str) -> Optional[float]:
    if precision in value_dict:
        return value_dict[precision]
    for synonym in PRECISION_SYNONYMS.get(precision, []):
        if synonym in value_dict:
            return value_dict[synonym]
    if "fp32" in value_dict:
        return value_dict["fp32"]
    for v in value_dict.values():
        return v
    return None


def resolve_peak(spec: dict, key: str, precision: str) -> Optional[float]:
    value = spec.get(key)
    if isinstance(value, dict):
        return _resolve_precision_key(value, precision)
    return value


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
        env = resolve_env_spec(label, row, peaks)
        if env is None:
            print(f"[warn] No peak spec for label '{label}', skipping")
            continue
        env_key, env_spec = env

        flops = None
        dram_bytes = None
        if row["backend"] == "cuda":
            ncu_path = args.ncu_dir / f"{label}.csv"
            ncu_vals = parse_ncu_csv(ncu_path, metric_names)
            if ncu_vals:
                flop_sp = ncu_vals.get("flop_count_sp", 0.0)
                flop_hp = ncu_vals.get("flop_count_hp", 0.0)
                flop_dp = ncu_vals.get("flop_count_dp", 0.0)
                total_flops = flop_sp + flop_hp + flop_dp
                if total_flops > 0:
                    flops = total_flops
                read_bytes = ncu_vals.get("dram__bytes_read.sum")
                write_bytes = ncu_vals.get("dram__bytes_write.sum")
                if read_bytes is not None and write_bytes is not None:
                    dram_bytes = read_bytes + write_bytes
        model = row["arch"]
        batch_key = str(row["batch_size"])
        model_entry = complexity.get(model, {})
        batch_entry = model_entry.get(batch_key)
        if flops is None and batch_entry is not None:
            per_iter = estimate_training_flops_per_iter(batch_entry, row["batch_size"], TRAIN_FACTOR)
            if per_iter is not None:
                flops = per_iter * row["measured_iters"]

        if flops is None or row["elapsed_sec"] <= 0:
            continue

        attained = flops / row["elapsed_sec"] / 1e9
        ai = (flops / dram_bytes) if (dram_bytes and dram_bytes > 0) else None

        peak_gflops = resolve_peak(env_spec, "peak_gflops", row["precision"])
        peak_gbps = resolve_peak(env_spec, "peak_gbps", row["precision"])
        if peak_gflops is None or peak_gbps is None:
            print(f"[warn] Peak specs missing for '{label}', skipping")
            continue

        if ai is None and peak_gbps:
            ai_lb = attained / peak_gbps if peak_gbps > 0 else None
            if ai_lb is not None:
                knee = peak_gflops / peak_gbps if peak_gbps > 0 else None
                if knee and peak_gflops and peak_gflops > 0 and (attained / peak_gflops) >= 0.7:
                    ai = max(ai_lb, knee)
                else:
                    ai = ai_lb

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
                "peak_gflops": peak_gflops,
                "peak_gbps": peak_gbps,
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
        writer.writerows(out_rows)
    print(f"Wrote {args.output} with {len(out_rows)} points")


if __name__ == "__main__":
    main()
