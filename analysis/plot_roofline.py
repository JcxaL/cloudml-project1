#!/usr/bin/env python3
"""Plot roofline charts from analysis/roofline_points.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=Path("analysis/roofline_points.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    return parser.parse_args()


def load_points(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key in ("ai", "attained_gflops", "peak_gflops", "peak_gbps"):
                value = row.get(key)
                row[key] = float(value) if value not in ("", None, "None") else None
            rows.append(row)
    return rows


def plot_env(env: str, points: list[dict], outdir: Path) -> None:
    if not points:
        return

    peak_gflops = points[0]["peak_gflops"]
    peak_gbps = points[0]["peak_gbps"]
    if peak_gflops is None or peak_gbps is None:
        return

    pts_with_ai = [p for p in points if p["ai"] is not None]
    pts_without_ai = [p for p in points if p["ai"] is None]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = [1e-3, 1e4]
    mem_curve = [peak_gbps * xs[0], peak_gbps * xs[1]]
    ax.plot(xs, mem_curve, linestyle="--", color="gray", label=f"Memory roof ({peak_gbps:.0f} GB/s)")
    ax.hlines(peak_gflops, xmin=xs[0], xmax=xs[1], linestyles="--", color="gray", label=f"Compute roof ({peak_gflops:.0f} GFLOP/s)")

    for p in pts_with_ai:
        ax.scatter(
            p["ai"],
            p["attained_gflops"],
            label=f"{p['arch']}-{p['precision']}-bs{p['batch_size']}",
            s=40,
        )

    for p in pts_without_ai:
        ax.scatter(
            1e-3,
            p["attained_gflops"],
            marker="x",
            s=60,
            label=f"{p['arch']}-{p['precision']}-bs{p['batch_size']} (AI n/a)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel("Attained performance (GFLOP/s)")
    ax.set_title(f"Roofline — {env}")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=8, ncol=2, loc="best")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"roofline_{env}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_points(args.points)
    by_env: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_env[row["env"]].append(row)
    for env, pts in by_env.items():
        plot_env(env, pts, args.outdir)
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
