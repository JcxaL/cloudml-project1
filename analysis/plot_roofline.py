#!/usr/bin/env python3
"""Plot roofline charts per environment using analysis/roofline_points.csv."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

COLOR_MAP = {
    "resnet50": "#1f77b4",
    "vgg16": "#2ca02c",
    "mobilenet_v2": "#ff7f0e",
}
MARKERS = {"resnet50": "o", "vgg16": "s", "mobilenet_v2": "^"}
DISPLAY_ARCH = {"resnet50": "ResNet-50", "vgg16": "VGG-16", "mobilenet_v2": "MobileNetV2"}
CUDA_ENVS = {"gcp", "gcp_l4", "rtx4090"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=Path("analysis/roofline_points.csv"))
    parser.add_argument("--peaks", type=Path, default=Path("analysis/peaks.json"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--labels", type=int, default=3, help="Annotations per model")
    return parser.parse_args()


def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["env"] = df["env"].str.lower()
    df["arch"] = df["arch"].str.lower()
    df["precision"] = df["precision"].str.lower()
    df["backend"] = df["backend"].str.lower()
    df["batch_size"] = df["batch_size"].astype(int)
    df = df[(df["attained_gflops"].notna()) & (df["ai"].notna())]
    return df


def marker_size(batch: int) -> float:
    return 20 + 4 * math.sqrt(batch)


def resolve_peak(peaks: dict, env: str):
    spec = peaks.get(env) or peaks.get(env.split("_")[0])
    if not spec:
        raise KeyError(f"No peaks for env {env}")
    peak_gbps = spec.get("peak_gbps")
    peak_gflops_info = spec.get("peak_gflops")
    if isinstance(peak_gflops_info, dict):
        peak_gflops = max(peak_gflops_info.values())
    else:
        peak_gflops = peak_gflops_info
    return peak_gflops, peak_gbps


def annotate_top(ax, df_env: pd.DataFrame, max_labels: int) -> None:
    for arch, group in df_env.groupby("arch"):
        top = group.sort_values("attained_gflops", ascending=False).head(max_labels)
        for _, row in top.iterrows():
            label = f"{DISPLAY_ARCH.get(arch, arch)} bs{int(row['batch_size'])}"
            ax.annotate(label, (row["ai"], row["attained_gflops"]), xytext=(4, 4), textcoords="offset points", fontsize=7)


def plot_env(df: pd.DataFrame, peaks: dict, env: str, outdir: Path, max_labels: int) -> None:
    df_env = df[df["env"] == env]
    if df_env.empty:
        return
    peak_gflops, peak_gbps = resolve_peak(peaks, env)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(right=0.72)

    xmin = max(1e-3, 0.5 * df_env["ai"].min())
    xmax = min(1e4, 1.5 * df_env["ai"].max())
    ymin = max(1e-2, 0.5 * df_env["attained_gflops"].min())
    ymax = peak_gflops * 1.2

    ax.set_xscale("log")
    ax.set_yscale("log")

    xs = np.logspace(np.log10(xmin), np.log10(xmax), 256)
    ax.plot(xs, peak_gbps * xs, linestyle="--", color="gray", linewidth=1, label=f"Memory roof ({int(peak_gbps)} GB/s)")
    ax.hlines(peak_gflops, xmin, xmax, linestyles="--", color="gray", linewidth=1, label=f"Compute roof ({int(peak_gflops)} GFLOP/s)")
    knee = peak_gflops / peak_gbps
    ax.vlines(knee, ymin, peak_gflops, linestyles=":", color="gray", linewidth=0.8)
    ax.text(knee, peak_gflops * 0.75, f"knee≈{knee:.1f} F/B", rotation=90, va="top", ha="right", fontsize=8, color="gray")

    for _, row in df_env.iterrows():
        arch = row["arch"]
        marker = MARKERS.get(arch, "o")
        color = COLOR_MAP.get(arch, "C0")
        est = row["backend"] == "cuda"
        ax.scatter(
            row["ai"],
            row["attained_gflops"],
            marker=marker,
            s=36,
            facecolors="none" if est else color,
            edgecolors=color,
            linewidth=1.0,
            zorder=3,
        )

    annotate_top(ax, df_env, max_labels)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel("Attained performance (GFLOP/s)")
    ax.set_title(f"Roofline — {env.upper()}")
    ax.grid(which="both", alpha=0.25)

    model_handles = [
        Line2D([0], [0], marker=MARKERS.get(a, "o"), color=COLOR_MAP.get(a, "C0"), linestyle="", label=DISPLAY_ARCH.get(a, a))
        for a in sorted(df_env["arch"].unique())
    ]
    est_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", label="Estimated intensity")
    leg1 = ax.legend(model_handles, [h.get_label() for h in model_handles], title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    leg2 = ax.legend([est_handle], [est_handle.get_label()], loc="upper left", bbox_to_anchor=(1.02, 0.7), frameon=False)
    ax.add_artist(leg1)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    fig.savefig(outdir / f"roofline_{env}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_points(args.points)
    peaks = json.loads(args.peaks.read_text())
    for env in sorted(df["env"].unique()):
        plot_env(df, peaks, env, args.outdir, args.labels)
    print(f"Wrote roofline figures to {args.outdir}")


if __name__ == "__main__":
    main()
