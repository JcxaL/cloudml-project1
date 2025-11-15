#!/usr/bin/env python3
"""Plot roofline charts per environment using analysis/roofline_points.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MARKERS = {"resnet50": "o", "vgg16": "s", "mobilenet_v2": "^"}
DISPLAY_ARCH = {"resnet50": "ResNet-50", "vgg16": "VGG-16", "mobilenet_v2": "MobileNetV2"}
CUDA_ENVS = {"gcp", "gcp_l4", "rtx4090"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=Path("analysis/roofline_points.csv"))
    parser.add_argument("--peaks", type=Path, default=Path("analysis/peaks.json"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--max-labels", type=int, default=3, help="Annotations per model")
    return parser.parse_args()


def load_data(points_path: Path) -> pd.DataFrame:
    df = pd.read_csv(points_path)
    df = df.copy()
    df = df.dropna(subset=["attained_gflops", "ai"])
    df["env"] = df["env"].str.lower()
    df["arch"] = df["arch"].str.lower()
    df["precision"] = df["precision"].str.lower()
    df["batch_size"] = df["batch_size"].astype(int)
    return df


def color_map_for_arches(arches):
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    mapping = {}
    for idx, arch in enumerate(sorted(set(arches))):
        mapping[arch] = colors[idx % len(colors)]
    return mapping


def draw_roof(ax, peak_gflops, peak_gbps, x_min, x_max):
    knee = peak_gflops / peak_gbps
    xs = np.logspace(np.log10(x_min), np.log10(min(knee, x_max)), 100)
    ax.plot(xs, peak_gbps * xs, linestyle="--", color="gray", linewidth=1, label=f"Memory roof ({int(peak_gbps)} GB/s)")
    if x_max > knee:
        ax.hlines(peak_gflops, knee, x_max, colors="gray", linestyles="--", linewidth=1, label=f"Compute roof ({int(peak_gflops)} GFLOP/s)")
    ax.axvline(knee, linestyle=":", color="gray", linewidth=0.8)
    ax.text(knee, peak_gflops * 0.6, "knee", rotation=90, va="top", ha="right", fontsize=8, color="gray")
    return knee


def annotate_points(ax, df_env, max_labels):
    for arch, group in df_env.groupby("arch"):
        top = group.sort_values("attained_gflops", ascending=False).head(max_labels)
        for _, row in top.iterrows():
            label = f"{DISPLAY_ARCH.get(arch, arch)} bs{int(row['batch_size'])}"
            ax.annotate(label, (row["ai"], row["attained_gflops"]), xytext=(4, 4), textcoords="offset points", fontsize=7)


def plot_env(df: pd.DataFrame, peaks: dict, env: str, args):
    df_env = df[df["env"] == env]
    if df_env.empty:
        return
    peak_spec = peaks.get(env) or peaks.get(env.split("_")[0])
    if not peak_spec:
        return
    peak_gbps = peak_spec.get("peak_gbps", 1.0)
    peak_gflops_info = peak_spec.get("peak_gflops")
    if isinstance(peak_gflops_info, dict):
        peak_gflops = max(peak_gflops_info.values())
    else:
        peak_gflops = peak_gflops_info

    color_map = color_map_for_arches(df_env["arch"].unique())

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.subplots_adjust(right=0.72)

    xmin = max(1e-3, 0.5 * df_env["ai"].min())
    xmax = min(1e4, 2.0 * df_env["ai"].max())
    ymin = max(1e-2, 0.5 * df_env["attained_gflops"].min())
    ymax = 1.5 * peak_gflops

    knee = draw_roof(ax, peak_gflops, peak_gbps, xmin, xmax)

    for _, row in df_env.iterrows():
        arch = row["arch"]
        est = env in CUDA_ENVS
        ax.plot(
            row["ai"],
            row["attained_gflops"],
            marker=MARKERS.get(arch, "o"),
            color=color_map[arch],
            markersize=4 + math.sqrt(row["batch_size"]),
            markerfacecolor="none" if est else color_map[arch],
            markeredgecolor=color_map[arch],
            linestyle="",
        )

    annotate_points(ax, df_env, args.max_labels)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel("Attained performance (GFLOP/s)")
    ax.set_title(f"Roofline — {env.upper()}")
    ax.grid(which="both", alpha=0.25)

    model_handles = [
        Line2D([0], [0], marker=MARKERS.get(a, "o"), linestyle="", color=color_map[a], label=DISPLAY_ARCH.get(a, a))
        for a in sorted(color_map.keys())
    ]
    hollow_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", label="Estimated intensity")
    leg1 = ax.legend(model_handles, [h.get_label() for h in model_handles], title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    leg2 = ax.legend([hollow_handle], ["Estimated intensity"], loc="upper left", bbox_to_anchor=(1.02, 0.7), frameon=False)
    ax.add_artist(leg1)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.45), frameon=False)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    fig.savefig(args.outdir / f"roofline_{env}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.points)
    peaks = json.loads(args.peaks.read_text())
    for env in sorted(df["env"].unique()):
        plot_env(df, peaks, env, args)
    print(f"Wrote roofline figures to {args.outdir}")


if __name__ == "__main__":
    main()
