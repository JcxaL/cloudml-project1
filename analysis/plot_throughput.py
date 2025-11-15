#!/usr/bin/env python3
"""Generate throughput vs. batch plots per environment (and host) from logs/metrics.csv."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ENV_HOST_VIEW = {
    "gcp_l4": ["p1-gpu-l4"],
    "rtx4090": ["RTX4090"],
}

MARKERS = {"resnet50": "o", "vgg16": "s", "mobilenet_v2": "^"}
LINESTYLES = {"fp32": "-", "amp": "--"}
DISPLAY_ARCH = {"resnet50": "ResNet-50", "vgg16": "VGG-16", "mobilenet_v2": "MobileNetV2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("logs/metrics.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--envs", nargs="*", help="Optional list of env keys to plot (default: infer)")
    return parser.parse_args()


def fallback_env(label: str) -> str:
    if not label:
        return "unknown"
    parts = label.split("_")
    if parts[0] == "mac" and len(parts) > 1:
        return "_".join(parts[:2])
    if parts[0] == "gcp" and len(parts) > 1:
        return "_".join(parts[:2])
    return parts[0]


def derive_env(row: pd.Series) -> str:
    host = (row.get("hostname") or "").lower()
    backend = (row.get("backend") or "").lower()
    label = row.get("label", "")
    if host == "alvins-macbook":
        if backend == "mps":
            return "mac_mps"
        if backend == "cpu":
            return "mac_cpu"
    if host == "p1-gpu-l4":
        return "gcp_l4"
    if "rtx4090" in host:
        return "rtx4090"
    return fallback_env(label)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["env"] = df.apply(derive_env, axis=1)
    df["host"] = df["hostname"].fillna("unknown")
    df["batch_size"] = df["batch_size"].astype(int)
    df["images_per_sec"] = pd.to_numeric(df["images_per_sec"], errors="coerce")
    df = df.dropna(subset=["images_per_sec"])
    return df


def size_for_batch(batch: int) -> float:
    return 4.0 + math.sqrt(batch)


def color_map_for_hosts(hosts: list[str]) -> dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    mapping: dict[str, str] = {}
    for idx, host in enumerate(sorted(set(hosts))):
        mapping[host] = colors[idx % len(colors)]
    return mapping


def legend_handles(color_map, hosts_present, arches_present, precisions_present):
    model_handles = [
        Line2D([0], [0], marker=MARKERS.get(a, "o"), linestyle="", color="k", label=DISPLAY_ARCH.get(a, a))
        for a in sorted(arches_present)
    ]
    precision_handles = [
        Line2D([0], [0], linestyle=LINESTYLES.get(p, "-"), color="k", label=p.upper())
        for p in sorted(precisions_present)
    ]
    host_handles = [
        Line2D([0], [0], color=color_map[h], lw=2, label=h)
        for h in sorted(hosts_present)
    ]
    return model_handles, precision_handles, host_handles


def plot_env(df: pd.DataFrame, env: str, outdir: Path, host_filter: str | None = None) -> None:
    subset = df[df["env"] == env]
    if host_filter:
        subset = subset[subset["host"] == host_filter]
    if subset.empty:
        return

    agg = (
        subset.groupby(["host", "arch", "precision", "batch_size"])["images_per_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    color_map = color_map_for_hosts(agg["host"].tolist())

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (host, arch, precision), group in agg.groupby(["host", "arch", "precision"]):
        group = group.sort_values("batch_size")
        x = group["batch_size"].to_numpy()
        y = group["mean"].to_numpy()
        yerr = group["std"].to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=MARKERS.get(arch, "o"),
            linestyle=LINESTYLES.get(precision, "-"),
            color=color_map.get(host, "C0"),
            capsize=3,
            linewidth=1.4,
            markersize=5,
        )
        for xi, yi, bs in zip(x, y, group["batch_size"]):
            ax.plot(
                xi,
                yi,
                marker=MARKERS.get(arch, "o"),
                color=color_map.get(host, "C0"),
                linestyle="",
                markersize=size_for_batch(int(bs)) * 0.25,
            )

    title = f"Throughput vs Batch ({env})"
    if host_filter:
        title += f" — {host_filter}"
    ax.set_title(title)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Images / sec")
    batches = sorted(agg["batch_size"].unique())
    ax.set_xticks(batches)
    ax.margins(y=0.10)
    ax.grid(True, which="both", alpha=0.3)

    model_handles, precision_handles, host_handles = legend_handles(
        color_map, agg["host"].unique(), agg["arch"].unique(), agg["precision"].unique()
    )
    leg1 = ax.legend(model_handles, [h.get_label() for h in model_handles], title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    leg2 = ax.legend(precision_handles, [h.get_label() for h in precision_handles], title="Precision", loc="upper left", bbox_to_anchor=(1.02, 0.7), frameon=False)
    leg3 = ax.legend(host_handles, [h.get_label() for h in host_handles], title="Host", loc="upper left", bbox_to_anchor=(1.02, 0.4), frameon=False)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{safe_name(host_filter)}" if host_filter else ""
    fig.tight_layout()
    fig.savefig(outdir / f"throughput_{env}{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.metrics)
    envs = args.envs or sorted(df["env"].unique())
    for env in envs:
        plot_env(df, env, args.outdir, host_filter=None)
        for host in ENV_HOST_VIEW.get(env, []):
            plot_env(df, env, args.outdir, host_filter=host)
    print(f"Wrote throughput figures to {args.outdir}")


if __name__ == "__main__":
    main()
