#!/usr/bin/env python3
"""Minimal driver around torchvision ImageNet models for short, timed runs.

Adds knobs for reproducibility (seeds, deterministic algorithms), data loading
(no augment vs. training augment, prefetch factor, pinning), performance toggles
(channels_last, grad accumulation), and optional NVTX markers for Nsight.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root folder with train/ subdir")
    parser.add_argument("--arch", default="resnet50", help="torchvision model name (e.g., resnet50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Iterations to warm up (not timed)")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32", help="Math precision mode")
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Torch device backend to target",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label", default="", help="Optional label stored in metrics.csv")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Where to write metrics.csv")
    parser.add_argument("--pin-memory", action="store_true", help="Enable CUDA pinned memory for DataLoader")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (workers > 0)")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable RandomResizedCrop/HorizontalFlip; use Resize+CenterCrop",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Run model/inputs in channels_last format (recommended on CUDA ConvNets)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of microbatches to accumulate before stepping",
    )
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX range over measured region (CUDA only)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch deterministic algorithms (less variance, may hurt perf)",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn.benchmark (great for fixed shapes, off by default)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use torchvision FakeData instead of reading from disk (sanity checks)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")

    return torch.device("cpu")


def resolve_model(arch: str) -> nn.Module:
    if not hasattr(models, arch):
        raise ValueError(f"torchvision.models has no architecture named '{arch}'")
    model_fn = getattr(models, arch)
    model = model_fn(weights=None)
    return model


def build_dataloader(
    data_root: Path,
    batch_size: int,
    workers: int,
    use_pin_mem: bool,
    device: torch.device,
    *,
    synthetic: bool,
    total_steps: int,
    no_augment: bool,
    prefetch_factor: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    train_dir = data_root / "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if synthetic:
        samples_needed = max(batch_size * max(total_steps, 1), batch_size)
        dataset = datasets.FakeData(
            size=samples_needed,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    else:
        if not train_dir.exists():
            raise FileNotFoundError(f"Expected ImageNet-style train folder at {train_dir}")
        if no_augment:
            tfm = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            tfm = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        dataset = datasets.ImageFolder(train_dir, tfm)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but batch_size={batch_size}. "
            "Reduce batch size or use --synthetic for dry runs."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = {}
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["worker_init_fn"] = _worker_init_fn
    else:
        loader_kwargs["worker_init_fn"] = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not synthetic,
        num_workers=workers,
        pin_memory=use_pin_mem and device.type == "cuda",
        drop_last=True,
        persistent_workers=workers > 0,
        generator=generator,
        **loader_kwargs,
    )
    return loader


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def _build_grad_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module else None
    if grad_scaler_cls is None:
        cuda_mod = getattr(torch, "cuda", None)
        amp_cuda = getattr(cuda_mod, "amp", None) if cuda_mod else None
        grad_scaler_cls = getattr(amp_cuda, "GradScaler", None) if amp_cuda else None
    if grad_scaler_cls is None:
        return None
    try:
        return grad_scaler_cls(device_type=device.type, enabled=enabled)
    except TypeError:
        return grad_scaler_cls(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    amp_module = getattr(torch, "amp", None)
    autocast_fn = getattr(amp_module, "autocast", None) if amp_module else None
    if autocast_fn is None:
        cuda_mod = getattr(torch, "cuda", None)
        amp_cuda = getattr(cuda_mod, "amp", None) if cuda_mod else None
        autocast_fn = getattr(amp_cuda, "autocast", None) if amp_cuda else None
    if autocast_fn is None:
        return nullcontext()
    try:
        return autocast_fn(device_type=device.type, enabled=enabled)
    except TypeError:
        return autocast_fn(enabled=enabled)


def main() -> int:
    args = parse_args()
    grad_accum_steps = max(1, args.grad_accum_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    device = select_device(args.backend)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if args.deterministic:
            torch.backends.cudnn.deterministic = True

    model = resolve_model(args.arch).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    amp_enabled = args.precision == "amp"
    if amp_enabled and device.type != "cuda":
        print("[warn] AMP requested but CUDA backend is not active; falling back to fp32", file=sys.stderr)
        amp_enabled = False

    scaler = _build_grad_scaler(device, amp_enabled)

    total_steps = args.warmup_iters + args.iters
    if total_steps == 0:
        raise ValueError("At least one warmup or measured iteration is required")

    loader = build_dataloader(
        args.data,
        args.batch_size,
        args.workers,
        args.pin_memory,
        device,
        synthetic=args.synthetic,
        total_steps=total_steps,
        no_augment=args.no_augment,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed + 1337,
    )
    non_blocking = device.type in {"cuda", "mps"}

    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch

    def train_microbatch(batch):
        images, target = batch
        images = images.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        if args.channels_last and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        with _autocast_context(device, amp_enabled):
            output = model(images)
            loss = criterion(output, target)
        return loss

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            batch = next_batch()
            loss = train_microbatch(batch) / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    for _ in range(args.warmup_iters):
        train_step()

    maybe_sync(device)
    nvtx_active = False
    if args.nvtx and device.type == "cuda":
        import torch.cuda.nvtx as nvtx

        nvtx.range_push("measured")
        nvtx_active = True
    start = time.monotonic()
    for _ in range(args.iters):
        train_step()
    maybe_sync(device)
    elapsed = time.monotonic() - start
    if nvtx_active:
        nvtx.range_pop()

    if elapsed == 0:
        raise RuntimeError("Measured elapsed time is zero; increase --iters or check system timers")

    images_processed = args.batch_size * args.iters * grad_accum_steps
    throughput = images_processed / elapsed

    print(
        f"MEASURED_SEC={elapsed:.6f} IMGS_PER_SEC={throughput:.2f} "
        f"BATCH={args.batch_size} ARCH={args.arch} BACKEND={device.type.upper()}"
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "metrics.csv"
    fieldnames = [
        "timestamp",
        "hostname",
        "arch",
        "backend",
        "device_name",
        "batch_size",
        "warmup_iters",
        "measured_iters",
        "precision",
        "lr",
        "momentum",
        "weight_decay",
        "elapsed_sec",
        "images_per_sec",
        "images_processed",
        "data_root",
        "label",
        "torch_version",
        "channels_last",
        "grad_accum_steps",
        "deterministic",
        "cudnn_benchmark",
    ]
    row = {
        "timestamp": int(time.time()),
        "hostname": socket.gethostname(),
        "arch": args.arch,
        "backend": device.type,
        "device_name": device_name(device),
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "measured_iters": args.iters,
        "precision": "amp" if amp_enabled and scaler is not None else "fp32",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "elapsed_sec": elapsed,
        "images_per_sec": throughput,
        "images_processed": images_processed,
        "data_root": str(args.data.resolve()),
        "label": args.label,
        "torch_version": torch.__version__,
        "channels_last": bool(args.channels_last and device.type == "cuda"),
        "grad_accum_steps": grad_accum_steps,
        "deterministic": bool(args.deterministic),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
    }
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
