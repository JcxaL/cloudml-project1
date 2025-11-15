# Project 1 — DNN Performance Modeling & Roofline Analysis

This repo captures everything needed for my Project 1 submission: a fixed-iteration
training driver, environment/run scripts, profiler recipes, log aggregation, and a
report skeleton. The goal is to compare short runs of ResNet‑50, VGG‑16, and
MobileNetV2 across three environments (Mac CPU/MPS, local RTX 4090, and a GCP
GPU), quantify the limits via roofline modeling, and write up the findings.

---

## 1. Project Requirements & Deliverables

- **Experiment design (10%)** — Describe objectives, hypotheses, and the run matrix.
- **Complexity estimation (20%)** — Capture parameter/activation sizes and FLOP counts.
- **Measurement (15%)** — Collect throughput, timing, and system counters.
- **Roofline modeling (20%)** — Place every point on a roofline, using Nsight metrics on CUDA.
- **Analysis (35%)** — Explain how environments and model choices affect performance.

The written report (`report.md`) embeds throughput tables, roofline plots, and
environment notes. Logs and figures produced by the scripts are referenced
directly from the report.

---

## 2. Repository Layout

```
p1-roofline/
├─ analysis/               # Aggregation scripts, plots, peak specs
│  ├─ roofline.py          # Build roofline_points.csv from logs + Nsight
│  ├─ plot_roofline.py     # Render roofline_<env>.png
│  ├─ peaks.json           # Peak GFLOP/s + GB/s per environment
│  └─ summary/             # Throughput + complexity tables (CSV/MD)
├─ code/
│  ├─ run_train.py         # Training driver with timing/logging hooks
│  ├─ metric_names_ncu.txt # Nsight metrics requested by ncu
│  └─ examples/…           # torchvision ImageNet example (submodule/clone)
├─ data/imagenet-mini/     # Small ImageFolder subset for quick runs
├─ docs/                   # Run matrix + checklist
├─ env/                    # Per-environment virtualenvs
├─ figures/                # Throughput + roofline plots consumed by the report
├─ logs/
│  ├─ metrics.csv          # Appended by run_train.py for every run
│  ├─ ncu/                 # Nsight CSV exports
│  ├─ time/                # `/usr/bin/time` outputs (per run)
│  └─ env_info.md          # Manual environment annotations
├─ scripts/                # Helpers for repeating runs, profiling, env setup
│  ├─ run_repeat.sh        # RUNS× loops with `/usr/bin/time`
│  ├─ run_mac_matrix.sh    # CPU/MPS batch runner
│  ├─ run_cuda_matrix.sh   # CUDA batch runner (fp32 + amp)
│  ├─ run_cuda_nsight.sh   # Nsight Compute helper
│  ├─ model_complexity.py  # torchinfo-based complexity dump
│  └─ setup_env.sh         # venv + requirements install
├─ report.md               # Project write-up template (filled as data arrives)
├─ requirements.txt        # Torch + tooling versions used everywhere
└─ tmp/                    # Scratch space (ignored)
```

Everything is organized so the same scripts can run on macOS, Linux, WSL, or GCP.

---

## 3. Dataset & Environment Setup

1. Create a small, class-balanced ImageNet subset under
   `data/imagenet-mini/{train,val}/class/*.jpg`. Use `--synthetic` for dry runs.
2. Create a virtual environment per machine (`env/venv<machine>`) and install the
   pinned requirements:

   ```bash
   ./scripts/setup_env.sh env/venv_mps   # or env/venv_gcp, env/venv_4090, …
   source env/venv_mps/bin/activate
   ```

3. For CUDA machines, verify `nvidia-smi`, Nsight Compute (`ncu`), and
   `/usr/bin/time -v` availability. For macOS, verify `/usr/bin/time -l`.

Environment-specific details, including OS versions, CUDA/cuDNN builds, and zone
information, belong in `logs/env_info.md` for reproducibility.

---

## 4. Training & Timing Workflow

`code/run_train.py` wraps torchvision classification models with the knobs needed
for the project:

- Device selection: `--backend {auto,cuda,mps,cpu}`.
- Precision: `--precision {fp32,amp}` with safe fallbacks when AMP is requested
  on non-CUDA devices.
- Data loader controls: `--workers`, `--pin-memory`, `--prefetch-factor`,
  `--synthetic`, `--no-augment`.
- Performance toggles: `--channels-last`, `--grad-accum-steps`, `--deterministic`,
  `--cudnn-benchmark`, optional NVTX ranges.

Each run prints a summary (`MEASURED_SEC=… IMGS_PER_SEC=…`) and appends a row to
`logs/metrics.csv`. Use `scripts/run_repeat.sh` to repeat a configuration with a
consistent label and `/usr/bin/time` capture.

Example (Mac MPS):

```bash
RUNS=3 LABEL=mac_mps_resnet50_fp32_bs32 \
  bash scripts/run_repeat.sh \
  --data data/imagenet-mini --arch resnet50 --batch-size 32 \
  --warmup-iters 10 --iters 100 --workers 0 \
  --backend mps --precision fp32 --no-augment --deterministic
```

### Batch runners

- `scripts/run_mac_matrix.sh cpu|mps` sweeps the CPU or MPS matrix (models × batch sizes).
- `scripts/run_cuda_matrix.sh` sweeps the CUDA matrix (includes AMP for ResNet-50).

### Nsight Compute

`scripts/run_cuda_nsight.sh` launches shorter runs (default 5 warmup / 30 measured
iters) under Nsight Compute and writes CSV logs to `logs/ncu/`. Update
`code/metric_names_ncu.txt` if additional counters are needed—any entries in that
file are parsed automatically by `analysis/roofline.py`.

---

## 5. Complexity Estimation

Run `scripts/model_complexity.py` to capture parameter counts, activation sizes,
and per-batch FLOPs using `torchinfo`. The command writes per-model text summaries
plus a JSON aggregate consumed by the roofline script:

```bash
python scripts/model_complexity.py \
  --models resnet50 vgg16 mobilenet_v2 \
  --batch-sizes 16 32 64 128 \
  --output-dir logs/model_summaries
```

The resulting CSV/JSON under `logs/model_summaries/` also drives the tables in
`analysis/summary/` and feeds non-CUDA roofline points (where FLOP counts come
from theoretical estimates instead of profiler counters).

---

## 6. Analysis Pipeline

Once timing + profiler logs exist:

1. **Aggregate metrics** (mean/std per configuration) with the helper notebooks or
   Python scripts in `analysis/summary/`.
2. **Roofline points** — combine metrics, Nsight CSVs, peak specs, and complexity
   JSON into a single CSV:

   ```bash
   python analysis/roofline.py \
     --metrics logs/metrics.csv \
     --ncu-dir logs/ncu \
     --peaks analysis/peaks.json \
     --complexity-json logs/model_summaries/summary.json \
     --output analysis/roofline_points.csv
   ```

3. **Plot rooflines** for each environment:

   ```bash
   python analysis/plot_roofline.py --points analysis/roofline_points.csv --outdir figures
   ```

4. **Throughput plots** — notebooks/scripts under `analysis/summary/` turn
   `logs/metrics.csv` into `figures/throughput_<env>.png` and Markdown tables.

All generated figures live in `figures/` so the report can reference them
directly (e.g., `![Roofline — Mac MPS](figures/roofline_mps.png)`).

---

## 7. Report & Documentation

- `report.md` contains the full write-up structure (abstract, experiment design,
  complexity, measurement, roofline modeling, analysis, limitations, appendix).
- `docs/run_matrix.md` and `docs/run_checklist.md` capture execution plans and to-dos.

When filling the report, cite the generated tables/figures by file name and link
to commands/labels in `logs/metrics.csv` so another reviewer can replay any run.

---

## 8. Repro Checklist

1. Clone repo + run `scripts/setup_env.sh` on the target machine.
2. Populate `data/imagenet-mini` or use `--synthetic` for sanity checks.
3. Run the appropriate matrix script (Mac CPU/MPS, local CUDA, or GCP CUDA).
4. Capture Nsight CSVs for each CUDA label using `scripts/run_cuda_nsight.sh`.
5. Document environment specifics in `logs/env_info.md`.
6. Generate complexity summaries, throughput tables, and roofline plots.
7. Update `report.md` with the new figures/tables and key observations.

Following this workflow keeps the project reproducible and aligned with the
grading rubric: every number in the report traces back to version-controlled
scripts, logs, and figures inside this repository.

