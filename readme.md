⸻

Project 1 — DNN Performance Modeling with Roofline (3-Env Comparison)

Goal. Measure and model DNN training performance across three environments, using short, fixed-iteration runs. Produce a roofline model and a written analysis that explains what limits performance and why.  ￼

⸻

1) What you must deliver (explicit requirements)

From Lecture 7 “Project 1 – Performance modeling and analysis”:
	•	Environments: Choose 2–3 distinct setups (e.g., GCP GPU, local RTX 4090, local CPU or Apple M-series/MPS).
	•	Models: Use 2–3 NN models (e.g., ResNet-50, VGG-16, MobileNetV2).
	•	Runs: Perform short, representative runs (no need to fully train; accuracy isn’t the focus).
	•	Report sections & grading rubric (percent of Project 1):
	•	Experiment Design (10%) — objective & hypothesis.
	•	Complexity Estimation (20%) — params/activations, memory & compute footprints.
	•	Measurement (15%) — throughput/time/counters.
	•	Roofline Modeling (20%) — build rooflines from your measurements (optionally compare with Nsight’s built-in roofline).
	•	Analysis (35%) — interpret results across environments & models.
	•	Due: Nov 7, 11:59 pm.  ￼

Dataset & starter code pointers (Lecture 7):
	•	Use the PyTorch ImageNet example: https://github.com/pytorch/examples/tree/master/imagenet.
	•	Do not download full ImageNet-1k; a small extracted subset is fine (ImageFolder layout).
	•	You may use Nsight Roofline to assist.  ￼

Course logistics (Lecture 1): Project 1 is individual, graded as part of the course projects; reports must follow good writing etiquette (citations, clarity).  ￼

⸻

2) Implicit expectations (what the lectures assume you’ll do)
	•	Time correctly, and name your time: Record real/wall, user, sys, and report how you timed the measured section. Use time.monotonic() for code timing and /usr/bin/time -v for process stats.  ￼
	•	Measure bytes and FLOPs for NVIDIA GPUs: Collect dram__bytes_* and pcie__* for data movement and flop_count_* for math from Nsight Compute (ncu). These drive Arithmetic Intensity (AI) and roofline placement.  ￼
	•	Compute the roofline properly:
\text{Attainable FLOP/s}=\min\big(\text{AI}\times \text{peak GB/s},\ \text{peak GFLOP/s}\big),
\quad \text{Transition at}\ \text{AI}=\frac{\text{peak GFLOP/s}}{\text{peak GB/s}}
Use log-log axes and discuss whether points are bandwidth- or compute-bound.  ￼
	•	Estimate complexity & memory: Summarize parameter counts and activation sizes; note extra memory for gradients/momentum. Use torchsummary (or similar) to justify batch-size choices and memory pressure.  ￼
	•	Reproducibility matters: Containers are encouraged where practical (especially on GCP). Document software versions and commands.  ￼
	•	Cloud reality: Region/zone capacity and virtualization layers affect performance and even availability; record the zone/GPU you actually used (the “GPU chase” point).  ￼
	•	Optional precision study: Exploring AMP/mixed precision on NVIDIA is fair game for performance factors (tensor core speed, memory traffic). Note tradeoffs.

⸻

3) Repository layout

p1-roofline/
├─ code/
│  ├─ run_train.py                # fixed-iteration driver (+ --synthetic FakeData mode)
│  ├─ metric_names_ncu.txt        # Nsight Compute metrics (create from snippet below)
│  └─ Dockerfile.gpu              # optional for GCP/local NVIDIA
├─ data/
│  └─ imagenet-mini/{train,val}/  # small ImageFolder subset
├─ env/                           # per-env virtualenvs or conda
├─ logs/
│  ├─ metrics.csv                 # auto-appended by run_train.py
│  └─ env_info.md                 # you fill this out (system details)
└─ README.md

Create code/metric_names_ncu.txt:

flop_count_sp,flop_count_hp,dram__bytes_read.sum,dram__bytes_write.sum,pcie__read_bytes.sum,pcie__write_bytes.sum

(These match Lecture 7’s NVIDIA metrics mapping.)  ￼

⸻

4) Quick start (clone & smoke test)

# workspace
mkdir -p ~/p1-roofline/{code,env,logs,data,tmp}
cd ~/p1-roofline

# clone PyTorch examples; link ImageNet example (Lecture 7)
git clone https://github.com/pytorch/examples.git code/examples
ln -s code/examples/imagenet code/train_imagenet_src

# python deps (macOS, Linux, WSL, cloud)
python3 -m venv env/venv
source env/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # torch+torchvision/torchaudio pinned for portability

# or run ./scripts/setup_env.sh (creates env/venv, installs requirements)
./scripts/setup_env.sh

# bring in your driver (already present in this repo):
# code/run_train.py

# dataset: place a small, class-balanced subset here:
# data/imagenet-mini/train/<class>/*.jpg
# data/imagenet-mini/val/<class>/*.jpg

# no dataset handy yet? add --synthetic to use FakeData tensors

# smoke test (CPU/MPS ok):
python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 8 --warmup-iters 1 --iters 2 --workers 0 \
  --precision fp32 --synthetic

The ImageNet example and partial-dataset guidance come straight from the Lecture 7 slide.  ￼

⸻

5) Exact steps — Local NVIDIA RTX 4090

Env setup

python3 -m venv env/venv4090
source env/venv4090/bin/activate
pip install --upgrade pip torch torchvision torchaudio
nvidia-smi

Run + time

/usr/bin/time -v python code/run_train.py \
  --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 10 --iters 100 \
  --workers 8 --precision fp32 --backend cuda \
  2>&1 | tee logs/4090_resnet50_fp32_bs128.txt

(Record real/user/sys and RSS from /usr/bin/time as Lecture 7 suggests.)  ￼

Profile with Nsight Compute (CSV export)

```bash
mkdir -p logs/ncu
LABEL=rtx4090_resnet50_fp32_prof
ncu --profile-from-start off --target-processes all \
  --metrics $(cat code/metric_names_ncu.txt) \
  --csv --page raw --log-file logs/ncu/${LABEL}.csv \
  python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 5 --iters 30 --workers 8 --precision fp32 \
  --backend cuda --label ${LABEL}
```

(These metrics map to dram/pcie bytes and flop counts used in roofline construction.)  ￼

Optional AMP run

/usr/bin/time -v python code/run_train.py \
  --data data/imagenet-mini --arch resnet50 \
  --batch-size 256 --warmup-iters 10 --iters 100 \
  --workers 8 --precision amp --backend cuda \
  2>&1 | tee logs/4090_resnet50_amp_bs256.txt

(Mention mixed-precision benefits/risks in the report.)

### Windows & WSL2 reminders (same RTX box)

- **WSL2 (Ubuntu)** — gives `/usr/bin/time` + Nsight CLI parity:

```bash
python3 -m venv env/venwsl
source env/venwsl/bin/activate
pip install --upgrade pip -r requirements.txt
RUNS=3 LABEL=rtx4090_resnet50_fp32 bash scripts/run_repeat.sh \
  --data data/imagenet-mini --arch resnet50 --batch-size 128 \
  --warmup-iters 10 --iters 100 --workers 8 --backend cuda --precision fp32

mkdir -p logs/ncu
ncu --csv --page raw --profile-from-start off --target-processes all \
  --metrics $(cat code/metric_names_ncu.txt) \
  --log-file logs/ncu/rtx4090_resnet50_fp32_prof.csv \
  python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 5 --iters 30 --workers 8 \
  --backend cuda --precision fp32 --label rtx4090_resnet50_fp32_prof
```

- **Native Windows (PowerShell)** — use `bash` (Git Bash/WSL) for the repeat helper and `Measure-Command` if you prefer native timing:

```powershell
python -m venv env\win4090
env\win4090\Scripts\Activate.ps1
pip install --upgrade pip -r requirements.txt
$env:RUNS = 3
$env:LABEL = "win4090_resnet50_fp32"
bash scripts/run_repeat.sh --data data/imagenet-mini --arch resnet50 `
  --batch-size 128 --warmup-iters 10 --iters 100 --workers 8 `
  --backend cuda --precision fp32

New-Item -ItemType Directory -Path logs\ncu -Force | Out-Null
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3\ncu.exe" `
  --profile-from-start off --target-processes all `
  --metrics flop_count_sp,flop_count_hp,dram__bytes_read.sum,dram__bytes_write.sum,pcie__read_bytes.sum,pcie__write_bytes.sum `
  --csv --page raw --log-file logs\ncu\${env:LABEL}_prof.csv `
  python code\run_train.py --data data\imagenet-mini --arch resnet50 `
    --batch-size 128 --warmup-iters 5 --iters 30 --workers 8 `
    --backend cuda --precision fp32 --label ${env:LABEL}_prof `
  | Out-File -FilePath logs\${env:LABEL}_prof.txt -Encoding ascii
```

Log OS build, driver/CUDA/cuDNN, and whether you used WSL2 vs native Windows in `logs/env_info.md`.

⸻

6) Exact steps — Local CPU or Apple Silicon (MPS)

A) CPU-only (Linux/x86)

python3 -m venv env/venvcpu
source env/venvcpu/bin/activate
pip install --upgrade pip torch torchvision torchaudio

/usr/bin/time -v python code/run_train.py \
  --data data/imagenet-mini --arch resnet50 \
  --batch-size 32 --warmup-iters 5 --iters 50 --workers 8 --precision fp32 \
  2>&1 | tee logs/cpu_resnet50_fp32_bs32.txt

# IPC/CPI & cache behavior
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 32 --warmup-iters 5 --iters 50 --workers 8 --precision fp32 \
  2>&1 | tee logs/cpu_resnet50_perf.txt

(Real/user/sys timing and derived metrics are part of the “metrics of performance” toolbox.)  ￼

B) Apple Silicon (macOS, M-series)

python3 -m venv env/venvmps
source env/venvmps/bin/activate
pip install --upgrade pip torch torchvision torchaudio

/usr/bin/time -l python code/run_train.py \
  --data data/imagenet-mini --arch resnet50 \
  --batch-size 64 --warmup-iters 5 --iters 50 --workers 8 \
  --precision fp32 --backend mps \
  2>&1 | tee logs/mps_resnet50_bs64.txt

(Use Instruments → Metal System Trace if you collect GPU counters; your wall-time still anchors the roofline placement for this env.)  ￼

⸻

7) Exact steps — GCP GPU (cloud)

Why cloud notes matter: Availability varies by zone/GPU, and virtualization can affect perf; document zone, GPU type, image, driver/CUDA, and storage. (The course even set a “GPU Chase” exercise to drive this home.)

### Project / quota prep (run once)
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com
# GPU quota requests happen in the console (IAM & Admin → Quotas); target NVIDIA_L4_GPUS/NVIDIA_T4_GPUS/etc.
```
Pick a region/zone with real capacity (`gcloud compute accelerator-types list`) and log project/region/zone/machine/GPU/disk/network notes in `logs/env_info.md`.

  ￼

VM & env

# pick zone and GPU that actually has capacity
export ZONE=us-central1-a
export INSTANCE=p1-gpu
gcloud compute ssh $INSTANCE --zone $ZONE

# on the VM:
python3 -m venv ~/p1-roofline/env/vmgpu
source ~/p1-roofline/env/vmgpu/bin/activate
pip install --upgrade pip torch torchvision torchaudio
sudo apt -y install git
mkdir -p ~/p1-roofline && cd ~/p1-roofline
git clone https://github.com/pytorch/examples.git code/examples
ln -s code/examples/imagenet code/train_imagenet_src
mkdir -p data/imagenet-mini

Run + profile (same pattern as 4090)

nvidia-smi
which ncu

/usr/bin/time -v python code/run_train.py \
  --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 10 --iters 100 \
  --workers 8 --precision fp32 \
  2>&1 | tee logs/gcp_resnet50_fp32_bs128.txt

mkdir -p logs/ncu
ncu --profile-from-start off --target-processes all \
  --metrics $(cat code/metric_names_ncu.txt) \
  --csv --page raw --log-file logs/ncu/gcp_resnet50_fp32_prof.csv \
  python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 5 --iters 30 --workers 8 --precision fp32 \
  --backend cuda --label gcp_resnet50_fp32_prof

# cost hygiene
gcloud compute instances stop $INSTANCE --zone $ZONE
# delete when finished
# gcloud compute instances delete $INSTANCE --zone $ZONE

Optional containerization (recommended for reproducibility)

# code/Dockerfile.gpu
FROM pytorch/pytorch:<CUDA_TAG>
WORKDIR /workspace
RUN pip install gpustat torchsummary
COPY run_train.py /workspace/run_train.py

cd code
docker build -t p1-gpu -f Dockerfile.gpu .
docker run --rm --gpus all \
  -v ~/p1-roofline/data:/data -v ~/p1-roofline/logs:/logs \
  p1-gpu /usr/bin/time -v python /workspace/run_train.py \
    --data /data/imagenet-mini --arch resnet50 \
    --batch-size 128 --warmup-iters 10 --iters 100 --workers 8 --precision fp32

(Containers + version pinning align with the course’s Docker content.)  ￼

⸻

8) Driver usage (code/run_train.py)

Key flags:
	•	--data PATH (ImageFolder with train/), --arch resnet50, --batch-size, --workers
	•	--warmup-iters, --iters (measured loop)
	•	--precision {fp32,amp}, --backend {auto,cuda,cpu,mps}
	•	--pin-memory (CUDA loaders), --synthetic (FakeData sanity runs before dataset sync)
	•	--no-augment / --prefetch-factor / --channels-last / --grad-accum-steps / --nvtx / --deterministic / --cudnn-benchmark for finer control over pipelines.
	•	--label STRING, --log-dir logs (appends to logs/metrics.csv)

It prints a one-line summary and appends a CSV row with host, device, timing, and throughput. This CSV is your primary timing ledger.

⸻

9) What to capture per run (for the roofline & analysis)
	•	Timing: elapsed (monotonic) + /usr/bin/time real/user/sys, RSS.  ￼
	•	Throughput: images/sec (already logged).
	•	NVIDIA metrics: dram__bytes_{read,write}.sum, pcie__{read,write}_bytes.sum, flop_count_{sp,hp} via ncu.  ￼
	•	CPU stats (if applicable): perf stat IPC/CPI/cache-miss.  ￼
	•	Memory footprint: nvidia-smi/gpustat, and model summary for params/activations.  ￼

Compute:
	•	AI = FLOPs / bytes moved (use DRAM bytes; mention PCIe if host-device copies dominate).
	•	Attained GFLOP/s = FLOPs / elapsed.
	•	Place each point on the roofline using your environment’s peak GB/s and peak GFLOP/s; explain whether each point is bandwidth- or compute-bound and why.  ￼

### Tooling to keep this reproducible
	1.	Run `scripts/model_complexity.py --batch-sizes 16 32 64 128 256` to dump `logs/model_summaries/*.txt` plus `summary.json` (FLOP/param references for the report + CPU/MPS roofline points). `analysis/roofline.py` assumes the summary encodes per-sample forward MACs and multiplies by 2× for training (adjust TRAIN_FACTOR if you justify a different value).
	2.	Capture Nsight Compute metrics as CSV per CUDA run: `ncu --csv --page raw --metrics $(cat code/metric_names_ncu.txt) --log-file logs/ncu/$LABEL.csv ...` (always pass the same --label used for run_train.py).
	3.	Fill real peak specs (GFLOP/s & GB/s) for each environment inside `analysis/peaks.json` (script refuses to run with `null`).
	4.	After measurements, aggregate everything into roofline-ready rows:
```
python analysis/roofline.py --metrics logs/metrics.csv --ncu-dir logs/ncu \
  --complexity-json logs/model_summaries/summary.json \
  --peaks analysis/peaks.json --output analysis/roofline_points.csv
```
	5.	Use `analysis/roofline_points.csv` in a notebook (or run `python analysis/plot_roofline.py --points analysis/roofline_points.csv`) to plot rooflines + measured points.

⸻

10) Run matrix (keep it short but comparable)
	•	Models: resnet50, vgg16, mobilenet_v2 (see `docs/run_matrix.md` for the table)
	•	Batch sizes (indicative):
	•	4090: 64/128/256
	•	GCP GPU: 32/64/128 (depends on GPU)
	•	CPU/MPS: 16/32/64
	•	Precision: FP32 everywhere; add AMP on NVIDIA.
	•	Iters: --warmup-iters 10, --iters 100 (use 30 iters for profiled runs to bound overhead).
	•	Repeats: 3 per configuration with labels `{env}_{model}_{precision}_bs{batch}_run{N}`; keep `/usr/bin/time`, `perf`, and `ncu --csv` logs under `logs/{time,perf,ncu}/`.
	Repeat 3× per point to gauge variability (especially on cloud).

⸻

11) Report structure (map to rubric)
	1.	Title & Abstract (context + key findings).
	2.	Experiment Design (10%) — Objective, hypothesis, environments, models, run matrix.  ￼
	3.	Complexity Estimation (20%) — Params/activations per model; expected memory traffic vs compute; batch-size rationale.  ￼
	4.	Measurement (15%) — How timed and profiled; tables for throughput, time breakdown, FLOPs/bytes.  ￼
	5.	Roofline Modeling (20%) — Method, peak specs, plots, interpretation of bandwidth- vs compute-bound regions.  ￼
	6.	Analysis (35%) — Cross-env comparisons (cloud vs bare-metal, virtualization), model sensitivity, precision effects (AMP), limitations.
	7.	Reproducibility — Software/driver versions, zones, instance types, exact commands, container digest.  ￼
	8.	References & appendix — Cite slides/tools/data.

⸻

12) Bookkeeping & checklists
	•	logs/metrics.csv appended by every run.
	•	logs/env_info.md — Fill in: OS/Kernel; Python; CUDA/driver/cuDNN; PyTorch; CPU/GPU model & memory; region/zone; storage & network; container image tag.  ￼
	•	Keep code/metric_names_ncu.txt under version control.
	•	Save ncu outputs and /usr/bin/time outputs alongside CSV rows.

⸻

13) Notes on AI Workflow & containers (course context)

The course frames ML work as a pipeline (data → training → adaptation → inference) and uses containers to make environments consistent and fast to spin up; your Project 1 asks you to apply those principles to measurement & performance modeling.  ￼

⸻

Appendix: Roofline recap (Lecture 7)
	•	The transition (machine balance) sits at AI = peakGFLOP/s ÷ peakGB/s.
	•	Your point’s attainable GFLOP/s is limited by min(memory roof, compute roof).
Explain each point’s placement and what would move it up/right (e.g., data reuse, kernel fusion, mixed precision).  ￼

⸻

Cited course materials: Project brief & rubric, ImageNet pointer, Nsight metrics and roofline math (Lecture 7); mixed precision benefits (Lecture 8); containerization & cloud realities incl. “GPU Chase” (Lecture 4); course grading/logistics (Lecture 1).

⸻

One-line command index (handy)

# CPU/MPS timing
/usr/bin/time -v python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 32 --warmup-iters 5 --iters 50 --workers 8 --precision fp32

# NVIDIA FP32
/usr/bin/time -v python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 10 --iters 100 --workers 8 --precision fp32 --backend cuda

# NVIDIA AMP
/usr/bin/time -v python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 256 --warmup-iters 10 --iters 100 --workers 8 --precision amp --backend cuda

# Nsight Compute metrics
LABEL=rtx4090_resnet50_fp32_prof
ncu --profile-from-start off --target-processes all \
  --metrics $(cat code/metric_names_ncu.txt) \
  --csv --page raw --log-file logs/ncu/${LABEL}.csv \
  python code/run_train.py --data data/imagenet-mini --arch resnet50 \
  --batch-size 128 --warmup-iters 5 --iters 30 --workers 8 --precision fp32 \
  --backend cuda --label ${LABEL}

⸻

## Supporting files in this repo

- `docs/run_matrix.md` — explicit run matrix (models × batch sizes × envs × repeats) for experiment design.
- `scripts/setup_env.sh` — reproducible venv bootstrap for macOS/Linux/WSL/cloud.
- `scripts/model_complexity.py` — dumps per-model parameter/activation summaries and JSON used for CPU/MPS roofline points.
- `scripts/run_repeat.sh` — wraps `/usr/bin/time` runs with consistent labels (set RUNS/LABEL env vars).
- `analysis/peaks.json` — fill with per-precision peak GFLOP/s + GB/s per environment before running `analysis/roofline.py`.
- `analysis/roofline.py` — combines `logs/metrics.csv`, Nsight CSV logs, and complexity data into `analysis/roofline_points.csv`.
- `analysis/plot_roofline.py` — renders roofline_{env}.png from `analysis/roofline_points.csv`.
