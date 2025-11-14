# Project 1 — Performance Modeling & Roofline Analysis across Mac (CPU/MPS), RTX 4090, and GCP GPU

Author: _Your Name_
Course / Semester: _e.g., NYU Cloud & ML, Fall 2025_
Repository: _link or commit hash_
Data subset: _Imagenette / ImageNet mini (brief description)_

---

## Abstract (≤150 words)

_Summarize goal, method, main results (throughput ranges, bandwidth vs compute limits), and one actionable insight (e.g., AMP shifts ResNet-50 into compute-bound regime on 4090 at large batches)._  
**Keywords:** roofline, throughput, arithmetic intensity, GPU profiling, mixed precision

---

## 1. Introduction & Research Questions

**Objective.** Compare short, fixed-iteration training runs of ResNet-50, VGG-16, and MobileNetV2 across Mac CPU/MPS, local RTX 4090, and a GCP GPU, then explain the limits via roofline modeling.

**Questions.**
1. Where is each model compute-bound vs. bandwidth-bound across environments?
2. How do batch size and precision (FP32 vs. AMP) shift arithmetic intensity and throughput?
3. What is the effect of cloud virtualization/placement (zone, GPU type) vs. bare-metal?

---

## 2. Experiment Design (Rubric: Experiment Design, 10%)

**Models.** ResNet-50, VGG-16, MobileNetV2 (different reuse/memory patterns).  
**Dataset.** ImageNet-style subset (describe classes/count).  
**Run matrix.**

| Env | Backend | Precision | Batch sizes | Iters (warmup/measured) | Repeats |
| --- | --- | --- | --- | --- | --- |
| Mac CPU | cpu | fp32 | 16, 32, 64 | 10 / 100 | 3 |
| Mac MPS | mps | fp32 | 32, 64 | 10 / 100 | 3 |
| RTX 4090 | cuda | fp32, amp | 64, 128, 256 | 10 / 100 | 3 |
| GCP GPU (specify) | cuda | fp32, amp | 32, 64, 128 | 10 / 100 | 3 |

**Timing & profiling.**
- Python: `time.monotonic()` over measured loop (with CUDA/MPS sync).
- System: `/usr/bin/time -v` (Linux) / `-l` (macOS) for real/user/sys, RSS.
- Profiling: Nsight Compute CLI `--csv --page raw --metrics $(cat code/metric_names_ncu.txt)`.

**Reproducibility.** Full batches (`drop_last=True`), `--deterministic`, `--no-augment` to isolate compute, consistent gradient accumulation when needed.

---

## 3. Model Complexity (Rubric: Complexity Estimation, 20%)

Command:
```bash
python scripts/model_complexity.py \
  --models resnet50 vgg16 mobilenet_v2 \
  --batch-sizes 16 32 64 128 \
  --output-dir logs/model_summaries
```
Generates `logs/model_summaries/summary.json` with `mult_adds_per_sample_forward`, `flops_per_batch_forward`, `params`.

**Table 1 — Parameter & Activation Summary**

| Model | Params (M) | Forward MACs / sample (G) | Forward FLOPs / batch (G) | Notes |
| --- | --- | --- | --- | --- |
| ResNet-50 | ... | ... | ... | activation discussion |
| VGG-16 | ... | ... | ... | |
| MobileNetV2 | ... | ... | ... | |

_Write 4–6 sentences interpreting which models stress memory most and how that shapes AI/bandwidth sensitivity._

---

## 4. Measurement Results (Rubric: Measurement, 15%)

**Data sources.** `logs/metrics.csv`, `/usr/bin/time` outputs under `logs/*`, Nsight CSVs under `logs/ncu/`.

### 4.1 Throughput vs Batch Size
- Plot per environment (A1–A3): X=batch size, Y=images/sec, error bars = ±1 SD.
- Use figures `figures/throughput_env-<ENV>.png`.
- Describe trends & AMP uplift.

### 4.2 Variability (Cloud vs Bare Metal)
- Plot B: distribution for fixed config (e.g., ResNet-50 bs128 FP32) across environments.
- Figure `figures/variance_fixed_config.png`.
- Comment on virtualization noise.

---

## 5. Roofline Modeling (Rubric: Roofline Modeling, 20%)

**Peaks.** Fill `analysis/peaks.json` with GFLOP/s & GB/s (per precision). Cite vendor docs.

**Aggregation.**
```bash
python analysis/roofline.py \
  --metrics logs/metrics.csv \
  --ncu-dir logs/ncu \
  --peaks analysis/peaks.json \
  --complexity-json logs/model_summaries/summary.json \
  --output analysis/roofline_points.csv
```
- CUDA points: Nsight FLOPs/bytes → AI + GFLOP/s.
- CPU/MPS points: complexity JSON + TRAIN_FACTOR.

**Plots.**
- Plot C (per env): `analysis/plot_roofline.py --points analysis/roofline_points.csv --outdir figures` → `figures/roofline_<ENV>.png`.
- Optional Plot D: AI vs batch for CUDA models (`figures/ai_vs_batch_<model>.png`).

_Write 2–3 paragraphs tying points to roofs, balance point AI, AMP shifts, and batch effects._

---

## 6. Analysis & Discussion (Rubric: Analysis, 35%)

Discuss factors: algorithm/model, framework/runtime knobs (channels-last, AMP, dataloader settings), virtualization (GCP zone/GPU type), hardware ceilings, optional cost/$ per image.

**Table 2 — Summary (mean ± SD)**

| Env | Model | Precision | Batch | Images/s | Attained GFLOP/s | AI | Bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | ... | ... | mem/compute |

_Write main takeaways (2–3)._

---

## 7. Reproducibility & Environment (Rubric: Writing & Reproducibility, 15%)

Re-run commands documented in repo / `logs/env_info.md`.

```bash
python analysis/roofline.py ...
python analysis/plot_roofline.py ...
```

**Environment table.**

| Env key | Host | OS/Kernel | Device | Driver/CUDA/cuDNN | Python/Torch | Storage | Region/Zone |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rtx4090 | ... | ... | RTX 4090 | ... | ... | ... | n/a |
| gcp_<gpu> | ... | ... | ... | ... | ... | ... | ... |
| mac_mps | ... | ... | Apple M-series | n/a | ... | ... | n/a |

Include peaks.json content + references, dataset location, `logs/requirements.txt` snapshot.

---

## 8. Limitations & Future Work
- Small subset vs full ImageNet, TRAIN_FACTOR approximation, missing DRAM counters on CPU/MPS, future instrumentation ideas.

---

## 9. Conclusion
- 3–5 sentences tying back to questions and main findings.

---

## References
- List vendor docs, Nsight metrics, other citations.

---

## Figures Checklist
- A1–A3: Throughput vs batch per env (`figures/throughput_env-<ENV>.png`).
- B: Variability (`figures/variance_fixed_config.png`).
- C1–C3: Roofline (`figures/roofline_<ENV>.png`).
- D (optional): AI vs batch.

Embed example:
```markdown
![Roofline — RTX 4090](figures/roofline_rtx4090.png){ width=70% }
```

---

## Appendix
- Commands per run.
- Nsight metric list (contents of `code/metric_names_ncu.txt`).
- Additional tables/plots.

---

_Replace prompts with real content before submission._
