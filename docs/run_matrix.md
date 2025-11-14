# Project 1 Run Matrix

The same matrix applies to every environment; adjust batch sizes to fit memory. Each configuration runs **3 repeats** (report mean ± std).

| Environment | Backend | Models                   | Batch sizes | Precisions | Instrumentation |
|-------------|---------|--------------------------|-------------|------------|-----------------|
| Mac CPU     | cpu     | resnet50, vgg16, mobilenet_v2 | 16, 32, 64  | fp32       | `/usr/bin/time -v`, `perf stat` (cycles, instructions, cache-misses, branches, branch-misses) |
| Mac MPS     | mps     | resnet50, vgg16, mobilenet_v2 | 16, 32, 64  | fp32       | `/usr/bin/time -l`, Instruments (Metal System Trace as needed) |
| RTX 4090    | cuda    | resnet50, vgg16, mobilenet_v2 | 64, 128, 256| fp32 & amp | `/usr/bin/time -v`, `ncu --csv --metrics $(cat code/metric_names_ncu.txt)` |
| GCP GPU     | cuda    | resnet50, vgg16, mobilenet_v2 | 32, 64, 128 | fp32 & amp | `/usr/bin/time -v`, `ncu --csv --metrics $(cat code/metric_names_ncu.txt)` |

## Command Naming Convention

Use the `--label` flag to tag every run as `{env}_{model}_{precision}_bs{batch}_run{N}`. Save matching profiler logs under:

```
logs/time/{label}.txt   # stdout from /usr/bin/time
logs/ncu/{label}.csv    # Nsight Compute metrics (csv mode)
logs/perf/{label}.txt   # perf stat output (CPU-only)
```

This naming lets the roofline prep script correlate time + profiler metrics automatically.
