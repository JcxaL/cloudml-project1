# Run Checklist & TODOs

- [ ] **Mac CPU:** run `scripts/run_mac_matrix.sh cpu` (uses RUNS env, defaults to 3). Verify `/usr/bin/time -l` logs under `logs/time/` and `logs/metrics.csv` rows exist for every model/batch.
- [ ] **Mac MPS:** run `scripts/run_mac_matrix.sh mps` (same conventions) once CPU runs finish.
- [ ] **RTX 4090 (WSL2 or native):** for each model × batch × precision, run `RUNS=3 LABEL=rtx4090_<model>_<precision>_bs<B>` via `scripts/run_repeat.sh` with `--backend cuda --channels-last --no-augment --deterministic`. Capture Nsight CSV for each profiled label using instructions in README §5.
- [ ] **GCP GPU:** after provisioning, repeat the CUDA runs/labels using the `gcp_` prefix, plus Nsight CSV per model.
- [ ] **Aggregation:** once all logs exist, run `analysis/roofline.py` then `analysis/plot_roofline.py`; drop figures into the report.
- [ ] **Report filling:** use `report.md` template; insert throughput tables/figures, variance charts, and roofline analysis.

Use these boxes to track progress as you execute the matrix.
