#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/imagenet-mini}
RUNS=${RUNS:-3}
WORKERS=${WORKERS:-4}
BACKEND=cuda
COMMON=(--data "$DATA" --warmup-iters 10 --iters 100 --workers "$WORKERS" \
        --backend "$BACKEND" --channels-last --no-augment)

ENTRIES=(
  "resnet50:fp32:128"
  "resnet50:amp:256"
  "vgg16:fp32:64"
  "mobilenet_v2:fp32:128"
)

for entry in "${ENTRIES[@]}"; do
  IFS=: read -r model precision batches <<<"$entry"
  for bs in ${batches//,/ }; do
    LABEL="gcp_${model}_${precision}_bs${bs}"
    echo "[run_cuda_matrix] $LABEL"

    extra_opts=()
    if [[ $model != "vgg16" ]]; then
      extra_opts+=(--deterministic)
    else
      echo "[run_cuda_matrix] Skipping deterministic flag for $model (non-deterministic ops)"
    fi

    RUNS=$RUNS LABEL=$LABEL bash scripts/run_repeat.sh \
      "${COMMON[@]}" "${extra_opts[@]}" \
      --arch "$model" --precision "$precision" --batch-size "$bs"
  done
done
