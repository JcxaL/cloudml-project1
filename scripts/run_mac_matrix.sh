#!/usr/bin/env bash
# Helper to run the Mac CPU or MPS matrix via run_repeat.sh.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cpu|mps>" >&2
  exit 1
fi

BACKEND=$1
if [[ $BACKEND != "cpu" && $BACKEND != "mps" ]]; then
  echo "Backend must be cpu or mps" >&2
  exit 1
fi

DATA_ROOT=${DATA_ROOT:-data/imagenet-mini}
COMMON_ARGS=(
  --data "$DATA_ROOT"
  --warmup-iters 10
  --iters 100
  --workers 0
  --precision fp32
  --no-augment
  --deterministic
)

if [[ $BACKEND == "cpu" ]]; then
  MODELS=("resnet50:16 32" "vgg16:16" "mobilenet_v2:16 32 64")
else
  MODELS=("resnet50:16 32" "vgg16:16" "mobilenet_v2:16 32 64")
fi

for entry in "${MODELS[@]}"; do
  model=${entry%%:*}
  batches=${entry##*:}
  for bs in $batches; do
    label_prefix="mac_${BACKEND}_${model}_fp32_bs${bs}"
    RUNS=${RUNS:-3} LABEL=$label_prefix bash scripts/run_repeat.sh \
      "${COMMON_ARGS[@]}" --arch "$model" --batch-size "$bs" --backend "$BACKEND"
  done
done
