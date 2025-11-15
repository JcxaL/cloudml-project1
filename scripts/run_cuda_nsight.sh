#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/imagenet-mini}
WORKERS=${WORKERS:-4}
BACKEND=cuda
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-30}
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUBLAS_WORKSPACE_CONFIG

mkdir -p logs/ncu

profile_run() {
  local model=$1
  local precision=$2
  local batch=$3
  local label=$4
  shift 4
  echo "[run_cuda_nsight] $label"
  ncu --target-processes all \
    --metrics $(cat code/metric_names_ncu.txt) \
    --csv --page raw --log-file logs/ncu/${label}.csv \
    python code/run_train.py \
      --data "$DATA" --arch "$model" --batch-size "$batch" \
      --warmup-iters "$WARMUP" --iters "$ITERS" --workers "$WORKERS" \
      --backend "$BACKEND" --channels-last --no-augment "$@" \
      --precision "$precision" --label "$label"
}

profile_run resnet50 fp32 128 gcp_resnet50_fp32_bs128_prof --deterministic
profile_run resnet50 amp 256 gcp_resnet50_amp_bs256_prof --deterministic
profile_run vgg16 fp32 64 gcp_vgg16_fp32_bs64_prof
profile_run mobilenet_v2 fp32 128 gcp_mobilenet_v2_fp32_bs128_prof --deterministic
