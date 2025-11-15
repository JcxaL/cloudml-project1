#!/usr/bin/env bash
# Try creating a T4 VM with 200GB disk in the first zone with capacity.
# Usage: INSTANCE=name [PROJECT=...] [CANDIDATE_ZONES="zone1 zone2 ..."] ./scripts/create_gcp_t4.sh
set -euo pipefail

: "${INSTANCE:?Set INSTANCE (e.g., export INSTANCE=p1-gpu-t4)}"
PROJECT=${PROJECT:-$(gcloud config get-value core/project)}
MACHINE=${MACHINE:-n1-standard-8}
GPU_TYPE=${GPU_TYPE:-nvidia-tesla-t4}
DISK=${DISK:-200}
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
CANDIDATE_ZONES=${CANDIDATE_ZONES:-"us-central1-a us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b us-west1-c us-west2-b us-west2-c us-west3-b us-west4-a us-west4-b us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c northamerica-northeast1-b northamerica-northeast1-c"}

for Z in $CANDIDATE_ZONES; do
  echo "[create_gcp_t4] Trying zone $Z ..."
  if gcloud compute instances create "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$Z" \
      --machine-type="$MACHINE" \
      --accelerator=count=1,type="$GPU_TYPE" \
      --maintenance-policy=TERMINATE \
      --provisioning-model=STANDARD \
      --boot-disk-size="$DISK" \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --scopes=storage-full,compute-rw; then
    echo "[create_gcp_t4] Success in zone $Z"
    exit 0
  else
    echo "[create_gcp_t4] No capacity in $Z"
  fi
done

echo "[create_gcp_t4] Failed in all zones: $CANDIDATE_ZONES" >&2
exit 1
