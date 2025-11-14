#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: RUNS=3 LABEL=mylabel $0 [python args...]" >&2
  exit 1
fi

RUNS=${RUNS:-3}
LABEL=${LABEL:-run}
LOG_DIR=${LOG_DIR:-logs/time}
mkdir -p "$LOG_DIR"

for i in $(seq 1 "$RUNS"); do
  run_label="${LABEL}_r${i}"
  echo "[run_repeat] Starting $run_label"
  /usr/bin/time -v python code/run_train.py "$@" --label "$run_label" \
    2>&1 | tee "$LOG_DIR/${run_label}.txt"
  echo "[run_repeat] Finished $run_label"
  echo
  sleep ${SLEEP_BETWEEN:-0}
done
