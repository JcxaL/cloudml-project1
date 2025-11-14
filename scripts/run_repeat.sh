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

choose_time_cmd(){
  if /usr/bin/time -v true >/dev/null 2>&1; then
    echo "/usr/bin/time -v"
  elif /usr/bin/time -l true >/dev/null 2>&1; then
    echo "/usr/bin/time -l"
  elif command -v gtime >/dev/null 2>&1 && gtime -v true >/dev/null 2>&1; then
    echo "gtime -v"
  else
    echo "time"
  fi
}

if [[ -n ${TIME_CMD_OVERRIDE:-} ]]; then
  TIME_CMD_STR=$TIME_CMD_OVERRIDE
else
  TIME_CMD_STR=$(choose_time_cmd)
fi
read -r -a TIME_CMD <<<"$TIME_CMD_STR"

for i in $(seq 1 "$RUNS"); do
  run_label="${LABEL}_r${i}"
  timestamp=$(date +%Y%m%d_%H%M%S)
  log_filename="${run_label}_${timestamp}.txt"
  echo "[run_repeat] Starting $run_label (log: $log_filename)"
  "${TIME_CMD[@]}" python code/run_train.py "$@" --label "$run_label" \
    2>&1 | tee "$LOG_DIR/${log_filename}"
  echo "[run_repeat] Finished $run_label"
  echo
  sleep ${SLEEP_BETWEEN:-0}
done
