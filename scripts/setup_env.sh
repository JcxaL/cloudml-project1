#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${1:-env/venv}
PYTHON_BIN=${PYTHON:-python3}

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found; run from repo root" >&2
  exit 1
fi

if [ ! -d "$(dirname "$VENV_PATH")" ]; then
  mkdir -p "$(dirname "$VENV_PATH")"
fi

if [ ! -d "$VENV_PATH" ]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print(f"Torch {torch.__version__}")
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
PY
