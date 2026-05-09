#!/usr/bin/env bash
# Execute notebooks/eda_reglab_deep_dive.ipynb headlessly (all plots embedded in outputs).
# For Narval / Alliance when `jupyter lab` is not available.
#
# Prerequisite:
#   module load scipy-stack/2024b
#
# Usage (from LegalRAG repo root):
#   bash scripts/run_eda_notebook.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv_notebook"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV" --system-site-packages
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install -q -U pip
python -m pip install -q "nbconvert>=7.0" "ipykernel>=6.0"

IN="${ROOT}/notebooks/eda_reglab_deep_dive.ipynb"
OUT="${ROOT}/notebooks/eda_reglab_deep_dive_executed.ipynb"

python -m jupyter nbconvert \
  --to notebook \
  --execute "$IN" \
  --output "${OUT##*/}" \
  --output-dir "${OUT%/*}" \
  --Execute.timeout=3600 \
  --Execute.kernel_name=python3

echo "Wrote: $OUT"

python -m jupyter nbconvert \
  --to html \
  "$OUT" \
  --output-dir "${OUT%/*}" \
  --no-input

echo "HTML: ${OUT%.ipynb}.html"
