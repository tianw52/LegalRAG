#!/usr/bin/env bash
# Run on a LOGIN node with internet, or submit:
#   sbatch scripts/run_reglab_prepare_data.slurm
#
# Optional:
#   LEGALRAG_VENV=...   (default: sibling PyTorch venv next to LegalRAG in ram112 tree)
#   REGLAB_ROOT=...     (default: <repo>/data/reglab_eval) — e.g. put HF exports on $SCRATCH
#   MAX_CORPUS_DOCS=N   — smoke test; passes --max-corpus-docs to prepare
#   SKIP_PIP_INSTALL=1  — do not run pip (venv already has deps)

set -euo pipefail

LEGALRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LEGALRAG_ROOT"

LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
REGLAB_ROOT="${REGLAB_ROOT:-$LEGALRAG_ROOT/data/reglab_eval}"

# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  pip install -e ".[eval]" -q
else
  echo "SKIP_PIP_INSTALL=1 — skipping pip install -e .[eval]"
fi

mkdir -p "$REGLAB_ROOT"
if [[ -n "${MAX_CORPUS_DOCS:-}" ]]; then
  python -m evaluation.reglab.prepare barexam_qa --out-dir "$REGLAB_ROOT/barexam_qa" --max-corpus-docs "$MAX_CORPUS_DOCS"
  python -m evaluation.reglab.prepare housing_qa --out-dir "$REGLAB_ROOT/housing_qa" --max-corpus-docs "$MAX_CORPUS_DOCS"
else
  python -m evaluation.reglab.prepare barexam_qa --out-dir "$REGLAB_ROOT/barexam_qa"
  python -m evaluation.reglab.prepare housing_qa --out-dir "$REGLAB_ROOT/housing_qa"
fi

echo "Done. Data under: $REGLAB_ROOT/{barexam_qa,housing_qa}"
echo "Then submit GPU jobs, e.g.:"
echo "  cd $LEGALRAG_ROOT && sbatch --export=ALL,REGLAB_ROOT=$REGLAB_ROOT scripts/run_reglab_char_eval.slurm"
echo "  sbatch --export=ALL,REGLAB_ROOT=$REGLAB_ROOT scripts/run_reglab_paper_eval.slurm"
