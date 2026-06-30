#!/bin/bash
#SBATCH --job-name=table22-baseline
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Reproduce Table 22 (Historical MBE subset, baseline rows) from:
#   "A Reasoning-Focused Legal Retrieval Benchmark" (Zheng et al., CS&Law 2025)
#
# PREREQUISITES — must be done on login node before submitting:
#   1. pip install rank-bm25
#   2. All E5 models downloaded into HuggingFace cache
#   3. Run: python3 reproduction/preflight_check.py   (should print all ✓)
#
# ARRAY TASK → MODEL:
#   0 = BM25                   (CPU, ~1 hr, no GPU needed)
#   1 = E5-small-v2            (~1 hr on A100)
#   2 = E5-base-v2             (~2 hr on A100)
#   3 = E5-large-v2            (~3 hr on A100)
#   4 = E5-mistral-7b-instruct (~10 hr on A100)
#
# Submit all five tasks in parallel:
#   bash reproduction/submit_table22.sh
#
# Or run a single model:
#   sbatch --array=0 reproduction/run_table22_baseline.slurm  # BM25 only
#   sbatch --array=3 reproduction/run_table22_baseline.slurm  # E5-large only
#
# Results written to:
#   ${RESULTS_DIR}/bm25_metrics.json
#   ${RESULTS_DIR}/e5-{small,base,large,mistral}_metrics.json
#   ${RESULTS_DIR}/passages_cache.npz   (shared across tasks, built by first task)
#
set -euo pipefail

REPO_ROOT="/home/ram112/projects/def-jieliang/ram112"
LEGALRAG_VENV="${LEGALRAG_VENV:-${REPO_ROOT}/PyTorch}"
SCRIPT="${REPO_ROOT}/reproduction/reproduce_table21_baseline.py"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/reproduction/table21_results}"
PASSAGE_CACHE="${RESULTS_DIR}/passages_cache.npz"

# ── Enforce offline mode: compute nodes have no internet access ──────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0   # datasets reads from local cache files (parquet), not network

# ── Point to project-local HF cache if set ───────────────────────────────────
# Uncomment and set if you use a non-default HF cache location:
# export HF_HOME="/path/to/your/hf_cache"

module --force purge
# arrow/17.0.0 provides pyarrow, which is required by sentence-transformers and datasets.
# python/3.12 is the base interpreter matching the venv.
module load StdEnv/2023 gcc arrow/17.0.0 cuda/12.2 python/3.12 2>/dev/null || true
source "${LEGALRAG_VENV}/bin/activate"

# ── Sanity checks (fail fast instead of wasting GPU time) ───────────────────
echo "=== Table 22 Baseline — task ${SLURM_ARRAY_TASK_ID} ==="
echo "  Host   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Python : $(python3 --version)"
echo "  Results: ${RESULTS_DIR}"
echo ""

python3 - <<'PYCHECK'
import sys, importlib

missing = []
for pkg in ("rank_bm25", "numpy", "sentence_transformers", "datasets", "tqdm"):
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"ERROR: Missing packages: {missing}", file=sys.stderr)
    print("Run 'pip install <pkg>' on the login node first.", file=sys.stderr)
    sys.exit(1)

# faiss is optional; numpy fallback is used if absent
try:
    import faiss
    print(f"  faiss    : OK (v{faiss.__version__})")
except ImportError:
    print("  faiss    : absent — numpy fallback will be used")

import rank_bm25, numpy, sentence_transformers, datasets
print(f"  rank_bm25: OK")
print(f"  numpy    : {numpy.__version__}")
print(f"  st       : {sentence_transformers.__version__}")
print(f"  datasets : {datasets.__version__}")
PYCHECK

mkdir -p "${RESULTS_DIR}"

# ── Run the retriever for this array task ────────────────────────────────────
python3 "${SCRIPT}" \
    --model-idx "${SLURM_ARRAY_TASK_ID}" \
    --results-dir "${RESULTS_DIR}" \
    --passage-cache "${PASSAGE_CACHE}"

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
