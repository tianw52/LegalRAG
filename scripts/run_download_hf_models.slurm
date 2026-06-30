#!/bin/bash
#SBATCH --job-name=hf-models-dl
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/hf_models_dl_%j.out
#SBATCH --error=slurm_logs/hf_models_dl_%j.err
#
# Download six Hugging Face embedding models for evaluation jobs (cache in ~/.cache/huggingface).
# Requires internet; if your compute nodes have no outbound access, this job will fail —
# run on a login node or a partition with network instead.
#
#   sbatch scripts/run_download_hf_models.slurm
#
# Optional:
#   LEGALRAG_VENV — venv (default: ../PyTorch relative to LegalRAG)
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"

mkdir -p "$LEGALRAG_ROOT/slurm_logs"

# Eval jobs usually run offline; Hub must be reachable here
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true

# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"

cd "$LEGALRAG_ROOT"
echo "=== HF model download job ==="
echo "  HF_HOME effective: ${HF_HOME:-$HOME/.cache/huggingface (hub default)}"
echo "  Venv: $LEGALRAG_VENV"
date

python3 - <<'EOF'
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

models_hf = [
    "jhu-clsp/BERT-DPR-CLERC-ft",
    "nlpaueb/legal-bert-base-uncased",
]
models_st = [
    "axondendriteplus/Legal-Embed-bge-base-en-v1.5",
    "Octen/Octen-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-mpnet-base-v2",
]

for m in models_hf:
    print(f"\n--- {m} (snapshot_download) ---")
    snapshot_download(m)
    print("    OK")

for m in models_st:
    print(f"\n--- {m} (SentenceTransformer) ---")
    SentenceTransformer(m)
    print("    OK")

print("\n=== All models ready ===")
EOF

echo "=== Done ==="
date
