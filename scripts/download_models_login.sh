#!/bin/bash
# Cache Hugging Face models required by the eval array jobs.
# Preferred on cluster:  sbatch scripts/run_download_hf_models.slurm
# Or interactively on login (needs internet):  bash scripts/download_models_login.sh
set -euo pipefail

LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
PYTHON="$LEGALRAG_VENV/bin/python3"

module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true
source "$LEGALRAG_VENV/bin/activate"

echo "=== Downloading HuggingFace models for eval ==="
"$PYTHON" - <<'EOF'
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import subprocess

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
    print(f"\n--- {m} (transformers snapshot) ---")
    snapshot_download(m)
    print(f"    OK")

for m in models_st:
    print(f"\n--- {m} (SentenceTransformer) ---")
    SentenceTransformer(m)
    print(f"    OK")

print("\n=== All models ready ===")
EOF
