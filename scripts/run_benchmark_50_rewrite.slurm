#!/bin/bash
#SBATCH --job-name=benchmark50_rewrite
#SBATCH --output=slurm_logs/benchmark_50_rewrite_%j.out
#SBATCH --error=slurm_logs/benchmark_50_rewrite_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Rewrite 200 queries in benchmarks_50 using one model.
# No GPU needed — API calls only.
#
# Submit (pick one model per job):
#   sbatch --export=MODEL=gemini scripts/run_benchmark_50_rewrite.slurm
#   sbatch --export=MODEL=mistral scripts/run_benchmark_50_rewrite.slurm
#   sbatch --export=MODEL=qwen72b scripts/run_benchmark_50_rewrite.slurm
#
# Resume after 402 — prefer repairing the checked-in JSONs (no query_rewrite.py required):
#   export HF_TOKEN=hf_...
#   python3 scripts/rewrite_benchmark_50_qwen_hf.py
#   python3 scripts/build_benchmark_50_reformated_processed.py --models qwen72b
#
# Or resume the old pipeline if you still have a partial results/*.json:
#   sbatch --export=MODEL=qwen72b,RESUME_FROM=results/query_rewrites_compare_1vars_Qwen-Qwen2.5-72B-Instruct.json scripts/run_benchmark_50_rewrite.slurm
#
# Resume Gemini after quota (set new key in .env as GOOGLE_API_KEY or GOOGLE_API_KEY_BACKUP):
#   sbatch --export=MODEL=gemini,RESUME_FROM=results/query_rewrites_compare_1vars_gemini-2.5-flash-lite.json scripts/run_benchmark_50_rewrite.slurm
#
# After 200 rows, split + merge for eval (benchmark_50_reformated_gemini.json):
#   python3 scripts/split_compare_json_to_benchmark_50_reformated.py \
#     --input results/query_rewrites_compare_1vars_gemini-2.5-flash-lite.json --subdir gemini
#   python3 scripts/merge_benchmark_50_reformated_rewrites.py

set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"

module --force purge
module load StdEnv/2023 python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip install openai google-genai -q 2>/dev/null || true
mkdir -p slurm_logs

# Load .env (must have API keys)
set -a
[[ -f .env ]] && source .env
set +a

MODEL="${MODEL:-gemini}"

case "$MODEL" in
    gemini)
        export LLM_PROVIDER=gemini
        export LLM_MODEL="gemini-2.5-flash-lite"
        [[ -z "$GOOGLE_API_KEY" && -z "$GEMINI_API_KEY" ]] && { echo "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set (see .env.example)"; exit 1; }
        ;;
    mistral)
        export LLM_PROVIDER=mistral
        export LLM_MODEL="mistral-small-latest"
        [[ -z "$MISTRAL_API_KEY" ]] && { echo "ERROR: MISTRAL_API_KEY not set"; exit 1; }
        ;;
    qwen72b)
        export LLM_PROVIDER=""
        export LLM_MODEL="Qwen/Qwen2.5-72B-Instruct"
        export LLM_BASE_URL="${LLM_BASE_URL:-https://router.huggingface.co/v1}"
        [[ -z "$LLM_API_KEY" ]] && { echo "ERROR: LLM_API_KEY not set"; exit 1; }
        ;;
    *)
        echo "ERROR: MODEL must be gemini, mistral, or qwen72b"
        exit 1
        ;;
esac

echo "=== Benchmark 50 — Rewrite ($MODEL) ==="
echo "Model: $LLM_MODEL"
[[ -n "$RESUME_FROM" ]] && echo "Resume from: $RESUME_FROM"
echo ""

EXTRA=()
[[ -n "$RESUME_FROM" ]] && EXTRA+=(--resume-from "$RESUME_FROM")

python3 -u scripts/query_rewrite.py --mode compare --benchmarks-dir benchmarks_50 --variant v4_reddit_style "${EXTRA[@]}"

echo ""
echo "=== Done — check results/query_rewrites_compare_1vars_*.json ==="
