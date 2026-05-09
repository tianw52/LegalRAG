#!/bin/bash
#SBATCH --job-name=query_rewrite
#SBATCH --output=slurm_logs/query_rewrite_%j.out
#SBATCH --error=slurm_logs/query_rewrite_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Runs query_rewrite.py --mode compare using qwen3:14b via a local Ollama server.
# No internet access required on the compute node — everything is pre-downloaded.
#
# Prerequisites (one-time, run from the login node):
#   1. bin/ollama       — binary already at LegalRAG/bin/ollama  ✓
#   2. bin/ollama_models — model already pulled to LegalRAG/bin/ollama_models  ✓
#
# Submit:
#   cd /home/ram112/projects/def-jieliang/ram112/LegalRAG
#   sbatch scripts/run_query_rewrite.slurm

set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
QUERY_SCRIPT="$LEGALRAG_ROOT/scripts/query_rewrite.py"
ENV_FILE="$LEGALRAG_ROOT/.env"

# Pre-downloaded assets — no curl/network needed on compute node
OLLAMA_BIN="$LEGALRAG_ROOT/bin/ollama"
OLLAMA_MODELS_DIR="$LEGALRAG_ROOT/bin/ollama_models"

# Unique port per job to avoid collisions on shared nodes
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
OLLAMA_LOG="${SLURM_TMPDIR:-/tmp}/ollama_${SLURM_JOB_ID}.log"

echo "================================================================"
echo " LegalRAG Query Rewrite  (SLURM job $SLURM_JOB_ID)"
echo " Node     : $(hostname)"
echo " Port     : $OLLAMA_PORT"
echo " Binary   : $OLLAMA_BIN"
echo " Models   : $OLLAMA_MODELS_DIR"
echo "================================================================"

# ── Load modules ──────────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e . -q
pip install openai -q

mkdir -p slurm_logs

# ── Step 1: Sanity-check pre-downloaded assets ────────────────────────────────
echo ""
echo "[1/3] Checking pre-downloaded assets..."

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: Ollama binary not found at $OLLAMA_BIN"
    exit 1
fi
if [[ ! -d "$OLLAMA_MODELS_DIR/blobs" ]]; then
    echo "ERROR: Model cache not found at $OLLAMA_MODELS_DIR"
    exit 1
fi

echo "      Binary : $($OLLAMA_BIN --version 2>/dev/null | grep -v Warning || echo 'ok')"
echo "      Models : $(du -sh "$OLLAMA_MODELS_DIR" | cut -f1) on disk"

# ── Step 2: Start Ollama server ───────────────────────────────────────────────
echo ""
echo "[2/3] Starting Ollama server on port $OLLAMA_PORT..."

OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
CUDA_VISIBLE_DEVICES=0 \
"$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
echo "      pid: $OLLAMA_PID  log: $OLLAMA_LOG"

echo "      Waiting for server..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" &>/dev/null; then
        echo "      Ready after ${i}s."
        break
    fi
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" &>/dev/null; then
    echo "ERROR: Ollama server did not start. Last log:"
    tail -20 "$OLLAMA_LOG"
    exit 1
fi

# Verify qwen3:14b is visible and warm up the model (first request loads into GPU, can take 2-5 min)
echo "      Models available:"
curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" | python3 -c \
    "import sys,json; [print('       -', m['name']) for m in json.load(sys.stdin).get('models',[])]" \
    2>/dev/null || echo "      (could not list models)"
echo "      Warming up model (first load into GPU, may take 2-5 min)..."
curl -sf --max-time 600 -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3:14b","prompt":"Hi","stream":false}' \
    >/dev/null && echo "      Warmup done." || echo "      Warmup failed (non-fatal)."

# ── Step 3: Run query rewriting ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Running compare mode: v4_reddit_style | v5_everyday_words | v7_google_punchy"
echo " 4 queries × 4 datasets × 3 variants = 48 LLM calls"
echo "================================================================"
echo ""

# Pass the Ollama URL directly via env — no need to modify .env on disk
LLM_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1" \
LLM_API_KEY="ollama" \
LLM_MODEL="qwen3:14b" \
    python3 -u "$QUERY_SCRIPT" --mode compare --n 4

echo ""
echo "Results saved to $LEGALRAG_ROOT/results/query_rewrites_compare.json"

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "Stopping Ollama (pid $OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true
echo "Done."
