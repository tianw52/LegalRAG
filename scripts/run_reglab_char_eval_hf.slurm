#!/bin/bash
#SBATCH --job-name=reglab-char-hf
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=reglab_char_eval_hf_%j.out
#SBATCH --error=reglab_char_eval_hf_%j.err

# RegLab char-level eval from Hugging Face (no corpus/*.txt or benchmarks/*.json).
# Does not run run_reglab_prepare_data.sh. Uses evaluation.reglab.ingest_hf + eval_char_hf.
#
# Optional exports:
#   HF_HOME / HF_DATASETS_CACHE — local Hugging Face cache
#   HF_HUB_OFFLINE=1           — use only cached reglab/* (recommended on compute nodes)
#   LIMIT_CORPUS=N             — passed as --limit-corpus to ingest_hf (smoke / cap)
#   LIMIT_QUERIES_INGEST=N     — benchmark-filter ingest uses first N HF QA rows for citation set
#   LIMIT_QUERIES_EVAL=N       — eval_char_hf --limit-queries (default: all)
#   LIMIT_CORPUS_MAPS=N        — eval_char_hf --limit-corpus-maps (smoke only; may drop gold)
#   INGEST_ALL=1               — ingest full HF split (heavy)
#   SKIP_INGEST=1
#   INDEX_BAREXAM / INDEX_HOUSING — default *-hf to avoid clobbering disk-based indices
#   SMOKE=1                    — sets small limits for a quick run (see below)
#
# Smoke (interactive or short job):
#   SMOKE=1 sbatch scripts/run_reglab_char_eval_hf.slurm
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
REGLAB_REGLOG="${REGLAB_REGLOG:-${SCRATCH:-$HOME}/legalrag_reglab_logs}"

EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
INDEX_BAREXAM="${INDEX_BAREXAM:-legalrag-reglab-barexam-hf}"
INDEX_HOUSING="${INDEX_HOUSING:-legalrag-reglab-housing-hf}"
RUN_BAREXAM="${RUN_BAREXAM:-1}"
RUN_HOUSING="${RUN_HOUSING:-1}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export LIMIT_CORPUS="${LIMIT_CORPUS:-1000}"
  export LIMIT_QUERIES_INGEST="${LIMIT_QUERIES_INGEST:-20}"
  export LIMIT_QUERIES_EVAL="${LIMIT_QUERIES_EVAL:-20}"
  export LIMIT_CORPUS_MAPS="${LIMIT_CORPUS_MAPS:-8000}"
  export KS="${KS:-20 40}"
fi

LIMIT_CORPUS="${LIMIT_CORPUS:-}"
LIMIT_QUERIES_INGEST="${LIMIT_QUERIES_INGEST:-}"
LIMIT_QUERIES_EVAL="${LIMIT_QUERIES_EVAL:-}"
LIMIT_CORPUS_MAPS="${LIMIT_CORPUS_MAPS:-}"
INGEST_ALL="${INGEST_ALL:-0}"

module --force purge
module load StdEnv/2023 gcc cuda python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"

cd "$LEGALRAG_ROOT"
python3 -c "import legalrag" || {
    echo "ERROR: legalrag not importable. pip install -e '.[eval]' on a login node."
    exit 1
}

# CUDA_VISIBLE_DEVICES را Slurm تنظیم می‌کند
mkdir -p "$REGLAB_REGLOG"

if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch not at $OPENSEARCH_DIR"
    exit 1
fi

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
    echo "" >> "$CONFIG"
    echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_STAMP="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
        echo "OpenSearch ready."
        break
    fi
    sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
    echo "ERROR: OpenSearch failed to start."
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    exit 1
fi

INGEST_ARGS=(
    --chunker hierarchical
    --parent-size 2048
    --chunk-size 512
    --chunk-overlap 64
    --embedding-provider "$EMBED_PROVIDER"
    --embedding-model "$EMBED_MODEL"
)
ingest_extra=()
[[ -n "$LIMIT_CORPUS" ]] && ingest_extra+=(--limit-corpus "$LIMIT_CORPUS")
[[ -n "$LIMIT_QUERIES_INGEST" ]] && ingest_extra+=(--limit-queries "$LIMIT_QUERIES_INGEST")
if [[ "$INGEST_ALL" == "1" ]]; then
    ingest_extra=(--ingest-all "${ingest_extra[@]}")
fi

if [[ -z "${SKIP_INGEST:-}" ]]; then
    if [[ "$RUN_BAREXAM" == "1" ]]; then
        echo "=== HF ingest barexam_qa → $INDEX_BAREXAM ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_BAREXAM" 2>/dev/null || true
        python3 -m evaluation.reglab.ingest_hf \
            --dataset barexam_qa \
            --index-name "$INDEX_BAREXAM" \
            "${INGEST_ARGS[@]}" \
            "${ingest_extra[@]}"
    fi
    if [[ "$RUN_HOUSING" == "1" ]]; then
        echo "=== HF ingest housing_qa → $INDEX_HOUSING ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_HOUSING" 2>/dev/null || true
        python3 -m evaluation.reglab.ingest_hf \
            --dataset housing_qa \
            --index-name "$INDEX_HOUSING" \
            "${INGEST_ARGS[@]}" \
            "${ingest_extra[@]}"
    fi
else
    echo "=== SKIP_INGEST set — using existing indices ==="
fi

EVAL_EXTRA=(--log-level INFO)
[[ -n "${KS:-}" ]] && EVAL_EXTRA+=(--ks $KS)
[[ -n "$LIMIT_QUERIES_EVAL" ]] && EVAL_EXTRA+=(--limit-queries "$LIMIT_QUERIES_EVAL")
[[ -n "$LIMIT_CORPUS_MAPS" ]] && EVAL_EXTRA+=(--limit-corpus-maps "$LIMIT_CORPUS_MAPS")

if [[ "$RUN_BAREXAM" == "1" ]]; then
    echo "=== Char eval HF: barexam_qa ==="
    python3 -m evaluation.reglab.eval_char_hf \
        --benchmark barexam_qa \
        --index-name "$INDEX_BAREXAM" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$REGLAB_REGLOG/barexam_char_hf_${SLURM_JOB_ID}.jsonl" \
        "${EVAL_EXTRA[@]}"
fi
if [[ "$RUN_HOUSING" == "1" ]]; then
    echo "=== Char eval HF: housing_qa ==="
    python3 -m evaluation.reglab.eval_char_hf \
        --benchmark housing_qa \
        --index-name "$INDEX_HOUSING" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$REGLAB_REGLOG/housing_char_hf_${SLURM_JOB_ID}.jsonl" \
        "${EVAL_EXTRA[@]}"
fi

if [[ -f "$OPENSEARCH_STAMP" ]]; then
    kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    rm -f "$OPENSEARCH_STAMP"
fi
echo "=== RegLab char eval (HF) done ==="
