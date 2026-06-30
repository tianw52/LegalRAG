#!/bin/bash
#SBATCH --job-name=rgl-hou-flat
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/rgl_housing_paper_flat_%A_%a.out
#SBATCH --error=slurm_logs/rgl_housing_paper_flat_%A_%a.err
#
# Paper-aligned Housing Statute QA — flat/statute-level retrieval (no child chunking).
# Compare directly to Zheng et al. Table 3 upper-bound passage Recall@K / MRR@10.
#
# Array tasks:
#   0  flat_passage + mpnet  → BM25 lexical           — paper BM25 R@10 = 40.8%
#   1  flat_passage + e5     → dense semantic         — paper E5-large-v2 R@10 = 50.6%
#   2  flat_passage + mpnet  → BM25 + structured qexp — paper BM25+reasoning ≈ 51.1%
#   3  flat_passage + e5     → dense + structured qexp — paper E5+reasoning ≈ 52.8%
#
# Tasks 2–3 require housing_qa_expanded.json (run run_reglab_housing_expand.slurm first).
# Results: $SCRATCH/reglab_eval/results_housing_paper_flat/
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

HOUSING_DIR="${HOUSING_DIR:-${SCRATCH}/reglab_eval/housing_qa}"
PARQUET_META_DIR="${PARQUET_META_DIR:-${HOUSING_DIR}/passages_meta}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/reglab_eval/results_housing_paper_flat}"
TOP_CHUNKS="${TOP_CHUNKS:-10000}"
HOUSING_RECALL="${HOUSING_RECALL:-upper}"
MIN_INDEX_DOCS="${MIN_INDEX_DOCS:-1500000}"
EXPANDED_BENCHMARK="${HOUSING_DIR}/benchmarks/housing_qa_expanded.json"

# flat_passage statutes can be very long → small batches (same as BarExam flat diag)
CHILD_CHUNK_BATCH="${CHILD_CHUNK_BATCH:-32}"
ST_ENCODE_BATCH="${ST_ENCODE_BATCH:-16}"

declare -a EMBED_SHORTS=(mpnet e5 mpnet e5)
declare -a EMBED_PROVIDERS=(
    sentence_transformers
    sentence_transformers
    sentence_transformers
    sentence_transformers
)
declare -a EMBED_MODELS=(
    "sentence-transformers/all-mpnet-base-v2"
    "intfloat/e5-large-v2"
    "sentence-transformers/all-mpnet-base-v2"
    "intfloat/e5-large-v2"
)
declare -a RETRIEVAL_MODES=(lexical semantic lexical semantic)
declare -a DIAG_LABELS=(
    "flat-bm25-baseline"
    "flat-e5-baseline"
    "flat-bm25-qexp"
    "flat-e5-qexp"
)
declare -a USE_QEXP=(0 0 1 1)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
EMBED_SHORT="${EMBED_SHORTS[$TASK_ID]}"
EMBED_PROVIDER="${EMBED_PROVIDERS[$TASK_ID]}"
EMBED_MODEL="${EMBED_MODELS[$TASK_ID]}"
RETRIEVAL_MODE="${RETRIEVAL_MODES[$TASK_ID]}"
DIAG_LABEL="${DIAG_LABELS[$TASK_ID]}"
QEXP="${USE_QEXP[$TASK_ID]}"

INDEX_NAME="rglhou-flat-${DIAG_LABEL}"

echo "=== Housing paper-flat — task ${TASK_ID}: ${DIAG_LABEL} ==="
echo "  CHUNKER         = flat_passage (one statute/passage per doc)"
echo "  EMBED_MODEL     = $EMBED_MODEL"
echo "  RETRIEVAL_MODE  = $RETRIEVAL_MODE"
echo "  QUERY_EXPANSION = $([[ $QEXP -eq 1 ]] && echo yes || echo no)"
echo "  INDEX_NAME      = $INDEX_NAME"
echo "  MIN_INDEX_DOCS  = $MIN_INDEX_DOCS"
echo "  TOP_CHUNKS      = $TOP_CHUNKS"
date

mkdir -p slurm_logs "$RESULTS_DIR"

[[ -d "$PARQUET_META_DIR" ]] && compgen -G "$PARQUET_META_DIR/passages_part_*.parquet" >/dev/null \
    || { echo "ERROR: No parquet under $PARQUET_META_DIR"; exit 1; }
[[ -f "$HOUSING_DIR/benchmarks/housing_qa.json" ]] \
    || { echo "ERROR: Missing $HOUSING_DIR/benchmarks/housing_qa.json"; exit 1; }

if [[ "$QEXP" -eq 1 ]]; then
    [[ -f "$EXPANDED_BENCHMARK" ]] \
        || { echo "ERROR: Query expansion benchmark missing: $EXPANDED_BENCHMARK"; exit 1; }
    BENCHMARK_FILE_ARG=(--benchmark-file "$EXPANDED_BENCHMARK")
else
    BENCHMARK_FILE_ARG=()
fi

OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    for _cand in "${SCRATCH}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
        [[ -f "$_cand/bin/opensearch" ]] && { OPENSEARCH_DIR="$_cand"; break; }
    done
fi
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "ERROR: OpenSearch not found"; exit 1; }

OS_CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
grep -q "plugins.security.disabled" "$OS_CONFIG" \
    || echo "plugins.security.disabled: true" >> "$OS_CONFIG"

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 cuda python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

export HF_HOME="/scratch/ram112/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env 2>/dev/null || true
grep -q "OPENSEARCH_PORT" .env \
    && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env \
    || echo "OPENSEARCH_PORT=9200" >> .env

OS_DATA_DIR="${SLURM_TMPDIR}/opensearch-housing-flat-${TASK_ID}"
OS_LOGS_DIR="${SLURM_TMPDIR}/opensearch-housing-flat-logs-${TASK_ID}"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"
OPENSEARCH_STAMP="${SLURM_TMPDIR}/opensearch_housing_flat_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"

cleanup_os() {
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
}
trap cleanup_os EXIT

echo "--- Starting OpenSearch ---"
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
    -Epath.data="$OS_DATA_DIR" -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
curl -s http://localhost:9200/_cluster/health | grep -q '"status"' \
    || { echo "ERROR: OpenSearch failed to start"; exit 1; }
echo "OpenSearch ready."

if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-knn" ]]; then
    for i in $(seq 1 30); do
        _ST=$(curl -s "http://localhost:9200/_cluster/health" \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
        [[ "$_ST" == "yellow" || "$_ST" == "green" ]] && break
        sleep 5
    done
    sleep 10
fi

echo "=== Ingest → $INDEX_NAME (flat_passage, ~1.8M statutes) ==="
curl -s -X DELETE "http://localhost:9200/$INDEX_NAME" 2>/dev/null || true

python3 tools/full_corpus_fast/fast_ingest.py \
    --parquet "$PARQUET_META_DIR" \
    --index-name "$INDEX_NAME" \
    --embedding-provider "$EMBED_PROVIDER" \
    --embedding-model "$EMBED_MODEL" \
    --chunker flat_passage \
    --st-encode-batch "$ST_ENCODE_BATCH" \
    --child-chunk-batch "$CHILD_CHUNK_BATCH" \
    --log-level INFO

INDEX_COUNT=$(curl -s "http://localhost:9200/${INDEX_NAME}/_count" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)
echo "=== Post-ingest index count: ${INDEX_COUNT} documents in ${INDEX_NAME} ==="

if [[ "${INDEX_COUNT:-0}" -lt "$MIN_INDEX_DOCS" ]]; then
    RESULTS_PAPER="$RESULTS_DIR/paper_housing_${HOUSING_RECALL}_${DIAG_LABEL}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.json"
    python3 - <<PY
import json
from pathlib import Path
payload = {
    "status": "invalid",
    "invalid_reason": "empty_or_underpopulated_index",
    "benchmark": "housing_qa",
    "paper": "Zheng et al. CS&Law 2025",
    "n_queries": 6853,
    "top_chunks": ${TOP_CHUNKS},
    "retrieval_mode": "${RETRIEVAL_MODE}",
    "housing_recall_mode": "${HOUSING_RECALL}",
    "jurisdiction_filter": "enabled",
    "chunker": "flat_passage",
    "recall_passage_percent": None,
    "mrr_10_percent": None,
    "index_name": "${INDEX_NAME}",
    "index_document_count": int("${INDEX_COUNT:-0}"),
    "min_index_docs_required": ${MIN_INDEX_DOCS},
    "embedding_model": "${EMBED_MODEL}",
    "diag_label": "${DIAG_LABEL}",
    "query_expansion": bool(${QEXP}),
}
path = Path("${RESULTS_PAPER}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote invalid stub → {path}")
PY
    echo "ERROR: Index underpopulated (${INDEX_COUNT} < ${MIN_INDEX_DOCS}). Aborting eval."
    exit 2
fi

RESULTS_PAPER="$RESULTS_DIR/paper_housing_${HOUSING_RECALL}_${DIAG_LABEL}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.json"
echo "=== Paper eval (${RETRIEVAL_MODE}, upper bound, jurisdiction filter) → $RESULTS_PAPER ==="

python3 -m evaluation.reglab.paper_eval \
    --source disk \
    --data-dir "$HOUSING_DIR" \
    --benchmark housing_qa \
    "${BENCHMARK_FILE_ARG[@]}" \
    --index-name "$INDEX_NAME" \
    --embedding-provider "$EMBED_PROVIDER" \
    --embedding-model "$EMBED_MODEL" \
    --retrieval-mode "$RETRIEVAL_MODE" \
    --top-chunks "$TOP_CHUNKS" \
    --housing-recall "$HOUSING_RECALL" \
    --min-index-docs "$MIN_INDEX_DOCS" \
    --fail-on-zero-recall \
    --ks 1 10 100 1000 \
    --results-json "$RESULTS_PAPER" \
    --log-level WARNING

python3 - <<PY
import json
from pathlib import Path
p = Path("${RESULTS_PAPER}")
d = json.loads(p.read_text())
d["diag_label"] = "${DIAG_LABEL}"
d["chunker"] = "flat_passage"
d["query_expansion"] = bool(${QEXP})
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"Annotated {p}")
PY

echo "=== Housing paper-flat task ${TASK_ID} (${DIAG_LABEL}) complete ==="
date
