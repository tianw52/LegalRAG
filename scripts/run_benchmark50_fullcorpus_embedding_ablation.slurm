#!/bin/bash
#SBATCH --job-name=lbrag-b50-dpr
#SBATCH --output=slurm_logs/benchmark50_fullcorpus_emb_%j.out
#SBATCH --error=slurm_logs/benchmark50_fullcorpus_emb_%j.err
#SBATCH --time=18:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Full-corpus ingest (--all) + retrieval eval on benchmark_50 reformatted queries,
# two embeddings: (1) code default legal-bert, (2) JHU CLERC DPR:
#   https://huggingface.co/jhu-clsp/BERT-DPR-CLERC-ft
#
# Corpus (ingest):  $LEGALRAG_ROOT/data/LegalBenchRAG/corpus
# Eval benchmarks:  $EVAL_BENCHMARK_ROOT/benchmark_50_reformated_proccessed/<variant>
#   Default EVAL_BENCHMARK_ROOT = $LEGALRAG_ROOT/data/LegalBenchRAG (see below).
#
# Submit:
#   mkdir -p slurm_logs
#   sbatch scripts/run_benchmark50_fullcorpus_embedding_ablation.slurm
#
# Use eval data from LegalRAG_OLD instead (override default):
#   sbatch --export=EVAL_BENCHMARK_ROOT=/home/ram112/projects/def-jieliang/ram112/LegalRAG_OLD scripts/run_benchmark50_fullcorpus_embedding_ablation.slurm
#
# mistral vs qwen72b rewrite folder:
#   sbatch --export=REWRITE_VARIANT=qwen72b scripts/run_benchmark50_fullcorpus_embedding_ablation.slurm
#
# Override second embedding only:
#   sbatch --export=EMBED_SECOND=other/model scripts/run_benchmark50_fullcorpus_embedding_ablation.slurm
#
# Separate OpenSearch index per experiment (no shared INDEX_NAME between exp1/exp2):
#   INDEX_EXP1  default: legalrag-legalbenchrag-exp1-default-bert
#   INDEX_EXP2  default: legalrag-legalbenchrag-exp2-dpr-clerc
#   sbatch --export=INDEX_EXP1=my-index-a,INDEX_EXP2=my-index-b ...
#
# HuggingFace gated models / rate limits:
#   sbatch --export=HF_TOKEN=hf_... scripts/run_benchmark50_fullcorpus_embedding_ablation.slurm
#
# Prerequisites:
#   - OpenSearch tarball (see scripts/setup_opensearch_tarball.sh), OPENSEARCH_TARBALL_DIR
#   - Corpus at DATA_DIR/corpus
#   - Eval JSONs under ${EVAL_BENCHMARK_ROOT}/benchmark_50_reformated_proccessed/${REWRITE_VARIANT}/
#
# NOTE: Full corpus ingest twice is very long; increase --time if needed.

set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
# Eval JSONs: .../data/LegalBenchRAG/benchmark_50_reformated_proccessed/<mistral|qwen72b>/
EVAL_BENCHMARK_ROOT="${EVAL_BENCHMARK_ROOT:-$LEGALRAG_ROOT/data/LegalBenchRAG}"
REWRITE_VARIANT="${REWRITE_VARIANT:-mistral}"

DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
EVAL_BENCHMARKS_DIR="$EVAL_BENCHMARK_ROOT/benchmark_50_reformated_proccessed/${REWRITE_VARIANT}"

# Default embedding = legalrag EmbeddingSettings when no .env (see legalrag/core/config.py)
EMBED_DEFAULT="${EMBED_DEFAULT:-nlpaueb/legal-bert-base-uncased}"
# Second experiment: CLERC DPR fine-tuned BERT (768-dim typical). Override with EMBED_SECOND or legacy EMBED_LEGAL.
EMBED_SECOND="${EMBED_SECOND:-${EMBED_LEGAL:-jhu-clsp/BERT-DPR-CLERC-ft}}"

# One OpenSearch index name per experiment (never reuse the same name for two models).
INDEX_EXP1="${INDEX_EXP1:-legalrag-legalbenchrag-exp1-default-bert}"
INDEX_EXP2="${INDEX_EXP2:-legalrag-legalbenchrag-exp2-dpr-clerc}"
KS="${KS:-1 5 10 20}"

OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e . -q

export CUDA_VISIBLE_DEVICES=0
# Avoid OpenSearch _bulk failures on large batches (legacy client default was 30s).
export OPENSEARCH_TIMEOUT="${OPENSEARCH_TIMEOUT:-600}"
# Cap JVM heap on shared 64G nodes (default ~50% RAM can SIGBUS/OOM alongside PyTorch).
export OPENSEARCH_JAVA_OPTS="${OPENSEARCH_JAVA_OPTS:--Xms4g -Xmx10g}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-sentence_transformers}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  LEGALRAG_ROOT:     $LEGALRAG_ROOT"
echo "  DATA_DIR (corpus): $DATA_DIR/corpus  [ingest --all]"
echo "  EVAL benchmarks:   $EVAL_BENCHMARKS_DIR"
echo "  Embedding 1:       $EMBED_DEFAULT"
echo "  INDEX_EXP1:          $INDEX_EXP1"
echo "  Embedding 2:       $EMBED_SECOND"
echo "  INDEX_EXP2:          $INDEX_EXP2"
echo "  OPENSEARCH_TIMEOUT: ${OPENSEARCH_TIMEOUT}"
echo "  OPENSEARCH_JAVA_OPTS: ${OPENSEARCH_JAVA_OPTS}"
echo "================================================================================"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs/results

if [[ ! -d "$DATA_DIR/corpus" ]]; then
  echo "ERROR: corpus not found: $DATA_DIR/corpus"
  exit 1
fi
if [[ ! -f "$EVAL_BENCHMARKS_DIR/cuad.json" ]]; then
  echo "ERROR: eval benchmark JSONs not found under: $EVAL_BENCHMARKS_DIR"
  exit 1
fi
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
  echo "ERROR: OpenSearch not at $OPENSEARCH_DIR. Run: ./scripts/setup_opensearch_tarball.sh"
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

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-0}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" \
DISABLE_INSTALL_DEMO_CONFIG=true \
./bin/opensearch --daemonize -p "$OPENSEARCH_PID"
cd "$LEGALRAG_ROOT"

echo "Waiting for OpenSearch..."
for _i in $(seq 1 60); do
  if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
    echo "OpenSearch ready."
    break
  fi
  sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
  echo "ERROR: OpenSearch failed to start."
  [[ -f "$OPENSEARCH_PID" ]] && kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
  exit 1
fi

run_experiment() {
  local TAG="$1"
  local MODEL="$2"
  local INDEX="$3"

  local RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/benchmark50_emb_${SLURM_JOB_ID:-local}_${TAG}.txt"

  echo ""
  echo "██████████████████████████████████████████████████████████████████████████████"
  echo "  EXPERIMENT: ${TAG}"
  echo "  OPENSEARCH INDEX (this run only): ${INDEX}"
  echo "  EMBEDDING MODEL (ingest + eval): ${MODEL}"
  echo "  Snapshot file: ${RESULT_FILE}"
  echo "  Results below are ONLY for this embedding + index."
  echo "██████████████████████████████████████████████████████████████████████████████"
  echo ""

  echo ">>> Dropping index ${INDEX} if present (fresh ingest for this embedding)..."
  curl -sS -X DELETE "http://localhost:9200/${INDEX}" >/dev/null || true

  echo ">>> Ingest: full corpus (--all) with embedding ${MODEL} → index ${INDEX}"
  python3 -m evaluation.LegalBenchRAG.ingest \
    --data-dir "$DATA_DIR" \
    --all \
    --embedding-model "$MODEL" \
    --index-name "$INDEX" \
    --log-level INFO

  echo ""
  echo ">>> Eval: retrieval — EMBEDDING=${MODEL} INDEX=${INDEX}"
  python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir "$DATA_DIR" \
    --benchmarks-dir "$EVAL_BENCHMARKS_DIR" \
    --index-name "$INDEX" \
    --embedding-model "$MODEL" \
    --ks $KS \
    --log-level INFO 2>&1 | tee "$RESULT_FILE"

  echo ""
  echo ">>> Finished experiment: ${TAG}  (embedding=${MODEL}, index=${INDEX})"
  echo ""
}

run_experiment "1_DEFAULT_CODE_EMBEDDING" "$EMBED_DEFAULT" "$INDEX_EXP1"
run_experiment "2_BERT_DPR_CLERC_FT" "$EMBED_SECOND" "$INDEX_EXP2"

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
  kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
  rm -f "$OPENSEARCH_PID"
fi

echo "=== All experiments finished. ==="
