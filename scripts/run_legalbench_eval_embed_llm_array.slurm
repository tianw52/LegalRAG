#!/bin/bash
#SBATCH --job-name=LegalBenchRAG
#SBATCH --array=1-4
#SBATCH --output=slurm_logs/legalbench_eval_embed_llm_%A_%a.out
#SBATCH --error=slurm_logs/legalbench_eval_embed_llm_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# LegalBench-RAG evaluation: 4 configs (embedding + LLM).
# Job array tasks:
#   1: Legal-Embed-bge-base + Mistral 3 14B
#   2: Legal-Embed-bge-base + Gemma 2 9B
#   3: BERT-DPR-CLERC-ft + Mistral 3 14B
#   4: BERT-DPR-CLERC-ft + Gemma 2 9B
#
# Only config changes (EMBEDDING_MODEL, LEGALBENCH_INDEX_NAME, LLM_MODEL).
# OpenSearch setup unchanged (same startup, port, security).
#
# Submit: sbatch scripts/run_legalbench_eval_embed_llm_array.slurm

set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
# Uninstall any stale editable install before reinstalling (avoids __editable__.legalrag-0.1.0.pth error)
pip uninstall legalrag -y 2>/dev/null || true
pip install -e . -q

export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs

# Config per array task (embedding + LLM only; both embeddings are 768-dim)
export EMBEDDING_DIM=768
case $SLURM_ARRAY_TASK_ID in
  1)
    export EMBEDDING_MODEL="axondendriteplus/Legal-Embed-bge-base-en-v1.5"
    export LEGALBENCH_INDEX_NAME="legalrag-legalbenchrag"
    export LLM_MODEL="mistralai/Ministral-3-14B-Instruct-2512"
    CONFIG_LABEL="LegalEmbed_Mistral3-14B"
    ;;
  2)
    export EMBEDDING_MODEL="axondendriteplus/Legal-Embed-bge-base-en-v1.5"
    export LEGALBENCH_INDEX_NAME="legalrag-legalbenchrag"
    export LLM_MODEL="google/gemma-2-9b-it"
    CONFIG_LABEL="LegalEmbed_Gemma2-9B"
    ;;
  3)
    export EMBEDDING_MODEL="jhu-clsp/BERT-DPR-CLERC-ft"
    export LEGALBENCH_INDEX_NAME="legalrag-legalbenchrag-bert-dpr"
    export LLM_MODEL="mistralai/Ministral-3-14B-Instruct-2512"
    CONFIG_LABEL="BERT-DPR-CLERC_Mistral3-14B"
    ;;
  4)
    export EMBEDDING_MODEL="jhu-clsp/BERT-DPR-CLERC-ft"
    export LEGALBENCH_INDEX_NAME="legalrag-legalbenchrag-bert-dpr"
    export LLM_MODEL="google/gemma-2-9b-it"
    CONFIG_LABEL="BERT-DPR-CLERC_Gemma2-9B"
    ;;
  *)
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=== Config: $CONFIG_LABEL ==="
echo "  EMBEDDING_MODEL=$EMBEDDING_MODEL"
echo "  LEGALBENCH_INDEX_NAME=$LEGALBENCH_INDEX_NAME"
echo "  LLM_MODEL=$LLM_MODEL"

# Check data
if [[ ! -d "$DATA_DIR/corpus" ]]; then
  echo "ERROR: $DATA_DIR/corpus not found."
  exit 1
fi
if [[ ! -f "$DATA_DIR/benchmarks/cuad.json" ]]; then
  echo "ERROR: $DATA_DIR/benchmarks/*.json not found."
  exit 1
fi

# Check OpenSearch tarball
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
  echo "ERROR: OpenSearch not at $OPENSEARCH_DIR."
  exit 1
fi

# Ensure security disabled (unchanged from original)
CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
  echo "" >> "$CONFIG"
  echo "# LegalRAG: disable security for local dev" >> "$CONFIG"
  echo "plugins.security.disabled: true" >> "$CONFIG"
fi

# OpenSearch env (unchanged)
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

# Start OpenSearch (unchanged)
OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" \
DISABLE_INSTALL_DEMO_CONFIG=true \
./bin/opensearch --daemonize -p "$OPENSEARCH_PID"
cd "$LEGALRAG_ROOT"

echo "Waiting for OpenSearch..."
for i in $(seq 1 60); do
  if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
    echo "OpenSearch ready."
    break
  fi
  sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
  echo "ERROR: OpenSearch failed to start."
  [[ -f "$OPENSEARCH_PID" ]] && kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
  exit 1
fi

# Delete only this task's index (rebuild for this embedding)
echo "=== Deleting index $LEGALBENCH_INDEX_NAME ==="
curl -s -X DELETE "http://localhost:9200/$LEGALBENCH_INDEX_NAME" 2>/dev/null || true

echo "=== Ingesting (GPU) ==="
python3 -m evaluation.LegalBenchRAG.ingest --data-dir "$DATA_DIR" --all --index-name "$LEGALBENCH_INDEX_NAME"

echo "=== Evaluating ==="
python3 -m evaluation.LegalBenchRAG.eval_precision_recall --data-dir "$DATA_DIR" --ks 20 40 60 --index-name "$LEGALBENCH_INDEX_NAME"

# Stop OpenSearch
echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
  kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
  rm -f "$OPENSEARCH_PID"
fi

echo "=== Done ($CONFIG_LABEL) ==="
