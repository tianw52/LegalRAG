#!/bin/bash
#SBATCH --job-name=benchmark50_full
#SBATCH --output=slurm_logs/benchmark_50_full_pipeline_%j.out
#SBATCH --error=slurm_logs/benchmark_50_full_pipeline_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Full RAG pipeline (Formulator → Retriever → Router → Generator) on benchmark_50.
#
# Submit all three (raw + Mistral + Qwen):
#   bash scripts/submit_full_pipeline_all.sh
#
# Or individually:
#   sbatch scripts/run_benchmark_50_full_pipeline.slurm   # raw
#   sbatch --dependency=afterok:RAW_ID --export=REWRITES=results/query_rewrites_compare_1vars_mistral-small-latest.json scripts/run_benchmark_50_full_pipeline.slurm
#   sbatch --dependency=afterok:RAW_ID --export=REWRITES=results/query_rewrites_compare_1vars_Qwen-Qwen2.5-72B-Instruct.json scripts/run_benchmark_50_full_pipeline.slurm
#
# Prerequisites: LegalBench-RAG data, OpenSearch, .env with LLM_API_KEY

set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
BENCHMARKS_DIR="$LEGALRAG_ROOT/benchmarks_50"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
module load protobuf/24.4
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e ".[full]" -q

export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs

set -a
[[ -f .env ]] && source .env
set +a

if [[ ! -f "$BENCHMARKS_DIR/contractnli.json" ]]; then
    echo "ERROR: benchmarks_50 not found"
    exit 1
fi
if [[ ! -d "$DATA_DIR/corpus" ]]; then
    echo "ERROR: $DATA_DIR/corpus not found"
    exit 1
fi
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

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
OPENSEARCH_TMP="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}"
mkdir -p "$OPENSEARCH_TMP"/{data,logs,config}
# Job-specific config with path.data/path.logs in SLURM_TMPDIR (writable on compute nodes)
cp -a "$OPENSEARCH_DIR/config/"* "$OPENSEARCH_TMP/config/"
echo "path.data: $OPENSEARCH_TMP/data" >> "$OPENSEARCH_TMP/config/opensearch.yml"
echo "path.logs: $OPENSEARCH_TMP/logs" >> "$OPENSEARCH_TMP/config/opensearch.yml"
echo "Starting OpenSearch (data/logs in $OPENSEARCH_TMP)..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_TMP/config" \
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

if [[ -z "$SKIP_INGEST" ]]; then
    echo "=== Ingesting corpus ==="
    curl -s -X DELETE http://localhost:9200/legalrag-legalbenchrag 2>/dev/null || true
    python3 -m evaluation.LegalBenchRAG.ingest --data-dir "$DATA_DIR" --all
else
    echo "=== Skipping ingest (SKIP_INGEST=1) ==="
fi

echo "=== Running full pipeline ==="
bash scripts/run_benchmark_50_full_pipeline.sh

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
    kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
    rm -f "$OPENSEARCH_PID"
fi

echo "=== Done ==="
