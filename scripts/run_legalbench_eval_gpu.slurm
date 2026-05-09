#!/bin/bash
#SBATCH --job-name=LegalBenchRAG
#SBATCH --output=slurm_logs/legalbench_eval.out
#SBATCH --error=slurm_logs/legalbench_eval.err
#SBATCH --time=04:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# LegalBench-RAG evaluation on cluster GPU.
# Prerequisites:
#   1. Run setup_opensearch_tarball.sh once (downloads OpenSearch)
#   2. LegalBench-RAG data in data/LegalBenchRAG/ (corpus + benchmarks)
#   3. pip install -e . runs inside the job (no manual install needed)

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
# Install legalrag for cluster eval (no pydantic-ai - avoids protoc/temporalio build)
pip install -e . -q

# Ensure CUDA for embeddings
export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs

# Check data
if [[ ! -d "$DATA_DIR/corpus" ]]; then
    echo "ERROR: $DATA_DIR/corpus not found. Download LegalBench-RAG from Dropbox."
    exit 1
fi
if [[ ! -f "$DATA_DIR/benchmarks/cuad.json" ]]; then
    echo "ERROR: $DATA_DIR/benchmarks/*.json not found."
    exit 1
fi

# Check OpenSearch tarball
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch not at $OPENSEARCH_DIR. Run: ./scripts/setup_opensearch_tarball.sh"
    exit 1
fi

# Ensure security is disabled (required for cluster - no SSL certs)
CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
    echo "Adding plugins.security.disabled to OpenSearch config..."
    echo "" >> "$CONFIG"
    echo "# LegalRAG: disable security for local dev" >> "$CONFIG"
    echo "plugins.security.disabled: true" >> "$CONFIG"
fi

# Create .env for cluster (port 9200, no Cursor conflict)
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
# Override port for cluster (no Cursor)
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

# Start OpenSearch in background (security disabled via config from setup script)
OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" \
DISABLE_INSTALL_DEMO_CONFIG=true \
./bin/opensearch --daemonize -p "$OPENSEARCH_PID"
cd "$LEGALRAG_ROOT"

# Wait for OpenSearch
echo "Waiting for OpenSearch..."
for i in $(seq 1 60); do
    if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
        echo "OpenSearch ready."
        break
    fi
    sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
    echo "ERROR: OpenSearch failed to start. Check vm.max_map_count (may need: sudo sysctl -w vm.max_map_count=262144)"
    [[ -f "$OPENSEARCH_PID" ]] && kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
    exit 1
fi

# Backup existing Legal-Embed index (save to legalrag-legalbenchrag-legal-embed)
# Reindex copies the index; original legalrag-legalbenchrag is never deleted
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:9200/legalrag-legalbenchrag" | grep -q 200; then
    echo "=== Backing up Legal-Embed index to legalrag-legalbenchrag-legal-embed ==="
    curl -s -X DELETE "http://localhost:9200/legalrag-legalbenchrag-legal-embed" 2>/dev/null || true
    curl -s -X POST "http://localhost:9200/_reindex?wait_for_completion=true" \
        -H 'Content-Type: application/json' \
        -d '{"source":{"index":"legalrag-legalbenchrag"},"dest":{"index":"legalrag-legalbenchrag-legal-embed"}}' \
        || echo "Reindex backup failed (non-fatal)"
fi

# Delete only the BERT-DPR index (we are rebuilding it)
echo "=== Deleting LegalBench index (legalrag-legalbenchrag) ==="
curl -s -X DELETE http://localhost:9200/legalrag-legalbenchrag 2>/dev/null || true

echo "=== Ingesting (GPU) ==="
python3 -m evaluation.LegalBenchRAG.ingest --data-dir "$DATA_DIR" --all

echo "=== Evaluating ==="
python3 -m evaluation.LegalBenchRAG.eval_precision_recall --data-dir "$DATA_DIR" --ks 20 40 60

# Stop OpenSearch
echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
    kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
    rm -f "$OPENSEARCH_PID"
fi

echo "=== Done ==="
