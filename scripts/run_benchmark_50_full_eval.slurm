#!/bin/bash
#SBATCH --job-name=bm50_full
#SBATCH --output=slurm_logs/benchmark_50_full_eval_%j.out
#SBATCH --error=slurm_logs/benchmark_50_full_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# benchmarks_50 — character-level eval (aligned with ReyhanehAhani/LegalRAG eval).
# Upstream eval does not support --rewrites-file or chunk-level metrics in one script.
#
# Optional: SKIP_INGEST=1
#   sbatch --export=SKIP_INGEST=1 scripts/run_benchmark_50_full_eval.slurm

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
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e . -q

export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs

[[ ! -f "$BENCHMARKS_DIR/contractnli.json" ]] && { echo "ERROR: benchmarks_50 not found"; exit 1; }
[[ ! -d "$DATA_DIR/corpus" ]] && { echo "ERROR: corpus not found"; exit 1; }
[[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]] && { echo "ERROR: OpenSearch not found"; exit 1; }

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
[[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG" && echo -e "\nplugins.security.disabled: true" >> "$CONFIG"

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
./bin/opensearch --daemonize -p "$OPENSEARCH_PID"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
curl -s http://localhost:9200/_cluster/health | grep -q '"status"' || { echo "OpenSearch failed"; kill $(cat "$OPENSEARCH_PID") 2>/dev/null; exit 1; }
echo "OpenSearch ready."

if [[ -z "$SKIP_INGEST" ]]; then
    echo "=== Ingesting corpus ==="
    curl -s -X DELETE http://localhost:9200/legalrag-legalbenchrag 2>/dev/null || true
    python3 -m evaluation.LegalBenchRAG.ingest --data-dir "$DATA_DIR" --all
else
    echo "=== Skipping ingest ==="
fi

echo "=== benchmarks_50 — character-level (upstream eval) ==="
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir "$DATA_DIR" \
    --benchmarks-dir "$BENCHMARKS_DIR" \
    --ks 1 5 10 20

echo "Stopping OpenSearch..."
[[ -f "$OPENSEARCH_PID" ]] && kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
rm -f "$OPENSEARCH_PID"
echo "=== Done ==="
