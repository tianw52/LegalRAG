#!/bin/bash
#SBATCH --job-name=benchmark50
#SBATCH --output=slurm_logs/benchmark_50_eval_%j.out
#SBATCH --error=slurm_logs/benchmark_50_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Benchmark 50 evaluation on cluster.
# Evaluates LegalRAG on benchmarks_50 (200 queries) — character-level CharRecall/CharPrecision.
#
# Raw queries:
#   sbatch scripts/run_benchmark_50_eval.slurm
#
# With benchmark_50_reformated rewrites (merged JSON; run merge script first):
#   python3 scripts/merge_benchmark_50_reformated_rewrites.py
#   sbatch --export=REWRITES=results/benchmark_50_reformated_mistral.json scripts/run_benchmark_50_eval.slurm
#   sbatch --export=REWRITES=results/benchmark_50_reformated_qwen72b.json scripts/run_benchmark_50_eval.slurm
#   sbatch --export=REWRITES=results/benchmark_50_reformated_gemini.json scripts/run_benchmark_50_eval.slurm
#
# Optional: skip ingest if index already exists (e.g. from prior job):
#   sbatch --export=SKIP_INGEST=1 scripts/run_benchmark_50_eval.slurm
#
# Prerequisites: LegalBench-RAG data in data/LegalBenchRAG/, OpenSearch tarball

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

# Check benchmarks_50
if [[ ! -f "$BENCHMARKS_DIR/contractnli.json" ]]; then
    echo "ERROR: benchmarks_50 not found at $BENCHMARKS_DIR"
    exit 1
fi

# Check data
if [[ ! -d "$DATA_DIR/corpus" ]]; then
    echo "ERROR: $DATA_DIR/corpus not found."
    exit 1
fi

# Check OpenSearch tarball
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch not at $OPENSEARCH_DIR. Run: ./scripts/setup_opensearch_tarball.sh"
    exit 1
fi

# Ensure security disabled
CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
    echo "" >> "$CONFIG"
    echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

# Start OpenSearch
OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
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

# Ingest (unless skipped)
if [[ -z "$SKIP_INGEST" ]]; then
    echo "=== Ingesting corpus ==="
    # Default LegalBench index (matches evaluation.LegalBenchRAG.pipeline.DEFAULT_INDEX_NAME)
    curl -s -X DELETE http://localhost:9200/legalrag-legalbenchrag 2>/dev/null || true
    python3 -m evaluation.LegalBenchRAG.ingest --data-dir "$DATA_DIR" --all
else
    echo "=== Skipping ingest (SKIP_INGEST=1) ==="
fi

EXTRA_ARGS=()
if [[ -n "$REWRITES" ]]; then
    RW="$REWRITES"
    [[ "$RW" != /* ]] && RW="$LEGALRAG_ROOT/$RW"
    if [[ -f "$RW" ]]; then
        EXTRA_ARGS+=(--rewrites-file "$RW" --rewrite-variant "${REWRITE_VARIANT:-v4_reddit_style}")
        echo "=== Benchmark 50 — REWRITTEN ($REWRITES) ==="
    else
        echo "WARN: Rewrites not found: $RW — using raw queries"
    fi
fi
[[ ${#EXTRA_ARGS[@]} -eq 0 ]] && echo "=== Benchmark 50 — RAW queries ==="
echo "=== Evaluating ==="
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir "$DATA_DIR" \
    --benchmarks-dir "$BENCHMARKS_DIR" \
    --ks 1 5 10 20 \
    "${EXTRA_ARGS[@]}"

# Stop OpenSearch
echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
    kill $(cat "$OPENSEARCH_PID") 2>/dev/null || true
    rm -f "$OPENSEARCH_PID"
fi

echo "=== Done ==="
