#!/bin/bash
#SBATCH --job-name=lbrag-b50-4embed
#SBATCH --output=slurm_logs/b50_4embed_sweep_%j.out
#SBATCH --error=slurm_logs/b50_4embed_sweep_%j.err
#SBATCH --time=120:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#
# Full LegalBench-50 reformulated sweep: 12 configs (hier/rec × 6 embeddings), same K-grid as prior sweeps.
# Eval JSONs:  data/LegalBenchRAG/benchmark_50_reformated_proccessed/<REWRITE_VARIANT>/
#
# Single variant (default mistral):
#   mkdir -p slurm_logs
#   sbatch scripts/run_b50_4embed_sweep_one_variant.slurm
#
# Choose variant:
#   sbatch --export=REWRITE_VARIANT=qwen35_9b,HF_TOKEN=hf_... scripts/run_b50_4embed_sweep_one_variant.slurm
#
# All three in parallel (array 0–2):
#   sbatch --array=0-2 scripts/run_b50_4embed_sweep_all_variants.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"

REWRITE_VARIANT="${REWRITE_VARIANT:-mistral}"

export OPENSEARCH_TIMEOUT="${OPENSEARCH_TIMEOUT:-600}"
# Slightly lower JVM heap vs 10g to reduce cgroup OOM risk alongside Python+GPU; 128G job mem.
export OPENSEARCH_JAVA_OPTS="${OPENSEARCH_JAVA_OPTS:--Xms4g -Xmx8g}"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")" slurm_logs
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip uninstall legalrag -y 2>/dev/null || true
  pip install -e . -q
) 9>"$PIP_LOCK"

export CUDA_VISIBLE_DEVICES=0
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-sentence_transformers}"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "================================================================================"
echo "  b50 — 4-embedding × chunker sweep — Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-}"
echo "  REWRITE_VARIANT:  $REWRITE_VARIANT"
echo "  Eval:             $DATA_DIR/benchmark_50_reformated_proccessed/${REWRITE_VARIANT}"
echo "================================================================================"

[[ -d "$DATA_DIR/corpus" ]] || { echo "ERROR: missing corpus: $DATA_DIR/corpus"; exit 1; }
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "ERROR: OpenSearch not at $OPENSEARCH_DIR"; exit 1; }

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
  echo "" >> "$CONFIG"
  echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-0}.pid"
OPENSEARCH_DATA_DIR="${SLURM_TMPDIR:-/tmp}/opensearch_data_${SLURM_JOB_ID:-0}"
mkdir -p "$OPENSEARCH_DATA_DIR"
echo "OpenSearch path.data (job-local): $OPENSEARCH_DATA_DIR"
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
  ./bin/opensearch --daemonize -p "$OPENSEARCH_PID" \
  -E "path.data=$OPENSEARCH_DATA_DIR" \
  -E "cluster.name=legalrag-os-${SLURM_JOB_ID:-local}"
cd "$LEGALRAG_ROOT"

for _i in $(seq 1 60); do
  curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && { echo "OpenSearch ready."; break; }
  sleep 5
done
curl -s http://localhost:9200/_cluster/health | grep -q '"status"' || { echo "ERROR: OpenSearch failed"; exit 1; }

bash scripts/run_b50_4embed_sweep_one_variant.sh

[[ -f "$OPENSEARCH_PID" ]] && kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
rm -f "$OPENSEARCH_PID"
echo "=== Slurm job finished (4-embed sweep). ==="
