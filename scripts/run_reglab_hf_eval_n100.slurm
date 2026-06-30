#!/bin/bash
#SBATCH --job-name=reglab-hf-n100
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/reglab_hf_n100_%j.out
#SBATCH --error=slurm_logs/reglab_hf_n100_%j.err

# ارزیابی RegLab مستقیم از Hugging Face برای N=۱۰۰ سوال (بدون export دیسک).
# شامل: روشن کردن OpenSearch، ingest از HF، char eval و paper-style eval.
#
# قبل از اجرا (ترجیحاً روی login node):  pip install -e '.[eval]'  و کش datasets برای reglab/*
# روی نود محاسبه:  export HF_HUB_OFFLINE=1  اگر کش از قبل موجود است.
#
# Optional env:
#   LEGALRAG_VENV, OPENSEARCH_TARBALL_DIR (یا به‌صورت خودکار: $SCRATCH و سپس $HOME/opensearch-2.14.0)
#   HF_HOME, HF_DATASETS_CACHE
#   HF_HUB_OFFLINE=1
#   RUN_BAREXAM=0 / RUN_HOUSING=0  — غیرفعال کردن یکی از دو دیتاست
#   SKIP_INGEST=1 — فقط eval (ایندکس باید از قبل روی همان نود ساخته شده باشد؛ معمولاً نیست)
#   EVAL_N=100 — تعداد سوال (پیش‌فرض ۱۰۰)
#
set -e

EVAL_N="${EVAL_N:-100}"
LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
REGLAB_REGLOG="${REGLAB_REGLOG:-${SCRATCH:-$HOME}/legalrag_reglab_logs}"

# offline — همه مدل‌ها و datasets از قبل کَش شده‌اند
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# OpenSearch: پیش‌فرض home؛ اگر نبود از SCRATCH (مثل نصب روی Alliance)
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
  OPENSEARCH_DIR=""
  for cand in "${SCRATCH:-}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
    if [[ -n "$cand" && -f "$cand/bin/opensearch" ]]; then
      OPENSEARCH_DIR="$cand"
      break
    fi
  done
fi

EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
# ایندکس جدا برای اجرای ۱۰۰تایی تا با ایندکس کامل قاطی نشود
INDEX_BAREXAM="${INDEX_BAREXAM:-legalrag-reglab-barexam-hf-n${EVAL_N}}"
INDEX_HOUSING="${INDEX_HOUSING:-legalrag-reglab-housing-hf-n${EVAL_N}}"
RUN_BAREXAM="${RUN_BAREXAM:-1}"
RUN_HOUSING="${RUN_HOUSING:-0}"   # فعلاً فقط barexam — برای تست کامل RUN_HOUSING=1 set کن
TOP_CHUNKS="${TOP_CHUNKS:-50000}"
HOUSING_RECALL="${HOUSING_RECALL:-upper}"

echo "=== RegLab HF eval: N=${EVAL_N} ==="
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-not set}"

mkdir -p "$LEGALRAG_ROOT/slurm_logs"

module --force purge
# arrow لازم است تا datasets/pyarrow بتواند shared library پیدا کند
module load StdEnv/2023 gcc arrow/17.0.0 cuda python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"

cd "$LEGALRAG_ROOT"
# بررسی نصب کامل legalrag — نصب باید روی login node انجام شده باشد (compute node نت ندارد)
python3 -c "from legalrag.core.config import settings; from legalrag.ingestion.embedder import build_embedder" || {
    echo "ERROR: legalrag یا pydantic در دسترس نیست."
    echo "  روی login node اجرا کن:"
    echo "  module load StdEnv/2023 gcc arrow/17.0.0 python/3.12"
    echo "  source $LEGALRAG_VENV/bin/activate"
    echo "  cd $LEGALRAG_ROOT && pip install -e . --no-deps"
    echo "  pip install pydantic pydantic-settings opensearch-py sentence-transformers python-dotenv rich structlog tenacity tiktoken pyyaml pytrec_eval-terrier"
    exit 1
}

# CUDA_VISIBLE_DEVICES را Slurm روی Alliance/CC تنظیم می‌کند — دستی =0 باعث GPU اشتباه می‌شود
mkdir -p "$REGLAB_REGLOG"

if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch پیدا نشد. set کنید OPENSEARCH_TARBALL_DIR یا نصب کنید زیر یکی از:"
    echo "  ${SCRATCH:-}/opensearch-2.14.0  یا  $HOME/opensearch-2.14.0"
    exit 1
fi
echo "OpenSearch: $OPENSEARCH_DIR"

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
# اگر security plugin نصب باشد disabled=true لازم است؛ اگر نباشد باید حذف شود
if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-security" ]]; then
    grep -q "plugins.security.disabled" "$CONFIG" || echo "plugins.security.disabled: true" >> "$CONFIG"
else
    sed -i '/plugins\.security/d' "$CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_STAMP="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-manual}.pid"

# هر جاب data/logs خودش را در SLURM_TMPDIR می‌سازد (local نود، shared نیست)
OS_DATA_DIR="${SLURM_TMPDIR:-/tmp/os_${SLURM_JOB_ID:-manual}}/opensearch-data"
OS_LOGS_DIR="${SLURM_TMPDIR:-/tmp/os_${SLURM_JOB_ID:-manual}}/opensearch-logs"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"

echo "Starting OpenSearch (data=$OS_DATA_DIR)..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
    -Epath.data="$OS_DATA_DIR" \
    -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
        echo "OpenSearch cluster health OK."
        break
    fi
    sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
    echo "ERROR: OpenSearch بالا نیامد."
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    exit 1
fi

# منتظر می‌مانیم تا KNN plugin در _cat/plugins ظاهر شود
echo "Waiting for KNN plugin to appear..."
for i in $(seq 1 24); do
    if curl -s http://localhost:9200/_cat/plugins?v 2>/dev/null | grep -q "opensearch-knn"; then
        echo "KNN plugin appeared (try $i)."
        break
    fi
    sleep 5
done

# صبر برای اینکه cluster واقعاً yellow/green بشود (native KNN library initialize شود)
# این جلوگیری می‌کند از 503 هنگام ایجاد اولین KNN index
echo "Waiting for cluster yellow status (KNN native libs)..."
for i in $(seq 1 30); do
    STATUS=$(curl -s "http://localhost:9200/_cluster/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','none'))" 2>/dev/null || echo "none")
    if [[ "$STATUS" == "yellow" || "$STATUS" == "green" ]]; then
        echo "Cluster status: $STATUS (try $i) — ready."
        break
    fi
    echo "  cluster status=$STATUS, waiting... ($i/30)"
    sleep 5
done

# یک sleep کوچک اضافه برای اطمینان از پایداری KNN
sleep 15
echo "OpenSearch آماده است."

INGEST_ARGS=(
    --chunker hierarchical
    --parent-size 2048
    --chunk-size 512
    --chunk-overlap 64
    --embedding-provider "$EMBED_PROVIDER"
    --embedding-model "$EMBED_MODEL"
    --limit-queries "$EVAL_N"
)

cleanup_os() {
    if [[ -f "$OPENSEARCH_STAMP" ]]; then
        kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
        rm -f "$OPENSEARCH_STAMP"
    fi
}
trap cleanup_os EXIT

if [[ -z "${SKIP_INGEST:-}" ]]; then
    if [[ "$RUN_BAREXAM" == "1" ]]; then
        echo "=== HF ingest: barexam_qa (تا ${EVAL_N} سوال برای فیلتر citation) → $INDEX_BAREXAM ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_BAREXAM" 2>/dev/null || true
        python3 -m evaluation.reglab.ingest_hf \
            --dataset barexam_qa \
            --index-name "$INDEX_BAREXAM" \
            "${INGEST_ARGS[@]}"
    fi
    if [[ "$RUN_HOUSING" == "1" ]]; then
        echo "=== HF ingest: housing_qa → $INDEX_HOUSING ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_HOUSING" 2>/dev/null || true
        python3 -m evaluation.reglab.ingest_hf \
            --dataset housing_qa \
            --index-name "$INDEX_HOUSING" \
            "${INGEST_ARGS[@]}"
    fi
else
    echo "=== SKIP_INGEST=1 — فرض می‌کنیم ایندکس از قبل روی localhost:9200 موجود است ==="
fi

# Char eval: مپ‌های طول passage/statute کامل (بدون --limit-corpus-maps) برای طلای درست
CHAR_EXTRA=(--limit-queries "$EVAL_N" --log-level INFO --ks 20 40 60)

if [[ "$RUN_BAREXAM" == "1" ]]; then
    echo "=== Char eval HF: barexam_qa (N=${EVAL_N}) ==="
    python3 -m evaluation.reglab.eval_char_hf \
        --benchmark barexam_qa \
        --index-name "$INDEX_BAREXAM" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$REGLAB_REGLOG/barexam_char_hf_n${EVAL_N}_${SLURM_JOB_ID}.jsonl" \
        "${CHAR_EXTRA[@]}"
    echo "=== Paper eval HF: barexam_qa (N=${EVAL_N}) ==="
    python3 -m evaluation.reglab.paper_eval \
        --source hf \
        --benchmark barexam_qa \
        --index-name "$INDEX_BAREXAM" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --top-chunks "$TOP_CHUNKS" \
        --limit "$EVAL_N" \
        --results-json "$REGLAB_REGLOG/paper_barexam_hf_n${EVAL_N}_${SLURM_JOB_ID}.json" \
        --log-level INFO
fi

if [[ "$RUN_HOUSING" == "1" ]]; then
    echo "=== Char eval HF: housing_qa (N=${EVAL_N}) ==="
    python3 -m evaluation.reglab.eval_char_hf \
        --benchmark housing_qa \
        --index-name "$INDEX_HOUSING" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$REGLAB_REGLOG/housing_char_hf_n${EVAL_N}_${SLURM_JOB_ID}.jsonl" \
        "${CHAR_EXTRA[@]}"
    echo "=== Paper eval HF: housing_qa ($HOUSING_RECALL, N=${EVAL_N}) ==="
    python3 -m evaluation.reglab.paper_eval \
        --source hf \
        --benchmark housing_qa \
        --index-name "$INDEX_HOUSING" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --top-chunks "$TOP_CHUNKS" \
        --housing-recall "$HOUSING_RECALL" \
        --limit "$EVAL_N" \
        --results-json "$REGLAB_REGLOG/paper_housing_hf_${HOUSING_RECALL}_n${EVAL_N}_${SLURM_JOB_ID}.json" \
        --log-level INFO
fi

echo "=== تمام: RegLab HF eval N=${EVAL_N} ==="
