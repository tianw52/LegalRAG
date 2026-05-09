# Compute Canada (Fir) — راه‌اندازی LegalRAG

این سند خلاصهٔ همان روشی است که در `LegalRAG_OLD` استفاده شده: **بدون Docker**، OpenSearch از **tarball**، و اجرای کارها با **Slurm**.

## ۱. مخزن و remotes

- مسیر پروژه (مثال): `~/projects/def-jieliang/ram112/LegalRAG`
- **origin**: فورک شما — `https://github.com/ReyhanehAhani/LegalRAG.git`
- **upstream** (اختیاری برای sync): `https://github.com/tianw52/LegalRAG.git`

```bash
git remote add upstream https://github.com/tianw52/LegalRAG.git   # اگر هنوز نیست
git fetch upstream
```

## ۲. OpenSearch (یک بار)

اسکریپت: `scripts/setup_opensearch_tarball.sh`

- نسخه پیش‌فرض: **OpenSearch 2.14.0**
- نصب در: `$HOME/opensearch-2.14.0` (یا با `OPENSEARCH_INSTALL_DIR` عوض کنید)
- در `config/opensearch.yml` مقدار `plugins.security.disabled: true` اضافه می‌شود (مثل docker-compose محلی)

اگر قبلاً نصب کرده‌اید، دوباره لازم نیست.

اگر خطای security دیدید:

```bash
./scripts/fix_opensearch_security.sh
```

در jobهای Slurm معمولاً این متغیر استفاده می‌شود:

```bash
export OPENSEARCH_TARBALL_DIR="$HOME/opensearch-2.14.0"
```

## ۳. دادهٔ LegalBench-RAG

مسیر مورد انتظار اسکریپت‌ها:

- `data/LegalBenchRAG/corpus/`
- `data/LegalBenchRAG/benchmarks/*.json`

اگر در `LegalRAG_OLD` داده دارید، می‌توانید همان را لینک یا کپی کنید (حجم زیاد است؛ بهتر است از همان محل مشترک استفاده کنید).

## ۴. محیط Python (مثال همان که در Slurm است)

```bash
module --force purge
module load StdEnv/2023 gcc cuda python/3.12   # طبق نیاز حساب/پارتیشن
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate
cd ~/projects/def-jieliang/ram112/LegalRAG
cp .env.example .env
# API keys و OPENSEARCH_* را در .env پر کنید (هرگز commit نکنید)
pip install -e ".[dev,eval]"   # یا حداقل pip install -e .
```

## ۵. اجرای ارزیابی با GPU (Slurm)

نمونه: `scripts/run_legalbench_eval_gpu.slurm`

- قبل از اجرا: `mkdir -p slurm_logs`
- مسیر ریشهٔ پروژه در فایل Slurm باید با محل clone شما یکی باشد (`LEGALRAG_ROOT`).
- Job داخل خودش OpenSearch را روی نود بالا می‌آورد و بعد ارزیابی را اجرا می‌کند.

```bash
cd ~/projects/def-jieliang/ram112/LegalRAG
mkdir -p slurm_logs
sbatch scripts/run_legalbench_eval_gpu.slurm
```

سایر jobها (benchmark_50، query rewrite، …) هم در `scripts/*.slurm` هستند؛ همان الگو: ماژول‌ها، venv، `OPENSEARCH_TARBALL_DIR`.

## ۶. مرجع قدیمی

برای جزئیات بیشتر یا تنظیمات قبلی، پوشهٔ مرجع:

`/home/ram112/projects/def-jieliang/ram112/LegalRAG_OLD`

---

## English summary

- **OpenSearch**: run `scripts/setup_opensearch_tarball.sh` once; use `OPENSEARCH_TARBALL_DIR` in jobs.
- **Data**: `data/LegalBenchRAG/` with corpus + benchmarks.
- **Jobs**: Slurm scripts under `scripts/*.slurm`; adjust `LEGALRAG_ROOT` and account/GPU lines for your allocation.
- **Secrets**: copy `.env.example` → `.env`; never commit `.env`.
