# ------------------------- CLERC Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --trace-file logs/eval/gemini_3/lbr_hier_clerc.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60
## ========== Recursive Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-clerc \
    --chunker recursive \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --trace-file logs/eval/gemini_3/lbr_rec_clerc.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

# ------------------------- LegalBert Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-legalbert \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-legalbert \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --trace-file logs/eval/gemini_3/lbr_hier_legalbert.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

## ========== Recursive Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-legalbert \
    --chunker recursive \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-legalbert \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --trace-file logs/eval/gemini_3/lbr_rec_legalbert.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

# ------------------------- Legal-Embed-BGE Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-legal-embed-bge-base-en-v1.5 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-legal-embed-bge-base-en-v1.5 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --trace-file logs/eval/gemini_3/lbr_hier_legal-embed-bge.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

## ========== Recursive Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-legal-embed-bge-base-en-v1.5 \
    --chunker recursive \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-legal-embed-bge-base-en-v1.5 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --trace-file logs/eval/gemini_3/lbr_rec_legal-embed-bge.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

# ------------------------- all-mpnet-base-v2 Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --trace-file logs/eval/gemini_3/lbr_hier_all-mpnet.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

## ========== Recursive Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-all-mpnet-base-v2 \
    --chunker recursive \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --trace-file logs/eval/gemini_3/lbr_rec_all-mpnet.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

# ------------------------- Octen-Embedding-0.6B --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-octen \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model Octen/Octen-Embedding-0.6B \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-octen \
    --embedding-provider sentence_transformers \
    --embedding-model Octen/Octen-Embedding-0.6B \
    --trace-file logs/eval/gemini_3/lbr_hier_octen.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60

# ------------------------- Qwen3-Embedding-0.6B --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-qwen3 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-qwen3 \
    --embedding-provider sentence_transformers \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --trace-file logs/eval/gemini_3/lbr_hier_qwen3.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmark_50_reformated_proccessed/gemini-3-flash \
    --ks 2 4 6 10 15 20 40 60