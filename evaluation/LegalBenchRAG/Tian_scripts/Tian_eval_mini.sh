# ------------------------- CLERC Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-clerc \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_clerc.jsonl \
    --ks 1 2 4 8 16 32 64


# ------------------------- LegalBert Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-legalbert \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \         
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-legalbert \
    --embedding-provider huggingface \
    --embedding-model nlpaueb/legal-bert-base-uncased \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_legalbert.jsonl \
    --ks 1 2 4 8 16 32 64

# ------------------------- Legal-Embed-BGE Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-legal-embed-bge-base-en-v1.5 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-legal-embed-bge-base-en-v1.5 \
    --embedding-provider sentence_transformers \
    --embedding-model axondendriteplus/Legal-Embed-bge-base-en-v1.5 \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_legal-embed-bge-base-en-v1.5.jsonl \
    --ks 1 2 4 8 16 32 64

# ------------------------- all-mpnet-base-v2 Embedding --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-all-mpnet-base-v2 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_all-mpnet-base-v2.jsonl \
    --ks 1 2 4 8 16 32 64


# ------------------------- Octen-Embedding-0.6B --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-octen \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model Octen/Octen-Embedding-0.6B \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-octen \
    --embedding-provider sentence_transformers \
    --embedding-model Octen/Octen-Embedding-0.6B \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_octen.jsonl \
    --ks 1 2 4 8 16 32 64

# ------------------------- Qwen3-Embedding-0.6B --------------------------
## ========== Hierarchical Chunking ==========
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-qwen3 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/legalbenchrag-mini \
    --index-name lbr-mini-hier-qwen3 \
    --embedding-provider sentence_transformers \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --trace-file logs/eval/lbr-mini/original/lbr_hier_qwen3.jsonl \
    --ks 1 2 4 8 16 32 64