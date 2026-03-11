# LegalBench-RAG Evaluation

Retrieval evaluation pipeline for [zeroentropy-ai/legalbenchrag](https://github.com/zeroentropy-ai/legalbenchrag) — a character-level IR benchmark covering legal contracts (CUAD, ContractNLI, MAUD, PrivacyQA).

Unlike CLERC (which uses NDCG/MRR over ranked document IDs), LegalBench-RAG measures retrieval quality at the **exact character level**:

| Metric    | Formula |
|-----------|---------|
| Recall    | `|retrieved chars ∩ ground-truth chars|` / `|ground-truth chars|` |
| Precision | `|retrieved chars ∩ ground-truth chars|` / `|retrieved chars|`    |

Retrieved chunk spans are merged per file before computing the intersection with ground-truth spans, following the methodology in the [original paper](https://arxiv.org/abs/2408.10343).

---

## Data Download

Download the benchmark data from the [official Dropbox link](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0).

Extract to:

```
data/LegalBenchRAG/
    corpus/
        contractnli/   *.txt
        cuad/          *.txt
        maud/          *.txt
        privacy_qa/    *.txt
    benchmarks/
        contractnli.json
        cuad.json
        maud.json
        privacy_qa.json
```

---

## Prerequisites

### 1. Ensure OpenSearch is running

```bash
curl http://localhost:9200/_cluster/health
```

---

## Quickstart

### Step 1 — Ingest corpus

By default, only the corpus files referenced by benchmark test cases are ingested (much smaller than the full corpus). This is the recommended first run.

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG
```

For a fast smoke test (10 test cases per sub-benchmark → minimal corpus):

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --limit 10
```

### Step 2 — Evaluate

```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG
```

Expected output:

```
────────────────────────────────────────────
  LegalBench-RAG Evaluation — char-level
────────────────────────────────────────────
  Benchmark          Recall  Precision      N
  ────────────────── ────────  ─────────  ─────
  contractnli        0.xxxx     0.xxxx    NNN
  cuad               0.xxxx     0.xxxx    NNN
  maud               0.xxxx     0.xxxx    NNN
  privacy_qa         0.xxxx     0.xxxx    NNN
  ────────────────── ────────  ─────────  ─────
  OVERALL            0.xxxx     0.xxxx    NNN
────────────────────────────────────────────
  Index : legalrag-legalbenchrag
```

---

## Options

### Ingest options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | required | Root of downloaded data dir |
| `--benchmarks NAME [NAME ...]` | all four | Restrict which sub-benchmarks determine which corpus files to ingest |
| `--limit N` | None (all) | Cap test cases per benchmark when selecting corpus files |
| `--all` | false | Ingest every `*.txt` file under `corpus/` (ignores benchmark filter) |
| `--log-level` | INFO | Verbosity |

### Evaluation options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | required | Root of downloaded data dir |
| `--benchmarks NAME [NAME ...]` | all four | Sub-benchmarks to evaluate |
| `--limit N` | None (all) | Cap test cases per benchmark (for fast iteration) |
| `--ks K [K ...]` | 20 40 60 | Rank cutoffs to evaluate; retrieves `max(ks)` chunks |
| `--log-level` | WARNING | Verbosity |

---

## Evaluate a single sub-benchmark

```bash
# Ingest only CUAD corpus files
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad

# Evaluate CUAD only, retrieve top-50 chunks
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad \
    --ks 20 40 60
```

---

## Re-ingestion

Delete the index and re-run ingestion when you change `EMBEDDING_MODEL` / `EMBEDDING_DIM` in `.env`:

```bash
curl -X DELETE http://localhost:9200/legalrag-legalbenchrag
python -m evaluation.LegalBenchRAG.ingest --data-dir data/LegalBenchRAG
```

---

## File Reference

| File | Description |
|------|-------------|
| `loader.py` | `LegalBenchRAGCorpusLoader`, `load_benchmark()`, `BenchmarkTestCase`, `BenchmarkSnippet` |
| `pipeline.py` | `LegalBenchRAGIngestionPipeline` — HierarchicalChunker → embed → OpenSearch |
| `ingest.py` | CLI for corpus ingestion |
| `eval_precision_recall.py` | CLI for character-level Precision & Recall evaluation |

---

## How chunk-to-span mapping works

1. Each corpus document is ingested with its **relative file path** (e.g. `cuad/contract_001.txt`) stored as `metadata.citation`.
2. `HierarchicalChunker` records `char_start` / `char_end` (character offsets in the original document text) on every child chunk.
3. At evaluation time, retrieved chunks are grouped by `metadata.citation` (= file path).
4. Spans within the same file are **sorted and merged** (overlapping spans collapsed).
5. Merged retrieved spans are intersected with ground-truth spans from the benchmark JSON.
6. Recall and precision are computed from total character counts.

---

## Sub-benchmarks

| Name | Source | Domain |
|------|--------|--------|
| `cuad` | [CUAD](https://arxiv.org/abs/2103.06268) | Commercial contracts (expert-annotated) |
| `contractnli` | [ContractNLI](https://arxiv.org/abs/2110.01799) | NDA / contract NLI |
| `maud` | [MAUD](https://arxiv.org/abs/2301.00876) | Merger agreements |
| `privacy_qa` | [PrivacyQA](https://aclanthology.org/D19-1500/) | App privacy policies |

---

## Citation

```bibtex
@article{pipitone2024legalbenchrag,
  title={LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain},
  author={Pipitone, Nicholas and Houir Alami, Ghita},
  journal={arXiv preprint arXiv:2408.10343},
  year={2024},
  url={https://arxiv.org/abs/2408.10343}
}
```
