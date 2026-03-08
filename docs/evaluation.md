# Evaluation

---

## CLERC Dataset

[jhu-clsp/CLERC](https://huggingface.co/datasets/jhu-clsp/CLERC) is a legal retrieval and generation benchmark from JHU CLSP (106k rows). It provides query–passage pairs from US federal/state case law, making it useful for evaluating retrieval pipeline quality.

---

## Folder Structure

```
jhu-clsp/CLERC/
├── collection/
│   ├── collection.doc.tsv.gz         (7.6 GB)  — full documents (one per row)
│   ├── collection.passage.tsv.gz     (8.6 GB)  — passages / chunks
│   ├── mapping.did2pid.tsv           (231 MB)  — doc ID → passage IDs
│   └── mapping.pid2did.tsv           (374 MB)  — passage ID → doc ID
├── generation/
│   ├── all.jsonl                     (5.0 GB)  — full dataset (query + positives + negatives)
│   ├── test.jsonl                    (99  MB)  — test split
│   └── train.jsonl                   (488 MB)  — train split
├── qrels/
│   ├── qrels-doc.test.direct.tsv         — ground truth at doc level (direct citations)
│   ├── qrels-doc.test.indirect.tsv       — ground truth at doc level (indirect citations)
│   ├── qrels-passage.test.direct.tsv     — ground truth at passage level (direct)
│   └── qrels-passage.test.indirect.tsv   — ground truth at passage level (indirect)
├── queries/
│   ├── test.all-removed.direct.tsv       — queries with all citations removed
│   ├── test.all-removed.indirect.tsv
│   ├── test.single-removed.direct.tsv    — queries with one citation removed
│   └── test.single-removed.indirect.tsv
└── triples/
    └── triples.train.tsv             — (query, positive, negative) triples for model training
```

---

## File Roles


| File                                     | Use in LegalRAG                                            |
| ---------------------------------------- | ---------------------------------------------------------- |
| `collection/collection.passage.tsv.gz`   | Corpus to ingest into OpenSearch (passage-level chunks)    |
| `collection/collection.doc.tsv.gz`       | Full documents (parent chunks equivalent)                  |
| `queries/test.single-removed.direct.tsv` | Evaluation queries — run through pipeline                  |
| `qrels/qrels-passage.test.direct.tsv`    | Ground truth relevance — compute Recall@K, MRR, NDCG       |
| `generation/test.jsonl`                  | All-in-one format: query + positives + negatives bundled   |
| `triples/triples.train.tsv`              | Fine-tuning an embedding model (not needed for evaluation) |


**Direct vs indirect**: Direct citations are explicitly named in the query text; indirect are
supporting precedents referenced in context.

---

## Loading the Dataset

```python
from datasets import load_dataset

# Generation task — query + positive + 20 hard negatives per row
dataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "generation/test.jsonl"},
    split="data",
)

# IR task — streaming to avoid downloading the full 8.6 GB corpus
corpus = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "collection/collection.passage.tsv.gz"},
    split="data",
    streaming=True,
)
```

---

## Evaluating the Retrieval Pipeline

### Option A — Using `generation/test.jsonl` (quickest)

Each row has `query`, `positive_passages` (1 item), and `negative_passages` (20 hard negatives).
Compute Recall@K: did the retriever return the positive passage within the top K results?

```python
from datasets import load_dataset
from legalrag.query.formulator import PassthroughQueryFormulator
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient

dataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "generation/test.jsonl"},
    split="data",
)

os_client = OpenSearchClient.from_settings()
embedder = build_embedder()
retriever = OpenSearchRetriever(os_client, embedder, mode="hybrid")
formulator = PassthroughQueryFormulator()

hits_at_k = {5: 0, 10: 0, 20: 0}
total = 0

for row in dataset:
    sq = formulator.formulate(row["query"])
    results = retriever.retrieve(sq)
    positive_id = row["positive_passages"][0]["docid"]

    retrieved_ids = [r.chunk.doc_id for r in results]
    for k in hits_at_k:
        if positive_id in retrieved_ids[:k]:
            hits_at_k[k] += 1
    total += 1

for k, hits in hits_at_k.items():
    print(f"Recall@{k}: {hits/total:.4f}")
```

### Option B — BEIR-style evaluation with `pytrec_eval`

Uses `queries/test.single-removed.direct.tsv` + `qrels/qrels-passage.test.direct.tsv`.
Standard IR evaluation computing NDCG@10, MRR@10, MAP.

```bash
pip install pytrec_eval-terrier
```

```python
import pytrec_eval
from legalrag.query.formulator import PassthroughQueryFormulator
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient

# Load queries: tsv with columns query_id, query_text
queries = {}
with open("queries/test.single-removed.direct.tsv") as f:
    for line in f:
        qid, text = line.strip().split("\t", 1)
        queries[qid] = text

# Load qrels: tsv with columns query_id, 0, passage_id, relevance
qrels = {}
with open("qrels/qrels-passage.test.direct.tsv") as f:
    for line in f:
        qid, _, pid, rel = line.strip().split("\t")
        qrels.setdefault(qid, {})[pid] = int(rel)

# Run retrieval
os_client = OpenSearchClient.from_settings()
embedder = build_embedder()
retriever = OpenSearchRetriever(os_client, embedder, mode="hybrid")
formulator = PassthroughQueryFormulator()

run = {}
for qid, text in queries.items():
    sq = formulator.formulate(text)
    results = retriever.retrieve(sq)
    run[qid] = {r.chunk.chunk_id: r.semantic_score or 0.0 for r in results}

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recip_rank", "map"})
metrics = evaluator.evaluate(run)

avg = {m: sum(v[m] for v in metrics.values()) / len(metrics) for m in ["ndcg_cut_10", "recip_rank", "map"]}
print(f"NDCG@10: {avg['ndcg_cut_10']:.4f}")
print(f"MRR:     {avg['recip_rank']:.4f}")
print(f"MAP:     {avg['map']:.4f}")
```

### Option C — Reranker quality (hard negatives)

Test whether `CrossEncoderReranker` correctly promotes the positive passage above the 20
hard negatives. No OpenSearch needed — pass the passages directly.

```python
from datasets import load_dataset
from legalrag.query.reranker import CrossEncoderReranker
from legalrag.core.models import Chunk, LegalDocumentMetadata, RetrievedChunk

dataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "generation/test.jsonl"},
    split="data",
)
reranker = CrossEncoderReranker()

correct = 0
total = 0

for row in dataset.select(range(200)):   # sample 200 rows
    query = row["query"]
    positive = row["positive_passages"][0]
    negatives = row["negative_passages"]

    candidates = []
    for p in [positive] + negatives:
        meta = LegalDocumentMetadata(source_path=p["docid"], doc_id=p["docid"])
        chunk = Chunk(chunk_id=p["docid"], doc_id=p["docid"], text=p["text"], metadata=meta)
        candidates.append(RetrievedChunk(chunk=chunk))

    reranked = reranker.rerank(query, candidates, top_n=len(candidates))
    if reranked[0].chunk.chunk_id == positive["docid"]:
        correct += 1
    total += 1

print(f"Reranker Precision@1: {correct/total:.4f}")
```

---

## LegalBench-RAG

[zeroentropy-ai/legalbenchrag](https://github.com/zeroentropy-ai/legalbenchrag) is an IR
benchmark for legal contract understanding (paper: [arXiv:2408.10343](https://arxiv.org/abs/2408.10343)).
Unlike CLERC (case law), LegalBench-RAG covers contracts: ContractNLI, CUAD, MAUD, PrivacyQA.

**Key differentiator**: evaluation is at the **exact character level** — ground truth is a
file path + character index range, giving deterministic precision and recall without
ambiguous passage boundary issues.

### Dataset Structure

```
data/
├── corpus/       # raw .txt contract files, may have subdirectory hierarchy
└── benchmarks/   # JSON benchmark files, one per dataset (CUAD, MAUD, etc.)
```

Each benchmark JSON has this shape:

```json
{
  "test_cases": [
    {
      "query": "Does the contract include a non-compete clause?",
      "ground_truth": [
        {
          "file_path": "corpus/contract_123.txt",
          "char_start": 4521,
          "char_end": 4892
        }
      ]
    }
  ]
}
```

### Setup

```bash
# Download the pre-built benchmark + corpus
# (see README for download link — avoids re-running expensive LLM generation step)
# Place corpus/ under data/corpus/ and benchmarks/ under data/benchmarks/
```

### Adapting for LegalRAG

Ingest the `data/corpus/` files via your existing `IngestionPipeline`, then map retrieved
chunks back to `(source_path, char_start, char_end)` spans for the benchmark evaluation.

```python
import json
from legalrag.query.formulator import PassthroughQueryFormulator
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient

# Ingest the corpus first:
#   python scripts/ingest.py /path/to/legalbenchrag/data/corpus/

# Load a benchmark file
with open("/path/to/legalbenchrag/data/benchmarks/cuad.json") as f:
    benchmark = json.load(f)

os_client = OpenSearchClient.from_settings()
embedder = build_embedder()
retriever = OpenSearchRetriever(os_client, embedder, mode="hybrid")
formulator = PassthroughQueryFormulator()

precision_scores, recall_scores = [], []

for case in benchmark["test_cases"]:
    sq = formulator.formulate(case["query"])
    results = retriever.retrieve(sq)

    # Ground truth character spans (file path + char range)
    gt_spans = {
        (gt["file_path"], gt["char_start"], gt["char_end"])
        for gt in case["ground_truth"]
    }

    # Retrieved spans from chunk metadata
    retrieved_spans = {
        (r.chunk.metadata.source_path, r.chunk.char_start, r.chunk.char_end)
        for r in results
    }

    # Exact boundary match (simplified — see note below for partial credit)
    tp = len(gt_spans & retrieved_spans)
    precision_scores.append(tp / len(retrieved_spans) if retrieved_spans else 0)
    recall_scores.append(tp / len(gt_spans) if gt_spans else 0)

print(f"Precision: {sum(precision_scores)/len(precision_scores):.4f}")
print(f"Recall:    {sum(recall_scores)/len(recall_scores):.4f}")
```

> **Note on span matching**: the above does exact boundary matching. LegalBench-RAG's  
> official scorer uses character-level overlap (partial credit for chunks that partially  
> cover a ground truth span). Use the official `benchmark.py` runner for canonical numbers.

### Comparison: CLERC vs LegalBench-RAG


|                         | CLERC                      | LegalBench-RAG                                       |
| ----------------------- | -------------------------- | ---------------------------------------------------- |
| Domain                  | US case law                | Legal contracts (CUAD, MAUD, ContractNLI, PrivacyQA) |
| Granularity             | Passage-level              | Exact character-level                                |
| Evaluation metrics      | NDCG / Recall@K / MRR      | Precision / Recall (character overlap)               |
| Corpus size             | 8.6 GB passages            | Smaller (contract text files)                        |
| Canada CanLII relevance | Medium                     | Low (US contracts)                                   |
| Best for testing        | Semantic retrieval quality | Exact span localisation quality                      |


---

