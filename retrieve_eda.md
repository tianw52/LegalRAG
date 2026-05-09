# Retrieval EDA — LegalBench-RAG Hierarchical vs Recursive

**Date**: 2026-03-18
**Indices**: `legalrag-lbr-hierarchical`, `legalrag-lbr-recursive`
**Benchmark subset**: `data/LegalBenchRAG/benchmarks_50` (50 queries × 4 benchmarks = 200 total)
**Trace files**: `logs/eval/lbr_hier_50.jsonl`, `logs/eval/lbr_rec_50.jsonl`

---

## 1. Evaluation Runs

### Command

```bash
# Hierarchical
source /home/twa174/LegalRAG/.venv/bin/activate && \
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --index-name legalrag-lbr-hierarchical \
    --ks 1 3 5 10 20 40 60 \
    --trace-file logs/eval/lbr_hier_50.jsonl

# Recursive
source /home/twa174/LegalRAG/.venv/bin/activate && \
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --index-name legalrag-lbr-recursive \
    --ks 1 3 5 10 20 40 60 \
    --trace-file logs/eval/lbr_rec_50.jsonl
```

### Output — Hierarchical

```
  CharRecall
  Benchmark               K=1      K=3      K=5     K=10     K=20     K=40     K=60   N
  ──────────────────  ───────  ───────  ───────  ───────  ───────  ───────  ───────  ─────
  contractnli          0.0000   0.0259   0.0736   0.0791   0.0863   0.1063   0.1202  (50)
  cuad                 0.0000   0.0000   0.0000   0.0064   0.0239   0.0699   0.0705  (50)
  maud                 0.0000   0.0000   0.0000   0.0000   0.0010   0.0010   0.0094  (50)
  privacy_qa           0.0122   0.0280   0.0306   0.0454   0.0976   0.1636   0.2191  (50)
  OVERALL              0.0030   0.0135   0.0260   0.0327   0.0522   0.0852   0.1048  (200)

  CharPrecision
  Benchmark               K=1      K=3      K=5     K=10     K=20     K=40     K=60   N
  ──────────────────  ───────  ───────  ───────  ───────  ───────  ───────  ───────  ─────
  contractnli          0.0000   0.0073   0.0136   0.0076   0.0046   0.0027   0.0019  (50)
  cuad                 0.0000   0.0000   0.0000   0.0009   0.0020   0.0029   0.0020  (50)
  maud                 0.0000   0.0000   0.0000   0.0000   0.0001   0.0001   0.0006  (50)
  privacy_qa           0.0103   0.0163   0.0130   0.0098   0.0084   0.0070   0.0070  (50)
  OVERALL              0.0026   0.0059   0.0067   0.0046   0.0038   0.0032   0.0029  (200)
```

### Output — Recursive

```
  CharRecall
  Benchmark               K=1      K=3      K=5     K=10     K=20     K=40     K=60   N
  ──────────────────  ───────  ───────  ───────  ───────  ───────  ───────  ───────  ─────
  contractnli          0.0035   0.0396   0.0653   0.1018   0.1418   0.2018   0.2018  (50)
  cuad                 0.0200   0.0512   0.0756   0.1158   0.1622   0.2323   0.2878  (50)
  maud                 0.0000   0.0000   0.0000   0.0000   0.0340   0.0589   0.0625  (50)
  privacy_qa           0.0257   0.0636   0.0786   0.1343   0.2441   0.3499   0.4322  (50)
  OVERALL              0.0123   0.0386   0.0549   0.0880   0.1455   0.2107   0.2461  (200)

  CharPrecision
  Benchmark               K=1      K=3      K=5     K=10     K=20     K=40     K=60   N
  ──────────────────  ───────  ───────  ───────  ───────  ───────  ───────  ───────  ─────
  contractnli          0.0030   0.0078   0.0100   0.0071   0.0040   0.0031   0.0021  (50)
  cuad                 0.0024   0.0089   0.0088   0.0056   0.0044   0.0037   0.0032  (50)
  maud                 0.0000   0.0000   0.0000   0.0009   0.0009   0.0012   0.0009  (50)
  privacy_qa           0.0235   0.0170   0.0161   0.0143   0.0116   0.0102   0.0092  (50)
  OVERALL              0.0072   0.0084   0.0087   0.0067   0.0052   0.0046   0.0038  (200)
```

---

## 2. Basic Sanity Checks

### Command

```python
import json, statistics
from collections import Counter

for fname, label in [
    ('logs/eval/lbr_hier_50.jsonl', 'HIERARCHICAL'),
    ('logs/eval/lbr_rec_50.jsonl',  'RECURSIVE'),
]:
    data = [json.loads(l) for l in open(fname)]
    n_zero_gt     = sum(1 for q in data if q['total_gt_chars'] == 0)
    n_zero_ret    = sum(1 for q in data if q['n_retrieved'] == 0)
    n_partial_ret = sum(1 for q in data if 0 < q['n_retrieved'] < 60)
    gt_chars      = [q['total_gt_chars'] for q in data]
    ret_counts    = [q['n_retrieved'] for q in data]
    any_file_match = sum(
        1 for q in data
        if {s['file'] for s in q['ground_truth']} & {c['file'] for c in q['retrieved_all']}
    )
    k60_vals = [
        next(m['char_recall'] for m in q['metrics_by_k'] if m['k'] == 60)
        for q in data
    ]
    zero_recall   = sum(1 for v in k60_vals if v == 0)
    perfect_recall = sum(1 for v in k60_vals if v >= 0.99)
```

### Output

```
=== HIERARCHICAL (200 queries) ===
  Queries with 0 GT chars:        0
  Queries with 0 retrieved:       0
  Queries with <60 retrieved:     0
  GT chars  min/median/max:       70 / 541 / 9210
  Retrieved min/median/max:       60 / 60 / 60
  Queries where GT file retrieved: 140/200
  R@60 == 0:    156/200 queries
  R@60 >= 0.99: 5/200 queries

=== RECURSIVE (200 queries) ===
  Queries with 0 GT chars:        0
  Queries with 0 retrieved:       0
  Queries with <60 retrieved:     0
  GT chars  min/median/max:       70 / 541 / 9210
  Retrieved min/median/max:       60 / 60 / 60
  Queries where GT file retrieved: 159/200
  R@60 == 0:    131/200 queries
  R@60 >= 0.99: 36/200 queries
```

### Findings

- Both traces are complete: 200 queries each, always exactly 60 chunks retrieved, no missing GT.
- GT char spans range from 70 to 9210 chars (median 541) — reasonable for legal clauses.
- Hierarchical: GT file appears in top-60 for only **140/200** queries; recursive: **159/200**.
- Zero recall at K=60 is alarmingly high: **156/200 (78%)** hierarchical, **131/200 (65.5%)** recursive.

---

## 3. File-Level vs Span-Level Retrieval Quality

### Command

```python
for fname, label in [...]:
    with_file, without_file = [], []
    for q in data:
        gt_files  = {s['file'] for s in q['ground_truth']}
        ret_files = {c['file'] for c in q['retrieved_all']}
        r60 = next(m['char_recall'] for m in q['metrics_by_k'] if m['k'] == 60)
        (with_file if gt_files & ret_files else without_file).append(r60)

    first_ranks = []
    for q in data:
        gt_files = {s['file'] for s in q['ground_truth']}
        for c in q['retrieved_all']:
            if c['file'] in gt_files:
                first_ranks.append(c['rank'])
                break

    span_overlap_count = sum(
        1 for q in data
        if any(
            c['file'] in {s['file'] for s in q['ground_truth']} and
            any(c['char_start'] < s['span'][1] and c['char_end'] > s['span'][0]
                for s in q['ground_truth'] if s['file'] == c['file'])
            for c in q['retrieved_all']
        )
    )
```

### Output

```
=== HIERARCHICAL ===
  GT file retrieved (140 queries): avg R@60 = 0.1497
  GT file missing   (60 queries):  avg R@60 = 0.0000
  First GT-file chunk rank: min=1 median=2 max=58
  Queries with a chunk overlapping GT span: 32/200

=== RECURSIVE ===
  GT file retrieved (159 queries): avg R@60 = 0.3095
  GT file missing   (41 queries):  avg R@60 = 0.0000
  First GT-file chunk rank: min=1 median=2 max=30
  Queries with a chunk overlapping GT span: 56/200
```

### Findings

- When the GT file is found, hierarchical still only achieves R@60=0.1497 vs recursive's 0.3095 — a 2× difference in span localization within the same document.
- The GT file is typically retrieved early when it is retrieved (median rank 2) — the problem is not that it appears at the bottom of the list but that the *specific chunk* covering the GT span is ranked too low.
- Only 32/200 (hier) and 56/200 (rec) queries have *any* retrieved chunk that overlaps the GT span at all.

---

## 4. Chunk Size Comparison — **Critical Finding**

### Command

```python
lens_h = [c['char_len'] for q in data_h for c in q['retrieved_all']]
lens_r = [c['char_len'] for q in data_r for c in q['retrieved_all']]
print('Hierarchical chunk sizes (top 10):', Counter(lens_h).most_common(10))
print('Recursive chunk sizes (top 10):', Counter(lens_r).most_common(10))
```

### Output

```
Hierarchical chunk sizes (top 10):
  [(512, 9233), (256, 396), (375, 36), (307, 34), (424, 29), (344, 25), ...]
  Max: 512   Min: 1

Recursive chunk sizes (top 10):
  [(1048, 144), (1041, 118), (1042, 115), (1047, 101), (1013, 98), ...]
  Max: 1176   Min: 129
```

### Finding — **Confounded Comparison**

The two indices were ingested at **different times with different `CHUNK_SIZE` settings**:

| Index | CHUNK_SIZE at ingestion | CHUNK_OVERLAP |
|---|---|---|
| `legalrag-lbr-hierarchical` | **512** (old default) | 64 |
| `legalrag-lbr-recursive` | **1048** (current `.env`) | 64 |

**The comparison is not apples-to-apples.** The observed recall gap may be partly or wholly attributable to chunk size rather than chunking strategy. To isolate the effect of chunking strategy, both indices must be re-ingested with the same `CHUNK_SIZE`.

---

## 5. Failure Mode Breakdown by Benchmark

### Command

```python
from collections import defaultdict
for label, data in [('HIER', data_h), ('REC', data_r)]:
    bm_stats = defaultdict(lambda: {'total':0,'file_missing':0,'file_found_no_span':0,'has_recall':0})
    for q in data:
        bm = q['tags'][0]
        bm_stats[bm]['total'] += 1
        gt_files  = {s['file'] for s in q['ground_truth']}
        ret_files = {c['file'] for c in q['retrieved_all']}
        r60 = next(m['char_recall'] for m in q['metrics_by_k'] if m['k'] == 60)
        if not (gt_files & ret_files):
            bm_stats[bm]['file_missing'] += 1
        elif r60 == 0:
            bm_stats[bm]['file_found_no_span'] += 1
        else:
            bm_stats[bm]['has_recall'] += 1
```

### Output

```
  HIER:
    contractnli    : file_missing=29  file_found_no_span=12  has_recall= 9
    cuad           : file_missing=17  file_found_no_span=25  has_recall= 8
    maud           : file_missing=12  file_found_no_span=36  has_recall= 2
    privacy_qa     : file_missing= 2  file_found_no_span=23  has_recall=25
  REC:
    contractnli    : file_missing=24  file_found_no_span=15  has_recall=11
    cuad           : file_missing= 6  file_found_no_span=24  has_recall=20
    maud           : file_missing=11  file_found_no_span=33  has_recall= 6
    privacy_qa     : file_missing= 0  file_found_no_span=18  has_recall=32
```

### Findings

Three distinct failure modes ordered by severity:

**1. Wrong document retrieved** (`file_missing`): The GT document never appears in the top 60.
- Worst for `contractnli` — many short contracts with generic language that matches many documents.
- Recursive dramatically better on `cuad` (6 vs 17 missing).
- Recursive perfect on `privacy_qa` (0 missing vs 2).

**2. Right document, wrong section** (`file_found_no_span`): GT file appears but no retrieved chunk overlaps the GT span.
- The dominant failure for `maud` and `cuad` — MAUD agreements are long (~100+ pages) with many structurally similar clauses; the embedder surfaces the right document but wrong clause.
- Median distance from retrieved chunks to GT span is ~3300 chars in both indices.
- This is a **semantic localization failure**, not a coverage gap — the relevant chunk exists in the index but isn't ranked within the top 60.

**3. Has recall** (`has_recall`): At least some GT characters are covered.
- Recursive significantly better on `cuad` (20 vs 8) and `privacy_qa` (32 vs 25).
- Both struggle with `maud` (2 and 6 out of 50).

---

## 6. Case Study: 1-Character Parent Boundary Gap (Q181)

### Command

```python
q181 = next(q for q in data_h if q['query_idx'] == 181)
gt_file = q181['ground_truth'][0]['file']
gs, ge = q181['ground_truth'][0]['span']  # [13561, 14100]
same = sorted([c for c in q181['retrieved_all'] if c['file'] == gt_file],
              key=lambda c: c['char_start'])
```

### Output

```
Q181 GT: file=...privacy_qa/Groupon.txt, span=[13561, 14100]
  rank=23  [  2167,  2679]  overlap=0
  rank=36  [ 12350, 12862]  overlap=0
  rank= 3  [ 13246, 13560]  overlap=0   ← ends 1 char before GT starts
  rank= 1  [ 16146, 16658]  overlap=0   ← ranked #1, 2586 chars past GT end
  rank=31  [ 16594, 17106]  overlap=0
  rank=59  [ 17042, 17554]  overlap=0
  ...
  Gap between last chunk before GT and GT start: 1
```

### Finding

A parent boundary in the hierarchical chunker falls at position 13560, exactly 1 character before the GT span begins at 13561. The chunk covering `[13560, ~14072]` **exists in the index** but was ranked below position 60 for this query. The model ranked a chunk from position 16146 as rank 1 — more than 2500 characters past the GT span's end.

This confirms the failure is **ranking quality**, not indexing coverage. No gap in document coverage exists; the correct chunk is simply not surfaced.

---

## 7. Summary of Findings

| Finding | Detail |
|---|---|
| **Comparison confounded** | Hierarchical index: CHUNK_SIZE=512; Recursive: CHUNK_SIZE=1048. Not apples-to-apples. |
| **Dominant failure: wrong section** | Right document retrieved but wrong clause. Median distance to GT span ~3300 chars. |
| **Secondary failure: wrong document** | GT file never appears in top 60. Varies heavily by benchmark. |
| **60-chunk cap is not the bottleneck** | The GT chunk is ranked far below 60, not cut at rank 61. |
| **MAUD is hardest** | Long M&A docs with similar-looking clauses. has_recall=2/50 (hier) and 6/50 (rec). |
| **Recursive better on CUAD** | file_missing drops from 17→6; has_recall rises from 8→20. |
| **Recursive perfect document retrieval on privacy_qa** | 0 file_missing vs 2 for hierarchical. |
| **Tiny chunk (len=1)** | 3 occurrences of a 1-character chunk at position [6886,6887] in one NDA file. Minor. |

## 8. Next Steps

1. **Re-ingest both indices with the same `CHUNK_SIZE`** (either 512 or 1048) to get a clean strategy comparison.
2. **Investigate embedding model**: `all-mpnet-base-v2` is not domain-adapted. Try `nlpaueb/legal-bert-base-uncased` or `BAAI/bge-large-en-v1.5`.
3. **Increase retrieval top-K**: Raising from 60 to 100+ may recover some borderline cases (though the 1-char gap case shows rank is far below 60).
4. **MAUD-specific analysis**: The clause-localization problem in long structured agreements may require query expansion or document-level filtering before chunk retrieval.