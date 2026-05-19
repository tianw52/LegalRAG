# New Dataset evaluation (barexam_qa / RegLab)

Character-level **CharRecall@K** and **CharPrecision@K** evaluation for large RegLab benchmarks (e.g. `reglab/barexam_qa`).

## Install

Copy the evaluator into the main LegalRAG tree (replaces the stock file):

```bash
cp New_Dataset/eval_precision_recall.py evaluation/LegalBenchRAG/eval_precision_recall.py
```

Requires the rest of [LegalRAG](https://github.com/tianw52/LegalRAG) (`legalrag`, `evaluation.LegalBenchRAG.loader`, OpenSearch index already built).

## Run

From the repo root, with benchmark JSON under `$DATA_DIR/benchmarks/` and a populated index:

```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
  --data-dir /path/to/barexam_qa \
  --benchmarks barexam_qa \
  --index-name YOUR_INDEX \
  --ks 2 4 6 10 15 20 40 60
```

Or via the RegLab wrapper (same K defaults as Tian unless `--ks` is passed):

```bash
python -m evaluation.reglab.eval_recall \
  --data-dir /path/to/barexam_qa \
  --benchmarks barexam_qa \
  --index-name YOUR_INDEX \
  --ks 2 4 6 10 15 20 40 60
```

## Changes vs upstream

- Optional `gt_text` in evaluation traces when `corpus/` `.txt` files are available
- `chunk_text` included in per-query traces
- Jurisdiction / `court_filter` handling for state-scoped corpora (e.g. housing_qa)
