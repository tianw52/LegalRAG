Copy over `CanLegalRAGBench/evaluate_retrieval` (same paths), build the index with `scripts/run_indexing.py --config <config>`, then:

    python sweep.py --name gemma_4b --config configs/experiments/local_gemma_recursive_8192.yaml --reranker qwen4b --pool 40
    python compare_sweeps.py gemma_4b gemma_06b plain

- `sweep.py`: runs one embedder + reranker setup on the 532 queries, prints nDCG@10, MRR, DocR@10, Hit@10. `--reranker`: none, qwen06b, qwen4b, qwen8b. `SWEEP_LIMIT=20` runs 20 queries only.
- `compare_sweeps.py`: paired bootstrap confidence interval between two runs.
- `benchmark_rag/components/rerankers/qwen3.py`: Qwen3-Reranker.
- `configs/`: Gemma (best) and Qwen3-0.6B (start point). Put them in `configs/experiments/`.
