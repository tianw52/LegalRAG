import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.pipeline.rag_pipeline import RAGPipeline
from benchmark_rag.registry import build_from_component_config

RR = {
    "qwen06b": {"type": "rerankers.qwen3.Qwen3Reranker", "model_name": "Qwen/Qwen3-Reranker-0.6B", "batch_size": 4},
    "qwen4b": {"type": "rerankers.qwen3.Qwen3Reranker", "model_name": "Qwen/Qwen3-Reranker-4B", "batch_size": 2},
    "qwen8b": {"type": "rerankers.qwen3.Qwen3Reranker", "model_name": "Qwen/Qwen3-Reranker-8B", "batch_size": 1, "quantization": "8bit"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--reranker", default="none")
ap.add_argument("--pool", type=int, default=40)
a = ap.parse_args()

OUT = f"runs/sweep_{a.name}"
os.makedirs(OUT, exist_ok=True)
cfg = ExperimentConfig.from_yaml(a.config)
cfg.reranker = None
pipe = RAGPipeline.from_config(cfg)
reranker = build_from_component_config(dict(RR[a.reranker])) if a.reranker != "none" else None
queries = [json.loads(l) for l in open("data/queries.json", encoding="utf-8") if l.strip()][: int(os.environ.get("SWEEP_LIMIT", 10**9))]
pf = OUT + "/progress.jsonl"
done = {}
if os.path.exists(pf):
    for l in open(pf, encoding="utf-8"):
        r = json.loads(l)
        done[r["query_id"]] = r
print(f"{a.name}: queries {len(queries)} | already done {len(done)}", flush=True)
raw = f"runs/raw_sweep_{a.name}.log"
t0 = time.time()
with open(pf, "a", encoding="utf-8") as fh, open(raw, "a", encoding="utf-8") as lg:
    for q in queries:
        if q["query_id"] in done:
            continue
        text = f"I am in {q['province']}. {q['query_text']}" if q.get("province") else q["query_text"]
        cands = pipe.retriever.retrieve(pipe.embedder.embed([text])[0], k=a.pool)
        rec = {"query_id": q["query_id"], "gold_citations": q["ground_truth_citations"], "pool": len(cands),
               "dense": [(c.doc_id, c.chunk_idx, c.score) for c in cands[:10]]}
        if reranker is not None:
            ranked = reranker.rerank(text, list(cands))
            seen, best = set(), []
            for c in ranked:
                if c.doc_id not in seen:
                    seen.add(c.doc_id)
                    best.append(c)
            rec["all"] = [(c.doc_id, c.chunk_idx, c.score) for c in ranked]
            rec["plain"] = [(c.doc_id, c.chunk_idx, c.score) for c in ranked[:10]]
            rec["dedup"] = [(c.doc_id, c.chunk_idx, c.score) for c in best[:10]]
        done[q["query_id"]] = rec
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n = len(done)
        lg.write(f"Querying: {100 * n // len(queries):3d}%|##| {n}/{len(queries)} [{(time.time() - t0) / 60:.0f}m]\n")
        lg.flush()
        print(f"{n}/{len(queries)}", flush=True)

def dcg(items, rel):
    return sum(1 / math.log2(i + 1) for i, d in enumerate(items, 1) if d in rel)

def score(variant):
    out = {k: [] for k in ("ndcg", "mrr", "docr", "hit", "p", "uniq")}
    for q in queries:
        r = done[q["query_id"]]
        rel = set(r["gold_citations"])
        top = list(dict.fromkeys(x[0] for x in r[variant]))
        hits = [d for d in top if d in rel]
        ideal = dcg(list(rel)[:10], rel)
        out["ndcg"].append(dcg(top, rel) / ideal if ideal else 0)
        out["mrr"].append(next((1 / i for i, d in enumerate(top, 1) if d in rel), 0))
        out["docr"].append(len(hits) / len(rel))
        out["hit"].append(float(bool(hits)))
        out["p"].append(len(hits) / len(top) if top else 0)
        out["uniq"].append(len(top))
    return {k: float(np.mean(v)) for k, v in out.items()}, out

res = {}
for v in ("dense", "plain", "dedup"):
    if v not in next(iter(done.values())):
        continue
    m, per = score(v)
    res[v] = m
    np.save(f"{OUT}/per_query_{v}.npy", np.array([per["ndcg"], per["mrr"]]))
    print(f"{v:6} nDCG@10 {m['ndcg']:.4f} MRR {m['mrr']:.4f} DocR@10 {m['docr']:.4f} Hit@10 {m['hit']:.4f} unique docs {m['uniq']:.2f}")
json.dump({"results": res, "n_queries": len(done), "reranker": a.reranker, "pool": a.pool, "config": a.config,
           "pool_median": float(np.median([r["pool"] for r in done.values()])),
           "minutes_this_session": (time.time() - t0) / 60}, open(OUT + "/metrics.json", "w"), indent=1)
open(OUT + "/DONE", "w").write("ok")
