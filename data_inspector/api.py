"""Data loading and API response builders — no HTTP concerns here."""
from __future__ import annotations
import json
import pathlib

BASE              = pathlib.Path(__file__).resolve().parent.parent
EVAL_LOGS_DIR     = BASE / "logs/eval"
NEW_DATASET_DIR   = BASE / "data_inspector/data/reglab"

# Translate RegLab file-level embedder names → LBR canonical display names.
# LBR naming is used as the standard in the UI.
_ND_TO_LBR: dict[str, str] = {
    "mpnet":     "all-mpnet",
    "clerc-ft":  "clerc",
    "legal-bge": "legal-embed-bge",
}
_LBR_TO_ND: dict[str, str] = {v: k for k, v in _ND_TO_LBR.items()}

# Corpus roots for RegLab datasets that don't embed GT text in the JSON.
# housing_qa embeds full `answer` text; barexam_qa does not (corpus may be empty).
REGLAB_CORPUS_ROOTS: dict[str, pathlib.Path] = {
    "barexam_qa": BASE / "data/reglab_eval/barexam_qa/corpus",
}

DATASET_META: dict[str, dict] = {
    "legalbenchrag-mini": {
        "label":           "LegalBenchRAG",
        "subtitle":        "LegalBenchRAG-mini",
        "description":     "https://arxiv.org/pdf/2408.10343",
        "logs_dir":        EVAL_LOGS_DIR,   # model dirs are directly under logs/eval/
        "corpus_dir":      BASE / "data/legalbenchrag-mini/corpus",
        "benchmarks_dir":  BASE / "data/legalbenchrag-mini/benchmarks",
        "benchmark_names": ["contractnli", "cuad", "maud", "privacy_qa"],
        "color":           "blue",
        "format":          "lbr",
        "placeholder":     False,
    },
    "reglab": {
        "label":           "RegLab",
        "subtitle":        "Stanford",
        "description":     "https://arxiv.org/pdf/2505.03970",
        "benchmark_names": ["barexam_qa", "housing_qa"],
        "color":           "purple",
        "format":          "new_dataset",
        "placeholder":     False,
    },
}

# Caches
_trace_cache:  dict[tuple, list]          = {}
_ds_offsets:   dict[tuple, dict[str,int]] = {}
_corpus_cache: dict[tuple, str]           = {}
_gt_cache:     dict[tuple, list]          = {}


def _meta(dataset_id: str) -> dict:
    return DATASET_META.get(dataset_id, {
        "label":           dataset_id,
        "subtitle":        "",
        "description":     "",
        "corpus_dir":      BASE / f"data/{dataset_id}/corpus",
        "benchmarks_dir":  BASE / f"data/{dataset_id}/benchmarks",
        "benchmark_names": [],
        "color":           "gray",
        "format":          "lbr",
        "placeholder":     False,
    })


def _is_nd(dataset_id: str) -> bool:
    return _meta(dataset_id).get("format") == "new_dataset"


# ── dataset listing ───────────────────────────────────────────────────────────

def api_datasets() -> list[dict]:
    known = dict(DATASET_META)

    # Auto-discover LBR-format dirs under logs/eval/ with actual jsonl runs
    if EVAL_LOGS_DIR.exists():
        for d in sorted(EVAL_LOGS_DIR.iterdir()):
            if not d.is_dir() or d.name in known:
                continue
            has_runs = any(
                sub.is_dir() and any(sub.glob("*.jsonl"))
                for sub in d.iterdir()
            )
            if has_runs:
                known[d.name] = {
                    "label":           d.name,
                    "subtitle":        "",
                    "description":     "",
                    "corpus_dir":      BASE / f"data/{d.name}/corpus",
                    "benchmarks_dir":  BASE / f"data/{d.name}/benchmarks",
                    "benchmark_names": [],
                    "color":           "gray",
                    "format":          "lbr",
                    "placeholder":     False,
                }

    out = []
    for ds_id, meta in known.items():
        if meta.get("format") == "new_dataset":
            models_file = NEW_DATASET_DIR / "models.json"
            available   = (not meta.get("placeholder")) and models_file.exists()
            n_runs      = len(json.loads(models_file.read_text())) if available else 0
        else:
            logs_dir  = meta.get("logs_dir", EVAL_LOGS_DIR / ds_id)
            n_runs    = sum(
                1 for d in logs_dir.iterdir()
                if d.is_dir() and any(d.glob("*.jsonl"))
            ) if logs_dir.exists() else 0
            available = (not meta.get("placeholder")) and n_runs > 0

        out.append({
            "id":              ds_id,
            "label":           meta["label"],
            "subtitle":        meta.get("subtitle", ""),
            "description":     meta.get("description", ""),
            "benchmark_names": meta.get("benchmark_names", []),
            "color":           meta.get("color", "gray"),
            "placeholder":     meta.get("placeholder", False),
            "available":       available,
            "n_runs":          n_runs,
        })
    return out


# ── LBR format helpers ────────────────────────────────────────────────────────

def _load_lbr_trace(dataset_id: str, model: str, embedder: str) -> list[dict]:
    key = (dataset_id, model, embedder)
    if key not in _trace_cache:
        logs_dir = _meta(dataset_id).get("logs_dir", EVAL_LOGS_DIR / dataset_id)
        path = logs_dir / model / f"lbr_hier_{embedder}.jsonl"
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "query" in obj:
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
        ds_min: dict[str, int] = {}
        for row in rows:
            ds = row.get("tags", ["unknown"])[0]
            qi = row.get("query_idx", 1)
            if ds not in ds_min or qi < ds_min[ds]:
                ds_min[ds] = qi
        _trace_cache[key]  = rows
        _ds_offsets[key]   = ds_min
    return _trace_cache[key]


def _lbr_gt_tests(dataset_id: str, benchmark: str) -> list[dict]:
    key = (dataset_id, benchmark)
    if key not in _gt_cache:
        benchmarks_dir = _meta(dataset_id)["benchmarks_dir"]
        path = benchmarks_dir / f"{benchmark}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        _gt_cache[key] = data["tests"]
    return _gt_cache[key]


def _lbr_corpus_text(dataset_id: str, file: str, s: int, e: int) -> str:
    key = (dataset_id, file)
    if key not in _corpus_cache:
        corpus_dir = _meta(dataset_id)["corpus_dir"]
        p = corpus_dir / file
        try:
            _corpus_cache[key] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _corpus_cache[key] = ""
    return _corpus_cache[key][s:e]


# ── new_dataset format helpers ────────────────────────────────────────────────

def _nd_raw_index() -> dict[str, list[str]]:
    """Raw models.json: keys like 'barexam_hier', values are embedder lists."""
    path = NEW_DATASET_DIR / "models.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _nd_models() -> dict[str, list[str]]:
    """Return {"original": [embedders in LBR canonical names]} — one LLM model for RegLab."""
    raw = _nd_raw_index()
    seen: list[str] = []
    for embs in raw.values():
        for e in embs:
            display = _ND_TO_LBR.get(e, e)
            if display not in seen:
                seen.append(display)
    return {"original": seen} if seen else {}


def _nd_chunkers() -> list[str]:
    """Extract unique chunker names from models.json keys (e.g. 'hier', 'rec')."""
    chunkers: list[str] = []
    for key in _nd_raw_index():
        chunker = key.rsplit("_", 1)[-1]
        if chunker not in chunkers:
            chunkers.append(chunker)
    return sorted(chunkers)


def _nd_dataset_shorts() -> list[str]:
    """Extract unique dataset short names from models.json keys (e.g. 'barexam', 'housing')."""
    shorts: list[str] = []
    for key in _nd_raw_index():
        short = key.rsplit("_", 1)[0]
        if short not in shorts:
            shorts.append(short)
    return shorts


def _nd_benchmark_to_short(benchmark: str | None) -> str | None:
    """'barexam_qa' → 'barexam', 'housing_qa' → 'housing', None → None."""
    return benchmark.replace("_qa", "") if benchmark else None


def _load_nd_rows_for(dataset_short: str, chunker: str, embedder: str) -> list[dict]:
    # embedder may be a LBR display name — translate back to RegLab file name
    file_embedder = _LBR_TO_ND.get(embedder, embedder)
    key = ("nd", dataset_short, chunker, file_embedder)
    if key not in _trace_cache:
        path = NEW_DATASET_DIR / f"{dataset_short}_{chunker}__{file_embedder}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        _trace_cache[key] = data.get("queries", [])
    return _trace_cache[key]


def _load_nd_rows(chunker: str, embedder: str) -> list[dict]:
    """Load and merge rows for all datasets (barexam + housing) in stable order."""
    file_embedder = _LBR_TO_ND.get(embedder, embedder)
    key = ("nd_all", chunker, file_embedder)
    if key not in _trace_cache:
        rows: list[dict] = []
        for ds in sorted(_nd_dataset_shorts()):
            try:
                rows.extend(_load_nd_rows_for(ds, chunker, embedder))
            except OSError:
                pass
        _trace_cache[key] = rows
    return _trace_cache[key]


def _nd_gt_text(dataset_name: str, file_path: str, s: int, e: int) -> str:
    """Read GT text from corpus for datasets that store empty `answer` in JSON."""
    key = ("nd_corpus", dataset_name, file_path)
    if key not in _corpus_cache:
        root = REGLAB_CORPUS_ROOTS.get(dataset_name)
        if root:
            p = root / file_path
            try:
                _corpus_cache[key] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                _corpus_cache[key] = ""
        else:
            _corpus_cache[key] = ""
    return _corpus_cache[key][s:e]


# ── public API ────────────────────────────────────────────────────────────────

def api_chunkers(dataset_id: str) -> list[str]:
    if _is_nd(dataset_id):
        return _nd_chunkers()
    return ["hier"]


def api_models(dataset_id: str) -> dict[str, list[str]]:
    if _is_nd(dataset_id):
        return _nd_models()
    logs_dir = _meta(dataset_id).get("logs_dir", EVAL_LOGS_DIR / dataset_id)
    result: dict[str, list[str]] = {}
    if not logs_dir.exists():
        return result
    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        embs = sorted(
            f.stem.removeprefix("lbr_hier_")
            for f in d.glob("lbr_hier_*.jsonl")
        )
        if embs:
            result[d.name] = embs
    return result


def api_queries(dataset_id: str, model: str, embedder: str, benchmark: str | None,
                chunker: str = "hier") -> list:
    if _is_nd(dataset_id):
        return _nd_api_queries(embedder, benchmark, chunker)
    return _lbr_api_queries(dataset_id, model, embedder, benchmark)


def api_query(dataset_id: str, model: str, embedder: str, idx: int,
              chunker: str = "hier") -> dict:
    if _is_nd(dataset_id):
        return _nd_api_query(embedder, idx, chunker)
    return _lbr_api_query(dataset_id, model, embedder, idx)


# ── LBR query responses ───────────────────────────────────────────────────────

def _lbr_api_queries(dataset_id: str, model: str, embedder: str, benchmark: str | None) -> list:
    rows = _load_lbr_trace(dataset_id, model, embedder)
    out = []
    for i, e in enumerate(rows):
        tags = e.get("tags", [])
        ds = tags[0] if tags else "unknown"
        if benchmark and ds != benchmark:
            continue
        mbk  = e.get("metrics_by_k", [])
        last = mbk[-1] if mbk else {}
        out.append({
            "idx":              i,
            "dataset":          ds,
            "query":            e["query"][:200],
            "n_gt_snippets":    last.get("n_gt_snippets", 0),
            "n_gt_hit":         last.get("n_gt_hit", 0),
            "total_gt_chars":   e.get("total_gt_chars", 0),
            "char_recall_max":  round(last.get("char_recall", 0), 4),
            "chunk_recall_max": round(last.get("chunk_recall", 0), 4),
            "k_values":         [m["k"] for m in mbk],
            "char_recalls":     [round(m.get("char_recall", 0), 4) for m in mbk],
        })
    return out


def _lbr_api_query(dataset_id: str, model: str, embedder: str, idx: int) -> dict:
    rows = _load_lbr_trace(dataset_id, model, embedder)
    e    = rows[idx]
    tags = e.get("tags", [])
    ds   = tags[0] if tags else "unknown"

    query_idx = e.get("query_idx", 1)
    ds_offset = _ds_offsets.get((dataset_id, model, embedder), {}).get(ds, 1)
    test_idx  = query_idx - ds_offset
    tests     = _lbr_gt_tests(dataset_id, ds)
    test      = tests[test_idx] if 0 <= test_idx < len(tests) else {}
    snippets  = test.get("snippets", [])

    mbk  = e.get("metrics_by_k", [])
    last = mbk[-1] if mbk else {}
    hit_by_id: dict[str, dict] = {
        c["chunk_id"]: c for c in last.get("chunk_hits", [])
    }

    retrieved = []
    for chunk in e.get("retrieved_all", []):
        cid   = chunk["chunk_id"]
        hinfo = hit_by_id.get(cid, {})
        text  = _lbr_corpus_text(dataset_id, chunk["file"], chunk["char_start"], chunk["char_end"])
        retrieved.append({
            "rank":        chunk["rank"],
            "file":        chunk["file"],
            "char_start":  chunk["char_start"],
            "char_end":    chunk["char_end"],
            "score":       chunk["score"],
            "text":        text,
            "is_hit":      hinfo.get("is_chunk_hit", False),
            "gt_overlaps": hinfo.get("gt_overlaps", []),
        })

    return {
        "dataset":        ds,
        "query":          e["query"],
        "original_query": test.get("query", ""),
        "ground_truth":   snippets,
        "total_gt_chars": e.get("total_gt_chars", 0),
        "retrieved":      retrieved,
        "metrics_by_k":   mbk,
    }


# ── new_dataset query responses ───────────────────────────────────────────────

def _nd_api_queries(embedder: str, benchmark: str | None, chunker: str) -> list:
    rows = _load_nd_rows(chunker, embedder)
    out = []
    for i, q in enumerate(rows):
        ds = q.get("dataset", "unknown")
        if benchmark and ds != benchmark:
            continue
        mbk  = q.get("metrics_by_k", [])
        last = mbk[-1] if mbk else {}
        out.append({
            "idx":              i,
            "dataset":          ds,
            "query":            q["query"][:200],
            "n_gt_snippets":    q.get("n_gt_snippets", 0),
            "n_gt_hit":         q.get("n_gt_hit", 0),
            "total_gt_chars":   sum(
                max(0, s["span"][1] - s["span"][0])
                for s in q.get("ground_truth", [])
            ),
            "char_recall_max":  round(q.get("char_recall_max", last.get("char_recall", 0)), 4),
            "chunk_recall_max": round(last.get("chunk_recall", 0), 4),
            "k_values":         [m["k"] for m in mbk],
            "char_recalls":     [round(m.get("char_recall", 0), 4) for m in mbk],
        })
    return out


def _nd_api_query(embedder: str, idx: int, chunker: str) -> dict:
    rows = _load_nd_rows(chunker, embedder)
    q    = rows[idx]
    ds   = q.get("dataset", "unknown")

    # Build file → retrieved chunks lookup for GT text fallback
    retrieved_by_file: dict[str, list[dict]] = {}
    for chunk in q.get("retrieved", []):
        retrieved_by_file.setdefault(chunk["file"], []).append(chunk)

    def _gt_text_from_retrieved(file_path: str, s: int, e: int) -> str:
        """Slice GT text out of the retrieved chunk whose span contains [s, e)."""
        for chunk in sorted(retrieved_by_file.get(file_path, []),
                            key=lambda c: c["char_start"]):
            cs, ce = chunk["char_start"], chunk["char_end"]
            if cs <= s and ce >= e:
                return chunk["text"][s - cs: e - cs]
        # Partial overlap fallback: stitch together overlapping chunks
        parts, covered = [], s
        for chunk in sorted(retrieved_by_file.get(file_path, []),
                            key=lambda c: c["char_start"]):
            cs, ce = chunk["char_start"], chunk["char_end"]
            if ce <= covered or cs >= e:
                continue
            take_s = max(cs, covered)
            take_e = min(ce, e)
            parts.append(chunk["text"][take_s - cs: take_e - cs])
            covered = take_e
            if covered >= e:
                break
        return "".join(parts)

    # Populate answer text where JSON stores empty string (e.g. barexam_qa)
    ground_truth = []
    for snip in q.get("ground_truth", []):
        answer = (snip.get("answer")
                  or _nd_gt_text(ds, snip["file_path"], snip["span"][0], snip["span"][1])
                  or _gt_text_from_retrieved(snip["file_path"], snip["span"][0], snip["span"][1]))
        ground_truth.append({**snip, "answer": answer})

    total_gt_chars = sum(max(0, s["span"][1] - s["span"][0]) for s in ground_truth)

    return {
        "dataset":        ds,
        "query":          q["query"],
        "original_query": q.get("original_query", q["query"]),
        "ground_truth":   ground_truth,
        "total_gt_chars": total_gt_chars,
        "retrieved":      q.get("retrieved", []),
        "metrics_by_k":   q.get("metrics_by_k", []),
    }
