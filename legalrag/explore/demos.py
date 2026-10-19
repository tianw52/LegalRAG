"""Benchmark example questions for the Home page and their precomputed demos.

Which questions a user sees depends only on their location
(``legalrag/prompts/explore_examples.yaml``): two local questions then two
federal ones, or one Ontario + one BC + two federal when the location has no
province (all of Canada, outside Canada, prefer not to say) or no questions.

Each listed question is precomputed by ``scripts/build_demos.py`` into
``data/demos/<query_id>.json``: the full exploration event stream plus its
trace, so the UI replays it instantly instead of running the pipeline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from legalrag.explore.profile import UserProfile, filter_province, profile_options
from legalrag.prompts.loader import load_prompt

QUERIES_PATH = Path("data/CanLegalRAGBench/queries.json")
DEMO_DIR = Path("data/demos")


@lru_cache(maxsize=1)
def dataset_queries() -> dict[int, dict]:
    if not QUERIES_PATH.exists():
        return {}
    with QUERIES_PATH.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {q["query_id"]: q for q in rows}


def _config() -> dict:
    return load_prompt("explore_examples")


def _province_codes() -> dict[str, str]:
    """Province name (as in queries.json) → code."""
    return {p["label"]: p["code"] for p in profile_options()["provinces"] if not p.get("all_of_canada")}


def example_ids(province: str) -> list[tuple[int, str]]:
    """[(query_id, kind)] for a user's location code; kind is "local" or "federal"."""
    cfg = _config()
    local = cfg["provinces"].get(filter_province(province) or "") or cfg["default"]
    return [(qid, "local") for qid in local] + [(qid, "federal") for qid in cfg["federal"]]


def examples(province: str) -> list[dict[str, Any]]:
    queries = dataset_queries()
    codes = _province_codes()
    out = []
    for qid, kind in example_ids(province):
        q = queries.get(qid)
        if q is None:
            continue
        out.append({
            "query_id": qid,
            "text": q["query_text"],
            "kind": kind,
            "province": codes.get(q.get("province", ""), ""),
            "demo": demo_path(qid).exists(),
        })
    return out


def all_demo_ids() -> list[tuple[int, str]]:
    """Every configured question once, with its kind."""
    cfg = _config()
    ids = [(qid, "federal") for qid in cfg["federal"]]
    ids += [(qid, "local") for qids in cfg["provinces"].values() for qid in qids]
    seen: set[int] = set()
    return [(q, k) for q, k in ids if not (q in seen or seen.add(q))]


def demo_profile(query_id: int, kind: str) -> tuple[UserProfile, str]:
    """(profile, scope) a demo is built with: the question's province, or all of Canada for federal ones."""
    q = dataset_queries()[query_id]
    cfg = _config()["demo_profile"]
    province = "all" if kind == "federal" else _province_codes()[q["province"]]
    profile = UserProfile(
        user_id="demo", display_name="Demo", province=province,
        familiarity=cfg["familiarity"], goal=cfg["goal"],
    )
    return profile, "all" if kind == "federal" else "profile"


def demo_path(query_id: int) -> Path:
    return DEMO_DIR / f"{query_id}.json"


def load_demo(query_id: int) -> dict[str, Any] | None:
    path = demo_path(query_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("trace", None)  # served separately by demo_trace()
    return data


@lru_cache(maxsize=1)
def _trace_index() -> dict[str, Path]:
    index = {}
    for path in DEMO_DIR.glob("*.json"):
        trace_id = json.loads(path.read_text(encoding="utf-8")).get("trace", {}).get("trace_id")
        if trace_id:
            index[trace_id] = path
    return index


def demo_trace(trace_id: str) -> dict[str, Any] | None:
    """The trace saved with a demo, for "Under the hood" on a replayed demo."""
    path = _trace_index().get(trace_id)
    if path is None:  # demos built after the index was cached
        _trace_index.cache_clear()
        path = _trace_index().get(trace_id)
    return json.loads(path.read_text(encoding="utf-8"))["trace"] if path else None
