"""Precompute the Home page's benchmark questions as instant demos.

For every question in legalrag/prompts/explore_examples.yaml, run the full
exploration (same engine as the API) and save its event stream + trace to
data/demos/<query_id>.json.  The UI replays these instead of running live.

Needs OpenSearch (index canlegal-qwen3) and the LLM server from .env (lab vLLM).
Run from the repo root:

    python scripts/build_demos.py                 # build the missing ones
    python scripts/build_demos.py --force         # rebuild all (e.g. after a prompt change)
    python scripts/build_demos.py --ids 684 56    # only these

A run with a failed view (view_error / error event) is not saved unless
--keep-partial, so a demo never shows a broken view.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legalrag.core.config import settings  # noqa: E402
from legalrag.explore import demos  # noqa: E402
from legalrag.explore.engine import ExplorationEngine  # noqa: E402
from legalrag.utils.logging import configure_logging  # noqa: E402


async def build_one(engine: ExplorationEngine, query_id: int, kind: str) -> tuple[dict, list[str]]:
    question = demos.dataset_queries()[query_id]["query_text"]
    profile, scope = demos.demo_profile(query_id, kind)
    t0 = time.perf_counter()
    events = [ev async for ev in engine.explore(question, profile, scope)]
    problems = [
        f"{ev.get('view_id', 'run')}: {ev['message']}"
        for ev in events
        if ev["event"] in ("view_error", "error")
    ]
    trace = dict(engine.traces.get(events[0]["trace_id"]))
    # Raw pydantic-ai messages duplicate the prompts/responses already in each call record
    trace["llm_calls"] = [{k: v for k, v in c.items() if k != "messages"} for c in trace["llm_calls"]]
    trace.pop("file", None)
    bundle = {
        "query_id": query_id,
        "kind": kind,
        "question": question,
        "scope": scope,
        "province": profile.province,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.explore.llm_model or settings.llm.model,
        "total_s": round(time.perf_counter() - t0, 1),
        "events": events,
        "trace": trace,
    }
    return bundle, problems


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="*", help="only these query ids")
    ap.add_argument("--force", action="store_true", help="rebuild demos that already exist")
    ap.add_argument("--keep-partial", action="store_true", help="save runs that had a failed view")
    ap.add_argument("--view-timeout", type=float, help="seconds per extra view (default: explore_policy.yaml); "
                    "raise it on a slow local LLM, where parallel calls queue")
    args = ap.parse_args()

    configure_logging(level="INFO", log_file=Path("logs/build_demos.log"))
    todo = [(q, k) for q, k in demos.all_demo_ids() if not args.ids or q in args.ids]
    if not args.force:
        todo = [(q, k) for q, k in todo if not demos.demo_path(q).exists()]
    print(f"{len(todo)} demo(s) to build with {settings.llm.model} at {settings.llm.base_url}")
    if not todo:
        return 0

    engine = ExplorationEngine()
    if args.view_timeout:
        engine._extra_view_timeout = args.view_timeout
    demos.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (query_id, kind) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {query_id} ({kind}) …", flush=True)
        bundle, problems = await build_one(engine, query_id, kind)
        views = sum(1 for ev in bundle["events"] if ev["event"] == "view")
        if problems and not args.keep_partial:
            failed.append(query_id)
            print(f"    NOT saved ({bundle['total_s']} s): " + "; ".join(problems))
            continue
        path = demos.demo_path(query_id)
        path.write_text(json.dumps(bundle, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"    saved {path} · {views} views · {bundle['total_s']} s"
              + (f" · problems: {'; '.join(problems)}" if problems else ""))
    if failed:
        print(f"Failed: {' '.join(map(str, failed))} — rerun with --ids {' '.join(map(str, failed))}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
