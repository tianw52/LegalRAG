#!/usr/bin/env python3
"""
Run one exploration against the API and summarise the streamed events.

Dev/debug tool for the exploration app: exercises the full stack (formulate →
retrieve → views) without the web UI, prints a timeline of events, and
optionally saves the raw NDJSON stream for inspection.

Usage
-----
    # API must be running: uvicorn legalrag.api.app:app --port 8000
    python scripts/explore_cli.py "Can my landlord evict me for a pet in BC?"
    python scripts/explore_cli.py --query-id 792                 # a CanLegalRAGBench query
    python scripts/explore_cli.py --query-id 792 --user-id dev-tester --scope all \
        --out logs/explore_792.ndjson
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

QUERIES_PATH = Path("data/CanLegalRAGBench/queries.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("question", nargs="?", help="Question text")
    src.add_argument("--query-id", type=int, help="Use a CanLegalRAGBench query by id")
    p.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--user-id", default=None, help="Profile to explore as (affects jurisdiction + prompts)")
    p.add_argument("--scope", choices=["profile", "all"], default="profile")
    p.add_argument("--out", type=Path, default=None, help="Save raw NDJSON events here")
    p.add_argument("--timeout", type=int, default=900, help="Seconds to wait for the stream")
    return p.parse_args(argv)


def load_benchmark_query(query_id: int) -> dict:
    with QUERIES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            if q["query_id"] == query_id:
                return q
    raise SystemExit(f"query_id {query_id} not found in {QUERIES_PATH}")


def stream_events(api: str, question: str, user_id: str | None, scope: str, timeout: int):
    body = json.dumps({"question": question, "user_id": user_id, "scope": scope}).encode()
    req = urllib.request.Request(
        f"{api}/api/explore", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if line.strip():
                yield json.loads(line)


def describe(ev: dict) -> str:
    """One-line human summary of an event."""
    kind = ev["event"]
    if kind == "status":
        return ev["message"]
    if kind == "query":
        juris = "all of Canada" if not ev["jurisdictions"] else "+".join(ev["jurisdictions"])
        relaxed = " (relaxed: too few local results)" if ev["relaxed"] else ""
        return f"rewrite={ev['reformulated'][:90]!r} | sources: {juris}{relaxed}"
    if kind == "evidence":
        docs = {e.get("citation") for e in ev["items"] if e["kind"] == "passage"}
        return f"{len(ev['items']) - 1} passages from {len(docs)} documents"
    if kind == "view":
        v = ev["view"]
        audit = v.get("audit") or {}
        backed = f" | backed={audit.get('backed_items')} unbacked={audit.get('unbacked_items')}" if audit else ""
        return f"{v['view_id']} ({v['kind']}, {v['source']}): {v['title']}{backed}"
    if kind == "plan":
        kinds = ", ".join(pv["kind"] for pv in ev["views"])
        return f"topic={ev['topic']!r} | views: {kinds} | {len(ev['follow_up_questions'])} follow-ups"
    if kind in ("view_error", "error"):
        return f"{ev.get('view_id', '')} {ev['message'][:200]}"
    if kind == "trace":
        return f"trace_id={ev['trace_id']}"
    if kind == "done":
        return ""
    return ""


def print_trace_summary(api: str, trace_id: str) -> None:
    """Step timings + LLM calls from GET /api/traces/{id}."""
    with urllib.request.urlopen(f"{api}/api/traces/{trace_id}") as resp:
        trace = json.load(resp)
    print(f"\nTrace {trace_id} ({trace['status']}, {trace.get('total_s')}s) → {trace.get('file', 'in memory')}")
    for s in trace["steps"]:
        extra = " ERROR " + s["error"] if s.get("error") else ""
        print(f"  {s['start_s']:7.1f}s  +{s.get('duration_s', 0):6.1f}s  {s['name']}{extra}")
    for c in trace["llm_calls"]:
        u = c.get("usage") or {}
        print(f"  LLM {c['label']:<28} {c['duration_s']:6.1f}s  in={u.get('input_tokens')} out={u.get('output_tokens')}"
              f"  retries={len(c.get('validation_retries') or [])}{'  ERROR ' + c['error'] if c.get('error') else ''}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.query_id is not None:
        q = load_benchmark_query(args.query_id)
        question = q["query_text"]
        print(f"Benchmark query {q['query_id']} ({q.get('province')}); "
              f"ground truth: {', '.join(q['ground_truth_citations'][:6])}…\n")
    else:
        question = args.question

    events: list[dict] = []
    t0 = time.perf_counter()
    for ev in stream_events(args.api, question, args.user_id, args.scope, args.timeout):
        events.append(ev)
        print(f"{time.perf_counter() - t0:7.1f}s  {ev['event']:<10} {describe(ev)}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        print(f"\nSaved {len(events)} events → {args.out}")

    trace_id = next((e["trace_id"] for e in events if e["event"] == "trace"), None)
    if trace_id:
        print_trace_summary(args.api, trace_id)

    failed = [e for e in events if e["event"] in ("view_error", "error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
