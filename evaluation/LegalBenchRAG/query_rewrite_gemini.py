#!/usr/bin/env python3
"""
Gemini Query Rewriter for LegalBench-RAG (Vertex AI Edition)
=============================================================
Rewrites benchmark queries into plain-language Reddit-style phrasing using
Google Cloud Vertex AI.

Uses a single prompt variant (v4_reddit_style) with the human-style system
prompt.  Output is the same "compare" JSON schema consumed by
``scripts/build_benchmark_50_reformated_processed.py``.

Usage
-----
# Test: 4 queries per dataset (16 total)
python -m evaluation.LegalBenchRAG.query_rewrite_gemini --mode test

# Full: all queries from benchmarks_50/
python -m evaluation.LegalBenchRAG.query_rewrite_gemini --mode full

# Resume from partial output (fills missing/errored rewrites)
python -m evaluation.LegalBenchRAG.query_rewrite_gemini --mode full --resume-from data/temp/query_rewrites_gemini.json

Environment
-----------
GCP_PROJECT             Google Cloud Project ID (required)
GCP_LOCATION            Google Cloud Region (default: us-central1)
LLM_MODEL               Model name (default: gemini-1.5-flash-preview-0409)
GEMINI_SLEEP_SECONDS    Pause between requests for rate-limit compliance (default: 1)
GEMINI_MAX_ATTEMPTS     Max retries per request (default: 8)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Vertex AI SDK
# ---------------------------------------------------------------------------
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.api_core import exceptions as gcp_exceptions
except ImportError:
    sys.exit(
        "[ERROR] google-cloud-aiplatform is required. Install with:\n"
        "  pip install google-cloud-aiplatform"
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
print(f"ROOT: {ROOT}")
BENCHMARKS_DIR = ROOT / "data" / "legalbenchrag-mini" / "benchmarks"
BENCHMARKS_50_DIR = ROOT / "data" / "LegalBenchRAG" / "benchmarks_50"
RESULTS_DIR = ROOT / "data" / "temp"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["contractnli", "cuad", "maud", "privacy_qa"]

# ---------------------------------------------------------------------------
# Prompt (v4_reddit_style + human system prompt)
# ---------------------------------------------------------------------------
VARIANT = "v4_reddit_style"

SYSTEM_PROMPT = (
    "You rewrite formal legal questions into how a non-lawyer would actually ask them. "
    "Replace legal jargon with plain, everyday words. Use contractions (don't, it's, won't). "
    "Vary your phrasing — avoid repeating the same opening. Keep the exact same meaning. "
    "Output ONLY the rewritten query, nothing else."
)

USER_TEMPLATE = (
    "Rewrite this legal question as if a real person posted it on Reddit asking for help. "
    "Use plain language — replace legal jargon with everyday words. "
    "Be creative with how you start: mimic how real humans actually post — varied, natural, "
    "sometimes rambling, sometimes direct, never formulaic. Avoid repetitive openings. "
    "Keep the exact meaning.\n\n"
    "Original: {query}\n\n"
    "Reddit-style question:"
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class QuotaExhaustedError(Exception):
    """Vertex AI 429 — daily/minute quota hit."""

# ---------------------------------------------------------------------------
# Error check helpers
# ---------------------------------------------------------------------------
def _is_rate_or_transient(e: Exception) -> tuple[bool, bool]:
    is_rate = isinstance(e, gcp_exceptions.ResourceExhausted) or "429" in str(e)
    is_transient = isinstance(e, (
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.DeadlineExceeded,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.GatewayTimeout
    )) or "timeout" in str(e).lower()
    return is_rate, is_transient

# ---------------------------------------------------------------------------
# Core rewrite
# ---------------------------------------------------------------------------
def rewrite_query(model_obj: GenerativeModel, query: str) -> str:
    """Call Vertex AI Gemini to rewrite a single query. Retries on transient / rate errors."""
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(query=query)}"
    sleep_s = float(os.getenv("GEMINI_SLEEP_SECONDS", "2"))
    max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))

    time.sleep(sleep_s)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = model_obj.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2000,
                    "temperature": 1.0,
                }
            )
            return response.text.strip()
        except Exception as e:
            last_err = e
            is_rate, is_transient = _is_rate_or_transient(e)

            if is_rate:
                wait = 65  # Vertex quotas usually reset per minute
                if attempt < max_attempts - 1:
                    time.sleep(wait)
                    continue
                raise QuotaExhaustedError(
                    "Vertex AI quota exhausted (429). Check your Google Cloud Quotas."
                ) from last_err

            if is_transient and attempt < max_attempts - 1:
                time.sleep(min(20 + 15 * attempt, 120))
                continue

            raise
    raise last_err

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_queries(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text())
    return [t["query"] for t in data["tests"]]

def sample_queries(
    datasets: list[str],
    source_dir: Path,
    n_per_dataset: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    for ds in datasets:
        path = source_dir / f"{ds}.json"
        if not path.exists():
            print(f"[WARN] {path} not found, skipping.")
            continue
        queries = load_queries(path)
        chosen = rng.sample(queries, min(n_per_dataset, len(queries)))
        for q in chosen:
            samples.append({"dataset": ds, "original": q})
    return samples

# ---------------------------------------------------------------------------
# Output helpers (compare-format JSON)
# ---------------------------------------------------------------------------
def _make_metadata(model: str, n_per: int, seed: int, datasets: list[str]) -> dict:
    return {
        "model": model,
        "n_per_dataset": n_per,
        "seed": seed,
        "datasets": datasets,
        "prompts": {
            VARIANT: {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": USER_TEMPLATE,
            }
        },
    }

def _flush(out_path: Path, metadata: dict, results: list[dict]) -> None:
    out_path.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, ensure_ascii=False)
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite LegalBench-RAG queries with Vertex AI Gemini (v4_reddit_style)."
    )
    parser.add_argument(
        "--mode", choices=["test", "full"], default="test",
        help="'test' samples --n per dataset; 'full' uses all queries.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Which datasets to process (default: all four).",
    )
    parser.add_argument(
        "--n", type=int, default=4,
        help="Queries per dataset in test mode (default: 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--benchmarks-dir", type=str, default=None,
        help="Override benchmarks directory (default: benchmarks_50 if present, else benchmarks).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from partial JSON, filling only missing/errored rewrites.",
    )
    args = parser.parse_args()

    # --- GCP / Vertex AI Initialization -------------------------------------
    project_id = os.getenv("GCP_PROJECT")
    if not project_id:
        sys.exit(
            "[ERROR] GCP_PROJECT is not set.\n"
            "  Please set your Google Cloud Project ID to use Vertex AI."
        )
    location = os.getenv("GCP_LOCATION", "global")
    model_name = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

    vertexai.init(project=project_id, location=location)
    model_obj = GenerativeModel(model_name)

    # --- Benchmarks dir ----------------------------------------------------
    if args.benchmarks_dir:
        bdir = Path(args.benchmarks_dir)
        if not bdir.is_absolute():
            bdir = ROOT / bdir
    # elif BENCHMARKS_50_DIR.exists():
    #     bdir = BENCHMARKS_50_DIR
    else:
        bdir = BENCHMARKS_DIR

    if not bdir.exists():
        sys.exit(f"[ERROR] Benchmarks dir not found: {bdir}")

    n_per = 9999 if args.mode == "full" else args.n

    print(f"Project      : {project_id} ({location})")
    print(f"Model        : {model_name}")
    print(f"Benchmarks   : {bdir}")
    print(f"Mode         : {args.mode}")
    print(f"Prompt       : {VARIANT}\n")

    # --- Connectivity check ------------------------------------------------
    print("Testing Vertex AI API... ", end="", flush=True)
    try:
        rewrite_query(model_obj, "test")
        print("OK\n")
    except QuotaExhaustedError as e:
        print(f"\n\n{e}\n")
        sys.exit(1)
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            print(f"FAILED: Model '{model_name}' not found in {location}.\n")
        else:
            print(f"FAILED: {e}\n")
        sys.exit(1)

    # --- Resume mode -------------------------------------------------------
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            sys.exit(f"[ERROR] --resume-from not found: {resume_path}")

        data = json.loads(resume_path.read_text())
        loaded = {r["original"]: r for r in data["results"]}
        meta = data["metadata"]
        samples = sample_queries(
            meta.get("datasets", args.datasets), bdir,
            n_per, meta.get("seed", args.seed),
        )

        results: list[dict] = []
        for item in samples:
            row = loaded.get(item["original"])
            if row is None:
                row = {"dataset": item["dataset"], "original": item["original"], "rewrites": {}}
            results.append(row)

        out_path = RESULTS_DIR / resume_path.name
        filled = 0
        for i, item in enumerate(samples, 1):
            row = results[i - 1]
            val = row.get("rewrites", {}).get(VARIANT, "")
            if val and not val.startswith("[ERROR") and val.strip():
                continue
            filled += 1
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] [{i}/{len(samples)}] {item['dataset'][:12]} ... ", end="", flush=True)
            t0 = time.time()
            try:
                rw = rewrite_query(model_obj, item["original"])
                row.setdefault("rewrites", {})[VARIANT] = rw
                print(f"done ({time.time() - t0:.1f}s)")
            except Exception as e:
                row.setdefault("rewrites", {})[VARIANT] = f"[ERROR: {e}]"
                print(f"ERROR ({time.time() - t0:.1f}s)")
            data["results"] = results
            _flush(out_path, meta, results)

        print(f"\nSaved → {out_path} ({filled} missing filled)")
        return

    # --- Normal mode -------------------------------------------------------
    samples = sample_queries(args.datasets, bdir, n_per, args.seed)
    print(f"Queries: {len(samples)}\n")

    model_safe = model_name.replace("/", "-").replace(":", "-")
    out_path = RESULTS_DIR / f"mini_rewrites_{model_safe}.json"
    metadata = _make_metadata(model_name, args.n, args.seed, args.datasets)

    results = []
    for i, item in enumerate(samples, 1):
        row = {"dataset": item["dataset"], "original": item["original"], "rewrites": {}}
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"  [{ts}] [{i:3d}/{len(samples)}] {item['dataset']:<12} "
            f"{item['original'][:80]}...",
        )
        t0 = time.time()
        try:
            rw = rewrite_query(model_obj, item["original"])
            row["rewrites"][VARIANT] = rw
            print(f"    → {rw[:90]}  ({time.time() - t0:.1f}s)")
        except QuotaExhaustedError as e:
            results.append(row)
            _flush(out_path, metadata, results)
            print(f"\n\n{e}\nPartial results saved to {out_path}")
            sys.exit(1)
        except Exception as e:
            row["rewrites"][VARIANT] = f"[ERROR: {e}]"
            print(f"    ERROR ({time.time() - t0:.1f}s): {e}")
        results.append(row)
        _flush(out_path, metadata, results)

    print(f"\nSaved {len(results)} rewrites → {out_path}")

if __name__ == "__main__":
    main()