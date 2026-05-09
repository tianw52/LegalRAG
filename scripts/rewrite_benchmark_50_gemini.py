#!/usr/bin/env python3
"""Rewrite all benchmark_50 queries (200 total) with Gemini via Google AI (google-genai).

Writes ``data/LegalBenchRAG/benchmark_50_reformated/gemini25_flash_lite/*.json`` in the same
shape as ``qwen72b/`` (``metadata`` + ``results`` with ``v4_reddit_style``).

Then run::

  python3 scripts/build_benchmark_50_reformated_processed.py --models gemini25_flash_lite

Environment (``.env`` in LegalRAG root, or export)::

  GOOGLE_API_KEY=...               # or GEMINI_API_KEY
  # If ``cd`` is under /home/... but the script resolves under /project/..., put the key in
  # ``.env`` in your current directory or set: ``export LEGALRAG_ROOT=/path/to/LegalRAG``
  # Optional:
  LLM_MODEL=gemini-2.5-flash-lite
  GEMINI_SLEEP_SECONDS=12
  GEMINI_MAX_ATTEMPTS=8

Requires: pip install google-genai  (or pip install -e '.[eval]')
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    """LegalRAG root: ``$LEGALRAG_ROOT`` if set and valid, else parent of ``scripts/``."""
    root = os.environ.get("LEGALRAG_ROOT", "").strip()
    if root:
        p = Path(root).expanduser().resolve()
        if (p / "data" / "LegalBenchRAG" / "benchmarks_50").is_dir():
            return p
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _repo_root()
DATA_DIR = REPO_ROOT / "data" / "LegalBenchRAG"
BENCHMARKS_50 = DATA_DIR / "benchmarks_50"
OUT_DIR = DATA_DIR / "benchmark_50_reformated" / "gemini25_flash_lite"
DATASETS = ("contractnli", "cuad", "maud", "privacy_qa")
VARIANT = "v4_reddit_style"
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _ordered_env_paths(*, cwd_first: bool) -> list[Path]:
    """Unique ``.env`` paths. Order matters.

    On clusters, ``__file__`` often resolves under ``/project/...`` while you ``cd`` to
    ``/home/...`` (same tree via symlink or two checkouts). Prefer **cwd** first when
    parsing keys manually; load **repo then cwd** for dotenv so the shell directory wins.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    order = (
        (Path.cwd() / ".env", REPO_ROOT / ".env")
        if cwd_first
        else (REPO_ROOT / ".env", Path.cwd() / ".env")
    )
    for p in order:
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _load_dotenv() -> None:
    """Load ``.env``: repo root first, then cwd (``override=True`` so cwd overrides)."""
    try:
        from dotenv import load_dotenv

        for path in _ordered_env_paths(cwd_first=False):
            load_dotenv(path, override=True)
    except ImportError:
        pass


def _parse_key_from_env_text(text: str) -> str:
    """Extract first non-empty GOOGLE_API_KEY / GEMINI_API_KEY value from ``.env`` body."""
    for raw in text.splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        m = re.match(r"^(GOOGLE_API_KEY|GEMINI_API_KEY)\s*=\s*(.*)$", line)
        if not m:
            continue
        val = m.group(2).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        else:
            # strip trailing inline comment: KEY=value  # comment
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
        if val:
            return val
    return ""


def _api_key_from_env_file() -> str:
    """Fallback: read GOOGLE_API_KEY / GEMINI_API_KEY from ``.env`` files (Slurm-safe).

    Accepts ``export KEY=...``, spaces around ``=``, UTF-8 BOM. Tries **cwd** before ``REPO_ROOT``.
    """
    for path in _ordered_env_paths(cwd_first=True):
        text = path.read_text(encoding="utf-8-sig")
        key = _parse_key_from_env_text(text)
        if key:
            return key
    return ""


def _prompt_meta() -> tuple[str, str]:
    system_prompt = (
        "You rewrite formal legal questions into how a non-lawyer would actually ask them. "
        "Replace legal jargon with plain, everyday words. Use contractions (don't, it's, won't). "
        "Vary your phrasing — avoid repeating the same opening. Keep the exact same meaning. "
        "Output ONLY the rewritten query, nothing else."
    )
    user_tmpl = (
        "Rewrite this legal question as if a real person posted it on Reddit asking for help. "
        "Use plain language — replace legal jargon with everyday words. Be creative with how you start: "
        "mimic how real humans actually post — varied, natural, sometimes rambling, sometimes direct, "
        "never formulaic. Avoid repetitive openings. Keep the exact meaning.\n\n"
        "Original: {query}\n\n"
        "Reddit-style question:"
    )
    return system_prompt, user_tmpl


def _is_bad(text: str | None) -> bool:
    if not text or not str(text).strip():
        return True
    return str(text).strip().startswith("[ERROR")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        help=f"Gemini model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows that already have a non-empty v4_reddit_style rewrite (no [ERROR).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Rewrite every row (ignore existing rewrites). Overrides --resume.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="Which benchmark JSON files to process.",
    )
    args = p.parse_args()

    _load_dotenv()

    only_missing = args.resume and not args.all

    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        key = _api_key_from_env_file().strip()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
    if not key:
        uniq: list[Path] = []
        seen_rp: set[Path] = set()
        for p in _ordered_env_paths(cwd_first=True) + _ordered_env_paths(cwd_first=False):
            r = p.resolve()
            if r not in seen_rp:
                seen_rp.add(r)
                uniq.append(p)
        tried = ", ".join(str(p) for p in uniq) or "(no .env file found)"
        print(
            "ERROR: No GOOGLE_API_KEY / GEMINI_API_KEY.\n"
            f"  REPO_ROOT={REPO_ROOT}\n"
            f"  Tried .env paths: {tried}\n"
            "  Set LEGALRAG_ROOT to your LegalRAG checkout if paths are wrong.\n"
            "  Ensure one line: GOOGLE_API_KEY=your_key (non-empty after =).",
            file=sys.stderr,
        )
        sys.exit(1)

    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = args.model

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.LegalBenchRAG.query_rewrite import build_client, rewrite_query

    system_prompt, user_tmpl = _prompt_meta()

    try:
        client, model, provider = build_client()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not build Gemini client: {e}", file=sys.stderr)
        sys.exit(1)

    if provider != "gemini":
        print("ERROR: expected gemini provider (set GOOGLE_API_KEY).", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        bench_path = BENCHMARKS_50 / f"{ds}.json"
        if not bench_path.exists():
            print(f"ERROR: missing {bench_path}", file=sys.stderr)
            sys.exit(1)

        out_path = OUT_DIR / f"{ds}.json"
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        tests = bench.get("tests", [])
        if len(tests) != 50:
            print(f"WARN: {ds}: expected 50 tests, got {len(tests)}", flush=True)

        by_orig: dict[str, dict] = {}
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for row in existing.get("results", []):
                o = row.get("original")
                if o:
                    by_orig[o] = row

        new_results: list[dict] = []
        for i, t in enumerate(tests, start=1):
            orig = t["query"]
            row = by_orig.get(orig)
            if row is None:
                row = {"dataset": ds, "original": orig, "rewrites": {}}
            else:
                row = {
                    "dataset": ds,
                    "original": orig,
                    "rewrites": dict(row.get("rewrites") or {}),
                }

            rw = (row.get("rewrites") or {}).get(VARIANT, "")
            if only_missing and not _is_bad(rw):
                new_results.append(row)
                continue

            try:
                text = rewrite_query(client, model, orig, VARIANT, provider)
            except Exception as e:  # noqa: BLE001
                text = f"[ERROR: {e}]"
            row.setdefault("rewrites", {})[VARIANT] = text
            new_results.append(row)

            payload = {
                "metadata": {
                    "model": model,
                    "prompt_variant": VARIANT,
                    "datasets": list(DATASETS),
                    "prompts": {
                        VARIANT: {
                            "system_prompt": system_prompt,
                            "user_prompt_template": user_tmpl,
                        }
                    },
                    "dataset": ds,
                },
                "results": new_results,
            }
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  {ds} {i}/{len(tests)}", flush=True)

        print(f"Wrote {out_path}", flush=True)

    print("\nNext: python3 scripts/build_benchmark_50_reformated_processed.py --models gemini25_flash_lite")


if __name__ == "__main__":
    main()
