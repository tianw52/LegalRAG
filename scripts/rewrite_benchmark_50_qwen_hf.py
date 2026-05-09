#!/usr/bin/env python3
"""Rewrite benchmark_50 queries with Qwen2.5-72B via Hugging Face Inference Router (OpenAI-compatible).

Reads/writes ``data/LegalBenchRAG/benchmark_50_reformated/qwen72b/*.json``.

Repair HF 402 / error placeholders (default: only broken rows):
  export HF_TOKEN=hf_...
  python3 scripts/rewrite_benchmark_50_qwen_hf.py

Uses stdlib only (no ``openai`` package).
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "LegalBenchRAG"
REFORMATED_QWEN = DATA_DIR / "benchmark_50_reformated" / "qwen72b"
DATASETS = ("contractnli", "cuad", "maud", "privacy_qa")
VARIANT = "v4_reddit_style"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


def _prompts() -> tuple[str, str]:
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
        "Original: {query}\n\nReddit-style question:"
    )
    return system_prompt, user_tmpl


def _needs_rewrite(text: str | None, only_errors: bool) -> bool:
    if not only_errors:
        return True
    if not text or not str(text).strip():
        return True
    s = str(text).strip()
    return s.startswith("[ERROR") or s.startswith("[error")


def _chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_retries: int = 6,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.35,
            "max_tokens": 768,
        }
    ).encode("utf-8")
    last_err = "unknown"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    ctx = ssl.create_default_context()

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                last_err = f"no choices: {raw[:500]}"
            else:
                msg = choices[0].get("message") or {}
                out = (msg.get("content") or "").strip()
                if out:
                    return out
                last_err = "empty content"
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                err_body = str(e)
            last_err = f"HTTP {e.code}: {err_body}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

        if "429" in last_err or "402" in last_err or "rate" in last_err.lower():
            wait = min(120.0, 5.0 * (2**attempt)) + random.uniform(0, 2)
            time.sleep(wait)
        else:
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 0.5))

    return f"[ERROR: {last_err}]"


def process_file(
    dataset: str,
    path: Path,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_template: str,
    only_errors: bool,
    sleep_s: float,
) -> tuple[int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    rewritten = 0
    skipped = 0

    meta = data.setdefault("metadata", {})
    meta["model"] = MODEL_ID
    meta.setdefault("prompts", {})[VARIANT] = {
        "system_prompt": system_prompt,
        "user_prompt_template": user_template,
    }
    meta["dataset"] = dataset

    for row in results:
        orig = row.get("original", "")
        rw = (row.get("rewrites") or {}).get(VARIANT, "")
        if not _needs_rewrite(rw, only_errors):
            skipped += 1
            continue
        user_content = user_template.format(query=orig)
        new_text = _chat_completions(base_url, api_key, model, system_prompt, user_content)
        row.setdefault("rewrites", {})[VARIANT] = new_text
        rewritten += 1
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        time.sleep(sleep_s)

    return len(results), rewritten, skipped


def main() -> None:
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--all",
        action="store_true",
        help="Rewrite every row. Default: only rows with missing/empty/[ERROR rewrites.",
    )
    p.add_argument("--model", default=MODEL_ID, help="HF model id for chat completions.")
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.75,
        help="Seconds to sleep after each successful rewrite.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="Which benchmark JSON files to process.",
    )
    args = p.parse_args()
    only_errors = not args.all

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN") or ""
    if not api_key:
        print("ERROR: Set HF_TOKEN or LLM_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    system_prompt, user_template = _prompts()

    REFORMATED_QWEN.mkdir(parents=True, exist_ok=True)
    grand_r = grand_s = 0
    for ds in args.datasets:
        out_path = REFORMATED_QWEN / f"{ds}.json"
        if not out_path.exists():
            print(f"ERROR: missing {out_path}", file=sys.stderr)
            sys.exit(1)
        total, rw, sk = process_file(
            ds,
            out_path,
            args.base_url,
            api_key,
            args.model,
            system_prompt,
            user_template,
            only_errors,
            args.sleep,
        )
        grand_r += rw
        grand_s += sk
        print(f"{ds}: total={total} rewritten={rw} skipped_ok={sk} -> {out_path}")

    print(f"\nDone. rewritten={grand_r} skipped={grand_s}")


if __name__ == "__main__":
    main()
