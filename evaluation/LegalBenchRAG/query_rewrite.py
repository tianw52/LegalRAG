"""
Query Rewriting Script for LegalBench-RAG
==========================================
Rewrites benchmark queries to be more natural / informal using a large Qwen model
via the HuggingFace Inference Router.

Usage
-----
# Test run: 3-4 queries sampled per dataset (12-16 total)
python scripts/query_rewrite.py --mode test

# Full run: all 50 queries from benchmarks_subset/
python scripts/query_rewrite.py --mode full

# Custom: specific datasets and sample size
python scripts/query_rewrite.py --mode test --datasets contractnli cuad --n 4

# Compare ALL prompt variants on same 16 queries (for human preference)
python scripts/query_rewrite.py --mode compare

Output
------
results/query_rewrites_test.json      (test mode, single prompt)
results/query_rewrites_full.json      (full mode, single prompt)
results/query_rewrites_compare_Nvars_MODEL.json   (e.g. query_rewrites_compare_2vars_Qwen-Qwen2.5-7B-Instruct.json)

Each entry:
{
  "dataset":       "contractnli",
  "original":      "Consider the NDA between ...; Does it say X?",
  "rewritten":     "In the NDA between ..., does it mention X?"
}
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI


class QuotaExhaustedError(Exception):
    """Raised when Gemini API returns 429 after retries (daily/minute quota exhausted)."""


class CreditsExhaustedError(Exception):
    """Raised when Hugging Face API returns 402 (monthly credits depleted)."""


def _make_gemini_client(api_key: str):
    """Build google-genai Client with bounded HTTP timeout (avoids ~300s hangs)."""
    if not HAS_GEMINI:
        return None
    timeout_ms = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "120000"))
    try:
        from google.genai import types as genai_types

        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )
    except (TypeError, ValueError, Exception):
        return genai.Client(api_key=api_key)


def _gemini_rate_or_transient(err_str: str) -> tuple[bool, bool]:
    """Returns (is_rate_limit, is_transient_retryable)."""
    el = err_str.lower()
    rate = (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "quota" in el
        or "rate limit" in el
        or "too many requests" in el
    )
    transient = any(
        x in err_str
        for x in (
            "timeout",
            "Timeout",
            "timed out",
            "503",
            "DEADLINE",
            "UNAVAILABLE",
            "500",
            "502",
            "504",
        )
    ) or ("read timed out" in el)
    return rate, transient


def _extract_from_reasoning(reasoning: str) -> str:
    """Extract final answer from Qwen3.5 reasoning text when content is null."""
    if not reasoning or not isinstance(reasoning, str):
        return ""
    # Look for "Let's go with: \"...\"" 
    m = re.search(r'Let\'?s go with:\s*["\']([^"\']+)["\']', reasoning, re.I)
    if m:
        return m.group(1).strip()
    # Last *Option N:* that ends with ? (complete question)
    opts = re.findall(r'\*Option \d+\*:\s*([^\n*]+?\?)', reasoning)
    if opts:
        return opts[-1].strip()
    # Any *Option N:* (may be truncated)
    opts = re.findall(r'\*Option \d+\*:\s*([^\n*]+)', reasoning)
    if opts:
        return opts[-1].strip()
    # Last line ending with ?
    lines = [ln.strip() for ln in reasoning.split("\n") if ln.strip().endswith("?")]
    if lines:
        return lines[-1]
    return ""


# Gemini (Google AI) — optional, for LLM_PROVIDER=gemini
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root (this file is evaluation/LegalBenchRAG/query_rewrite.py)
BENCHMARKS_DIR = ROOT / "data" / "LegalBenchRAG" / "benchmarks"
SUBSET_DIR     = ROOT / "data" / "LegalBenchRAG" / "benchmarks_subset"
RESULTS_DIR    = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["contractnli", "cuad", "maud", "privacy_qa"]

# ---------------------------------------------------------------------------
# Prompt templates (try different prompts and pick the best manually)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_DEFAULT = """\
You are a helpful assistant that rewrites formal legal queries into more natural, \
conversational language while fully preserving their original meaning and intent. \
The rewritten query should sound like something a real person would type — informal, \
fluent, and less rigid than the original. Do NOT add extra information. \
Return ONLY the rewritten query, nothing else.\
"""

SYSTEM_PROMPT_HUMAN = """\
You rewrite formal legal questions into how a non-lawyer would actually ask them. \
Replace legal jargon with plain, everyday words. Use contractions (don't, it's, won't). \
Vary your phrasing — avoid repeating the same opening. Keep the exact same meaning. \
Output ONLY the rewritten query, nothing else.\
"""

# Prompt variants — each aims for a different "human" style
PROMPT_VARIANTS = {
    "v1_conversational": (
        "Rewrite the following legal query to sound more natural and conversational, "
        "as if a non-lawyer is asking it casually. Keep all the meaning intact.\n\n"
        "Original query: {query}\n\n"
        "Rewritten query:"
    ),
    "v2_search_bar": (
        "Transform this formal legal query into something a regular person would "
        "type into a search engine. Make it shorter and more direct if possible, "
        "without losing any important details.\n\n"
        "Original query: {query}\n\n"
        "Rewritten query:"
    ),
    "v3_friend_texting": (
        "Imagine a friend is texting you this question. Rewrite it exactly as they "
        "would type it — casual, maybe a bit lazy with grammar, using everyday words. "
        "Same meaning, just how a real person would actually ask.\n\n"
        "Original: {query}\n\n"
        "How they'd text it:"
    ),
    "v4_reddit_style": (
        "Rewrite this legal question as if a real person posted it on Reddit asking for help. "
        "Use plain language — replace legal jargon with everyday words. "
        "Be creative with how you start: mimic how real humans actually post — varied, natural, "
        "sometimes rambling, sometimes direct, never formulaic. Avoid repetitive openings. "
        "Keep the exact meaning.\n\n"
        "Original: {query}\n\n"
        "Reddit-style question:"
    ),
    "v5_everyday_words": (
        "Rewrite this formal legal query using everyday, plain English. Use contractions "
        "(don't, it's, won't, that's), simpler words instead of legalese, and a relaxed "
        "tone. A regular person should understand it immediately. Preserve the exact meaning.\n\n"
        "Original: {query}\n\n"
        "Plain English version:"
    ),
    "v6_curious_user": (
        "A curious user is looking something up. Rewrite this legal question the way "
        "they'd actually type it — short, direct, maybe a bit impatient. Like 'does X say Y?' "
        "or 'what does the contract say about Z?' Keep all the important details.\n\n"
        "Original: {query}\n\n"
        "How they'd ask:"
    ),
    "v7_google_punchy": (
        "Turn this into a short, punchy search query — like what someone would type "
        "into Google when they need a quick answer. No preamble, no 'consider the...'. "
        "Just the core question in 1–2 sentences max. Same meaning.\n\n"
        "Original: {query}\n\n"
        "Google-style query:"
    ),
    "v8_first_person": (
        "Rewrite this legal question as if the user is asking for themselves — "
        "'I want to know...', 'Can someone tell me...', 'Does it say that I...'. "
        "First-person, personal, human. Keep the full meaning.\n\n"
        "Original: {query}\n\n"
        "First-person version:"
    ),
    "v9_plain_varied": (
        "Rewrite this as a plain-English question a non-lawyer would ask. "
        "Replace ALL legal terms with everyday words. "
        "Be creative and varied — mimic how real people actually ask questions, "
        "not a fixed template. Keep the exact meaning.\n\n"
        "Original: {query}\n\n"
        "Plain question:"
    ),
}

# Which system prompt to use per variant (human-style variants get the more casual one)
VARIANT_SYSTEM = {
    "v3_friend_texting": SYSTEM_PROMPT_HUMAN,
    "v4_reddit_style": SYSTEM_PROMPT_HUMAN,
    "v5_everyday_words": SYSTEM_PROMPT_HUMAN,
    "v6_curious_user": SYSTEM_PROMPT_HUMAN,
    "v7_google_punchy": SYSTEM_PROMPT_HUMAN,
    "v8_first_person": SYSTEM_PROMPT_HUMAN,
    "v9_plain_varied": SYSTEM_PROMPT_HUMAN,
}
for k in PROMPT_VARIANTS:
    if k not in VARIANT_SYSTEM:
        VARIANT_SYSTEM[k] = SYSTEM_PROMPT_DEFAULT

# ---------------------------------------------------------------------------
# LLM client (reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from env / .env)
# ---------------------------------------------------------------------------
def _load_env() -> None:
    """Load .env file if python-dotenv is available, otherwise skip."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def build_client():
    """Returns (client, model, provider). provider is 'openai', 'gemini', or 'mistral'."""
    _load_env()
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            sys.exit(
                "\n[ERROR] MISTRAL_API_KEY is not set.\n"
                "  Get a key at: https://console.mistral.ai/api-keys/\n"
            )
        model_name = os.getenv("LLM_MODEL", "ministral-14b-2512")
        client = OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=api_key,
            timeout=120.0,
        )
        return client, model_name, "mistral"

    if provider == "gemini":
        if not HAS_GEMINI:
            sys.exit(
                "\n[ERROR] Gemini backend requires: pip install google-genai\n"
            )
        api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            sys.exit(
                "\n[ERROR] GOOGLE_API_KEY or GEMINI_API_KEY is not set.\n"
                "  Get a free key at: https://aistudio.google.com/app/apikey\n"
            )
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
        client = _make_gemini_client(api_key)
        return client, model_name, "gemini"

    # OpenAI-compatible (HF, Ollama, vLLM, OpenRouter, etc.)
    base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct:novita")
    if "openrouter.ai" in base_url and os.getenv("OPENROUTER_API_KEY"):
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key.startswith("hf_..."):
        sys.exit(
            "\n[ERROR] LLM_API_KEY is not set.\n"
            "  Get a token at: https://huggingface.co/settings/tokens\n"
        )

    provider = "openrouter" if "openrouter.ai" in base_url else "openai"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600.0)
    return client, model, provider


# ---------------------------------------------------------------------------
# Core rewrite function
# ---------------------------------------------------------------------------
def rewrite_query(client, model: str, query: str, prompt_variant: str, provider: str = "openai") -> str:
    user_msg = PROMPT_VARIANTS[prompt_variant].format(query=query)
    system = VARIANT_SYSTEM.get(prompt_variant, SYSTEM_PROMPT_DEFAULT)

    if provider == "gemini":
        prompt = f"{system}\n\n{user_msg}"
        # Free tier is strict on RPM; default 12s spacing (~5 RPM). Override: GEMINI_SLEEP_SECONDS
        sleep_s = float(os.getenv("GEMINI_SLEEP_SECONDS", "12"))
        time.sleep(sleep_s)
        last_err = None
        max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "8"))
        for attempt in range(max_attempts):
            try:
                try:
                    from google.genai import types

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=256,
                            temperature=0.3,
                        ),
                    )
                except (ImportError, TypeError):
                    response = client.models.generate_content(model=model, contents=prompt)
                if hasattr(response, "text"):
                    return response.text.strip()
                return response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_rate, is_transient = _gemini_rate_or_transient(err_str)

                if is_rate:
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.I)
                    wait_s = max(65, float(m.group(1)) + 5) if m else 65
                    wait_s = min(wait_s, 180)
                    if attempt < max_attempts - 1:
                        time.sleep(wait_s)
                        continue
                    raise QuotaExhaustedError(
                        "Gemini API quota exhausted (429). Wait until tomorrow or use a paid API key. "
                        "See https://ai.google.dev/gemini-api/docs/rate-limits"
                    ) from last_err

                if is_transient and attempt < max_attempts - 1:
                    time.sleep(min(20 + 15 * attempt, 120))
                    continue

                # Non-retryable or attempts exhausted
                raise last_err
        raise last_err

    if provider == "openrouter":
        # OpenRouter: Qwen3.5 puts output in reasoning, not content. Use raw HTTP + extract.
        import requests
        url = str(client.base_url or "").rstrip("/") + "/chat/completions"
        api_key = client.api_key
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 512,  # Qwen3.5 uses reasoning; need room for thinking + answer
                "temperature": 0.3,
            },
            timeout=120,
        )
        if r.status_code != 200:
            err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}
            raise RuntimeError(f"OpenRouter API error {r.status_code}: {err}")
        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content and msg.get("reasoning"):
            content = _extract_from_reasoning(msg["reasoning"])
        return (content or "").strip()

    # OpenAI-compatible (HF, Ollama, vLLM, etc.)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.3,
        )
    except Exception as e:
        err_str = str(e)
        if "402" in err_str or "depleted" in err_str.lower() or "credits" in err_str.lower():
            raise CreditsExhaustedError(
                "Hugging Face monthly credits depleted (402). "
                "Subscribe to PRO or purchase credits at https://huggingface.co/settings/billing"
            ) from e
        raise

    content = response.choices[0].message.content
    return (content or "").strip()


# ---------------------------------------------------------------------------
# Sampling helpers
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
    samples = []
    for ds in datasets:
        path = source_dir / f"{ds}.json"
        if not path.exists():
            print(f"[WARN] {path} not found, skipping.")
            continue
        queries = load_queries(path)
        chosen  = rng.sample(queries, min(n_per_dataset, len(queries)))
        for q in chosen:
            samples.append({"dataset": ds, "original": q})
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite legal benchmark queries with Qwen.")
    parser.add_argument(
        "--mode", choices=["test", "full", "compare"], default="test",
        help="'test' samples per dataset; 'full' uses all; 'compare' runs all prompt variants.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Which datasets to process (default: all four).",
    )
    parser.add_argument(
        "--n", type=int, default=4,
        help="Queries to sample per dataset in test/compare mode (default: 4 → 16 total).",
    )
    parser.add_argument(
        "--prompt", choices=list(PROMPT_VARIANTS.keys()), default="v1_conversational",
        help="Which prompt variant to use (ignored in compare mode).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from partial JSON: fill only missing rewrites using GOOGLE_API_KEY_BACKUP.",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Compare mode: use only this variant (e.g. v4_reddit_style). Default: both v4 and v5.",
    )
    parser.add_argument(
        "--benchmarks-dir", type=str, default=None,
        help="Override benchmarks directory (e.g. benchmarks_50 for 50-query subset). Default: data/LegalBenchRAG/benchmarks.",
    )
    args = parser.parse_args()

    client, model, provider = build_client()
    print(f"Model    : {model}")
    print(f"Provider : {provider}")
    print(f"Mode     : {args.mode}\n")

    # Resolve benchmarks directory (needed for resume and compare)
    benchmarks_dir = Path(args.benchmarks_dir) if args.benchmarks_dir else BENCHMARKS_DIR
    if args.benchmarks_dir and not benchmarks_dir.is_absolute():
        benchmarks_dir = ROOT / benchmarks_dir
    if not benchmarks_dir.exists():
        print(f"ERROR: Benchmarks dir not found: {benchmarks_dir}")
        sys.exit(1)

    # ── Resume mode: fill only missing rewrites using backup key ─────────────────
    if args.resume_from and args.mode == "compare":
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            print(f"ERROR: --resume-from file not found: {resume_path}")
            sys.exit(1)
        data = json.loads(resume_path.read_text())
        loaded_results = {r["original"]: r for r in data["results"]}
        meta = data["metadata"]
        variants = list(meta["prompts"].keys())
        model = meta["model"]
        n_per = 9999 if args.benchmarks_dir else meta.get("n_per_dataset", args.n)
        seed = meta.get("seed", args.seed)
        datasets = meta.get("datasets", args.datasets)
        if provider == "gemini":
            # Default: use primary GOOGLE_API_KEY (same as build_client). Only switch to
            # GOOGLE_API_KEY_BACKUP when explicitly requested — otherwise backup-only mode
            # breaks resume if backup shares quota or is exhausted (see job logs).
            backup_key = os.getenv("GOOGLE_API_KEY_BACKUP", "")
            use_backup_only = os.getenv("GEMINI_RESUME_USE_BACKUP_ONLY", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if backup_key and use_backup_only:
                client = _make_gemini_client(backup_key)
                print("Resume: using GOOGLE_API_KEY_BACKUP only (GEMINI_RESUME_USE_BACKUP_ONLY=1)")
            elif backup_key:
                print(
                    "Resume: using primary GOOGLE_API_KEY; set GEMINI_RESUME_USE_BACKUP_ONLY=1 "
                    "to force GOOGLE_API_KEY_BACKUP for all calls."
                )
        elif provider == "openai":
            backup_key = os.getenv("LLM_API_KEY_BACKUP", "")
            if backup_key:
                print("Resume: will use LLM_API_KEY_BACKUP if main key hits 402")
        # For HF/Mistral: use existing client to fill empty rewrites
        samples = sample_queries(datasets, benchmarks_dir, n_per, seed)
        print(f"Resume mode: filling missing/empty rewrites")
        print(f"Loaded: {len(loaded_results)} partial | Full: {len(samples)} queries\n")
        # Pre-build full results list (200 rows) so we can flush progress after each test
        results = []
        for item in samples:
            row = loaded_results.get(item["original"])
            if row is None:
                row = {"dataset": item["dataset"], "original": item["original"], "rewrites": {}}
            results.append(row)

        out_path = RESULTS_DIR / resume_path.name
        missing_count = 0
        for i, item in enumerate(samples, 1):
            row = results[i - 1]
            for v in variants:
                val = row.get("rewrites", {}).get(v, "")
                if not val or val.startswith("[ERROR") or val.strip() == "":
                    if missing_count == 0:
                        print("Testing Gemini API... ", end="", flush=True)
                        try:
                            rewrite_query(client, model, "test", v, provider)
                            print("OK\n")
                        except Exception as e:
                            print(f"FAILED: {e}\n")
                            sys.exit(1)
                    missing_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] [{i}/{len(samples)}] {item['dataset'][:12]} | {v} ... ", end="", flush=True)
                    t0 = time.time()
                    try:
                        rewritten = rewrite_query(client, model, item["original"], v, provider)
                        row.setdefault("rewrites", {})[v] = rewritten
                        print(f"done ({time.time() - t0:.1f}s)")
                    except CreditsExhaustedError as e:
                        backup_key = os.getenv("LLM_API_KEY_BACKUP", "")
                        if backup_key and provider == "openai":
                            print(f"402 — switching to LLM_API_KEY_BACKUP ... ", end="", flush=True)
                            base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
                            client = OpenAI(base_url=base_url, api_key=backup_key, timeout=600.0)
                            try:
                                rewritten = rewrite_query(client, model, item["original"], v, provider)
                                row.setdefault("rewrites", {})[v] = rewritten
                                print(f"done ({time.time() - t0:.1f}s)")
                            except Exception as retry_err:
                                row.setdefault("rewrites", {})[v] = f"[ERROR: {retry_err}]"
                                print(f"ERROR ({time.time() - t0:.1f}s)")
                        else:
                            row.setdefault("rewrites", {})[v] = f"[ERROR: {e}]"
                            print(f"ERROR ({time.time() - t0:.1f}s)")
                            if not backup_key:
                                print("\nTip: Add LLM_API_KEY_BACKUP=<new_HF_token> to .env and re-run with --resume-from")
                    except Exception as e:
                        row.setdefault("rewrites", {})[v] = f"[ERROR: {e}]"
                        print(f"ERROR ({time.time() - t0:.1f}s)")
            data["results"] = results
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\nSaved → {out_path} ({missing_count} missing filled)")
        return

    # Gemini: quick connectivity test (fail fast if quota/key invalid)
    if provider == "gemini":
        print("Testing Gemini API... ", end="", flush=True)
        try:
            rewrite_query(client, model, "test", "v4_reddit_style", provider)
            print("OK\n")
        except QuotaExhaustedError as e:
            print(f"\n\n{e}\n")
            sys.exit(1)
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                print(f"FAILED: Model '{model}' not found. Try: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-1.5-flash\n")
            else:
                print(f"FAILED: {e}\n")
            sys.exit(1)

    # ── Compare mode: run variants on same queries ───────────────────────────
    if args.mode == "compare":
        n_per = 9999 if args.benchmarks_dir else args.n  # custom dir = use all queries
        samples = sample_queries(args.datasets, benchmarks_dir, n_per, args.seed)
        variants = [args.variant] if args.variant else ["v4_reddit_style", "v5_everyday_words"]
        if args.variant and args.variant not in PROMPT_VARIANTS:
            print(f"ERROR: --variant '{args.variant}' not in {list(PROMPT_VARIANTS.keys())}")
            sys.exit(1)
        print(f"Queries: {len(samples)} | Variants: {len(variants)}\n")

        # Output filename: query_rewrites_compare_2vars_Qwen-Qwen2.5-7B-Instruct.json
        model_safe = model.replace("/", "-").replace(":", "-")
        compare_out_name = f"query_rewrites_compare_{len(variants)}vars_{model_safe}.json"
        compare_out_path = RESULTS_DIR / compare_out_name

        compare_results = []
        total_calls = len(samples) * len(variants)
        call_num = 0
        for i, item in enumerate(samples, 1):
            row = {"dataset": item["dataset"], "original": item["original"], "rewrites": {}}
            print(f"\n[{i:3d}/{len(samples)}] {item['dataset']}")
            print(f"         {item['original'][:90]}...")
            sys.stdout.flush()
            for v in variants:
                call_num += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] call {call_num:2d}/{total_calls} | {v} ... ", end="", flush=True)
                t0 = time.time()
                try:
                    rewritten = rewrite_query(client, model, item["original"], v, provider)
                    row["rewrites"][v] = rewritten
                    elapsed = time.time() - t0
                    print(f"done ({elapsed:.1f}s)")
                    print(f"           → {rewritten[:90]}")
                except CreditsExhaustedError as e:
                    # HF 402: try backup key if set
                    backup_key = os.getenv("LLM_API_KEY_BACKUP", "")
                    if backup_key and provider == "openai":
                        print(f"402 — switching to LLM_API_KEY_BACKUP ... ", end="", flush=True)
                        base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
                        client = OpenAI(base_url=base_url, api_key=backup_key, timeout=600.0)
                        try:
                            rewritten = rewrite_query(client, model, item["original"], v, provider)
                            row["rewrites"][v] = rewritten
                            print(f"done ({time.time() - t0:.1f}s)")
                            print(f"           → {rewritten[:90]}")
                        except Exception as retry_err:
                            compare_results.append(row)
                            compare_out_path.write_text(json.dumps({
                                "metadata": {"model": model, "n_per_dataset": args.n, "seed": args.seed,
                                             "datasets": args.datasets, "prompts": {k: {"system_prompt": VARIANT_SYSTEM.get(k, SYSTEM_PROMPT_DEFAULT), "user_prompt_template": PROMPT_VARIANTS[k]} for k in variants}},
                                "results": compare_results,
                            }, indent=2, ensure_ascii=False))
                            print(f"\n\nBackup key also failed: {retry_err}\nPartial results saved to {compare_out_path}")
                            sys.exit(1)
                    else:
                        compare_results.append(row)
                        compare_out_path.write_text(json.dumps({
                            "metadata": {"model": model, "n_per_dataset": args.n, "seed": args.seed,
                                         "datasets": args.datasets, "prompts": {k: {"system_prompt": VARIANT_SYSTEM.get(k, SYSTEM_PROMPT_DEFAULT), "user_prompt_template": PROMPT_VARIANTS[k]} for k in variants}},
                            "results": compare_results,
                        }, indent=2, ensure_ascii=False))
                        print(f"\n\n{e}\nPartial results saved to {compare_out_path}")
                        if not backup_key:
                            print("Tip: Add LLM_API_KEY_BACKUP=<new_HF_token> to .env to auto-switch when quota is depleted.")
                        sys.exit(1)
                except QuotaExhaustedError as e:
                    compare_results.append(row)
                    compare_out_path.write_text(json.dumps({
                        "metadata": {"model": model, "n_per_dataset": args.n, "seed": args.seed,
                                     "datasets": args.datasets, "prompts": {k: {"system_prompt": VARIANT_SYSTEM.get(k, SYSTEM_PROMPT_DEFAULT), "user_prompt_template": PROMPT_VARIANTS[k]} for k in variants}},
                        "results": compare_results,
                    }, indent=2, ensure_ascii=False))
                    print(f"\n\n{e}\nPartial results saved to {compare_out_path}")
                    sys.exit(1)
                except Exception as e:
                    row["rewrites"][v] = f"[ERROR: {e}]"
                    print(f"ERROR ({time.time() - t0:.1f}s): {e}")
                sys.stdout.flush()
            compare_results.append(row)

        out = {
            "metadata": {
                "model": model,
                "n_per_dataset": args.n,
                "seed": args.seed,
                "datasets": args.datasets,
                "prompts": {
                    k: {
                        "system_prompt": VARIANT_SYSTEM.get(k, SYSTEM_PROMPT_DEFAULT),
                        "user_prompt_template": PROMPT_VARIANTS[k],
                    }
                    for k in variants
                },
            },
            "results": compare_results,
        }
        compare_out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved comparison → {compare_out_path}")
        return

    # ── Single-prompt mode ───────────────────────────────────────────────────
    print(f"Prompt: {args.prompt}\n")
    if args.mode == "test":
        samples = sample_queries(args.datasets, benchmarks_dir, args.n, args.seed)
    else:
        src = benchmarks_dir if args.benchmarks_dir else (SUBSET_DIR if SUBSET_DIR.exists() else BENCHMARKS_DIR)
        samples = sample_queries(args.datasets, src, n_per_dataset=9999, seed=args.seed)

    print(f"Total queries to rewrite: {len(samples)}\n")

    results = []
    for i, item in enumerate(samples, 1):
        print(f"[{i:3d}/{len(samples)}] {item['dataset']} | {item['original'][:80]}...")
        try:
            rewritten = rewrite_query(client, model, item["original"], args.prompt, provider)
        except (QuotaExhaustedError, CreditsExhaustedError) as e:
            print(f"\n\n{e}\n")
            sys.exit(1)
        except Exception as e:
            print(f"         [ERROR] {e}")
            rewritten = ""
        item["rewritten"] = rewritten
        print(f"           → {rewritten[:100]}\n")
        results.append(item)

    out_path = RESULTS_DIR / f"query_rewrites_{args.mode}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} rewrites → {out_path}")

    print("\n" + "=" * 70)
    print(f"{'DATASET':<15} {'ORIGINAL':<50} {'REWRITTEN'}")
    print("=" * 70)
    for r in results:
        print(f"{r['dataset']:<15} {r['original'][:48]:<50} {r.get('rewritten','')[:60]}")


if __name__ == "__main__":
    main()
