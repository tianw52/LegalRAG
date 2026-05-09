#!/usr/bin/env python3
"""Print REPO_ROOT and whether a Gemini API key is visible (no secret printed). Run from LegalRAG root."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rw", ROOT / "scripts" / "rewrite_benchmark_50_gemini.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)

mod._load_dotenv()
k = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or mod._api_key_from_env_file()
    or ""
).strip()
print("REPO_ROOT:", mod.REPO_ROOT)
print("cwd:", Path.cwd())
print("key_len:", len(k), "OK" if len(k) > 20 else "EMPTY — put GOOGLE_API_KEY in .env here or under REPO_ROOT")
