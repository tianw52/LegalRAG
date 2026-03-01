"""
LLM client factory.

The openai Python library is used purely as an HTTP client for any server
that speaks the OpenAI Chat Completions API spec.  We are NOT locked into
the OpenAI service – the library just handles auth headers, retries, and
streaming for us.

Default backend: Qwen3.5-397B-A17B via HuggingFace Inference Router (Novita).
The HF Router proxies your request to a third-party GPU provider; you only
need a HuggingFace token (free tier available).

    Backend              | LLM_BASE_URL                                       | LLM_API_KEY / env
    ─────────────────────┼────────────────────────────────────────────────────┼──────────────────
    HF Router (default)  | https://router.huggingface.co/v1                  | HF_TOKEN
    vLLM (local GPU)     | http://localhost:8000/v1                           | EMPTY (any string)
    Ollama (local)       | http://localhost:11434/v1                          | ollama
    DashScope (cloud)    | https://dashscope.aliyuncs.com/compatible-mode/v1  | sk-...
    OpenAI (cloud)       | https://api.openai.com/v1                          | sk-...

API key resolution order:
  1. LLM_API_KEY in .env / environment
  2. HF_TOKEN in environment  (convenient when using the HF Router)
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI

from legalrag.core.config import settings


def _resolve_api_key() -> str:
    """Return LLM_API_KEY, falling back to HF_TOKEN if set."""
    key = settings.llm.api_key
    if key == "EMPTY":
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            return hf_token
    return key


@lru_cache(maxsize=1)
def get_sync_client() -> OpenAI:
    """Return a cached synchronous OpenAI-compatible client."""
    return OpenAI(api_key=_resolve_api_key(), base_url=settings.llm.base_url)


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    """Return a cached async OpenAI-compatible client."""
    return AsyncOpenAI(api_key=_resolve_api_key(), base_url=settings.llm.base_url)
