"""
LLM client factory.

The openai Python library is used purely as an HTTP client for any server
that speaks the OpenAI Chat Completions API spec.  We are NOT locked into
the OpenAI service – the library just handles auth headers, retries, and
streaming for us.

Default backend: local Ollama (see .env.example); lab vLLM and the HF Router
work the same way.

    Backend              | LLM_BASE_URL                                       | LLM_API_KEY / env
    ─────────────────────┼────────────────────────────────────────────────────┼──────────────────
    HF Router            | https://router.huggingface.co/v1                  | HF_TOKEN
    vLLM (local GPU)     | http://localhost:8000/v1                           | EMPTY (any string)
    Ollama (default)     | http://localhost:11434/v1                          | ollama
    DashScope (cloud)    | https://dashscope.aliyuncs.com/compatible-mode/v1  | sk-...
    OpenAI (cloud)       | https://api.openai.com/v1                          | sk-...

API key resolution order:
  1. LLM_API_KEY in .env / environment
  2. HF_TOKEN in environment  (convenient when using the HF Router)
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import AsyncOpenAI

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
def get_async_client() -> AsyncOpenAI:
    """Return a cached async OpenAI-compatible client."""
    return AsyncOpenAI(api_key=_resolve_api_key(), base_url=settings.llm.base_url)


async def served_model(model: str) -> str:
    """The checkpoint actually behind *model* on the LLM server.

    Servers can share one served name (lab and laptop vLLM both serve "qwen3"),
    so traces record what really ran: vLLM reports the checkpoint as ``root``
    in ``/v1/models`` (e.g. "Qwen/Qwen3-30B-A3B-GPTQ-Int4"); Ollama's id is
    already specific ("qwen3:8b").  Falls back to *model* if the server can't say.
    """
    try:
        page = await get_async_client().with_options(timeout=3, max_retries=0).models.list()
    except Exception:
        return model
    for m in page.data:
        if m.id == model:
            return getattr(m, "root", None) or m.id
    return model
