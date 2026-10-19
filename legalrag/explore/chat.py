"""Ask panel: answer a follow-up question about one exploration, streamed.

The client sends the exploration's evidence (e1..eN) with each message, so
the server stays stateless and precomputed demos can be chatted with too.
Answers cite ``[eN]``; ids not in the evidence are removed from the final
text.  Prompt: ``chat`` in legalrag/prompts/explore_views.yaml.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from legalrag.core.config import settings
from legalrag.explore.profile import UserProfile
from legalrag.explore.schemas import Evidence
from legalrag.explore.views import _THINKING_OFF, format_evidence, prompt_context
from legalrag.prompts.loader import load_prompt
from legalrag.utils.llm_client import get_async_client

logger = logging.getLogger(__name__)

_MAX_HISTORY = 8                      # earlier turns sent back to the model
_CITE = re.compile(r"\[(e\d+(?:\s*,\s*e\d+)*)\]")
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)     # the exploration's question
    message: str = Field(min_length=1, max_length=2000)
    context: str = Field("", max_length=300)                 # "Sources · R v Smith · e2"
    history: list[ChatTurn] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, max_length=60)
    user_id: str | None = None


def clean_citations(text: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Drop cited ids that aren't in the evidence; return (text, cited ids in order)."""
    cited: list[str] = []

    def keep(match: re.Match) -> str:
        ids = [i.strip() for i in match.group(1).split(",") if i.strip() in valid_ids]
        cited.extend(i for i in ids if i not in cited)
        return f"[{', '.join(ids)}]" if ids else ""

    text = _CITE.sub(keep, _THINK.sub("", text))
    return re.sub(r"\s+([.,;:])", r"\1", text).strip(), cited


async def answer(req: ChatRequest, profile: UserProfile | None) -> AsyncIterator[dict[str, Any]]:
    """Yield {"event": "delta", "text"} … then {"event": "done", "answer", "evidence_ids"}."""
    passages = [e for e in req.evidence if e.kind == "passage"]
    ctx = prompt_context(req.question, profile, format_evidence(passages, len(passages)))
    system = load_prompt("explore_views")["chat"]["system"].format(
        **ctx, context=req.context or "the overview"
    )
    messages = [{"role": "system", "content": system}]
    messages += [{"role": t.role, "content": t.text} for t in req.history[-_MAX_HISTORY:]]
    messages.append({"role": "user", "content": req.message})

    llm_cfg = load_prompt("explore_policy")["llm"]
    client = get_async_client().with_options(timeout=llm_cfg["timeout"], max_retries=llm_cfg["http_retries"])
    extra = _THINKING_OFF[settings.llm.thinking_off]
    model = settings.explore.llm_model or settings.llm.model
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_output_tokens"],
        stream=True,
        **({"extra_body": extra} if extra else {}),
    )
    parts: list[str] = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            parts.append(delta)
            yield {"event": "delta", "text": delta}

    text, cited = clean_citations("".join(parts), {e.evidence_id for e in passages})
    logger.info(
        "chat %.1fs | model=%s | context=%s | cited=%s | message=%s",
        time.perf_counter() - t0, model, req.context, ",".join(cited), req.message[:120],
    )
    yield {"event": "done", "answer": text, "evidence_ids": cited}
