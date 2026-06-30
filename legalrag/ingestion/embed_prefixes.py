"""Model-specific embedding input formatting.

Prefixes are applied only when documented by the model card / official usage
docs.  Do not guess prefixes for models without explicit guidance.
"""

from __future__ import annotations

from typing import Literal

EmbedRole = Literal["query", "passage"]

# Qwen3-Embedding README: one-sentence English task instruction on queries only.
QWEN3_EMBEDDING_DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _model_key(model_name: str) -> str:
    return model_name.casefold().replace("_", "-")


def is_e5_model(model_name: str) -> bool:
    key = _model_key(model_name)
    return "e5-" in key or key.endswith("/e5") or "/e5-" in key or key.startswith("intfloat/e5")


def is_qwen3_embedding_model(model_name: str) -> bool:
    return "qwen3-embedding" in _model_key(model_name)


def is_octen_embedding_model(model_name: str) -> bool:
    return "octen-embedding" in _model_key(model_name)


def format_embedding_inputs(
    texts: list[str],
    model_name: str,
    *,
    role: EmbedRole | None = None,
    qwen3_task: str | None = None,
) -> list[str]:
    """Return texts with model-specific prefixes when *role* is set.

    * **E5** (intfloat/e5-*): ``query: `` / ``passage: `` — required per model card.
    * **Qwen3-Embedding**: ``Instruct: …\\nQuery:…`` on queries only; documents raw.
    * **Octen-Embedding**: ``"- "`` document prefix only (model-card workaround); queries raw.
    """
    if not texts or role is None:
        return texts

    if is_e5_model(model_name):
        prefix = "query: " if role == "query" else "passage: "
        return [prefix + t for t in texts]

    if is_qwen3_embedding_model(model_name):
        if role == "query":
            task = qwen3_task or QWEN3_EMBEDDING_DEFAULT_TASK
            return [f"Instruct: {task}\nQuery:{t}" for t in texts]
        return texts

    if is_octen_embedding_model(model_name):
        if role == "passage":
            # Octen model card: prepend "- " to documents to avoid Qwen3 upstream issue.
            return ["- " + t if not t.startswith("- ") else t for t in texts]
        return texts

    return texts
