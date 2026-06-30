"""Answer generator: prompts the LLM with retrieved context to produce a final answer.

System prompt and message templates live in:

    legalrag/prompts/generator.yaml

Edit that file to tune tone, citation style, or context formatting without
touching Python.

Context assembly
----------------
For each retrieved chunk we optionally expand to its parent chunk for richer
context (small-to-big retrieval pattern).  Parent expansion is done lazily
via the OpenSearch client to avoid loading all parents upfront.

Backends
--------
* ``LLMGenerator``     — Any OpenAI-compatible HTTP server (vLLM, Ollama, HF
                         Router, OpenAI, DashScope, …).  Configured via .env.
* ``LocalHFGenerator`` — Loads a HuggingFace causal-LM (e.g. Qwen2.5/Qwen3)
                         directly onto the local GPU.  No API server required.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseGenerator
from legalrag.core.models import RAGResponse, RetrievedChunk
from legalrag.opensearch.client import OpenSearchClient
from legalrag.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class LLMGenerator(BaseGenerator):
    """Generates answers using an LLM conditioned on retrieved context.

    Prompt configuration (system prompt, context template, user turn template)
    is loaded from ``legalrag/prompts/generator.yaml`` at construction time.
    """

    def __init__(
        self,
        os_client: OpenSearchClient | None = None,
        expand_to_parent: bool = True,
        max_context_chunks: int = 5,
        model: str | None = None,
    ) -> None:
        from legalrag.utils.llm_client import get_async_client, get_sync_client

        self._sync_client = get_sync_client()
        self._async_client = get_async_client()
        self._model = model or settings.llm.model
        self._os_client = os_client
        self._expand_to_parent = expand_to_parent
        self._max_context_chunks = max_context_chunks

        # Load prompt config once; cached by loader for the process lifetime
        self._prompt_cfg = load_prompt("generator")

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> RAGResponse:
        messages = self._build_messages(query, context_chunks)
        cfg = self._prompt_cfg.get("model_params", {})
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=cfg.get("temperature", 0.1),
        )
        answer = response.choices[0].message.content or ""
        logger.debug("Generated answer (%d chars)", len(answer))
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=context_chunks,
        )

    async def stream(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        messages = self._build_messages(query, context_chunks)
        cfg = self._prompt_cfg.get("model_params", {})
        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=cfg.get("temperature", 0.1),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> list[dict]:
        system_prompt: str = self._prompt_cfg["system"]
        context_str = self._build_context(context_chunks)
        user_turn: str = self._prompt_cfg["user_turn_template"].format(
            question=query,
            context=context_str,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
        ]

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        selected = chunks[: self._max_context_chunks]
        item_tpl: str = self._prompt_cfg["context_item_template"]
        separator: str = self._prompt_cfg.get("context_separator", "\n\n")
        parts: list[str] = []
        for i, rc in enumerate(selected, start=1):
            text = self._get_text(rc)
            parts.append(
                item_tpl.format(
                    i=i,
                    court=rc.chunk.metadata.court if rc.chunk.metadata else "unknown",
                    citation=rc.chunk.metadata.citation if rc.chunk.metadata else "unknown",
                    text=text,
                )
            )
        return separator.join(parts)

    def _get_text(self, rc: RetrievedChunk) -> str:
        """Return parent text if expand_to_parent is enabled, else child text."""
        if (
            self._expand_to_parent
            and self._os_client is not None
            and rc.chunk.parent_chunk_id
        ):
            parent = self._os_client.get_parent(rc.chunk.parent_chunk_id)
            if parent:
                return parent.get("text", rc.chunk.text)
        return rc.chunk.text


# ── Local HuggingFace generator ───────────────────────────────────────────────


class LocalHFGenerator(BaseGenerator):
    """Generates answers using a local HuggingFace causal LM loaded onto GPU.

    No external API server is required — the model is loaded directly via
    ``transformers`` and runs on the available device (CUDA GPU preferred).

    Supports any instruction-tuned model that implements a HuggingFace chat
    template, including:

    * ``Qwen/Qwen2.5-7B-Instruct``   (default, 7B, strong legal reasoning)
    * ``Qwen/Qwen2.5-14B-Instruct``  (14B, higher quality, needs ~28 GB VRAM)
    * ``Qwen/Qwen3-8B``              (latest Qwen3 series)
    * ``Qwen/Qwen3-14B``
    * ``meta-llama/Llama-3.1-8B-Instruct``

    Example usage::

        gen = LocalHFGenerator("Qwen/Qwen2.5-7B-Instruct")
        response = gen.generate(query, retrieved_chunks)
        print(response.answer)
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | int = "auto",
        os_client: OpenSearchClient | None = None,
        expand_to_parent: bool = True,
        max_context_chunks: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = True,
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        hf_offline: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_id:
            HuggingFace model ID or local path. Defaults to
            ``Qwen/Qwen2.5-7B-Instruct``.
        device:
            ``"auto"`` (recommended — picks GPU if available), ``"cuda"``,
            ``"cpu"``, or an integer GPU index.
        os_client:
            OpenSearch client for parent-chunk expansion (small-to-big).
            Pass ``None`` to disable parent expansion.
        expand_to_parent:
            When ``True`` and ``os_client`` is provided, retrieved child
            chunks are expanded to their parent chunk for richer context.
        max_context_chunks:
            How many top retrieved chunks to include in the prompt.
        max_new_tokens:
            Maximum number of tokens the model may generate.
        temperature:
            Sampling temperature. Values near 0 give deterministic outputs.
        do_sample:
            Enable nucleus / temperature sampling. Set to ``False`` for
            greedy decoding (fastest, deterministic).
        torch_dtype:
            Precision for model weights — ``"bfloat16"`` (default, fast on
            A100/H100), ``"float16"``, or ``"float32"``.
        load_in_8bit:
            Load the model in 8-bit via ``bitsandbytes`` quantization to
            reduce VRAM usage. Requires ``bitsandbytes`` installed.
        hf_offline:
            If ``True``, sets ``TRANSFORMERS_OFFLINE=1`` before loading so
            the model is loaded purely from local cache.
        """
        import os

        if hf_offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalHFGenerator requires 'transformers' and 'torch'. "
                "Install them with: pip install transformers torch"
            ) from exc

        self._model_id = model_id or self.DEFAULT_MODEL
        self._os_client = os_client
        self._expand_to_parent = expand_to_parent
        self._max_context_chunks = max_context_chunks
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample

        self._prompt_cfg = load_prompt("generator")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        _torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        logger.info("LocalHFGenerator: loading tokenizer '%s'", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        logger.info(
            "LocalHFGenerator: loading model '%s' (dtype=%s, 8bit=%s, device=%s)",
            self._model_id, torch_dtype, load_in_8bit, device,
        )
        load_kwargs: dict = {
            "trust_remote_code": True,
            "torch_dtype": _torch_dtype,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif device == "auto":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            **load_kwargs,
        )

        # Move model to device when device_map is not "auto"
        if device != "auto" and not load_in_8bit:
            self._device = torch.device(
                device if isinstance(device, str) else f"cuda:{device}"
            )
            self._model = self._model.to(self._device)
        else:
            self._device = next(self._model.parameters()).device

        self._model.eval()
        logger.info(
            "LocalHFGenerator: model loaded on %s — ready.",
            self._device,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> RAGResponse:
        """Generate an answer using the local model."""
        messages = self._build_messages(query, context_chunks)
        answer = self._run_inference(messages)
        logger.debug("LocalHFGenerator: generated answer (%d chars)", len(answer))
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=context_chunks,
        )

    async def stream(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Streaming generation via a thread executor (non-blocking)."""
        import asyncio

        messages = self._build_messages(query, context_chunks)

        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, self._run_inference, messages)
        # Yield in one shot — true token streaming requires TextIteratorStreamer
        # which is a forward-compatible extension left for future work.
        yield answer

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_inference(self, messages: list[dict]) -> str:
        import torch

        # Apply the model's built-in chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(
            [text],
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample and self._temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )

        # Decode only the newly generated tokens (skip the input prompt)
        input_len = model_inputs["input_ids"].shape[1]
        new_ids = generated_ids[0][input_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _build_messages(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> list[dict]:
        system_prompt: str = self._prompt_cfg["system"]
        context_str = self._build_context(context_chunks)
        user_turn: str = self._prompt_cfg["user_turn_template"].format(
            question=query,
            context=context_str,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
        ]

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        selected = chunks[: self._max_context_chunks]
        item_tpl: str = self._prompt_cfg["context_item_template"]
        separator: str = self._prompt_cfg.get("context_separator", "\n\n")
        parts: list[str] = []
        for i, rc in enumerate(selected, start=1):
            text = self._get_text(rc)
            parts.append(
                item_tpl.format(
                    i=i,
                    court=rc.chunk.metadata.court if rc.chunk.metadata else "unknown",
                    citation=rc.chunk.metadata.citation if rc.chunk.metadata else "unknown",
                    text=text,
                )
            )
        return separator.join(parts)

    def _get_text(self, rc: RetrievedChunk) -> str:
        """Return parent text if expand_to_parent is enabled, else child text."""
        if (
            self._expand_to_parent
            and self._os_client is not None
            and rc.chunk.parent_chunk_id
        ):
            parent = self._os_client.get_parent(rc.chunk.parent_chunk_id)
            if parent:
                return parent.get("text", rc.chunk.text)
        return rc.chunk.text

    @property
    def model_id(self) -> str:
        return self._model_id
