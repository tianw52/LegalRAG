"""served_model(): the checkpoint behind a shared served name (vLLM `root`)."""

import asyncio
from types import SimpleNamespace

from legalrag.utils import llm_client


class _FakeClient:
    def __init__(self, models=None, fail=False):
        self._models, self._fail = models or [], fail

    def with_options(self, **_):
        return self

    @property
    def models(self):
        return self

    async def list(self):
        if self._fail:
            raise ConnectionError("server down")
        return SimpleNamespace(data=self._models)


def _run(monkeypatch, client, name="qwen3"):
    monkeypatch.setattr(llm_client, "get_async_client", lambda: client)
    return asyncio.run(llm_client.served_model(name))


def test_vllm_root(monkeypatch) -> None:
    client = _FakeClient([SimpleNamespace(id="qwen3", root="Qwen/Qwen3-30B-A3B-GPTQ-Int4")])
    assert _run(monkeypatch, client) == "Qwen/Qwen3-30B-A3B-GPTQ-Int4"


def test_ollama_id_without_root(monkeypatch) -> None:
    assert _run(monkeypatch, _FakeClient([SimpleNamespace(id="qwen3:8b")]), "qwen3:8b") == "qwen3:8b"


def test_falls_back_when_server_cannot_say(monkeypatch) -> None:
    assert _run(monkeypatch, _FakeClient(fail=True)) == "qwen3"
    assert _run(monkeypatch, _FakeClient([SimpleNamespace(id="other", root="x")])) == "qwen3"
