"""Per-exploration trace: what happened under the hood, step by step.

One ``Trace`` per call to ``ExplorationEngine.explore``. It records:
  * steps (formulate, retrieve, rerank, each view) with start offset + duration
    and step-specific data (query rewrite, filters, candidates + scores, …)
  * every LLM call: model, full prompts, raw response parts (text and any
    thinking), token usage, duration, validation retries, errors

Traces are kept in memory (``TraceStore``, recent N) so the UI can open them
while a run is in progress, and written to
``logs/explore_traces/<YYYY-MM-DD>/<HHMMSS>_<trace_id>.json`` when the run ends.

The same events are logged as they happen (logger ``legalrag.explore.trace``):
INFO = run start, each step with duration + key results, ranked rows, each LLM
call (model, time, tokens, retries, errors, thinking); DEBUG = full prompts and
raw LLM responses.  Every line is prefixed with ``[trace_id]``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACE_DIR = Path("logs/explore_traces")
_MAX_LOG_VALUE = 200  # chars per value in INFO log lines


def _compact(value: Any) -> str:
    """Short single-line rendering of a step field for INFO logs."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return f"<{len(value)} rows>"
    if isinstance(value, list):
        text = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, default=str)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= _MAX_LOG_VALUE else text[: _MAX_LOG_VALUE - 1] + "…"


class Trace:
    def __init__(self, question: str, profile: dict | None, scope: str) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now(UTC)
        self._t0 = time.perf_counter()
        self.data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "question": question,
            "profile": profile,
            "scope": scope,
            "status": "running",
            "steps": [],
            "llm_calls": [],
        }
        p = profile or {}
        logger.info(
            "[%s] explore start | scope=%s | province=%s familiarity=%s goal=%s | question=%s",
            self.trace_id, scope, p.get("province") or "-", p.get("familiarity") or "-",
            p.get("goal") or "-", _compact(question),
        )

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    @contextmanager
    def step(self, name: str, **data: Any) -> Iterator[dict[str, Any]]:
        """Time a step; the yielded dict is stored, so callers can add results to it."""
        record: dict[str, Any] = {"name": name, "start_s": self._elapsed(), **data}
        self.data["steps"].append(record)
        t = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            record["duration_s"] = round(time.perf_counter() - t, 3)
            fields = " | ".join(
                f"{k}={_compact(v)}"
                for k, v in record.items()
                if k not in ("name", "start_s", "duration_s", "error") and v not in (None, "", [])
            )
            log = logger.warning if "error" in record else logger.info
            log(
                "[%s] step %-26s %7.1fs%s%s",
                self.trace_id, name, record["duration_s"],
                f" | {fields}" if fields else "",
                f" | ERROR {record['error']}" if "error" in record else "",
            )

    def log_rows(self, title: str, rows: list[str]) -> None:
        """Log a ranked list (e.g. retrieval candidates) one row per line at INFO."""
        logger.info("[%s]   %s", self.trace_id, title)
        for row in rows:
            logger.info("[%s]     %s", self.trace_id, row)

    def add_llm_call(self, record: dict[str, Any]) -> None:
        self.data["llm_calls"].append({"start_s": self._elapsed(), **record})
        usage = record.get("usage") or {}
        retries = len(record.get("validation_retries") or [])
        log = logger.warning if record.get("error") else logger.info
        log(
            "[%s] LLM  %-26s %7.1fs | model=%s | tokens in=%s out=%s | retries=%d%s",
            self.trace_id, record.get("label"), record.get("duration_s", 0), record.get("model"),
            usage.get("input_tokens"), usage.get("output_tokens"), retries,
            f" | ERROR {record['error']}" if record.get("error") else "",
        )
        for reason in record.get("validation_retries") or []:
            logger.info("[%s]   retry reason: %s", self.trace_id, _compact(reason))
        for thought in record.get("thinking") or []:
            logger.info("[%s]   thinking: %s", self.trace_id, _compact(thought))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[%s]   system prompt:\n%s", self.trace_id, record.get("system_prompt"))
            logger.debug("[%s]   user prompt:\n%s", self.trace_id, record.get("user_prompt"))
            for text in record.get("responses") or []:
                logger.debug("[%s]   raw response:\n%s", self.trace_id, text)

    def finish(self, status: str = "done") -> Path | None:
        self.data["status"] = status
        self.data["total_s"] = self._elapsed()
        day = self.started_at.strftime("%Y-%m-%d")
        path = TRACE_DIR / day / f"{self.started_at.strftime('%H%M%S')}_{self.trace_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write trace %s: %s", path, exc)
            return None
        self.data["file"] = str(path)
        timings = ", ".join(
            f"{s['name']}={s.get('duration_s', '?')}s" for s in self.data["steps"]
        )
        logger.info("[%s] explore %s in %.1fs [%s] → %s", self.trace_id, status, self.data["total_s"], timings, path)
        return path


class TraceStore:
    """Recent traces in memory (live view in the UI) with disk fallback."""

    def __init__(self, max_items: int = 50) -> None:
        self._items: OrderedDict[str, Trace] = OrderedDict()
        self._max = max_items

    def add(self, trace: Trace) -> None:
        self._items[trace.trace_id] = trace
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        if trace_id in self._items:
            return self._items[trace_id].data
        matches = sorted(TRACE_DIR.glob(f"*/*_{trace_id}.json"))
        if matches:
            return json.loads(matches[-1].read_text(encoding="utf-8"))
        return None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Newest first: in-memory traces, then traces on disk."""
        out = [
            {k: t.data.get(k) for k in ("trace_id", "started_at", "question", "status", "total_s")}
            for t in reversed(self._items.values())
        ]
        seen = {o["trace_id"] for o in out}
        for path in sorted(TRACE_DIR.glob("*/*.json"), reverse=True):
            if len(out) >= limit:
                break
            trace_id = path.stem.split("_", 1)[-1]
            if trace_id in seen:
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            out.append({k: d.get(k) for k in ("trace_id", "started_at", "question", "status", "total_s")})
        return out[:limit]
