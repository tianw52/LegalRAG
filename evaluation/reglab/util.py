"""Helpers for stable corpus paths and dataset I/O."""

from __future__ import annotations

import hashlib

PREFIX_PASSAGES = "passages"
PREFIX_STATUTES = "statutes"


def corpus_relpath(prefix: str, idx: str | int) -> str:
    """Return path relative to *corpus/* dir from an external corpus ID (SHA256 hex filename)."""
    h = hashlib.sha256(str(idx).encode("utf-8")).hexdigest()
    return f"{prefix}/{h}.txt"


def statute_relpath(state: str, idx: str | int) -> str:
    """Housing corpus path with jurisdiction folder (Section 5.2 — per-state retrieval pool)."""
    h = hashlib.sha256(str(idx).encode("utf-8")).hexdigest()
    st = (state or "unknown").strip().replace("/", "_")
    return f"{PREFIX_STATUTES}/{st}/{h}.txt"
