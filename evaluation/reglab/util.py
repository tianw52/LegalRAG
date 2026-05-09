"""Helpers for stable corpus paths and dataset I/O."""

from __future__ import annotations

import hashlib

PREFIX_PASSAGES = "passages"
PREFIX_STATUTES = "statutes"


def corpus_relpath(prefix: str, idx: str | int) -> str:
    """Return path relative to *corpus/* dir from an external corpus ID (SHA256 hex filename)."""
    h = hashlib.sha256(str(idx).encode("utf-8")).hexdigest()
    return f"{prefix}/{h}.txt"
