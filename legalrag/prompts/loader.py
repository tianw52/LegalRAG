"""YAML prompt loader with file-level caching.

Loads prompt configuration from ``legalrag/prompts/<name>.yaml`` and caches
the parsed result so repeated accesses within a process pay no I/O cost.

Usage
-----
    from legalrag.prompts.loader import load_prompt

    cfg = load_prompt("formulator")
    system_prompt: str = cfg["system"]
    temperature: float = cfg["model_params"]["temperature"]
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, Any]:
    """Return the parsed YAML config for *name* (filename without extension).

    Results are cached for the lifetime of the process; restart the application
    or call ``load_prompt.cache_clear()`` to pick up edits during development.

    Raises
    ------
    FileNotFoundError
        If ``legalrag/prompts/<name>.yaml`` does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
