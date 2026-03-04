"""Structured logging configuration using structlog + stdlib logging.

Log output
----------
- Console (stdout): human-readable coloured output via structlog ConsoleRenderer.
- File (logs/queries.log): plain-text append log; one JSON-like line per record.
  Each run is separated by a header line so individual runs are easy to find.

Call ``configure_logging()`` once at application startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "queries.log"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure stdlib + structlog for the application.

    Parameters
    ----------
    level:
        Logging level string (DEBUG / INFO / WARNING / ERROR).
    log_file:
        Path to the append log file.  Defaults to ``logs/queries.log``
        relative to the project root.  Pass ``None`` to disable file logging.
    """
    import structlog

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    target_file = log_file if log_file is not None else _LOG_FILE

    # ── Handlers ──────────────────────────────────────────────────────────────
    handlers: list[logging.Handler] = []

    # stdout – coloured, human-readable
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    handlers.append(stream_handler)

    # file – plain text, appended across runs
    if target_file is not None:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        # Simpler format for the file (no ANSI codes)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
        force=True,  # override any previous basicConfig call
    )

    # ── Third-party library noise → WARNING only ───────────────────────────────
    # httpx logs every HTTP request at INFO; opensearch-py does the same.
    # sentence_transformers emits model load details at INFO.
    # These are useful at DEBUG but clutter INFO output.
    for noisy_logger in (
        "httpx",
        "httpcore",
        "opensearch",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
