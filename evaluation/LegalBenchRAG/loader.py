"""LegalBench-RAG corpus loader and benchmark reader.

Data layout (after downloading from Dropbox):

    data/LegalBenchRAG/
        corpus/                   # raw text files (nested sub-dirs allowed)
            contractnli/
                *.txt
            cuad/
                *.txt
            maud/
                *.txt
            privacy_qa/
                *.txt
        benchmarks/               # one JSON per sub-benchmark
            contractnli.json
            cuad.json
            maud.json
            privacy_qa.json

Benchmark JSON schema
---------------------
Each file is a JSON object matching::

    {
        "tests": [
            {
                "query": "...",
                "snippets": [
                    {
                        "file_path": "cuad/NNN.txt",   # relative to corpus/
                        "span": [char_start, char_end]  # half-open [start, end)
                    },
                    ...
                ],
                "tags": ["cuad"]   # optional
            },
            ...
        ]
    }

LegalBenchRAGCorpusLoader
    Discovers corpus files and yields one RawDocument per file.
    ``source`` is the root data dir (e.g. ``data/LegalBenchRAG``).
    File path relative to ``corpus/`` is stored in ``metadata.citation``
    so we can look it up during evaluation.

LegalBenchRAGBenchmarkReader
    Reads the four benchmark JSON files and returns flat list of test cases.
    Does NOT depend on OpenSearch — just pure data loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from legalrag.core.interfaces import BaseLoader, BaseMetadataExtractor
from legalrag.core.models import LegalDocumentMetadata, RawDocument, stable_id

logger = logging.getLogger(__name__)

# ── Benchmark types (self-contained; no dependency on legalbenchrag package) ──


class BenchmarkSnippet(BaseModel):
    """A ground-truth snippet: which file and which character span."""

    file_path: str               # relative to corpus/
    span: tuple[int, int]        # [char_start, char_end)


class BenchmarkTestCase(BaseModel):
    """One query with its ground-truth snippet(s)."""

    query: str
    snippets: list[BenchmarkSnippet]
    tags: list[str] = []


# ── Corpus loader ─────────────────────────────────────────────────────────────


class LegalBenchRAGCorpusLoader(BaseLoader):
    """Stream all corpus text files as RawDocuments.

    Parameters
    ----------
    corpus_dir:
        Absolute or relative path to the ``corpus/`` folder inside the
        downloaded LegalBench-RAG data directory.
    file_paths:
        Optional explicit list of relative file paths (relative to
        ``corpus_dir``) to load.  When given, only those files are loaded
        (useful when you only want to ingest the documents needed for a
        specific benchmark subset).  When ``None``, all ``*.txt`` files
        under ``corpus_dir`` are discovered.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        file_paths: list[str] | None = None,
    ) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._file_paths = file_paths

    def load(self, source: str = "") -> list[RawDocument]:
        """Return all corpus documents as a list (may be large)."""
        return list(self.iter())

    def iter(self):
        """Yield RawDocuments one at a time — constant memory."""
        if self._file_paths is not None:
            paths = [self._corpus_dir / fp for fp in self._file_paths]
        else:
            paths = sorted(self._corpus_dir.rglob("*.txt"))

        count = 0
        for path in paths:
            if not path.is_file():
                logger.warning("Corpus file not found: %s", path)
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)
                continue

            # file_path is relative to corpus_dir — used as the lookup key
            rel_path = str(path.relative_to(self._corpus_dir))
            doc_id = stable_id("legalbenchrag", rel_path)

            meta = LegalDocumentMetadata(
                doc_id=doc_id,
                source_path=str(path),
                citation=rel_path,   # relative path stored as citation for lookup
            )
            yield RawDocument(metadata=meta, text=text)
            count += 1

        logger.info(
            "LegalBenchRAGCorpusLoader: yielded %d documents from %s",
            count,
            self._corpus_dir,
        )


# ── Benchmark reader ──────────────────────────────────────────────────────────

_BENCHMARK_NAMES = ("contractnli", "cuad", "maud", "privacy_qa")


def load_benchmark(
    benchmarks_dir: str | Path,
    names: list[str] | None = None,
    limit_per_benchmark: int | None = None,
) -> list[BenchmarkTestCase]:
    """Read benchmark JSON files and return a flat list of test cases.

    Parameters
    ----------
    benchmarks_dir:
        Path to the ``benchmarks/`` folder (contains ``*.json`` files).
    names:
        Subset of benchmark names to load.  Defaults to all four:
        ``contractnli``, ``cuad``, ``maud``, ``privacy_qa``.
    limit_per_benchmark:
        Cap the number of test cases loaded per benchmark file.  Useful for
        quick iteration / smoke tests.  ``None`` loads everything.
    """
    benchmarks_dir = Path(benchmarks_dir)
    names = names or list(_BENCHMARK_NAMES)

    all_tests: list[BenchmarkTestCase] = []
    for name in names:
        json_path = benchmarks_dir / f"{name}.json"
        if not json_path.exists():
            logger.warning("Benchmark file not found: %s — skipping", json_path)
            continue
        with open(json_path, encoding="utf-8") as fh:
            raw = json.load(fh)

        tests_raw = raw.get("tests", [])
        if limit_per_benchmark is not None:
            tests_raw = tests_raw[:limit_per_benchmark]

        for t in tests_raw:
            # Normalise tag: always include the benchmark name
            tags = list(t.get("tags", []))
            if name not in tags:
                tags.append(name)
            all_tests.append(
                BenchmarkTestCase(
                    query=t["query"],
                    snippets=[
                        BenchmarkSnippet(
                            file_path=s["file_path"],
                            span=(s["span"][0], s["span"][1]),
                        )
                        for s in t.get("snippets", [])
                    ],
                    tags=tags,
                )
            )
        logger.info("Loaded %d tests from %s", len(tests_raw), json_path)

    logger.info("Total test cases loaded: %d", len(all_tests))
    return all_tests


def corpus_file_paths_for_tests(tests: list[BenchmarkTestCase]) -> list[str]:
    """Return sorted deduplicated list of corpus file paths needed by *tests*."""
    paths: set[str] = set()
    for test in tests:
        for snippet in test.snippets:
            paths.add(snippet.file_path)
    return sorted(paths)


# ── Passthrough metadata extractor ────────────────────────────────────────────


class PassthroughExtractor(BaseMetadataExtractor):
    """No-op extractor — LegalBench-RAG docs have no CanLII-style header."""

    def extract(self, document: RawDocument) -> RawDocument:
        return document
