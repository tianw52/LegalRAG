"""HuggingFace-backed RegLab corpus and benchmark loading (no on-disk export).

``metadata.citation`` and :class:`BenchmarkSnippet.file_path` both store the same
logical key so character- and passage-level eval can reuse disk-based scoring::

    barexam_qa:passage:<idx>           # idx = string ``row[\"idx\"]`` from passages config
    housing_qa:statute:<state>:<idx>   # state + statute idx from statutes config

OpenSearch chunk ``doc_id`` matches ``metadata.citation`` for stable IDs.

Requires ``datasets<3`` for dataset scripts (see ``pyproject.toml`` optional ``eval``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from legalrag.core.interfaces import BaseLoader
from legalrag.core.models import LegalDocumentMetadata, RawDocument

from evaluation.LegalBenchRAG.loader import BenchmarkSnippet, BenchmarkTestCase

logger = logging.getLogger(__name__)

RegLabHFDataset = Literal["barexam_qa", "housing_qa"]


def _load_dataset():
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit("Install `datasets` (pip install -e '.[eval]')") from e
    return load_dataset


def _pick_split(ds_dict: Any, preferred: tuple[str, ...]) -> str:
    for k in preferred:
        if k in ds_dict:
            return k
    return next(iter(ds_dict.keys()))


BAREXAM_PASSAGE_SPLITS = ("train", "validation", "test")
BAREXAM_QA_SPLITS = ("train", "validation", "test")


def load_barexam_all_passages(load_dataset_fn: Any, **load_kwargs: Any) -> Any:
    """Return all barexam_qa passage rows (train + validation + test concatenated).

    The HF dataset loader exposes three disjoint splits (~856,830 rows total).
    Using only ``train`` (~686,324 rows) omits ~20% of the public corpus.
    """
    from datasets import concatenate_datasets

    passage_ds = load_dataset_fn("reglab/barexam_qa", "passages", **load_kwargs)
    parts = [passage_ds[s] for s in BAREXAM_PASSAGE_SPLITS if s in passage_ds]
    if not parts:
        raise SystemExit("barexam_qa passages: no train/validation/test splits found")
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def iter_barexam_qa_rows(qa_ds: Any) -> Iterator[dict[str, Any]]:
    """Yield QA examples from all barexam_qa splits (train + validation + test)."""
    for split in BAREXAM_QA_SPLITS:
        if split in qa_ds:
            yield from qa_ds[split]


def barexam_passage_citation(idx: str) -> str:
    return f"barexam_qa:passage:{idx}"


def housing_statute_citation(state: str, idx: str) -> str:
    return f"housing_qa:statute:{state}:{idx}"


def barexam_passage_length_map(*, limit_rows: int | None = None) -> dict[str, int]:
    """Map passage ``idx`` → character length (full corpus unless ``limit_rows``)."""
    load_dataset = _load_dataset()
    passages = load_barexam_all_passages(load_dataset, trust_remote_code=True)
    column = passages.column_names
    if "idx" not in column or "text" not in column:
        raise SystemExit(f"Unexpected barexam passages schema: {column}")
    out: dict[str, int] = {}
    n = 0
    for row in passages:
        sid = str(row["idx"])
        out[sid] = len(row.get("text") or "")
        n += 1
        if limit_rows is not None and n >= limit_rows:
            break
    logger.info("barexam passages: built length map for %d rows", len(out))
    return out


def housing_statute_aux_maps(*, limit_rows: int | None = None) -> tuple[dict[str, int], dict[str, str]]:
    """Return ``(idx -> len(text), idx -> state)`` for housing statutes."""
    load_dataset = _load_dataset()
    ds = load_dataset("reglab/housing_qa", "statutes", trust_remote_code=True)
    split = _pick_split(ds, ("corpus", "train", "validation", "test"))
    col = ds[split].column_names
    if "idx" not in col or "text" not in col:
        raise SystemExit(f"Unexpected housing statutes schema: {col}")
    to_len: dict[str, int] = {}
    to_state: dict[str, str] = {}
    n = 0
    for row in ds[split]:
        sid = str(row["idx"])
        st = (row.get("state") or "").strip()
        to_len[sid] = len(row.get("text") or "")
        to_state[sid] = st
        n += 1
        if limit_rows is not None and n >= limit_rows:
            break
    logger.info("housing statutes: built maps for %d rows", len(to_len))
    return to_len, to_state


@dataclass
class HFLoadStats:
    source_mode: str = "hf"
    n_queries_loaded: int = 0
    n_gold_snippets: int = 0
    n_queries_missing_any_gold: int = 0
    n_corpus_rows_scanned_for_maps: int = 0


def load_barexam_hf_benchmark_tests(
    *,
    limit_queries: int | None = None,
    passage_map_limit: int | None = None,
) -> tuple[list[BenchmarkTestCase], HFLoadStats]:
    """Load QA rows from HF; spans are ``[0, len(passage)]`` from the passages table."""
    load_dataset = _load_dataset()
    stats = HFLoadStats()

    idx_to_len = barexam_passage_length_map(limit_rows=passage_map_limit)
    stats.n_corpus_rows_scanned_for_maps = len(idx_to_len)

    qa = load_dataset("reglab/barexam_qa", "qa", trust_remote_code=True)
    col = qa[BAREXAM_QA_SPLITS[0]].column_names
    for name in ("gold_idx", "question"):
        if name not in col:
            raise SystemExit(f"Unexpected barexam qa schema (missing {name}): {col}")

    tests: list[BenchmarkTestCase] = []
    missing = 0
    n_seen = 0

    for row in iter_barexam_qa_rows(qa):
        if limit_queries is not None and n_seen >= limit_queries:
            break
        n_seen += 1
        gid = str(row["gold_idx"])
        cite = barexam_passage_citation(gid)
        n_chars = idx_to_len.get(gid)
        if n_chars is None or n_chars == 0:
            missing += 1
            continue
        prompt = (row.get("prompt") or "").strip()
        question = (row.get("question") or "").strip()
        if prompt:
            qtext = f"{prompt}\n{question}"
        else:
            qtext = question
        tests.append(
            BenchmarkTestCase(
                query=qtext,
                snippets=[BenchmarkSnippet(file_path=cite, span=(0, n_chars))],
                tags=["barexam_qa"],
            )
        )

    stats.n_queries_loaded = len(tests)
    stats.n_gold_snippets = len(tests)
    stats.n_queries_missing_any_gold = missing
    if missing:
        logger.warning(
            "barexam HF: dropped %d QA rows (gold passage missing from passage map or empty)",
            missing,
        )
    return tests, stats


def load_housing_hf_benchmark_tests(
    *,
    limit_queries: int | None = None,
    statute_map_limit: int | None = None,
) -> tuple[list[BenchmarkTestCase], HFLoadStats]:
    load_dataset = _load_dataset()
    stats = HFLoadStats()
    to_len, to_state = housing_statute_aux_maps(limit_rows=statute_map_limit)
    stats.n_corpus_rows_scanned_for_maps = len(to_len)

    q_meta = load_dataset("reglab/housing_qa", "questions", trust_remote_code=True)
    split = _pick_split(q_meta, ("test", "train", "validation"))
    ds_split = q_meta[split]
    col = ds_split.column_names
    for name in ("state", "question", "statutes"):
        if name not in col:
            raise SystemExit(f"Unexpected housing questions schema (missing {name}): {col}")

    n = len(ds_split)
    if limit_queries is not None:
        n = min(n, limit_queries)

    tests: list[BenchmarkTestCase] = []
    missing = 0
    for i in range(n):
        row = ds_split[i]
        state = (row.get("state") or "").strip()
        question = (row.get("question") or "").strip()
        qtext = (
            f"Consider statutory law for {state} in the year 2021.\n"
            f"{question}\n"
            'Answer "Yes" or "No".'
        )
        snippets: list[BenchmarkSnippet] = []
        ok = True
        for st in row.get("statutes") or []:
            sid = st.get("statute_idx")
            if sid is None:
                continue
            sid_str = str(sid)
            st_for = to_state.get(sid_str)
            if not st_for:
                ok = False
                break
            cite = housing_statute_citation(st_for, sid_str)
            nch = to_len.get(sid_str, 0)
            if nch == 0:
                ok = False
                break
            snippets.append(BenchmarkSnippet(file_path=cite, span=(0, nch)))
        if not ok or not snippets:
            missing += 1
            continue
        extra_tags = ["housing_qa"]
        tests.append(
            BenchmarkTestCase(
                query=qtext,
                snippets=snippets,
                tags=extra_tags,
                jurisdiction=state,
            )
        )

    stats.n_queries_loaded = len(tests)
    stats.n_gold_snippets = sum(len(t.snippets) for t in tests)
    stats.n_queries_missing_any_gold = missing
    if missing:
        logger.warning("housing HF: dropped %d questions (incomplete gold statute lookup)", missing)
    return tests, stats


class RegLabHFCorpusLoader(BaseLoader):
    """Yield :class:`RawDocument` from HF reglab configs without writing ``.txt`` files."""

    def __init__(
        self,
        dataset: RegLabHFDataset,
        *,
        citation_filter: set[str] | None = None,
        limit_corpus: int | None = None,
    ) -> None:
        self._dataset = dataset
        self._citation_filter = citation_filter
        self._limit_corpus = limit_corpus

    def load(self, source: str = "") -> list[RawDocument]:
        return list(self.iter())

    def iter(self) -> Iterator[RawDocument]:
        load_dataset = _load_dataset()
        yielded = 0
        if self._dataset == "barexam_qa":
            ds = load_dataset("reglab/barexam_qa", "passages", trust_remote_code=True)
            split = _pick_split(ds, ("train", "validation", "test"))
            for row in ds[split]:
                idx = str(row["idx"])
                cite = barexam_passage_citation(idx)
                if self._citation_filter is not None and cite not in self._citation_filter:
                    continue
                text = row.get("text") or ""
                meta = LegalDocumentMetadata(
                    doc_id=cite,
                    source_path=f"hf://reglab/barexam_qa/passages/{idx}",
                    citation=cite,
                    extra={
                        "reglab_source": "hf",
                        "dataset": "barexam_qa",
                        "row_idx": idx,
                        "hf_idx": idx,
                    },
                )
                yield RawDocument(metadata=meta, text=text)
                yielded += 1
                if self._limit_corpus is not None and yielded >= self._limit_corpus:
                    break
        else:
            ds = load_dataset("reglab/housing_qa", "statutes", trust_remote_code=True)
            split = _pick_split(ds, ("corpus", "train", "validation", "test"))
            for row in ds[split]:
                idx = str(row["idx"])
                state = (row.get("state") or "").strip()
                cite = housing_statute_citation(state, idx)
                if self._citation_filter is not None and cite not in self._citation_filter:
                    continue
                text = row.get("text") or ""
                meta = LegalDocumentMetadata(
                    doc_id=cite,
                    source_path=f"hf://reglab/housing_qa/statutes/{state}/{idx}",
                    citation=cite,
                    court=state,
                    extra={
                        "reglab_source": "hf",
                        "dataset": "housing_qa",
                        "row_idx": idx,
                        "hf_idx": idx,
                        "state": state,
                        "jurisdiction": state,
                    },
                )
                yield RawDocument(metadata=meta, text=text)
                yielded += 1
                if self._limit_corpus is not None and yielded >= self._limit_corpus:
                    break

        logger.info(
            "RegLabHFCorpusLoader(%s): yielded %d documents (filter=%s, cap=%s)",
            self._dataset,
            yielded,
            "yes" if self._citation_filter else "no",
            self._limit_corpus,
        )


def citations_for_benchmark_tests(tests: list[BenchmarkTestCase]) -> set[str]:
    s: set[str] = set()
    for t in tests:
        for sn in t.snippets:
            s.add(sn.file_path)
    return s
