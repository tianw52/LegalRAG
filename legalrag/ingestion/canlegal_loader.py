"""Loader for the UBC-VL/CanLegalRAGBench corpus (documents.json, JSON lines).

Unlike the CanLII .txt corpus there is no header block to parse: every field
(citation, name, court code, year, url) is already structured, so this loader
populates ``LegalDocumentMetadata`` directly and no metadata extractor or
``clean_document_text()`` pass is needed.

Court names and jurisdictions come from ``canlegal_courts.yaml`` (editable).

Dataset: https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench
"""

from __future__ import annotations

import functools
import json
import logging
import re
from datetime import date
from pathlib import Path

import yaml

from legalrag.core.interfaces import BaseLoader
from legalrag.core.models import LegalDocumentMetadata, RawDocument, doc_id_from_citation

logger = logging.getLogger(__name__)

HF_REPO_ID = "UBC-VL/CanLegalRAGBench"
_COURTS_YAML = Path(__file__).with_name("canlegal_courts.yaml")


@functools.lru_cache(maxsize=1)
def _court_config() -> tuple[dict[str, tuple[str, str]], list[tuple[re.Pattern, str]]]:
    cfg = yaml.safe_load(_COURTS_YAML.read_text(encoding="utf-8"))
    courts = {code: (name, juris) for code, (name, juris) in cfg["courts"].items()}
    rules = [(re.compile(pattern), juris) for pattern, juris in cfg["citation_rules"]]
    return courts, rules


UNKNOWN_JURISDICTION = "unknown"


def resolve_court(original_source: str, citation: str, name: str = "") -> tuple[str | None, str]:
    """Return (court name, jurisdiction code) for a dataset row.

    Unmatched rows get ``"unknown"`` (not None) so retrieval filters can opt to
    include them explicitly with a ``terms`` query.
    """
    courts, rules = _court_config()
    if original_source in courts:
        return courts[original_source]
    haystack = f"{citation} {name}"
    for pattern, juris in rules:
        if pattern.search(haystack):
            return None, juris
    return None, UNKNOWN_JURISDICTION


class CanLegalRAGBenchLoader(BaseLoader):
    """Load ``documents.json`` (one JSON object per line) into RawDocuments."""

    def load(self, source: str) -> list[RawDocument]:
        path = Path(source)
        if path.is_dir():
            path = path / "documents.json"
        if not path.is_file():
            raise FileNotFoundError(f"documents.json not found: {path}")

        docs: list[RawDocument] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    docs.append(self._to_document(json.loads(line)))
        logger.info("Loaded %d CanLegalRAGBench documents from %s", len(docs), path)
        return docs

    @staticmethod
    def _to_document(row: dict) -> RawDocument:
        citation = row["citation"].strip()
        code = row.get("original_source") or "UNKNOWN"
        court, jurisdiction = resolve_court(code, citation, row.get("name") or "")
        year = row.get("year")
        source_path = f"CanLegalRAGBench/{citation}"
        meta = LegalDocumentMetadata(
            doc_id=doc_id_from_citation(citation, source_path),
            source_path=source_path,
            doc_type="statute" if code == "STATUTE" else "case",
            court=court,
            court_abbrev=None if code in ("UNKNOWN", "STATUTE") else code,
            citation=citation,
            case_name=row.get("name"),
            year=year,
            # Dataset gives year only; year-precision date keeps range filters usable
            decision_date=date(year, 1, 1) if year else None,
            url=row.get("url") or None,
            jurisdiction=jurisdiction,
            extra={"dataset_source": row.get("dataset_source")},
        )
        return RawDocument(metadata=meta, text=row["text"])


def download_dataset(data_dir: str | Path) -> Path:
    """Fetch documents.json + queries.json from the HF Hub into *data_dir*."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("documents.json", "queries.json"):
        if not (data_dir / filename).exists():
            logger.info("Downloading %s/%s …", HF_REPO_ID, filename)
            hf_hub_download(
                HF_REPO_ID, filename, repo_type="dataset", local_dir=str(data_dir)
            )
    return data_dir
