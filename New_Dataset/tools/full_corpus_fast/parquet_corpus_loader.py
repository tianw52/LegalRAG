"""Load RegLab/LegalBench-style corpus from Parquet (citation + text).

Supports a **single file** ``passages.parquet`` or a **shard directory** containing
``passages_part_*.parquet`` files (the output of ``build_passages_parquet.py --output-dir``).

The ``citation`` column must be the relative path under the corpus root
(e.g. ``passages/....txt`` for BarExam or ``statutes/Alabama/....txt`` for Housing)
so that it aligns with the benchmark JSON ``file_path`` field and OpenSearch filters.
For ``statutes/<State>/...`` paths, ``metadata.court`` is populated from the state folder
name (mirroring :class:`~evaluation.reglab.corpus_loader.RegLabCorpusLoader`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from legalrag.core.interfaces import BaseLoader
from legalrag.core.models import LegalDocumentMetadata, RawDocument, stable_id

logger = logging.getLogger(__name__)

_PART_RE = re.compile(r"^passages_part_(\d+)\.parquet$")


def _parquet_shards(parquet_path: Path) -> list[Path]:
    parts: list[tuple[int, Path]] = []
    for p in parquet_path.glob("passages_part_*.parquet"):
        m = _PART_RE.match(p.name)
        if m:
            parts.append((int(m.group(1)), p))
    parts.sort(key=lambda x: x[0])
    return [p for _, p in parts]


class ParquetCorpusLoader(BaseLoader):
    """Stream documents from Parquet built by ``build_passages_parquet.py``."""

    def __init__(self, parquet_path: str | Path, *, read_batch_rows: int = 4096) -> None:
        self._path = Path(parquet_path)
        self._read_batch_rows = read_batch_rows
        if self._path.is_dir():
            self._files = _parquet_shards(self._path)
            if not self._files:
                raise FileNotFoundError(
                    f"No passages_part_*.parquet under {self._path}"
                )
        elif self._path.is_file():
            self._files = [self._path]
        else:
            raise FileNotFoundError(self._path)

    def load(self, source: str = "") -> list[RawDocument]:
        return list(self.iter())

    def iter(self):
        import pyarrow.parquet as pq

        count = 0
        for pf_path in self._files:
            pf = pq.ParquetFile(pf_path)
            label = pf_path.name if self._path.is_dir() else pf_path.name
            for batch in pf.iter_batches(batch_size=self._read_batch_rows):
                citations = batch.column("citation").to_pylist()
                texts = batch.column("text").to_pylist()
                for cit, text in zip(citations, texts):
                    if text is None:
                        text = ""
                    c = str(cit)
                    doc_id = stable_id("legalbenchrag", c)
                    court: str | None = None
                    parts = c.split("/")
                    if len(parts) >= 3 and parts[0] == "statutes":
                        court = parts[1]

                    meta = LegalDocumentMetadata(
                        doc_id=doc_id,
                        source_path=f"parquet:{label}:{c}",
                        citation=c,
                        court=court,
                    )
                    yield RawDocument(metadata=meta, text=str(text))
                    count += 1

        logger.info(
            "ParquetCorpusLoader: yielded %d rows from %s",
            count,
            self._path,
        )
