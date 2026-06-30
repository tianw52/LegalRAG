"""Export `reglab/barexam_qa` and `reglab/housing_qa` to LegalBench-RAG on-disk layout.

Output layout (for :mod:`evaluation.LegalBenchRAG.loader` and ingestion)::

    <out_dir>/
        corpus/
            passages/*.txt      # barexam_qa only (hashed filenames)
            statutes/<State>/*.txt
                # housing_qa: jurisdiction subfolders (Section 5.2 — per-state pool)
        benchmarks/
            barexam_qa.json
            housing_qa.json

Each ground-truth unit is stored as one `.txt` file.  For **character-level**
LegalBench-style metrics use :mod:`evaluation.reglab.eval_recall`.  For **passage-level**
Recall@K / MRR@10 as in Zheng et al. (CS&Law 2025) use :mod:`evaluation.reglab.paper_eval`
after re-exporting with this script and ingesting with :mod:`evaluation.reglab.ingest_reglab`.

Query strings
-------------
* **barexam_qa**: ``prompt`` and ``question`` are concatenated (newline-separated) when
  ``prompt`` is non-empty, matching typical multi-part MBE prompts.
* **housing_qa**: questions use the dataset card prompt template with state and year 2021.

Corpus size
-----------
Full exports are large (~857k passages for barexam — train+validation+test; ~1.7M statutes for housing). Use
``--max-corpus-docs`` for smoke tests (writes the first *N* corpus rows in dataset order,
then keeps only QA items whose gold IDs lie in that set).

By default the corpus step iterates the **non-streaming** HuggingFace split so dataset scripts
are not required to open local TSV paths that may be missing on compute nodes. Set
``REGLAB_PREPARE_STREAMING=1`` to force streaming (only if those paths exist).

Dependencies::
    pip install -e ".[eval]"

Examples::

    python -m evaluation.reglab.prepare barexam_qa --out-dir data/reglab_eval/barexam_qa

    python -m evaluation.reglab.prepare housing_qa --out-dir data/reglab_eval/housing_qa

After a housing disk export, build Parquet shards over ``housing_qa/corpus`` (see
``tools/full_corpus_fast/run_build_housing_parquet.slurm``) before full GPU eval with
``scripts/run_reglab_housing_full_eval_array.slurm``.

    # smoke test
    python -m evaluation.reglab.prepare barexam_qa --out-dir data/reglab_eval/barexam_smoke \\
        --max-corpus-docs 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from evaluation.reglab.util import PREFIX_PASSAGES, corpus_relpath, statute_relpath
from evaluation.reglab.hf_reglab import iter_barexam_qa_rows, load_barexam_all_passages

logger = logging.getLogger(__name__)


def _require_datasets():  # pragma: no cover - import guard
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "Install the HuggingFace `datasets` package:  pip install -e '.[eval]'"
        ) from e
    return load_dataset


def _pick_split_name(ds_dict: dict, preferred: tuple[str, ...]) -> str:
    for k in preferred:
        if k in ds_dict:
            return k
    return next(iter(ds_dict.keys()))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_barexam_qa(out_dir: Path, max_corpus_docs: int | None) -> None:
    load_dataset = _require_datasets()

    out_corpus = out_dir / "corpus"
    out_bm = out_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_corpus.exists():
        shutil.rmtree(out_corpus)
    if out_bm.exists():
        shutil.rmtree(out_bm)
    out_corpus.mkdir(parents=True)
    out_bm.mkdir(parents=True)

    passages = load_barexam_all_passages(load_dataset, trust_remote_code=True)

    allowed: set[str] | None = set() if max_corpus_docs is not None else None
    n_written = 0

    # Streaming uses the dataset builder's local TSV paths; on cluster nodes those paths
    # often don't exist (different checkout / no HF download). Prefer the cached Dataset.
    passage_iter = passages
    if _env_flag("REGLAB_PREPARE_STREAMING"):
        logger.warning(
            "REGLAB_PREPARE_STREAMING=1 ignored for barexam_qa — full corpus requires "
            "concatenated train+validation+test splits."
        )

    for row in passage_iter:
        if max_corpus_docs is not None and n_written >= max_corpus_docs:
            break
        idx = str(row["idx"])
        text = row.get("text") or ""
        rel = corpus_relpath(PREFIX_PASSAGES, idx)
        _write_text(out_corpus / rel, text)
        if allowed is not None:
            allowed.add(idx)
        n_written += 1

    logger.info("Wrote %d passage files under %s", n_written, out_corpus)

    qa_meta = load_dataset("reglab/barexam_qa", "qa", trust_remote_code=True)

    tests: list[dict[str, Any]] = []
    for row in iter_barexam_qa_rows(qa_meta):
        gid = str(row["gold_idx"])
        if allowed is not None and gid not in allowed:
            continue
        rel = corpus_relpath(PREFIX_PASSAGES, gid)
        gold_path = out_corpus / rel
        if not gold_path.is_file():
            continue
        n_chars = len(gold_path.read_text(encoding="utf-8"))
        if n_chars == 0:
            logger.debug("Skipping QA with empty gold file: %s", gid)
            continue

        prompt = (row.get("prompt") or "").strip()
        question = (row.get("question") or "").strip()
        if prompt:
            qtext = f"{prompt}\n{question}"
        else:
            qtext = question

        tests.append({
            "query": qtext,
            "snippets": [{"file_path": rel, "span": [0, n_chars]}],
            "tags": ["barexam_qa"],
            "jurisdiction": None,
        })

    benchmark_path = out_bm / "barexam_qa.json"
    benchmark_path.write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    logger.info("Wrote %d benchmark tests → %s", len(tests), benchmark_path)


def prepare_housing_qa(out_dir: Path, max_corpus_docs: int | None) -> None:
    load_dataset = _require_datasets()

    out_corpus = out_dir / "corpus"
    out_bm = out_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_corpus.exists():
        shutil.rmtree(out_corpus)
    if out_bm.exists():
        shutil.rmtree(out_bm)
    out_corpus.mkdir(parents=True)
    out_bm.mkdir(parents=True)

    stat_meta = load_dataset("reglab/housing_qa", "statutes", trust_remote_code=True)
    split_name = _pick_split_name(stat_meta, ("corpus", "train", "validation", "test"))

    allowed: set[str] | None = set() if max_corpus_docs is not None else None
    n_written = 0
    statute_idx_to_state: dict[str, str] = {}

    stat_iter = stat_meta[split_name]
    if _env_flag("REGLAB_PREPARE_STREAMING"):
        try:
            stat_iter = load_dataset(
                "reglab/housing_qa",
                "statutes",
                split=split_name,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Streaming load failed (%s); using non-streaming split.", exc)
            stat_iter = stat_meta[split_name]

    for row in stat_iter:
        if max_corpus_docs is not None and n_written >= max_corpus_docs:
            break
        idx = row["idx"]
        text = row.get("text") or ""
        st = (row.get("state") or "").strip()
        rel = statute_relpath(st, idx)
        _write_text(out_corpus / rel, text)
        statute_idx_to_state[str(idx)] = st
        if allowed is not None:
            allowed.add(str(idx))
        n_written += 1

    logger.info("Wrote %d statute files under %s", n_written, out_corpus)

    q_meta = load_dataset("reglab/housing_qa", "questions", trust_remote_code=True)
    q_split_name = _pick_split_name(q_meta, ("test", "train", "validation"))
    q_rows = q_meta[q_split_name]

    tests: list[dict[str, Any]] = []
    for row in q_rows:
        state = (row.get("state") or "").strip()
        question = (row.get("question") or "").strip()
        qtext = (
            f"Consider statutory law for {state} in the year 2021.\n"
            f"{question}\n"
            'Answer "Yes" or "No".'
        )

        statutes = row.get("statutes") or []
        snippets: list[dict[str, Any]] = []
        ok = True
        for st in statutes:
            sid = st.get("statute_idx")
            if sid is None:
                continue
            sid_str = str(sid)
            if allowed is not None and sid_str not in allowed:
                ok = False
                break
            st_for_path = statute_idx_to_state.get(sid_str)
            if not st_for_path:
                ok = False
                break
            rel = statute_relpath(st_for_path, sid)
            gold_path = out_corpus / rel
            if not gold_path.is_file():
                ok = False
                break
            n_chars = len(gold_path.read_text(encoding="utf-8"))
            if n_chars == 0:
                ok = False
                break
            snippets.append({"file_path": rel, "span": [0, n_chars]})

        if not ok or not snippets:
            continue

        tests.append({
            "query": qtext,
            "snippets": snippets,
            "tags": ["housing_qa"],
            "jurisdiction": state,
        })

    benchmark_path = out_bm / "housing_qa.json"
    benchmark_path.write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    logger.info("Wrote %d benchmark tests → %s", len(tests), benchmark_path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export reglab HuggingFace datasets into LegalBench-RAG directory layout.",
    )
    p.add_argument(
        "dataset",
        choices=("barexam_qa", "housing_qa"),
        help="Which HuggingFace configuration to export.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Dataset root directory. Subdirs corpus/ and benchmarks/ are replaced.",
    )
    p.add_argument(
        "--max-corpus-docs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after writing N corpus documents (dataset order). "
            "QA rows whose gold IDs are not in this subset are dropped."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.dataset == "barexam_qa":
        prepare_barexam_qa(args.out_dir.resolve(), args.max_corpus_docs)
    else:
        prepare_housing_qa(args.out_dir.resolve(), args.max_corpus_docs)


if __name__ == "__main__":
    main(sys.argv[1:])
