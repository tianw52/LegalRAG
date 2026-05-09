from evaluation.reglab.paper_metrics import (
    mrr_at_cutoff,
    ranked_passage_citations,
    recall_at_k_lower,
    recall_at_k_upper,
)
from evaluation.reglab.util import corpus_relpath, statute_relpath


def test_corpus_relpath_stable() -> None:
    assert corpus_relpath("passages", "abc") == corpus_relpath("passages", "abc")
    assert corpus_relpath("passages", 42) == corpus_relpath("passages", "42")
    p1 = corpus_relpath("passages", "x")
    p2 = corpus_relpath("statutes", "x")
    assert p1.startswith("passages/")
    assert p2.startswith("statutes/")
    assert p1 != p2


def test_statute_relpath_includes_state() -> None:
    p = statute_relpath("Alabama", 99)
    assert p.startswith("statutes/Alabama/")
    assert p.endswith(".txt")


def test_passage_dedupe_order() -> None:
    class M:
        def __init__(self, citation: str) -> None:
            self.citation = citation

    class C:
        def __init__(self, citation: str) -> None:
            self.metadata = M(citation)

    class R:
        def __init__(self, citation: str) -> None:
            self.chunk = C(citation)

    order = ranked_passage_citations(
        [R("a"), R("a"), R("b"), R("a"), R("c")]
    )
    assert order == ["a", "b", "c"]


def test_recall_and_mrr() -> None:
    rank = ["x", "gold", "y", "z"]
    gold = {"gold"}
    assert recall_at_k_upper(rank, gold, 1) == 0.0
    assert recall_at_k_upper(rank, gold, 2) == 1.0
    assert mrr_at_cutoff(rank, gold, 10) == 0.5

    gold2 = {"a", "b"}
    rank2 = ["a", "z", "b"]
    assert recall_at_k_upper(rank2, gold2, 1) == 1.0
    assert recall_at_k_lower(rank2, gold2, 2) == 0.0
    assert recall_at_k_lower(rank2, gold2, 3) == 1.0
