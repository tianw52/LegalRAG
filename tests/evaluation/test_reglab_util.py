from evaluation.reglab.util import corpus_relpath


def test_corpus_relpath_stable() -> None:
    assert corpus_relpath("passages", "abc") == corpus_relpath("passages", "abc")
    assert corpus_relpath("passages", 42) == corpus_relpath("passages", "42")
    p1 = corpus_relpath("passages", "x")
    p2 = corpus_relpath("statutes", "x")
    assert p1.startswith("passages/")
    assert p2.startswith("statutes/")
    assert p1 != p2
