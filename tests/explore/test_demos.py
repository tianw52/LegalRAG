"""Home page example questions (explore_examples.yaml) and Ask-panel citation cleanup."""

from legalrag.explore.chat import clean_citations
from legalrag.explore.demos import all_demo_ids, example_ids
from legalrag.explore.profile import profile_options
from legalrag.prompts.loader import load_prompt


def test_province_gets_two_local_then_two_federal() -> None:
    cfg = load_prompt("explore_examples")
    picked = example_ids("BC")
    assert picked == [(q, "local") for q in cfg["provinces"]["BC"]] + [(q, "federal") for q in cfg["federal"]]


def test_no_province_gets_ontario_bc_and_federal() -> None:
    cfg = load_prompt("explore_examples")
    expected = [(q, "local") for q in cfg["default"]] + [(q, "federal") for q in cfg["federal"]]
    for code in ("all", "outside", "undisclosed", "", "MB", "QC", "NU"):
        assert example_ids(code) == expected, code
    assert set(cfg["default"]) == {cfg["provinces"]["ON"][0], cfg["provinces"]["BC"][0]}


def test_every_location_has_four_distinct_questions() -> None:
    for p in profile_options()["provinces"]:
        ids = [q for q, _ in example_ids(p["code"])]
        assert len(ids) == 4 and len(set(ids)) == 4, p["code"]


def test_config_province_codes_are_strings() -> None:
    """Unquoted YAML keys break silently (ON parses as True)."""
    assert all(isinstance(code, str) for code in load_prompt("explore_examples")["provinces"])


def test_all_demo_ids_are_unique() -> None:
    ids = [q for q, _ in all_demo_ids()]
    assert len(ids) == len(set(ids)) == 22


def test_clean_citations_drops_unknown_ids() -> None:
    text, cited = clean_citations("Notice is required [e3]. Courts disagree [e9, e2]. Also [e7] .", {"e2", "e3"})
    assert text == "Notice is required [e3]. Courts disagree [e2]. Also."
    assert cited == ["e3", "e2"]


def test_clean_citations_strips_thinking() -> None:
    text, cited = clean_citations("<think>hmm</think>Yes [e1].", {"e1"})
    assert (text, cited) == ("Yes [e1].", ["e1"])
