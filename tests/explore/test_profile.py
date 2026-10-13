"""Profile options (explore_policy.yaml) and location → jurisdiction filter."""

from legalrag.explore.profile import filter_province, profile_options
from legalrag.ingestion.canlegal_loader import _court_config


def test_province_codes_are_strings() -> None:
    """Unquoted YAML codes break silently (ON parses as True)."""
    for p in profile_options()["provinces"]:
        assert isinstance(p["code"], str), p


def test_province_codes_match_ingestion_jurisdictions() -> None:
    """Every province option must be a jurisdiction the index can contain."""
    courts, rules = _court_config()
    known = {j for _, j in courts.values()} | {j for _, j in rules}
    for p in profile_options()["provinces"]:
        if not p.get("all_of_canada"):
            assert p["code"] in known, p


def test_filter_province() -> None:
    assert filter_province("ON") == "ON"
    assert filter_province("BC") == "BC"
    for code in ("all", "outside", "undisclosed", "", "XX"):
        assert filter_province(code) is None
