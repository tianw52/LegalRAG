"""View builders, with the LLM call stubbed out."""

from legalrag.explore import views
from legalrag.explore.schemas import Fact, FactToggleOutput


def _fact(text: str, status: str) -> Fact:
    return Fact(text=text, status=status, why_it_matters="w", if_changed="c", evidence_ids=["q"])


async def test_fact_toggle_ids_are_unique(monkeypatch):
    # Several facts share a status; each toggle must still have its own id.
    out = FactToggleOutput(
        key_question="k",
        facts=[_fact("a", "stated"), _fact("b", "stated"), _fact("c", "unknown")],
    )

    async def fake_run(agent, prompt, trace, label):
        return out

    monkeypatch.setattr(views, "_agent", lambda *a, **k: None)
    monkeypatch.setattr(views, "_run", fake_run)
    ctx = {"question": "q", "evidence": "", "audience": "", "goal_hint": "", "interests": "", "province": ""}

    view = await views.fact_toggle(ctx, {"q"})

    assert [f["id"] for f in view.spec["facts"]] == ["f1", "f2", "f3"]
