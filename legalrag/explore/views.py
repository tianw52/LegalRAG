"""View builders.

* Built-in views (no LLM): case landscape, case timeline — derived from
  retrieval metadata, so they appear as soon as retrieval finishes.
* LLM views: fact toggles (pre-defined), the planner (agent decides which
  ad-hoc views to build), and one builder per ad-hoc catalog kind.

Prompts: legalrag/prompts/explore_views.yaml.  Schemas: explore/schemas.py.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from legalrag.core.config import settings
from legalrag.explore.profile import UserProfile, profile_options
from legalrag.explore.schemas import (
    ADHOC_OUTPUTS,
    Evidence,
    ExplorationPlan,
    FactToggleOutput,
    PlannedView,
    View,
)
from legalrag.prompts.loader import load_prompt
from legalrag.query.formulator import _FormulatorOutput
from legalrag.utils.llm_client import _resolve_api_key

logger = logging.getLogger(__name__)


# ── Built-in views ────────────────────────────────────────────────────────────


def _documents(evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Group passage evidence by document, ordered by best rank."""
    docs: dict[str, dict[str, Any]] = {}
    for ev in evidence:
        if ev.kind != "passage" or not ev.doc_id:
            continue
        d = docs.get(ev.doc_id)
        if d is None:
            d = docs[ev.doc_id] = {
                "doc_id": ev.doc_id,
                "citation": ev.citation,
                "case_name": ev.case_name,
                "court": ev.court,
                "court_abbrev": ev.court_abbrev,
                "jurisdiction": ev.jurisdiction,
                "year": ev.year,
                "doc_type": ev.doc_type,
                "url": ev.url,
                "best_rank": ev.rank,
                "evidence_ids": [],
            }
        d["evidence_ids"].append(ev.evidence_id)
    return sorted(docs.values(), key=lambda d: d["best_rank"] or 0)


def case_landscape(evidence: list[Evidence]) -> View:
    docs = _documents(evidence)
    return View(
        view_id="case_landscape",
        kind="case_landscape",
        source="builtin",
        title="Sources found",
        description="Cases and statutes behind this exploration, grouped by jurisdiction and court.",
        spec={"documents": docs},
    )


def case_timeline(evidence: list[Evidence]) -> View:
    docs = [d for d in _documents(evidence) if d["year"]]
    return View(
        view_id="case_timeline",
        kind="case_timeline",
        source="builtin",
        title="Decisions over time",
        description="When the retrieved decisions and statutes were issued.",
        spec={"documents": sorted(docs, key=lambda d: d["year"])},
    )


# ── LLM plumbing ──────────────────────────────────────────────────────────────

_OUTPUT_MODES = {"prompted": PromptedOutput, "native": NativeOutput, "tool": ToolOutput}


def _agent(output_type: type[BaseModel], system_prompt: str) -> Agent:
    llm_cfg = load_prompt("explore_policy")["llm"]
    model = OpenAIChatModel(
        settings.explore.llm_model or settings.llm.model,
        provider=OpenAIProvider(base_url=settings.llm.base_url, api_key=_resolve_api_key()),
    )
    model_settings: dict[str, Any] = {
        "temperature": llm_cfg["temperature"],
        "timeout": llm_cfg["timeout"],
    }
    if llm_cfg.get("qwen3_no_think"):
        model_settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return Agent(
        model,
        output_type=_OUTPUT_MODES[llm_cfg["output_mode"]](output_type),
        system_prompt=system_prompt,
        model_settings=model_settings,
        output_retries=llm_cfg["retries"],
    )


async def _run(agent: Agent, user_prompt: str) -> Any:
    if load_prompt("explore_policy")["llm"].get("qwen3_no_think"):
        user_prompt += "\n/no_think"
    return (await agent.run(user_prompt)).output


async def formulate(question: str) -> _FormulatorOutput:
    """Query rewrite for retrieval, using legalrag/prompts/formulator.yaml."""
    return await _run(_agent(_FormulatorOutput, load_prompt("formulator")["system"]), question)


def format_evidence(evidence: list[Evidence], limit: int) -> str:
    lines = []
    for ev in evidence[:limit]:
        if ev.kind != "passage":
            continue
        head = " · ".join(
            str(x) for x in (ev.citation, ev.case_name, ev.court, ev.jurisdiction, ev.year) if x
        )
        lines.append(f"[{ev.evidence_id}] {head}\n{ev.text.strip()}")
    return "\n\n".join(lines)


def prompt_context(question: str, profile: UserProfile | None, evidence_text: str) -> dict[str, str]:
    opts = profile_options()
    fam = {f["key"]: f for f in opts["familiarity"]}
    goals = {g["key"]: g for g in opts["goals"]}
    provinces = {p["code"]: p["label"] for p in opts["provinces"]}
    p = profile
    return {
        "question": question,
        "evidence": evidence_text,
        "audience": fam.get(p.familiarity if p else "", fam["some"])["audience"],
        "goal_hint": goals.get(p.goal if p else "", goals["explore"])["planner_hint"],
        "interests": ", ".join(p.interests) if p and p.interests else "not specified",
        "province": provinces.get(p.province, "not specified") if p and p.province else "not specified",
    }


def ground(spec: Any, valid_ids: set[str]) -> dict[str, int]:
    """Strip evidence ids not in *valid_ids* (in place); count backed/unbacked items."""
    audit = {"backed_items": 0, "unbacked_items": 0, "invalid_ids_removed": 0}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "evidence_ids" in node:
                ids = [i for i in node["evidence_ids"] or [] if i in valid_ids]
                audit["invalid_ids_removed"] += len(node["evidence_ids"] or []) - len(ids)
                node["evidence_ids"] = ids
                audit["backed_items" if ids else "unbacked_items"] += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec)
    return audit


# ── LLM views ─────────────────────────────────────────────────────────────────


async def fact_toggle(ctx: dict[str, str], valid_ids: set[str]) -> View:
    prompts = load_prompt("explore_views")
    cfg = prompts["fact_toggle"]
    agent = _agent(FactToggleOutput, prompts["preamble"].format(**ctx) + "\n" + cfg["system"])
    out = await _run(agent, cfg["user"].format(**ctx))
    spec = out.model_dump()
    audit = ground(spec, valid_ids)
    return View(
        view_id="fact_toggle",
        kind="fact_toggle",
        source="llm",
        title="Facts that matter",
        description="Toggle facts to see what the sources say would change.",
        spec=spec,
        audit=audit,
    )


async def plan(ctx: dict[str, str]) -> ExplorationPlan:
    prompts = load_prompt("explore_views")
    catalog = "\n".join(
        f"- {kind}: {c['description']}" for kind, c in prompts["catalog"].items()
    )
    system = prompts["preamble"].format(**ctx) + "\n" + prompts["planner"]["system"].format(
        **ctx, catalog=catalog
    )
    out: ExplorationPlan = await _run(
        _agent(ExplorationPlan, system), prompts["planner"]["user"].format(**ctx)
    )
    out.views = out.views[:2]
    return out


async def build_adhoc(
    planned: PlannedView, index: int, ctx: dict[str, str], valid_ids: set[str]
) -> View:
    prompts = load_prompt("explore_views")
    entry = prompts["catalog"][planned.kind]
    system = prompts["preamble"].format(**ctx) + "\n" + entry["system"]
    user = prompts["builder_user"].format(
        **ctx, view_title=planned.title, view_purpose=planned.purpose, view_focus=planned.focus
    )
    out = await _run(_agent(ADHOC_OUTPUTS[planned.kind], system), user)
    renderer = entry["renderer"]
    body = out.model_dump()
    # Shape into the Demo renderer's spec format
    spec: dict[str, Any] = {"type": renderer}
    if renderer == "network":
        spec.update(layout="hierarchical", items=[], network=body)
    else:
        spec.update(body)
    audit = ground(spec, valid_ids)
    return View(
        view_id=f"adhoc_{index}",
        kind=planned.kind,
        source="agent",
        title=planned.title,
        description=planned.purpose,
        renderer=renderer,
        spec=spec,
        audit=audit,
    )
