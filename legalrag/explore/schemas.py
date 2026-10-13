"""Schemas for the exploration engine.

Two groups:
  * LLM output schemas — what each view-generation call must return.  Field
    names match Demo/js/viz-renderers.js so the specs render unchanged.
    Field descriptions are sent to the LLM as part of the JSON schema.
  * API models — evidence, documents and the view envelope streamed to the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceIds = list[str]
_EV_DESC = 'Evidence ids supporting this item, e.g. ["e1", "e4"], or ["q"] if only from the question.'


# ── API models ────────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """One retrieved passage the views can cite (e1..eN), or the question (q)."""

    evidence_id: str
    kind: Literal["passage", "question"] = "passage"
    text: str
    rank: int | None = None
    rerank_score: float | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    citation: str | None = None
    case_name: str | None = None
    court: str | None = None
    court_abbrev: str | None = None
    jurisdiction: str | None = None
    year: int | None = None
    doc_type: str | None = None
    url: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class View(BaseModel):
    """Envelope for every view streamed to the UI.

    ``kind`` selects the React component; for agent-built views ``renderer``
    names the Demo renderer type and ``spec`` is passed to it unchanged.
    """

    view_id: str
    kind: str
    source: Literal["builtin", "llm", "agent"]
    title: str
    description: str = ""
    renderer: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)


# ── LLM output: query rewrite (prompt: formulator.yaml) ───────────────────────


class QueryRewriteOutput(BaseModel):
    """Search-ready rewrite of the user's question."""

    reformulated_query: str = Field(
        description="Concise, precise version of the question optimised for semantic search."
    )
    lexical_keywords: list[str] = Field(
        default_factory=list,
        description="Legal terms and keywords for BM25 retrieval.",
    )
    court_filter: str | None = Field(
        default=None,
        description="Exact court name filter; null when not specified.",
    )
    citation_filter: str | None = Field(
        default=None,
        description="Neutral citation filter (e.g. '2010 BCCA 220'); null when not specified.",
    )
    date_from: str | None = Field(
        default=None,
        description="Start of date range in YYYY-MM-DD format; null when not specified.",
    )
    date_to: str | None = Field(
        default=None,
        description="End of date range in YYYY-MM-DD format; null when not specified.",
    )


# ── LLM output: fact toggles (pre-defined view) ───────────────────────────────


class Fact(BaseModel):
    # No id field: small models fill it with the status ("stated", "unknown"), so ids
    # collide and the UI toggles every fact with that status at once.  views.fact_toggle
    # assigns f1..fN instead.
    text: str = Field(description="The fact, one short sentence.")
    status: Literal["stated", "assumed", "unknown"] = Field(
        description="stated = user said it; assumed = implied; unknown = decisive but not mentioned."
    )
    why_it_matters: str = Field(description="One sentence: why the sources treat this fact as relevant.")
    if_changed: str = Field(description="One sentence: what changes if this fact were different or false.")
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class FactToggleOutput(BaseModel):
    key_question: str = Field(description="The central legal question, phrased neutrally.")
    facts: list[Fact]


# ── LLM output: planner ───────────────────────────────────────────────────────

AdHocKind = Literal[
    "concept_map", "comparison", "legal_test", "rule_hierarchy", "process_flow", "scenario_timeline"
]


class PlannedView(BaseModel):
    kind: AdHocKind
    title: str = Field(description="Short title shown above the view.")
    purpose: str = Field(description="One sentence: what the user will understand from this view.")
    focus: str = Field(description="What the view should contain.")
    why_useful: str = Field(description="One sentence: what this view adds beyond the built-in views.")


class ExplorationPlan(BaseModel):
    topic: str = Field(description="Topic in 2-6 words.")
    orientation: str = Field(description="At most two neutral sentences describing what the sources cover.")
    extra_views_reason: str = Field(
        description="One sentence: why these extra views were chosen, or why none would help."
    )
    views: list[PlannedView] = Field(
        description="0 to the stated maximum extra views from the catalog; empty if none would add value."
    )
    follow_up_questions: list[str] = Field(description="3-5 questions the user could explore next.")


# ── LLM output: ad-hoc builders (one per catalog kind) ────────────────────────


class NetworkNode(BaseModel):
    id: str
    label: str = Field(description="Short label, max ~4 words.")
    entity_type: Literal["doctrine", "party", "plaintiff", "defendant", "third_party", "actor", "entity"]
    layer: int = Field(description="0 central issue, 1 rules, 2 cases/statutes, 3 parties/facts.")
    description: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class NetworkEdge(BaseModel):
    source: str = Field(description="Node id.")
    target: str = Field(description="Node id.")
    relation: str = Field(description="Short verb phrase.")
    description: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class ConceptMapOutput(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


class MatrixCell(BaseModel):
    id: str
    row: str
    column: str
    value: Literal["Yes", "No", "Maybe", "Likely", "Fails", "Missing", "N/A"]
    description: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class ComparisonOutput(BaseModel):
    items: list[MatrixCell]


class DecisionNode(BaseModel):
    id: str
    parent: str | None = Field(description="Parent node id; null for the root.")
    label: str = Field(description="Short question or outcome, max ~8 words.")
    node_type: Literal["question", "outcome"]
    outcome: Literal["allowed", "prohibited", "conditional"] | None = Field(
        default=None, description="Only for outcome nodes."
    )
    edge_label: str | None = Field(default=None, description="Label on the edge from the parent, e.g. Yes/No.")
    detail: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class LegalTestOutput(BaseModel):
    items: list[DecisionNode]


class RuleNode(BaseModel):
    id: str
    parent: str | None = Field(description="Parent node id; null for the root.")
    label: str = Field(description="Short label, max ~6 words.")
    category: Literal["grant", "right", "condition", "restriction", "exception"]
    detail: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class RuleHierarchyOutput(BaseModel):
    items: list[RuleNode]


class FlowStage(BaseModel):
    id: str
    label: str = Field(description="Short stage name.")
    location_name: str = Field(description="Same as label (display field).")
    category: Literal["location"] = "location"
    description: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class FlowEdge(BaseModel):
    source: str
    target: str
    relation: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class ProcessFlowOutput(BaseModel):
    items: list[FlowStage]
    flow_edges: list[FlowEdge]


class TimelineEvent(BaseModel):
    id: str
    date_label: str = Field(description="Display date; may be relative.")
    event: str = Field(description="Short event title.")
    category: Literal["event", "legal"]
    description: str
    evidence_ids: EvidenceIds = Field(description=_EV_DESC)


class ScenarioTimelineOutput(BaseModel):
    items: list[TimelineEvent]


ADHOC_OUTPUTS: dict[str, type[BaseModel]] = {
    "concept_map": ConceptMapOutput,
    "comparison": ComparisonOutput,
    "legal_test": LegalTestOutput,
    "rule_hierarchy": RuleHierarchyOutput,
    "process_flow": ProcessFlowOutput,
    "scenario_timeline": ScenarioTimelineOutput,
}
