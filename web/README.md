# LegalRAG Explore — web UI

React 18 + Vite 5 (plain JavaScript). Talks to the FastAPI backend in
`legalrag/api/app.py`; in dev, Vite proxies `/api` to `http://127.0.0.1:8000`.

```bash
npm ci           # exact versions from package-lock.json (Node ≥ 18)
npm run dev      # http://localhost:5173 (API must be running on :8000)
npm run build    # static bundle in dist/ (host anywhere; set VITE_API_BASE if the API is on another origin)
npm run smoke    # headless Chrome end-to-end test → ../logs/ui-smoke/*.png (FULL=1 for LLM views)
```

## Structure

Design: "LegalRAG Explorer Redesign" canvas (paper ground, ink type, one hot accent,
condensed display caps, mono labels). Tokens and all styles: `src/theme.css`.

```
src/
  main.jsx               entry; global styles (theme.css + vendor renderer CSS)
  App.jsx                session routing: Login → Onboarding → Home ↔ Exploration
  api.js                 REST calls + exploreStream() (NDJSON reader for POST /api/explore)
  theme.css              design tokens (:root) + every style
  pages/
    Login.jsx            dev-only name login
    Onboarding.jsx       one question per step: province tile map, familiarity, goal, topics
    Home.jsx             ask, benchmark questions, this session's explorations
    Exploration.jsx      title band (question, scope, build status, "under the hood") + tabs
  tabs/                  one per exploration tab; all read the same run state
    OverviewTab.jsx      orientation, key numbers, views list, follow-up questions
    WhatIfTab.jsx        fact toggles ("If not →")
    CasesTab.jsx         jurisdiction tile map filter + expandable document list
    TimelineTab.jsx      lanes per jurisdiction group on a year axis
    MapsTab.jsx          planner-chosen views (0–3) via the Demo renderers
    SourcesTab.jsx       passage list + full-text reader with highlights
  components/            reusable UI
    AppHeader.jsx        top bar; ProfileControls (province pill + account menu)
    ProvinceTiles.jsx    tile map of Canada (onboarding picker, cases filter)
    primitives.jsx       Dot, JurisdictionLegend, SplitBar, Switch, Segmented, SourceLinks, Pending
    HighlightedText.jsx  document text with passages marked by char offsets; minimap marks
    DemoViz.jsx          wrapper around the Demo renderers (window.VizRenderers)
    TracePanel.jsx       "Under the hood" drawer (GET /api/traces/{id})
  lib/                   framework-free logic
    useExplorations.js   hook: runs, active run, explore() streaming
    exploration.js       run state + applyEvent() reducer for the event stream
    stats.js             derived numbers (documents, group counts, year span, tab counts)
    jurisdiction.js      groups + colours, province tile layout, name → code
    selection.js         selection model for diagram items
    storage.js           localStorage wrappers that never throw
  vendor/                copied from ../Demo (Reyhaneh) — keep in sync, don't edit here
scripts/
  smoke.mjs              headless-Chrome end-to-end test (quick by default; FULL=1 waits for LLM views)
```

## Conventions

- **Evidence links are the shared interaction.** Anything that cites passages renders
  `<SourceLinks ids={…} onOpen={openEvidence} />`, which jumps to the Sources tab reader at
  that passage.
- **Adding a pre-defined view:**
  1. Return a `View` from `legalrag/explore/views.py`.
  2. Read it from `run.views[<view_id>]` in a tab under `src/tabs/` (or add a tab in
     `pages/Exploration.jsx`).
- **Adding an ad-hoc view kind:** no frontend change if it uses an existing Demo renderer.
  Add it to `catalog:` in `legalrag/prompts/explore_views.yaml` and to `ADHOC_OUTPUTS` in
  `legalrag/explore/schemas.py`.
- **Profile options and prompts are not in the frontend.** They come from
  `legalrag/prompts/explore_policy.yaml` through the API.
