import { useCallback, useRef, useState } from "react";
import { api, exploreStream } from "../api.js";
import { applyEvent, demoRun, newRun } from "./exploration.js";

// Session state for explorations: the list of runs (newest last), the active
// one, explore() which streams POST /api/explore into a new run, openDemo()
// which replays a precomputed example, and updateRun() for per-run UI state
// (fact answers, Ask-panel threads).  Starting a new exploration aborts the
// previous stream (the server then cancels its LLM calls).
export function useExplorations(userId) {
  const [runs, setRuns] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const abortRef = useRef(null);

  const updateRun = useCallback(
    (id, fn) => setRuns((rs) => rs.map((r) => (r.id === id ? fn(r) : r))),
    []
  );

  const explore = useCallback(
    async (question, scope = "profile", province = null) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const run = newRun(question, scope, province);
      setRuns((rs) => [...rs, run]);
      setActiveId(run.id);
      try {
        await exploreStream(
          { question, userId, scope },
          (ev) => updateRun(run.id, (r) => applyEvent(r, ev)),
          controller.signal
        );
      } catch (e) {
        if (e.name !== "AbortError") updateRun(run.id, (r) => applyEvent(r, { event: "error", message: e.message }));
      }
      return run.id;
    },
    [userId, updateRun]
  );

  // Precomputed example: reuse an already-open copy, else fetch and replay it.
  const openDemo = useCallback(
    async (queryId) => {
      const open = runs.find((r) => r.demo?.queryId === queryId);
      if (open) return setActiveId(open.id);
      const run = demoRun(await api.demo(queryId));
      setRuns((rs) => [...rs, run]);
      setActiveId(run.id);
    },
    [runs]
  );

  const active = runs.find((r) => r.id === activeId) || null;
  return { runs, active, activeId, setActiveId, explore, openDemo, updateRun };
}
