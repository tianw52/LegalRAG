import { useCallback, useRef, useState } from "react";
import { exploreStream } from "../api.js";
import { applyEvent, newRun } from "./exploration.js";

// Session state for explorations: the list of runs (newest last), the active
// one, and explore() which streams POST /api/explore into a new run.
// Starting a new exploration aborts the previous stream (the server then
// cancels its LLM calls).
export function useExplorations(userId) {
  const [runs, setRuns] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const abortRef = useRef(null);

  const explore = useCallback(
    async (question, scope = "profile") => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const run = newRun(question, scope);
      setRuns((rs) => [...rs, run]);
      setActiveId(run.id);
      const update = (fn) => setRuns((rs) => rs.map((r) => (r.id === run.id ? fn(r) : r)));
      try {
        await exploreStream(
          { question, userId, scope },
          (ev) => update((r) => applyEvent(r, ev)),
          controller.signal
        );
      } catch (e) {
        if (e.name !== "AbortError") update((r) => applyEvent(r, { event: "error", message: e.message }));
      }
      return run.id;
    },
    [userId]
  );

  const active = runs.find((r) => r.id === activeId) || null;
  return { runs, active, activeId, setActiveId, explore };
}
