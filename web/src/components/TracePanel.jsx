import { useEffect, useState } from "react";
import { api } from "../api.js";

// "Under the hood" drawer: renders GET /api/traces/{id} (legalrag/explore/trace.py).
// Polls while the run is still in progress.
export default function TracePanel({ traceId, live, onClose }) {
  const [trace, setTrace] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let stop = false;
    async function load() {
      try {
        const t = await api.trace(traceId);
        if (!stop) setTrace(t);
        return t;
      } catch (e) {
        if (!stop) setError(e.message);
      }
    }
    load();
    const timer = live ? setInterval(load, 3000) : null;
    return () => {
      stop = true;
      if (timer) clearInterval(timer);
    };
  }, [traceId, live]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const step = (name) => trace?.steps.find((s) => s.name === name);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer wide" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <div className="lbl">Under the hood · trace {traceId}</div>
            <h2>{trace ? `${trace.status} · ${trace.total_s ?? "…"} s` : "Loading…"}</h2>
            {trace?.file && <p className="muted small">Saved to {trace.file}</p>}
          </div>
          <button className="btn ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        {error && <div className="error">{error}</div>}
        {trace && (
          <div className="drawer-body trace">
            <StepTimeline steps={trace.steps} total={trace.total_s} />
            <QuerySection question={trace.question} formulate={step("formulate")} />
            <RetrievalSection retrieve={step("retrieve")} rerank={step("rerank")} />
            <LlmSection calls={trace.llm_calls} />
          </div>
        )}
      </aside>
    </div>
  );
}

function StepTimeline({ steps, total }) {
  const end = total || Math.max(1, ...steps.map((s) => s.start_s + (s.duration_s || 0)));
  return (
    <section>
      <h3>Steps</h3>
      <div className="steps">
        {steps.map((s, i) => (
          <div key={i} className={`step-row ${s.error ? "failed" : ""}`}>
            <span className="step-name">{s.name}</span>
            <span className="step-track">
              <span
                className="step-bar"
                style={{
                  left: `${(s.start_s / end) * 100}%`,
                  width: `${Math.max(0.5, ((s.duration_s ?? end - s.start_s) / end) * 100)}%`,
                }}
              />
            </span>
            <span className="step-time">
              {s.duration_s != null ? `${s.duration_s.toFixed(1)} s` : "running…"}
            </span>
          </div>
        ))}
      </div>
      {steps.filter((s) => s.error).map((s, i) => (
        <p key={i} className="error small">{s.name}: {s.error}</p>
      ))}
    </section>
  );
}

function QuerySection({ question, formulate }) {
  return (
    <section>
      <h3>Query</h3>
      <dl className="kv">
        <dt>Question</dt>
        <dd>{question}</dd>
        <dt>Rewritten for search</dt>
        <dd>{formulate?.reformulated || (formulate?.fallback ? `— (fallback: ${formulate.fallback})` : "—")}</dd>
        <dt>Keywords</dt>
        <dd>{formulate?.keywords?.join(", ") || "—"}</dd>
      </dl>
    </section>
  );
}

const MODE_LABEL = { semantic: "Dense (kNN)", lexical: "BM25", hybrid: "Hybrid (kNN + BM25)" };

function RetrievalSection({ retrieve, rerank }) {
  if (!retrieve) return null;
  const byRank = {};
  (rerank?.scores || []).forEach((s) => (byRank[s.retrieval_rank] = s));
  return (
    <section>
      <h3>Retrieval</h3>
      <p className="small">
        {MODE_LABEL[retrieve.mode] || retrieve.mode} search, top {retrieve.top_k}, jurisdictions:{" "}
        {retrieve.jurisdictions ? retrieve.jurisdictions.join(", ") : "all"}
        {retrieve.relaxed && ` → relaxed to all of Canada (only ${retrieve.filtered_hits} filtered hits)`}.
        {rerank && ` Reranked with ${rerank.model}; kept top ${rerank.keep}.`}
      </p>
      <div className="table-wrap">
        <table className="trace-table">
          <thead>
            <tr>
              <th>Retrieval #</th><th>Retrieval score</th><th>Rerank #</th><th>Rerank score</th><th>Kept</th>
              <th>Citation</th><th>Juris.</th><th>Passage</th>
            </tr>
          </thead>
          <tbody>
            {(retrieve.candidates || []).map((c) => {
              const r = byRank[c.retrieval_rank];
              return (
                <tr key={c.chunk_id} className={r?.kept ? "kept" : ""}>
                  <td>{c.retrieval_rank}</td>
                  <td>{c.retrieval_score != null ? c.retrieval_score.toFixed(3) : "—"}</td>
                  <td>{r?.rerank_rank ?? "—"}</td>
                  <td>{r?.rerank_score != null ? r.rerank_score.toFixed(3) : "—"}</td>
                  <td>{r?.kept ? "✓" : ""}</td>
                  <td>{c.citation}</td>
                  <td>{c.jurisdiction}</td>
                  <td className="preview">{c.text_preview}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LlmSection({ calls }) {
  return (
    <section>
      <h3>LLM calls ({calls.length})</h3>
      {calls.map((c, i) => (
        <details key={i} className="llm-call">
          <summary>
            <strong>{c.label}</strong> · {c.model} · {c.duration_s.toFixed(1)} s
            {c.usage && ` · ${c.usage.input_tokens} in / ${c.usage.output_tokens} out tokens`}
            {c.validation_retries?.length > 0 && ` · ${c.validation_retries.length} retries`}
            {c.error && <span className="error-inline"> · {c.error}</span>}
          </summary>
          <Block title="System prompt" text={c.system_prompt} />
          <Block title="User prompt" text={c.user_prompt} />
          <Block
            title="Thinking"
            text={c.thinking?.length ? c.thinking.join("\n\n---\n\n") : "(none — thinking is off: explore_policy.yaml → llm.thinking_off)"}
          />
          <Block title="Raw response" text={(c.responses || []).join("\n\n--- retry ---\n\n")} />
          {c.validation_retries?.length > 0 && (
            <Block title="Validation retries" text={c.validation_retries.join("\n\n")} />
          )}
          {c.output && <Block title="Parsed output" text={JSON.stringify(c.output, null, 2)} />}
        </details>
      ))}
    </section>
  );
}

function Block({ title, text }) {
  return (
    <div className="block">
      <div className="lbl">{title}</div>
      <pre>{text || "—"}</pre>
    </div>
  );
}
