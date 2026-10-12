import { useState } from "react";

export default function QuestionBox({ onSubmit, busy, examples }) {
  const [text, setText] = useState("");

  function submit(e) {
    e.preventDefault();
    const q = text.trim();
    if (q.length >= 3 && !busy) onSubmit(q);
  }

  return (
    <section className="question-box">
      <form onSubmit={submit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
          rows={3}
          placeholder="Describe a situation or ask about an area of Canadian law…"
        />
        <div className="row end">
          <span className="muted small">⌘/Ctrl + Enter</span>
          <button className="btn primary" disabled={busy || text.trim().length < 3}>
            {busy ? "Exploring…" : "Explore"}
          </button>
        </div>
      </form>
      {examples.length > 0 && (
        <div className="examples">
          <div className="lbl">Try a question from the benchmark</div>
          <div className="example-grid">
            {examples.map((ex) => (
              <button key={ex.query_id} className="example" onClick={() => setText(ex.text)}>
                <span className="muted small">{ex.province}</span>
                <span>{ex.text.length > 220 ? ex.text.slice(0, 220) + "…" : ex.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
