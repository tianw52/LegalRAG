import { useEffect, useRef, useState } from "react";
import { chatStream } from "../api.js";
import { TAB_LABEL } from "../lib/tabs.js";

const CITE = /\[(e\d+(?:\s*,\s*e\d+)*)\]/g;

// Ask panel: chat about one exploration, beside every tab.  Threads live in
// run.chat; each question records where it was asked ("asked on …" jumps
// back there).  The "Looking at" line follows the current tab and selection
// unless pinned.  Answers come from POST /api/chat and cite [eN] passages.
export default function ChatPanel({ run, userId, tab, context, suggestions, updateRun, onOpenEvidence, onGoTab, onClose }) {
  const [draft, setDraft] = useState("");
  const [pinned, setPinned] = useState(null); // {tab, context}
  const [threadsOpen, setThreadsOpen] = useState(false);
  const endRef = useRef(null);

  const { threads, cur } = run.chat;
  const thread = threads.find((t) => t.id === cur) || null;
  const msgs = thread?.msgs || [];
  const busy = msgs.some((m) => m.pending);
  const ctx = pinned?.context || context;

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [msgs.length, msgs[msgs.length - 1]?.text]);

  const setChat = (fn) => updateRun((r) => ({ ...r, chat: fn(r.chat) }));
  const editThread = (id, fn) =>
    setChat((c) => ({ ...c, threads: c.threads.map((t) => (t.id === id ? fn(t) : t)) }));

  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    setDraft("");
    let id = thread?.id;
    if (!id) {
      id = `t${Date.now()}`;
      setChat((c) => ({ threads: [...c.threads, { id, title: text.slice(0, 48), msgs: [] }], cur: id }));
    }
    const history = msgs.filter((m) => !m.pending && !m.error).map((m) => ({ role: m.role, text: m.text }));
    const where = pinned?.tab || tab;
    editThread(id, (t) => ({
      ...t,
      msgs: [...t.msgs, { role: "user", text, where, ctx }, { role: "assistant", text: "", pending: true, evidence_ids: [] }],
    }));
    const patchLast = (patch) =>
      editThread(id, (t) => ({ ...t, msgs: t.msgs.map((m, i) => (i === t.msgs.length - 1 ? { ...m, ...patch(m) } : m)) }));
    try {
      await chatStream(
        {
          question: run.question,
          message: text,
          context: ctx,
          history,
          evidence: run.evidence.filter((e) => e.kind === "passage"),
          user_id: userId,
        },
        (ev) => {
          if (ev.event === "delta") patchLast((m) => ({ text: m.text + ev.text }));
          else if (ev.event === "done") patchLast(() => ({ text: ev.answer, evidence_ids: ev.evidence_ids, pending: false }));
          else if (ev.event === "error") patchLast(() => ({ text: ev.message, error: true, pending: false }));
        }
      );
    } catch (e) {
      patchLast(() => ({ text: e.message, error: true, pending: false }));
    }
  }

  return (
    <aside className="chat" aria-label="Ask about this exploration">
      <div className="chat-head">
        <button type="button" className="chat-thread-btn" aria-expanded={threadsOpen} onClick={() => setThreadsOpen((o) => !o)} disabled={!threads.length}>
          <span className="chat-title">{thread?.title || "New thread"}</span>
          {threads.length > 0 && <span aria-hidden="true" className="small">▾</span>}
        </button>
        <button type="button" className="btn ghost tiny" onClick={() => setChat((c) => ({ ...c, cur: null }))}>+ New</button>
        <button type="button" className="icon-btn" aria-label="Close the Ask panel" onClick={onClose}>×</button>
      </div>

      {threadsOpen && (
        <div className="chat-threads">
          <span className="lbl">Threads in this exploration</span>
          {threads.map((t) => {
            const n = t.msgs.filter((m) => m.role === "user").length;
            return (
              <button
                key={t.id}
                type="button"
                className={t.id === cur ? "on" : ""}
                onClick={() => {
                  setChat((c) => ({ ...c, cur: t.id }));
                  setThreadsOpen(false);
                }}
              >
                <span className="chat-thread-title">{t.title}</span>
                <span className="small muted">
                  {n} question{n === 1 ? "" : "s"}
                  {t.msgs[0] ? ` · started on ${TAB_LABEL[t.msgs[0].where]}` : ""}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div className="chat-ctx">
        <span className="lbl">Looking at</span>
        <span className="chat-ctx-label" title={ctx}>{ctx}</span>
        <button
          type="button"
          className={`chat-pin ${pinned ? "on" : ""}`}
          aria-pressed={!!pinned}
          title={pinned ? "Follow what you look at again" : "Keep asking about this while you browse"}
          onClick={() => setPinned(pinned ? null : { tab, context })}
        >
          {pinned ? "Pinned" : "Pin"}
        </button>
      </div>

      <div className="chat-msgs" aria-live="polite">
        {!msgs.length && (
          <p className="small muted">
            Ask anything about what you're looking at. Answers only use this exploration's {run.evidence.length - 1} passages and cite them.
          </p>
        )}
        {msgs.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg-user">
              <div className="bubble">{m.text}</div>
              <button type="button" className="stamp" title="Go back to where this was asked" onClick={() => onGoTab(m.where)}>
                asked on {m.ctx}
              </button>
            </div>
          ) : (
            <div key={i} className={`msg-ai ${m.error ? "error" : ""}`}>
              {m.pending && !m.text ? (
                <span className="pending"><span className="pulse" /> Reading the passages…</span>
              ) : (
                <p>{withCitations(m.text, onOpenEvidence)}</p>
              )}
            </div>
          )
        )}
        <div ref={endRef} />
      </div>

      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          send(draft);
        }}
      >
        <div className="chips">
          {suggestions.map((s) => (
            <button key={s} type="button" className="sugg" disabled={busy} onClick={() => send(s)}>{s}</button>
          ))}
        </div>
        <div className="chat-box">
          <label htmlFor="chat-in" className="sr-only">Ask a question</label>
          <input id="chat-in" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Ask about this exploration…" />
          <button type="submit" aria-label="Send" disabled={busy || !draft.trim()}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          </button>
        </div>
        <span className="small muted">Answers use only this exploration's sources · not legal advice</span>
      </form>
    </aside>
  );
}

// "text [e3, e5] more" → text with clickable passage ids.
function withCitations(text, onOpen) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    out.push(text.slice(last, m.index));
    out.push(
      <span key={m.index} className="cites">
        {m[1].split(",").map((id) => (
          <button key={id} type="button" className="cite" onClick={() => onOpen(id.trim())}>{id.trim()}</button>
        ))}
      </span>
    );
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}
