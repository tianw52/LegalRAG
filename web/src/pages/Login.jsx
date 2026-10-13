import { useState } from "react";
import { api } from "../api.js";

// DEV-ONLY login: a display name identifies the profile (no password).
export default function Login({ onLogin }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError("");
    try {
      onLogin(await api.login(name.trim()));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page-center">
      <form className="login" onSubmit={submit}>
        <span className="brand static">
          <i className="brand-dot" />
          <span className="disp">LegalRAG Explore</span>
        </span>
        <h1 className="disp h-xl">See the law around your question</h1>
        <p className="lead muted">
          Canadian case law and statutes, mapped: the sources, the facts that matter, and how the
          rules fit together.
        </p>
        <label htmlFor="name" className="lbl">Your name</label>
        <div className="underline-input">
          <input
            id="name"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Alex"
            maxLength={80}
          />
          <button className="go-btn" aria-label="Continue" disabled={busy || !name.trim()}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          </button>
        </div>
        {error && <p className="error">{error}</p>}
        <p className="lbl">Development build · no password · profile stored on this server</p>
      </form>
    </div>
  );
}
