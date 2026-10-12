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
    <div className="auth-page">
      <form className="auth-card" onSubmit={submit}>
        <h1 className="brand">LegalRAG Explore</h1>
        <p className="muted">
          Explore Canadian case law visually: see the sources, the facts that matter, and how
          the rules fit together.
        </p>
        <label className="field">
          <span>Your name</span>
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Alex"
            maxLength={80}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="btn primary" disabled={busy || !name.trim()}>
          {busy ? "Signing in…" : "Continue"}
        </button>
        <p className="fineprint">Development build: no password; your profile is stored on this server.</p>
      </form>
    </div>
  );
}
