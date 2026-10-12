import { useState } from "react";
import { api } from "../api.js";

// Profile questions come from legalrag/prompts/explore_policy.yaml (GET /api/options).
export default function Onboarding({ options, profile, onDone, onCancel }) {
  const [province, setProvince] = useState(profile.province || "");
  const [familiarity, setFamiliarity] = useState(profile.familiarity || "");
  const [goal, setGoal] = useState(profile.goal || "");
  const [problem, setProblem] = useState(profile.problem_summary || "");
  const [interests, setInterests] = useState(profile.interests || []);
  const [custom, setCustom] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const chips = [...new Set([...options.interests, ...interests])];

  function toggleInterest(name) {
    setInterests((cur) => (cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]));
  }

  function addCustom() {
    const v = custom.trim();
    if (v && !interests.includes(v)) setInterests([...interests, v]);
    setCustom("");
  }

  async function save(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onDone(
        await api.saveProfile(profile.user_id, {
          province,
          familiarity,
          goal,
          problem_summary: goal === "specific_problem" ? problem : "",
          interests,
        })
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <form className="onboard-card" onSubmit={save}>
        <h1 className="brand">Hi {profile.display_name} 👋</h1>
        <p className="muted">
          A few questions so the explorer shows sources from your jurisdiction and explains them at
          the right level. You can change these any time.
        </p>

        <section>
          <h2>Where are you located?</h2>
          <p className="hint">Used to prioritise courts and laws from your province or territory.</p>
          <select value={province} onChange={(e) => setProvince(e.target.value)}>
            {options.provinces.map((p) => (
              <option key={p.code || "none"} value={p.code}>
                {p.label}
              </option>
            ))}
          </select>
        </section>

        <section>
          <h2>How familiar are you with the law?</h2>
          <div className="choice-grid">
            {options.familiarity.map((f) => (
              <label key={f.key} className={`choice ${familiarity === f.key ? "on" : ""}`}>
                <input
                  type="radio"
                  name="familiarity"
                  checked={familiarity === f.key}
                  onChange={() => setFamiliarity(f.key)}
                />
                <strong>{f.label}</strong>
                <span>{f.description}</span>
              </label>
            ))}
          </div>
        </section>

        <section>
          <h2>What brings you here?</h2>
          <div className="choice-grid">
            {options.goals.map((g) => (
              <label key={g.key} className={`choice ${goal === g.key ? "on" : ""}`}>
                <input
                  type="radio"
                  name="goal"
                  checked={goal === g.key}
                  onChange={() => setGoal(g.key)}
                />
                <strong>{g.label}</strong>
                <span>{g.description}</span>
              </label>
            ))}
          </div>
          {goal === "specific_problem" && (
            <textarea
              className="problem"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="Optional: describe your situation in a sentence or two"
              rows={3}
            />
          )}
        </section>

        <section>
          <h2>Topics you're interested in</h2>
          <div className="chips">
            {chips.map((name) => (
              <button
                type="button"
                key={name}
                className={`chip ${interests.includes(name) ? "on" : ""}`}
                onClick={() => toggleInterest(name)}
              >
                {name}
              </button>
            ))}
          </div>
          <div className="row">
            <input
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addCustom();
                }
              }}
              placeholder="Add another topic"
            />
            <button type="button" className="btn" onClick={addCustom}>
              Add
            </button>
          </div>
        </section>

        {error && <p className="error">{error}</p>}
        <div className="row end">
          {onCancel && (
            <button type="button" className="btn" onClick={onCancel}>
              Cancel
            </button>
          )}
          <button className="btn primary" disabled={busy || !familiarity || !goal}>
            {busy ? "Saving…" : "Start exploring"}
          </button>
        </div>
      </form>
    </div>
  );
}
