import { useState } from "react";
import { api } from "../api.js";
import ProvinceTiles from "../components/ProvinceTiles.jsx";

const STEPS = ["Location", "Familiarity", "Goal", "Topics"];

// One question per step.  Options come from legalrag/prompts/explore_policy.yaml (GET /api/options).
export default function Onboarding({ options, profile, onDone, onCancel }) {
  const [step, setStep] = useState(0);
  const [province, setProvince] = useState(profile.province || "");
  const [familiarity, setFamiliarity] = useState(profile.familiarity || "some");
  const [goal, setGoal] = useState(profile.goal || "explore");
  const [problem, setProblem] = useState(profile.problem_summary || "");
  const [interests, setInterests] = useState(profile.interests || []);
  const [custom, setCustom] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const provinceLabel = options.provinces.find((p) => p.code === province)?.label || "Choose one";
  const nonProvince = options.provinces.filter((p) => p.all_of_canada);
  const chips = [...new Set([...options.interests, ...interests])];
  const isLast = step === STEPS.length - 1;

  function toggleInterest(name) {
    setInterests((cur) => (cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]));
  }

  async function save() {
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
      setBusy(false);
    }
  }

  return (
    <div className="page onboarding">
      <header className="topbar">
        <span className="brand static">
          <i className="brand-dot" />
          <span className="disp">LegalRAG Explore</span>
        </span>
        <span className="spacer" />
        {onCancel ? (
          <button type="button" className="link" onClick={onCancel}>Cancel</button>
        ) : (
          <button type="button" className="link" onClick={save} disabled={busy}>Skip for now</button>
        )}
      </header>
      <div className="progress"><i style={{ width: `${((step + 1) / STEPS.length) * 100}%` }} /></div>

      <main className="onboard-main">
        <ol className="steps-list" aria-label="Setup steps">
          {STEPS.map((label, i) => (
            <li key={label} className={i === step ? "cur" : i < step ? "done" : ""}>
              <span className="num">{i < step ? "✓" : `0${i + 1}`}</span>
              <span>{label}</span>
            </li>
          ))}
        </ol>

        <section className="onboard-step">
          {step === 0 && (
            <>
              <h1 className="disp h-l">Where are you?</h1>
              <p className="muted">Sources from your province or territory come first, alongside federal law.</p>
              <ProvinceTiles
                ariaLabel="Province or territory"
                onPick={setProvince}
                tileProps={(code) => {
                  const on = code === province;
                  return {
                    on,
                    bg: on ? "var(--local)" : "transparent",
                    fg: "var(--ink)",
                    border: `1px solid ${on ? "var(--local)" : "var(--line-strong)"}`,
                  };
                }}
              />
              <div className="segmented" role="group" aria-label="Other choices">
                {nonProvince.map((p) => (
                  <button
                    key={p.code}
                    type="button"
                    className={`seg ${province === p.code ? "on" : ""}`}
                    aria-pressed={province === p.code}
                    onClick={() => setProvince(p.code)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <p className="row wrap baseline">
                <span className="disp h-s accent">{provinceLabel}</span>
                {nonProvince.some((p) => p.code === province) && (
                  <span className="muted small">Sources from every province and federal law.</span>
                )}
              </p>
            </>
          )}

          {step === 1 && (
            <>
              <h1 className="disp h-l">How familiar are you with the law?</h1>
              <div className="options">
                {options.familiarity.map((f, i) => (
                  <button
                    key={f.key}
                    type="button"
                    className={`opt ${familiarity === f.key ? "on" : ""}`}
                    aria-pressed={familiarity === f.key}
                    onClick={() => setFamiliarity(f.key)}
                  >
                    <span className="level" aria-hidden="true">
                      {[0, 1, 2].map((b) => (
                        <i key={b} className={b <= i ? "fill" : ""} style={{ height: 16 + b * 10 }} />
                      ))}
                    </span>
                    <span className="disp h-s">{f.label}</span>
                    <span className="muted">{f.description}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <h1 className="disp h-l">What brings you here?</h1>
              <div className="options">
                {options.goals.map((g, i) => (
                  <button
                    key={g.key}
                    type="button"
                    className={`opt ${goal === g.key ? "on" : ""}`}
                    aria-pressed={goal === g.key}
                    onClick={() => setGoal(g.key)}
                  >
                    <span className="num">0{i + 1}</span>
                    <span className="disp h-s">{g.label}</span>
                    <span className="muted">{g.description}</span>
                  </button>
                ))}
              </div>
              {goal === "specific_problem" && (
                <textarea
                  className="plain-area"
                  rows={3}
                  value={problem}
                  onChange={(e) => setProblem(e.target.value)}
                  placeholder="Optional: describe your situation in a sentence or two"
                />
              )}
            </>
          )}

          {step === 3 && (
            <>
              <h1 className="disp h-l">Pick a few topics</h1>
              <div className="chips">
                {chips.map((name) => (
                  <button
                    key={name}
                    type="button"
                    className={`chip ${interests.includes(name) ? "on" : ""}`}
                    aria-pressed={interests.includes(name)}
                    onClick={() => toggleInterest(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
              <div className="underline-input small-input">
                <input
                  value={custom}
                  onChange={(e) => setCustom(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && custom.trim()) {
                      e.preventDefault();
                      if (!interests.includes(custom.trim())) setInterests([...interests, custom.trim()]);
                      setCustom("");
                    }
                  }}
                  placeholder="Add another topic and press Enter"
                  aria-label="Add another topic"
                />
              </div>
            </>
          )}

          {error && <p className="error">{error}</p>}
          <div className="step-actions">
            {step > 0 && (
              <button type="button" className="btn ghost" onClick={() => setStep(step - 1)}>Back</button>
            )}
            <span className="spacer" />
            <span className="lbl">{step + 1} / {STEPS.length}</span>
            {isLast ? (
              <button type="button" className="btn primary" onClick={save} disabled={busy}>
                {busy ? "Saving…" : "Start exploring"} <span aria-hidden="true">→</span>
              </button>
            ) : (
              <button type="button" className="btn primary" onClick={() => setStep(step + 1)}>
                Next <span aria-hidden="true">→</span>
              </button>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
