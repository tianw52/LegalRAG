import { useEffect, useRef, useState } from "react";

// Top bar shared by every page: wordmark (→ home), a centre slot (nav or
// exploration tabs) and a right slot (province pill, account menu, actions).
export default function AppHeader({ onHome, center, right }) {
  return (
    <header className="topbar">
      <button type="button" className="brand" onClick={onHome}>
        <i className="brand-dot" />
        <span className="disp">LegalRAG Explore</span>
      </button>
      <div className="topbar-center">{center}</div>
      <div className="topbar-right">{right}</div>
    </header>
  );
}

// Province pill (omitted when provinceLabel is null) + account button with a small menu (edit profile / log out).
export function ProfileControls({ profile, provinceLabel, onEditProfile, onLogout }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const close = (e) => !ref.current?.contains(e.target) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <>
      {provinceLabel && (
        <button type="button" className="pill" onClick={onEditProfile} title="Change your location and profile">
          <i className="dot" style={{ background: "var(--local)" }} />
          {provinceLabel}
        </button>
      )}
      <div className="account" ref={ref}>
        <button
          type="button"
          className="account-btn"
          aria-label="Account"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <circle cx="12" cy="8" r="4" />
            <path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" />
          </svg>
        </button>
        {open && (
          <div className="menu" role="menu">
            <span className="lbl">Signed in as {profile.display_name}</span>
            <button type="button" role="menuitem" onClick={() => { setOpen(false); onEditProfile(); }}>
              Edit profile
            </button>
            <button type="button" role="menuitem" onClick={onLogout}>
              Log out
            </button>
          </div>
        )}
      </div>
    </>
  );
}
