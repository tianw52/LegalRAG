import { useEffect, useState } from "react";
import { api } from "./api.js";
import Login from "./components/Login.jsx";
import Onboarding from "./components/Onboarding.jsx";
import Workspace from "./components/Workspace.jsx";

const USER_KEY = "legalrag.userId";

function readStoredUser() {
  try {
    return localStorage.getItem(USER_KEY);
  } catch {
    return null;
  }
}

function storeUser(userId) {
  try {
    if (userId) localStorage.setItem(USER_KEY, userId);
    else localStorage.removeItem(USER_KEY);
  } catch {
    /* storage unavailable (private mode) — session-only login */
  }
}

export default function App() {
  const [options, setOptions] = useState(null);
  const [profile, setProfile] = useState(null);
  const [editingProfile, setEditingProfile] = useState(false);
  const [booting, setBooting] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setOptions(await api.options());
        const stored = readStoredUser();
        if (stored) setProfile(await api.getProfile(stored).catch(() => null));
      } catch (e) {
        setError(`Cannot reach the API (${e.message}). Is uvicorn running on :8000?`);
      } finally {
        setBooting(false);
      }
    })();
  }, []);

  function onLogin(p) {
    storeUser(p.user_id);
    setProfile(p);
  }

  function logout() {
    storeUser(null);
    setProfile(null);
  }

  if (booting) return <div className="center-msg">Loading…</div>;
  if (error) return <div className="center-msg error">{error}</div>;
  if (!profile) return <Login onLogin={onLogin} />;
  if (!profile.onboarded || editingProfile) {
    return (
      <Onboarding
        options={options}
        profile={profile}
        onDone={(p) => {
          setProfile(p);
          setEditingProfile(false);
        }}
        onCancel={profile.onboarded ? () => setEditingProfile(false) : null}
      />
    );
  }
  return (
    <Workspace
      profile={profile}
      options={options}
      onEditProfile={() => setEditingProfile(true)}
      onLogout={logout}
    />
  );
}
