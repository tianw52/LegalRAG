import { useEffect, useState } from "react";
import { api } from "./api.js";
import AppHeader, { ProfileControls } from "./components/AppHeader.jsx";
import { readStored, writeStored } from "./lib/storage.js";
import { useExplorations } from "./lib/useExplorations.js";
import Exploration from "./pages/Exploration.jsx";
import Home from "./pages/Home.jsx";
import Login from "./pages/Login.jsx";
import Onboarding from "./pages/Onboarding.jsx";

const USER_KEY = "legalrag.userId";

// Session routing: Login → Onboarding (first time / edit) → Home ↔ Exploration.
export default function App() {
  const [options, setOptions] = useState(null);
  const [profile, setProfile] = useState(null);
  const [editingProfile, setEditingProfile] = useState(false);
  const [page, setPage] = useState("home"); // home | explore
  const [booting, setBooting] = useState(true);
  const [error, setError] = useState("");
  const { runs, active, setActiveId, explore } = useExplorations(profile?.user_id);

  useEffect(() => {
    (async () => {
      try {
        setOptions(await api.options());
        const stored = readStored(USER_KEY);
        if (stored) setProfile(await api.getProfile(stored).catch(() => null));
      } catch (e) {
        setError(`Cannot reach the API (${e.message}). Is uvicorn running on :8000?`);
      } finally {
        setBooting(false);
      }
    })();
  }, []);

  if (booting) return <div className="page page-center"><span className="lbl">Loading…</span></div>;
  if (error) return <div className="page page-center"><p className="error">{error}</p></div>;
  if (!profile) {
    return (
      <Login
        onLogin={(p) => {
          writeStored(USER_KEY, p.user_id);
          setProfile(p);
        }}
      />
    );
  }
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

  // The non-province choices (all of Canada / outside Canada / prefer not to say) mean no filter
  const provinceOption = options.provinces.find((p) => p.code === profile.province);
  const allCanada = !provinceOption || !!provinceOption.all_of_canada;
  const provinceLabel = allCanada ? "All of Canada" : provinceOption.label;
  const goHome = () => setPage("home");
  const startExplore = (question, scope) => {
    setPage("explore");
    window.scrollTo({ top: 0 });
    explore(question, scope);
  };
  const header = (center, extra = null) => (
    <AppHeader
      onHome={goHome}
      center={center}
      right={
        <>
          {extra}
          <ProfileControls
            profile={profile}
            provinceLabel={provinceLabel}
            onEditProfile={() => setEditingProfile(true)}
            onLogout={() => {
              writeStored(USER_KEY, null);
              setProfile(null);
              setPage("home");
            }}
          />
        </>
      }
    />
  );

  if (page === "explore" && active) {
    return (
      <Exploration
        key={active.id}
        run={active}
        profile={profile}
        provinceLabel={provinceLabel}
        allCanada={allCanada}
        onExplore={startExplore}
        renderHeader={(tabs) =>
          header(tabs, <button type="button" className="pill" onClick={goHome}>New question</button>)
        }
      />
    );
  }

  return (
    <Home
      header={header(
        <nav className="navlinks" aria-label="Primary">
          <button type="button" className="navlink on">Explore</button>
          {runs.length > 0 && (
            <button type="button" className="navlink" onClick={() => document.getElementById("library")?.scrollIntoView({ behavior: "smooth" })}>
              Library
            </button>
          )}
        </nav>
      )}
      profile={profile}
      provinceLabel={provinceLabel}
      allCanada={allCanada}
      runs={runs}
      onExplore={startExplore}
      onOpenRun={(id) => {
        setActiveId(id);
        setPage("explore");
      }}
    />
  );
}
