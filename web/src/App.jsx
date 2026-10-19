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
  const { runs, active, setActiveId, explore, openDemo, updateRun } = useExplorations(profile?.user_id);

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
  const userProvince = allCanada ? null : profile.province;
  const goHome = () => setPage("home");
  const startExplore = (question, scope) => {
    setPage("explore");
    window.scrollTo({ top: 0 });
    explore(question, scope, userProvince);
  };
  // Benchmark examples open their precomputed demo instantly; live run otherwise
  const startExample = async (ex) => {
    if (!ex.demo) return startExplore(ex.text);
    try {
      await openDemo(ex.query_id);
      setPage("explore");
      window.scrollTo({ top: 0 });
    } catch {
      startExplore(ex.text);
    }
  };
  const header = (center, extra = null, showProvince = true) => (
    <AppHeader
      onHome={goHome}
      center={center}
      right={
        <>
          {extra}
          <ProfileControls
            profile={profile}
            provinceLabel={showProvince ? provinceLabel : null}
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
        options={options}
        userProvince={userProvince}
        onExplore={startExplore}
        updateRun={(fn) => updateRun(active.id, fn)}
        renderHeader={(tabs, actions) =>
          header(tabs, <>{actions}<button type="button" className="pill new-q" onClick={goHome}>New question</button></>, false)
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
      onExample={startExample}
      onOpenRun={(id) => {
        setActiveId(id);
        setPage("explore");
      }}
    />
  );
}
