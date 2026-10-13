// Thin client for the FastAPI backend (legalrag/api/app.py).
// In dev, Vite proxies /api → http://127.0.0.1:8000 (vite.config.js).
const BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  options: () => request("/api/options"),
  login: (display_name) =>
    request("/api/login", { method: "POST", body: JSON.stringify({ display_name }) }),
  getProfile: (userId) => request(`/api/profile/${encodeURIComponent(userId)}`),
  saveProfile: (userId, update) =>
    request(`/api/profile/${encodeURIComponent(userId)}`, {
      method: "PUT",
      body: JSON.stringify(update),
    }),
  examples: (province) =>
    request(`/api/examples?province=${encodeURIComponent(province || "")}&n=4`),
  document: (docId) => request(`/api/document/${encodeURIComponent(docId)}`),
  traces: (limit = 20) => request(`/api/traces?limit=${limit}`),
  trace: (traceId) => request(`/api/traces/${encodeURIComponent(traceId)}`),
};

// POST /api/explore returns NDJSON: one event object per line.
export async function exploreStream({ question, userId, scope }, onEvent, signal) {
  const res = await fetch(BASE + "/api/explore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, user_id: userId, scope }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`${res.status}: ${res.statusText}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}
