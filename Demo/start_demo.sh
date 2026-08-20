#!/usr/bin/env bash
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "Serving $(pwd) on http://127.0.0.1:${PORT}/answer-viz.html"
echo "If browsing from your laptop, open an SSH tunnel first:"
echo "  ssh -L ${PORT}:127.0.0.1:${PORT} ${USER}@$(hostname -f)"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
