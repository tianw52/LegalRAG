"""Threaded HTTP server — routes requests to api.py, serves static files."""
from __future__ import annotations
import http.server
import json
import mimetypes
import pathlib
import socketserver
import sys
import urllib.parse

from .api import api_chunkers, api_datasets, api_models, api_queries, api_query

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

# Per-dataset models cache: dataset_id → models dict
_models_cache: dict[str, dict] = {}


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p      = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))
        ds     = params.get("ds", "legalbenchrag-mini")

        try:
            if p == "/":
                self._static("start.html")

            elif p == "/inspect":
                self._static("index.html")

            elif p.startswith("/static/"):
                self._static(p.removeprefix("/static/"))

            elif p == "/api/datasets":
                self._json(api_datasets())

            elif p == "/api/models":
                if ds not in _models_cache:
                    _models_cache[ds] = api_models(ds)
                self._json(_models_cache[ds])

            elif p == "/api/chunkers":
                self._json(api_chunkers(ds))

            elif p == "/api/queries":
                self._json(api_queries(
                    ds,
                    params.get("model", ""),
                    params.get("embedder", ""),
                    params.get("dataset") or None,
                    params.get("chunker", "hier"),
                ))

            elif p == "/api/query":
                self._json(api_query(
                    ds,
                    params.get("model", ""),
                    params.get("embedder", ""),
                    int(params["idx"]),
                    params.get("chunker", "hier"),
                ))

            else:
                self.send_error(404)

        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _static(self, name: str):
        path = STATIC_DIR / name
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        ct   = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ct.startswith("text/"):
            ct += "; charset=utf-8"
        self._send(200, ct, body)

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self._send(status, "application/json", body)

    def _send(self, status: int, ct: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = fmt % args if args else fmt
        if any(c in msg for c in ('40', '50')):
            print(f"[inspector] {msg}", file=sys.stderr)
