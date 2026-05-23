"""
LegalBenchRAG Data Inspector — stdlib only, no extra installs.

Usage:
    python -m data_inspector [PORT]    # default 8765
    python -m data_inspector 9000

Then open http://<lab-machine-ip>:PORT in your browser.
"""
import sys
import threading
import webbrowser

from .server import ThreadedHTTPServer, Handler


def main():
    port   = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    url    = f"http://localhost:{port}"
    print(f"Data inspector →  {url}")
    threading.Timer(0.6, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
