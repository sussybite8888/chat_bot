"""Serve the browser frontend (`web/`).

    python -m sodachat.web              # http://127.0.0.1:8000

This is a static file server and nothing more — no model runs here. The page it
serves downloads the exported graphs and runs them itself, which is the whole
point of the web backend, so the only reason a server is involved at all is
that browsers will not `fetch` from `file://` and will not compile a WASM
module without a real MIME type.

Two headers matter and are why `python -m http.server` is not quite enough:

* **COOP/COEP.** Cross-origin isolation is what unlocks `SharedArrayBuffer`,
  and without it onnxruntime-web falls back to a single thread — several times
  slower on a reply that is already a few hundred forward passes.
* **Cache-Control.** A model file is 50-120MB and changes only when you
  re-export, so it is worth caching hard; the HTML and JS are not.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import socketserver
import webbrowser
from pathlib import Path

# Content that only changes when you re-export or rebuild: a whole model, one
# chunk of a split one (`chat.onnx.part000`), or the WASM runtime.
_IMMUTABLE_RE = re.compile(r"\.(?:onnx|wasm)$|\.onnx\.part\d+$")

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"

# .wasm and .onnx are the ones that matter: a WASM module served as
# text/plain will not stream-compile, and some proxies mangle unknown types.
EXTRA_TYPES = {
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **EXTRA_TYPES}

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Only the big opaque blobs are immutable. The manifest and the
        # tokenizer sit beside them and are rewritten by every re-export, so
        # caching those hard would pin the page to a stale model.
        # `.partNNN` is a chunked model from tools/build_dist.mjs — the largest
        # files served here, and the ones most worth not fetching twice.
        immutable = bool(_IMMUTABLE_RE.search(self.path))
        self.send_header(
            "Cache-Control",
            "public, max-age=31536000, immutable" if immutable else "no-cache",
        )
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    # Threaded so a 120MB model download does not block the page's other
    # requests, and reusable so a restart is not blocked by TIME_WAIT.
    daemon_threads = True
    allow_reuse_address = True


def _describe(root: Path) -> str:
    manifest = root / "models" / "manifest.json"
    if not manifest.exists():
        return (
            "no exported models yet — run `python -m sodachat.export_onnx` "
            "(the page will say the same thing)"
        )
    models = json.loads(manifest.read_text())["models"]
    total = sum(m["bytes"] for m in models.values()) / 1e6
    return f"{len(models)} model(s) exported, {total:.0f} MB: {', '.join(models)}"


def serve(host: str = "127.0.0.1", port: int = 8000, root: Path = WEB_ROOT,
          open_browser: bool = False) -> None:
    if not (root / "index.html").exists():
        raise SystemExit(f"no page at {root}/index.html")
    if not (root / "vendor" / "ort").exists():
        print("[sodachat] web/vendor/ort is missing — run `npm install && npm run vendor`")
    print(f"[sodachat] {_describe(root)}")
    handler = functools.partial(Handler, directory=str(root))
    with Server((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"[sodachat] serving {root} at {url} (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="sodachat.web", description="Serve the browser frontend (web/)."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--root", type=Path, default=WEB_ROOT)
    p.add_argument("--open", action="store_true", help="open a browser window")
    a = p.parse_args(argv)
    serve(a.host, a.port, a.root, a.open)


if __name__ == "__main__":
    main()
