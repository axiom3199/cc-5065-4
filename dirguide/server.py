"""Zero-dependency web UI server for the guiding kernel.

Launch:  python3 run_ui.py [--port 8765]  then open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .kernel import KERNEL, KernelError
from .experiment import RESULTS_DIR

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "DirGuide/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[ui] %s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8")

    def _file(self, path: str, ctype: str):
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except FileNotFoundError:
            self._json({"ok": False, "error": f"not found: {path}"}, 404)
            return
        # Artifacts and assets can be overwritten by new runs; never let a
        # browser reuse a stale copy without revalidation.
        self._send(200, body, ctype, cache=False)

    # --------------------------------------------------------------- GET
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        q = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._file(os.path.join(WEB_DIR, "index.html"),
                                  "text/html; charset=utf-8")
            if route == "/app.js":
                return self._file(os.path.join(WEB_DIR, "app.js"),
                                 "text/javascript; charset=utf-8")
            if route == "/style.css":
                return self._file(os.path.join(WEB_DIR, "style.css"),
                                  "text/css; charset=utf-8")
            if route == "/api/schema":
                return self._json({"ok": True, **KERNEL.schema()})
            if route == "/api/state":
                return self._json({"ok": True, "state": KERNEL.state()})
            if route == "/api/image.png":
                return self._send(200, KERNEL.png(), "image/png")
            if route == "/api/distribution":
                px = int(q.get("px", ["-1"])[0])
                py = int(q.get("py", ["-1"])[0])
                version = q.get("version", [None])[0]
                return self._json({"ok": True,
                                   "distribution": KERNEL.distribution(px, py, version)})
            if route == "/api/versions":
                return self._json({"ok": True, "versions": KERNEL.list_versions()})
            if route == "/api/results":
                return self._json(KERNEL.experiment_results())
            if route == "/api/acceptance":
                return self._json(KERNEL.acceptance_report())
            if route.startswith("/outputs/"):
                rel = route[len("/outputs/"):]
                root = os.path.abspath(RESULTS_DIR)
                path = os.path.abspath(os.path.join(root, rel))
                if not path.startswith(root + os.sep) and path != root:
                    return self._json({"ok": False, "error": "bad path"}, 403)
                ctype = ("image/png" if path.endswith(".png") else
                         "application/json" if path.endswith(".json") else
                         "text/csv" if path.endswith(".csv") else
                         "application/octet-stream")
                return self._file(path, ctype)
            return self._json({"ok": False, "error": "unknown route"}, 404)
        except KernelError as exc:
            return self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # never crash the server on a request
            return self._json({"ok": False,
                               "error": f"{type(exc).__name__}: {exc}"}, 500)

    # -------------------------------------------------------------- POST
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/command":
            return self._json({"ok": False, "error": "unknown route"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:
            return self._json({"ok": False, "error": f"bad JSON: {exc}"}, 400)
        try:
            result = KERNEL.command(payload)
            return self._json({"ok": True, "result": result})
        except KernelError as exc:
            # Failure is fed back to the guiding panel / control loop.
            state = None
            try:
                state = KERNEL.state()
            except Exception:
                pass
            return self._json({"ok": False, "error": str(exc), "state": state}, 400)
        except Exception as exc:
            return self._json({"ok": False,
                               "error": f"{type(exc).__name__}: {exc}"}, 500)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DirGuide 光照实验窗口")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print("=" * 64)
    print("  DirGuide 间接光方向引导实验台已启动")
    print(f"  本机浏览器打开: {url}")
    if args.host == "0.0.0.0":
        print(f"  局域网访问: http://<本机IP>:{args.port}/")
    print("  按 Ctrl+C 停止服务")
    print("=" * 64)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止中...")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
