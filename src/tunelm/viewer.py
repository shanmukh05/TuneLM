"""Serve the dataset viewer and one teacher query from the browser.

The page stays on this machine. API keys stay in the environment. The browser
sends only the provider, model id, and listener prompt.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tunelm.io import ROOT, load_env_file, project_path

PAGE = ROOT / "tools" / "dataset_viewer.html"
MAX_BODY = 100_000


class ViewerServer(ThreadingHTTPServer):
    """One local viewer. Tokenizer and memory load on the first query."""

    def __init__(self, address, handler, env_file, model_config, memory_dir, offline):
        super().__init__(address, handler)
        self.env_file = env_file
        self.model_config = model_config
        self.memory_dir = memory_dir
        self.offline = offline
        self._tokenizer = None
        self._memory = None
        self._ready = False
        self._lock = threading.Lock()

    def resources(self):
        with self._lock:
            return self._load()

    def _load(self):
        if self._ready:
            return self._tokenizer, self._memory
        from tunelm.io import load_config
        from tunelm.memory import Memory
        from tunelm.training import ModelSpec, load_tokenizer

        load_env_file(self.env_file)
        spec = ModelSpec.model_validate(load_config(self.model_config))
        self._tokenizer = load_tokenizer(spec, offline=self.offline)
        marker = project_path(Path(self.memory_dir) / "metadata.json")
        self._memory = Memory(self.memory_dir) if marker.exists() else None
        self._ready = True
        return self._tokenizer, self._memory


class ViewerHandler(BaseHTTPRequestHandler):
    """Serve the page and answer one composition query."""

    server: ViewerServer

    def do_GET(self):
        if self.path.split("?", 1)[0] not in {"/", "/dataset_viewer.html"}:
            self._send(404, {"error": "Not found"})
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/query":
            self._send(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": "Query body is empty or too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Query body must be an object")
            prompt = str(payload.get("prompt", "")).strip()
            model = str(payload.get("model", "")).strip()
            if not prompt or not model:
                raise ValueError("A prompt and a model id are required")
            tokenizer, memory = self.server.resources()
            from tunelm.synthesis import compose_request

            result = compose_request(
                prompt,
                str(payload.get("provider", "")),
                model,
                tokenizer=tokenizer,
                memory=memory,
            )
        except Exception as error:
            self._send(400, {"error": str(error)})
            return
        self._send(200, result)

    def log_message(self, fmt, *args):
        return

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_viewer(port: int, env_file: str, model_config: str, memory_dir: str, offline: bool) -> dict:
    """Block until interrupted. The page and the query endpoint share this port."""
    server = ViewerServer(
        ("127.0.0.1", port), ViewerHandler, env_file, model_config, memory_dir, offline
    )
    url = f"http://127.0.0.1:{port}/"
    print(f"Dataset viewer at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
    return {"url": url, "stopped": True}
