"""Serve the local viewer and route teacher and Modal requests from the browser."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tunelm.composition.responses import parse_completion
from tunelm.core.io import ROOT, load_env_file, project_path

PAGE = ROOT / "tools" / "dataset_viewer.html"
MAX_BODY = 100_000


class ViewerServer(ThreadingHTTPServer):
    """One local viewer. Tokenizer and memory load on the first query."""

    def __init__(self, address, handler, env_file, model_config, memory_dir, offline, modal_app):
        super().__init__(address, handler)
        self.env_file = env_file
        self.model_config = model_config
        self.memory_dir = memory_dir
        self.offline = offline
        self.modal_app = modal_app
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
        from tunelm.composition.memory import Memory
        from tunelm.core.io import load_config
        from tunelm.model.training import ModelSpec, load_tokenizer

        load_env_file(self.env_file)
        spec = ModelSpec.model_validate(load_config(self.model_config))
        self._tokenizer = load_tokenizer(spec, offline=self.offline)
        marker = project_path(Path(self.memory_dir) / "metadata.json")
        self._memory = Memory(self.memory_dir) if marker.exists() else None
        self._ready = True
        return self._tokenizer, self._memory


def predict_sft(prompt: str, app_name: str) -> dict:
    try:
        import modal
    except ImportError as error:
        raise RuntimeError(
            "Install the Modal extra in the viewer environment with pip install -e '.[modal]'"
        ) from error

    try:
        result = modal.Cls.from_name(app_name, "Composer")().predict.remote(prompt)
    except modal.exception.NotFoundError as error:
        raise RuntimeError(
            f"Modal app '{app_name}' was not found. Start it with "
            "modal deploy scripts/modal_inference.py and check your Modal profile."
        ) from error

    try:
        summary, code = parse_completion(result["completion"])
    except ValueError as error:
        return {**result, "prompt": prompt, "parse_error": str(error)}
    return {**result, "prompt": prompt, "summary": summary, "code": code}


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
        route = self.path.split("?", 1)[0]
        handlers = {"/api/query": self._query, "/api/predict": self._predict}
        if route not in handlers:
            self._send(404, {"error": "Not found"})
            return
        if route == "/api/predict" and self.headers.get_content_type() != "application/json":
            self._send(415, {"error": "Expected an application/json request"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Query body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Query body must be an object")
            result = handlers[route](payload)
        except ValueError as error:
            self._send(400, {"error": str(error)})
            return
        except Exception as error:
            self._send(502 if route == "/api/predict" else 400, {"error": str(error)})
            return
        self._send(200, result)

    def _query(self, payload):
        prompt = str(payload.get("prompt", "")).strip()
        model = str(payload.get("model", "")).strip()
        if not prompt or not model:
            raise ValueError("A prompt and a model id are required")
        tokenizer, memory = self.server.resources()
        from tunelm.data.synthesis import compose_request

        return compose_request(
            prompt, str(payload.get("provider", "")), model, tokenizer=tokenizer, memory=memory
        )

    def _predict(self, payload):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A listener prompt is required")
        return predict_sft(prompt.strip(), self.server.modal_app)

    def log_message(self, fmt, *args):
        return

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_viewer(
    port: int,
    env_file: str,
    model_config: str,
    memory_dir: str,
    offline: bool,
    modal_app: str = "tunelm-sft-inference",
) -> dict:
    """Block until interrupted. The page and the query endpoint share this port."""
    server = ViewerServer(
        ("127.0.0.1", port), ViewerHandler, env_file, model_config, memory_dir, offline, modal_app
    )
    url = f"http://127.0.0.1:{port}/"
    print(f"Dataset viewer at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
    return {"url": url, "stopped": True}
