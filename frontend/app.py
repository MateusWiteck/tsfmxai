"""Dependency-free local web interface for the multivariate training generator."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from collections import OrderedDict
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfmxai.visualization.studio_multiseries import (  # noqa: E402
    SYSTEM_DEFAULTS,
    TRAJECTORY_DEFAULTS,
    sample_studio_multiseries,
    sample_studio_trajectories,
    system_payload,
)


class SystemStore:
    """Thread-safe, bounded in-memory storage for systems shown in the browser."""

    def __init__(self, capacity: int = 64) -> None:
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, system: Any) -> str:
        system_id = uuid.uuid4().hex
        with self._lock:
            self._items[system_id] = system
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return system_id

    def get(self, system_id: str) -> Any:
        with self._lock:
            try:
                return self._items[system_id]
            except KeyError as error:
                raise ValueError(
                    "O sistema selecionado não existe mais; gere um novo sistema."
                ) from error


STORE = SystemStore()


class FrontendHandler(SimpleHTTPRequestHandler):
    """Serve static assets and the local JSON endpoints."""

    def end_headers(self):
        # Local development assets must reflect the currently installed generator UI.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 64_000:
            raise ValueError("Request body is too large.")
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        return payload

    def _write_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/defaults":
            self._write_json({"system": SYSTEM_DEFAULTS, "trajectory": TRAJECTORY_DEFAULTS})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/systems":
                self._sample_system(payload)
            elif self.path == "/api/trajectories":
                self._sample_trajectories(payload)
            else:
                self._write_json({"error": "Endpoint desconhecido."}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self.log_error("Unhandled API error: %s", error)
            self._write_json(
                {"error": "Erro interno durante a geração."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _sample_system(self, data: dict[str, Any]) -> None:
        system = sample_studio_multiseries(data)
        self._write_json(system_payload(system, STORE.add(system)))

    def _sample_trajectories(self, data: dict[str, Any]) -> None:
        system = STORE.get(str(data.get("system_id", "")))
        payload, final_system = sample_studio_trajectories(system, data)
        if final_system is not system:
            # The generator discarded the displayed system; hand the accepted one to the browser.
            payload["replacement_system"] = system_payload(final_system, STORE.add(final_system))
        self._write_json(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()

    handler = partial(FrontendHandler, directory=str(STATIC_ROOT))
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    print(f"tsfmxai frontend: http://{arguments.host}:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
