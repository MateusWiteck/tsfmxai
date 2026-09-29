"""Dependency-free local web interface for mechanism and trajectory generation."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(SOURCE_ROOT))

from tsfmxai.generators import (  # noqa: E402
    RandomTreeConfig,
    required_history_length,
    sample_random_mechanism,
    sample_univariate_trajectories,
)
from tsfmxai.representations import (  # noqa: E402
    Add,
    ExpressionNode,
    Lag,
    Multiply,
    Parameter,
    Time,
    children,
    leaf_count,
    node_count,
    operator_count,
    tree_depth,
)


class MechanismStore:
    """Thread-safe in-memory storage for mechanisms shown in the browser."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self._lock = threading.Lock()

    def add(self, mechanism: Any) -> str:
        mechanism_id = uuid.uuid4().hex
        with self._lock:
            self._items[mechanism_id] = mechanism
        return mechanism_id

    def get(self, mechanism_id: str) -> Any:
        with self._lock:
            try:
                return self._items[mechanism_id]
            except KeyError as error:
                raise ValueError(
                    "A árvore selecionada não existe mais; gere uma nova árvore."
                ) from error


STORE = MechanismStore()


def _node_payload(node: ExpressionNode, binding: Any) -> dict[str, Any]:
    if isinstance(node, Parameter):
        return {
            "type": "PARAMETER",
            "slot": node.slot,
            "value": float(binding.parameter_values[node.slot]),
            "children": [],
        }
    if isinstance(node, Time):
        return {"type": "TIME", "children": []}
    if isinstance(node, Lag):
        return {
            "type": "LAG",
            "series": node.variable,
            "slot": node.slot,
            "lag": int(binding.lag_values[node.slot]),
            "children": [],
        }
    if isinstance(node, Add):
        node_type = "ADD"
    elif isinstance(node, Multiply):
        node_type = "MULTIPLY"
    else:
        raise TypeError(type(node))
    return {
        "type": node_type,
        "children": [_node_payload(child, binding) for child in children(node)],
    }


def _mechanism_payload(
    mechanism_id: str,
    mechanism: Any,
    config: RandomTreeConfig,
) -> dict[str, Any]:
    history_length = required_history_length(mechanism.expression, mechanism.binding)
    return {
        "id": mechanism_id,
        "seed": mechanism.seed,
        "config": {
            "minimum_operators": config.minimum_operators,
            "maximum_operators": config.maximum_operators,
            "num_series": config.num_series,
            "minimum_lag": config.minimum_lag,
            "maximum_lag": config.maximum_lag,
            "require_lag": config.require_lag,
            "require_time": config.require_time,
        },
        "skeleton_formula": mechanism.skeleton_formula,
        "formula": mechanism.formula,
        "tree_text": mechanism.tree,
        "tree": _node_payload(mechanism.expression, mechanism.binding),
        "binding": {
            "parameters": dict(mechanism.binding.parameter_values),
            "lags": dict(mechanism.binding.lag_values),
        },
        "summary": {
            "operators": operator_count(mechanism.expression),
            "leaves": leaf_count(mechanism.expression),
            "nodes": node_count(mechanism.expression),
            "depth": tree_depth(mechanism.expression),
            "required_history": history_length,
        },
    }


def _as_int(data: dict[str, Any], key: str, default: int) -> int:
    return int(data.get(key, default))


def _as_float(data: dict[str, Any], key: str, default: float) -> float:
    return float(data.get(key, default))


class FrontendHandler(SimpleHTTPRequestHandler):
    """Serve static assets and the two local JSON endpoints."""

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 64_000:
            raise ValueError("Request body is too large.")
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        return payload

    def _write_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/mechanisms":
                self._sample_mechanism(payload)
                return
            if self.path == "/api/trajectories":
                self._sample_trajectories(payload)
                return
            self._write_json({"error": "Endpoint desconhecido."}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self.log_error("Unhandled API error: %s", error)
            self._write_json(
                {"error": "Erro interno durante a geração."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _sample_mechanism(self, data: dict[str, Any]) -> None:
        config = RandomTreeConfig(
            minimum_operators=_as_int(data, "minimum_operators", 1),
            maximum_operators=_as_int(data, "maximum_operators", 5),
            num_series=1,
            minimum_lag=_as_int(data, "minimum_lag", 1),
            maximum_lag=_as_int(data, "maximum_lag", 8),
            parameter_probability=_as_float(data, "parameter_probability", 0.45),
            time_probability=_as_float(data, "time_probability", 0.10),
            lag_probability=_as_float(data, "lag_probability", 0.45),
            add_probability=_as_float(data, "add_probability", 0.65),
            minimum_parameter_magnitude=_as_float(
                data, "minimum_parameter_magnitude", 0.05
            ),
            maximum_parameter_magnitude=_as_float(
                data, "maximum_parameter_magnitude", 0.90
            ),
            require_lag=bool(data.get("require_lag", True)),
            require_time=bool(data.get("require_time", False)),
        )
        mechanism = sample_random_mechanism(config, seed=_as_int(data, "seed", 20260930))
        mechanism_id = STORE.add(mechanism)
        self._write_json(_mechanism_payload(mechanism_id, mechanism, config))

    def _sample_trajectories(self, data: dict[str, Any]) -> None:
        mechanism = STORE.get(str(data.get("mechanism_id", "")))
        generated_steps = _as_int(data, "generated_steps", 60)
        if generated_steps < 1:
            raise ValueError("generated_steps must be positive in the visualization.")
        trajectories = sample_univariate_trajectories(
            mechanism.expression,
            mechanism.binding,
            trajectory_count=_as_int(data, "trajectory_count", 12),
            generated_steps=generated_steps,
            initial_value_minimum=_as_float(data, "initial_value_minimum", -1.0),
            initial_value_maximum=_as_float(data, "initial_value_maximum", 1.0),
            innovation_std=_as_float(data, "innovation_std", 0.0),
            root_seed=_as_int(data, "seed", 314159),
            maximum_absolute_value=None,
        )
        history_length = required_history_length(mechanism.expression, mechanism.binding)
        mean = trajectories.mean(axis=0)
        std = trajectories.std(axis=0)
        lower, upper = np.quantile(trajectories, [0.10, 0.90], axis=0)
        self._write_json(
            {
                "seed": _as_int(data, "seed", 314159),
                "times": list(range(trajectories.shape[1])),
                "generated_start": history_length,
                "trajectories": trajectories.tolist(),
                "summary": {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "p10": lower.tolist(),
                    "p90": upper.tolist(),
                    "minimum": float(trajectories.min()),
                    "maximum": float(trajectories.max()),
                },
            }
        )


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
