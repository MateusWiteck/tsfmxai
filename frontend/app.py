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
    SampledMechanism,
    expression_required_history_length,
    required_multivariate_history_length,
)
from tsfmxai.representations import (  # noqa: E402
    Add,
    ExpressionNode,
    Lag,
    Multiply,
    Parameter,
    Time,
    Tanh,
    children,
    leaf_count,
    node_count,
    operator_count,
    tree_depth,
    walk,
)
from tsfmxai.visualization.studio_generation import (  # noqa: E402
    sample_studio_system, sample_studio_trajectories,
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
                    "O sistema selecionado não existe mais; gere um novo sistema."
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
    elif isinstance(node, Tanh):
        node_type = "TANH"
    else:
        raise TypeError(type(node))
    return {
        "type": node_type,
        "children": [_node_payload(child, binding) for child in children(node)],
    }


def _equation_payload(
    mechanism: SampledMechanism,
    target_series: int,
) -> dict[str, Any]:
    history_length = expression_required_history_length(
        mechanism.expression,
        mechanism.binding,
    )
    dependencies: dict[int, set[int]] = {}
    for node in walk(mechanism.expression):
        if isinstance(node, Lag):
            dependencies.setdefault(node.variable, set()).add(
                int(mechanism.binding.lag_values[node.slot])
            )
    return {
        "target_series": target_series,
        "seed": mechanism.seed,
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
        "dependencies": [
            {"source_series": source, "lags": sorted(lags)}
            for source, lags in sorted(dependencies.items())
        ],
    }


def _system_payload(
    mechanism_id: str,
    mechanisms: list[SampledMechanism],
    config: RandomTreeConfig,
    root_seed: int,
) -> dict[str, Any]:
    equations = [
        _equation_payload(mechanism, target_series)
        for target_series, mechanism in enumerate(mechanisms)
    ]
    history_length = required_multivariate_history_length(
        [mechanism.expression for mechanism in mechanisms],
        [mechanism.binding for mechanism in mechanisms],
    )
    dependency_lags = [
        [
            next(
                (
                    item["lags"]
                    for item in equation["dependencies"]
                    if item["source_series"] == source_series
                ),
                [],
            )
            for source_series in range(config.num_series)
        ]
        for equation in equations
    ]
    return {
        "id": mechanism_id,
        "seed": root_seed,
        "num_series": config.num_series,
        "config": {
            "minimum_operators": config.minimum_operators,
            "maximum_operators": config.maximum_operators,
            "num_series": config.num_series,
            "minimum_lag": config.minimum_lag,
            "maximum_lag": config.maximum_lag,
            "require_lag": config.require_lag,
            "require_time": config.require_time,
        },
        "equations": equations,
        "dependency_lags": dependency_lags,
        "summary": {
            "equations": len(equations),
            "operators": sum(item["summary"]["operators"] for item in equations),
            "nodes": sum(item["summary"]["nodes"] for item in equations),
            "maximum_depth": max(item["summary"]["depth"] for item in equations),
            "required_history": history_length,
        },
    }


def _as_int(data: dict[str, Any], key: str, default: int) -> int:
    return int(data.get(key, default))


def _as_float(data: dict[str, Any], key: str, default: float) -> float:
    return float(data.get(key, default))


def _trajectory_payload(
    trajectories: np.ndarray,
    *,
    seed: int,
    history_length: int,
) -> dict[str, Any]:
    """Convert a ``[trajectory, series, time]`` array to the frontend contract."""

    if trajectories.ndim != 3:
        raise ValueError("Frontend trajectories must have three dimensions.")
    mean = trajectories.mean(axis=0)
    std = trajectories.std(axis=0)
    lower, upper = np.quantile(trajectories, [0.10, 0.90], axis=0)
    minimum = trajectories.min(axis=(0, 2))
    maximum = trajectories.max(axis=(0, 2))
    return {
        "seed": seed,
        "times": list(range(trajectories.shape[2])),
        "generated_start": history_length,
        "trajectories": trajectories.tolist(),
        "summary": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "p10": lower.tolist(),
            "p90": upper.tolist(),
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
        },
    }


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
                self._sample_system(payload)
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

    def _sample_system(self, data: dict[str, Any]) -> None:
        config = RandomTreeConfig(
            minimum_operators=_as_int(data, "minimum_operators", 1),
            maximum_operators=_as_int(data, "maximum_operators", 5),
            num_series=2,
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
        root_seed = _as_int(data, "seed", 20260930)
        system = sample_studio_system(config, method=str(data.get("stability_method", "none")),
                                     seed=root_seed, innovation=_as_float(data, "innovation_std", .05),
                                     shao_wu_mode=str(data.get("shao_wu_mode", "product_lags")))
        mechanism_id = STORE.add(system)
        payload = _system_payload(mechanism_id, system.equations, system.tree_config, root_seed)
        payload["stability_method"] = system.method
        if system.method.startswith("shao_wu"):
            payload["shao_wu_mode"] = system.retained.lag_transform
        payload["certificates"] = system.certificates
        payload["innovation"] = system.innovation
        self._write_json(payload)

    def _sample_trajectories(self, data: dict[str, Any]) -> None:
        system = STORE.get(str(data.get("mechanism_id", "")))
        generated_steps = _as_int(data, "generated_steps", 60)
        if generated_steps < 1:
            raise ValueError("generated_steps must be positive in the visualization.")
        trajectories = sample_studio_trajectories(
            system,
            trajectory_count=_as_int(data, "trajectory_count", 12),
            generated_steps=generated_steps,
            initial_value_minimum=_as_float(data, "initial_value_minimum", -1.0),
            initial_value_maximum=_as_float(data, "initial_value_maximum", 1.0),
            innovation_std=_as_float(data, "innovation_std", 0.0),
            root_seed=_as_int(data, "seed", 314159),
        )
        history_length = system.history_length
        self._write_json(
            _trajectory_payload(
                trajectories,
                seed=_as_int(data, "seed", 314159),
                history_length=history_length,
            )
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
