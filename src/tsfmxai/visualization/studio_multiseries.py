"""Preview adapter for the multivariate training generator.

The studio does not reimplement any scientific behavior: systems come from
``sample_multiseries_system`` and trajectories from ``TrainingBatchStream``, the
same stream used to build training batches. Every request field below maps to a
field of ``MultiseriesConfig``, ``NoiseConfig``, ``HistorySpec``,
``TrajectoryConfig`` or ``TrainingFormat``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsfmxai.datasets.multiseries import (
    TrainingBatchStream, TrainingFormat, TrajectoryConfig, generate_training_batches,
)
from tsfmxai.datasets.hybrid import generate_hybrid_training_batches
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.hybrid import HybridConfig, HybridSystem, kernel_formula, sample_hybrid_system
from tsfmxai.generators.initialization import HistorySpec, RandomHistorySpec
from tsfmxai.generators.multiseries import (
    DEPENDENCIES, MultiseriesConfig, MultiseriesSystem, NoiseConfig,
    sample_multiseries_system,
)
from tsfmxai.representations import (
    Add, ExpressionNode, Lag, Multiply, Parameter, Tanh, children,
    format_canonical_polynomial, leaf_count, node_count, operator_count, tree_depth, walk,
)

MAX_CHANNELS = 12
MAX_TRAJECTORIES = 64
MAX_STEPS = 1024
INITIALIZATIONS = ("random", "zero", "constant", "uniform", "cosine")

SYSTEM_DEFAULTS: dict[str, Any] = {
    "seed": 20261101,
    "channels": 6,
    "dependency": "random",
    "max_external_sources": 2,
    "self_source_weight": 3.0,
    "jacobian_share": 0.5,
    "kernel_synth_probability": 0.25,
    "target_gain": 0.8,
    "safety_factor": 0.95,
    "noise_minimum": 0.01,
    "noise_maximum": 0.1,
    "minimum_operators": 1,
    "maximum_operators": 6,
    "minimum_lag": 1,
    "maximum_lag": 16,
    "max_degree": 4,
    "minimum_parameter_magnitude": 0.05,
    "maximum_parameter_magnitude": 1.0,
}

TRAJECTORY_DEFAULTS: dict[str, Any] = {
    "seed": 314159,
    "trajectory_count": 8,
    "context_length": 96,
    "prediction_length": 96,
    "initialization": "random",
    "constant_probability": 0.5,
    "constant_minimum": 0.0,
    "constant_maximum": 1.0,
    "white_noise_probability": 0.5,
    "white_noise_minimum": 0.1,
    "white_noise_maximum": 1.0,
    "cosine_probability": 0.5,
    "cosine_amplitude_minimum": 0.1,
    "cosine_amplitude_maximum": 1.0,
    "cosine_period_minimum": 4,
    "cosine_period_maximum": 48,
    "amplitude": 1.0,
    "offset": 0.0,
    "period": 48,
    "fraction": 1.0,
    "resample_innovations": False,
}


def _number(data: Mapping[str, Any], defaults: Mapping[str, Any], key: str, kind: type):
    value = data.get(key, defaults[key])
    if kind is bool:
        if type(value) is not bool:
            raise ValueError(f"{key} deve ser verdadeiro ou falso.")
        return value
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{key} deve ser numérico.")
    if kind is int:
        if float(value) != int(value):
            raise ValueError(f"{key} deve ser um inteiro.")
        return int(value)
    if not np.isfinite(value):
        raise ValueError(f"{key} deve ser finito.")
    return float(value)


def build_system_config(data: Mapping[str, Any]) -> tuple[MultiseriesConfig, int]:
    """Translate the studio's request into the generator's own configuration."""

    d = SYSTEM_DEFAULTS
    channels = _number(data, d, "channels", int)
    if not 1 <= channels <= MAX_CHANNELS:
        raise ValueError(f"O número de canais deve estar entre 1 e {MAX_CHANNELS}.")
    dependency = data.get("dependency", d["dependency"])
    if dependency not in DEPENDENCIES:
        raise ValueError(f"Topologia desconhecida; use {', '.join(DEPENDENCIES)}.")
    # Channels with an external source always use Shao-Wu; this share splits
    # the self-only channels between Jacobian and Heaps.
    jacobian = _number(data, d, "jacobian_share", float)
    if not 0 <= jacobian <= 1:
        raise ValueError("A fração Jacobiano deve estar em [0, 1].")
    low, high = _number(data, d, "noise_minimum", float), _number(data, d, "noise_maximum", float)
    ast = CombinatorialTreeConfig(
        minimum_operators=_number(data, d, "minimum_operators", int),
        maximum_operators=_number(data, d, "maximum_operators", int),
        minimum_lag=_number(data, d, "minimum_lag", int),
        maximum_lag=_number(data, d, "maximum_lag", int),
        max_degree=_number(data, d, "max_degree", int),
        minimum_parameter_magnitude=_number(data, d, "minimum_parameter_magnitude", float),
        maximum_parameter_magnitude=_number(data, d, "maximum_parameter_magnitude", float),
        require_lag=False,
    )
    config = MultiseriesConfig(
        ast_config=ast,
        channel_count_probs={channels: 1.0},
        dependency_probs={dependency: 1.0},
        self_only_treatment_probs={"jacobian": jacobian, "heaps": 1.0 - jacobian},
        max_external_sources=_number(data, d, "max_external_sources", int),
        self_source_weight=_number(data, d, "self_source_weight", float),
        noise=NoiseConfig(std_range=(low, high)),
        target_gain=_number(data, d, "target_gain", float),
        safety_factor=_number(data, d, "safety_factor", float),
    )
    seed = _number(data, d, "seed", int)
    if seed < 0:
        raise ValueError("A seed deve ser não negativa.")
    return config, seed


def sample_studio_multiseries(data: Mapping[str, Any]) -> MultiseriesSystem | HybridSystem:
    """Sample with the training generator; KernelSynth channels use the hybrid one."""
    config, seed = build_system_config(data)
    probability = _number(data, SYSTEM_DEFAULTS, "kernel_synth_probability", float)
    if not 0 <= probability < 1:
        raise ValueError("A probabilidade KernelSynth deve estar em [0, 1).")
    if probability == 0:
        return sample_multiseries_system(config, seed=seed)
    hybrid = HybridConfig(config, {"ast": 1.0 - probability, "kernel_synth": probability})
    return sample_hybrid_system(hybrid, seed=seed)


def _families(system) -> tuple[str, ...]:
    return getattr(system, "channel_families", ("ast",) * system.num_channels)


def _node_payload(node: ExpressionNode, binding: Any) -> dict[str, Any]:
    if isinstance(node, Parameter):
        return {"type": "PARAMETER", "slot": node.slot,
                "value": float(binding.parameter_values[node.slot]), "children": []}
    if isinstance(node, Lag):
        return {"type": "LAG", "series": node.variable, "slot": node.slot,
                "lag": int(binding.lag_values[node.slot]), "children": []}
    names = {Add: "ADD", Multiply: "MULTIPLY", Tanh: "TANH"}
    if type(node) not in names:
        raise TypeError(type(node))
    return {"type": names[type(node)],
            "children": [_node_payload(child, binding) for child in children(node)]}


def _kernel_payload(system, target: int) -> dict[str, Any]:
    """A KernelSynth channel has no equation: describe its Gaussian-process kernel."""
    formula = kernel_formula(system.kernel_specs[target])
    return {
        "target": target,
        "family": "kernel_synth",
        "kernel_formula": formula,
        "original_formula": formula,
        "executed_formula": f"KernelSynth[{formula}]",
        "skeleton_formula": "",
        "canonical_formula": "sem equação: série de processo gaussiano (KernelSynth)",
        "canonical_terms": 0,
        "tree": {"type": "PARAMETER", "slot": "KERNEL_SYNTH", "value": 0.0, "children": []},
        "binding": {"parameters": {}, "lags": {}},
        "treatment": {"method": "kernel_synth", "scale": 1.0},
        "innovation": {"family": "none", "std": 0.0},
        "allowed_sources": [target],
        "effective_sources": [],
        "lags_by_source": {},
        "summary": {"operators": 0, "leaves": 0, "nodes": 0, "depth": 0, "degree": 0},
    }


def _equation_payload(system: MultiseriesSystem, target: int) -> dict[str, Any]:
    if _families(system)[target] == "kernel_synth":
        return _kernel_payload(system, target)
    equation, original = system.equations[target], system.original_equations[target]
    lags: dict[int, set[int]] = {}
    for node in walk(equation.expression):
        if isinstance(node, Lag):
            lags.setdefault(node.variable, set()).add(int(equation.binding.lag_values[node.slot]))
    polynomial = system.canonical[target]
    treatment, innovation = system.treatments[target], system.innovations[target]
    return {
        "target": target,
        "family": "ast",
        "original_formula": original.formula,
        "executed_formula": equation.formula,
        "skeleton_formula": equation.skeleton_formula,
        "canonical_formula": format_canonical_polynomial(polynomial),
        "canonical_terms": len(polynomial.terms),
        "tree": _node_payload(equation.expression, equation.binding),
        "binding": {"parameters": dict(equation.binding.parameter_values),
                    "lags": dict(equation.binding.lag_values)},
        "treatment": {"method": treatment.method, "scale": system.treatment_scales[target]},
        "innovation": {"family": innovation.family, "std": innovation.std},
        "allowed_sources": list(system.allowed_sources[target]),
        "effective_sources": list(system.effective_sources[target]),
        "lags_by_source": {str(s): sorted(v) for s, v in sorted(lags.items())},
        "summary": {
            "operators": operator_count(equation.expression),
            "leaves": leaf_count(equation.expression),
            "nodes": node_count(equation.expression),
            "depth": tree_depth(equation.expression),
            "degree": max((len(term.monomial) for term in polynomial.terms), default=0),
        },
    }


def system_payload(system: MultiseriesSystem, system_id: str) -> dict[str, Any]:
    n = system.num_channels
    equations = [_equation_payload(system, target) for target in range(n)]
    matrix = [[equation["lags_by_source"].get(str(source), []) for source in range(n)]
              for equation in equations]
    effective_edges = sum(1 for e in equations for s in e["effective_sources"] if s != e["target"])
    return {
        "id": system_id,
        "seed": system.seed,
        "num_channels": n,
        "dependency": system.dependency,
        "channel_order": list(system.channel_order),
        "history_length": system.history_length,
        "equations": equations,
        "dependency_lags": matrix,
        "summary": {
            "channels": n,
            "operators": sum(e["summary"]["operators"] for e in equations),
            "nodes": sum(e["summary"]["nodes"] for e in equations),
            "allowed_edges": sum(len(s) - 1 for s in system.allowed_sources),
            "effective_edges": effective_edges,
            "shao_wu_channels": sum(t.method == "shao_wu" for t in system.treatments),
            "kernel_synth_channels": sum(f == "kernel_synth" for f in _families(system)),
            "history_length": system.history_length,
        },
    }


def build_history_spec(data: Mapping[str, Any]) -> HistorySpec | RandomHistorySpec:
    d = TRAJECTORY_DEFAULTS
    kind = data.get("initialization", d["initialization"])
    if kind not in INITIALIZATIONS:
        raise ValueError(f"Inicialização desconhecida; use {', '.join(INITIALIZATIONS)}.")
    if kind == "random":
        def interval(name, kind_=float):
            return (_number(data, d, f"{name}_minimum", kind_), _number(data, d, f"{name}_maximum", kind_))
        try:
            return RandomHistorySpec(
                "studio random families",
                constant_probability=_number(data, d, "constant_probability", float),
                constant_magnitude=interval("constant"),
                noise_probability=_number(data, d, "white_noise_probability", float),
                noise_radius=interval("white_noise"),
                cosine_probability=_number(data, d, "cosine_probability", float),
                cosine_amplitude=interval("cosine_amplitude"),
                cosine_period=interval("cosine_period", int),
            )
        except ValueError as error:
            raise ValueError(f"Inicialização por famílias inválida: {error}") from None
    return HistorySpec(
        f"studio {kind}", kind,
        amplitude=_number(data, d, "amplitude", float),
        offset=_number(data, d, "offset", float),
        period=_number(data, d, "period", int),
        fraction=_number(data, d, "fraction", float),
    )


def _plain(array: np.ndarray) -> list:
    """Shortest float32 round-trip text keeps the payload compact and exact."""

    flat = [float(str(value)) for value in array.ravel()]
    return np.asarray(flat, dtype=np.float64).reshape(array.shape).tolist()


def trajectory_payload(trajectories: np.ndarray, *, seed: int, history_length: int,
                       **extra: Any) -> dict[str, Any]:
    """Convert a ``[trajectory, series, time]`` array to the frontend contract."""

    if trajectories.ndim != 3:
        raise ValueError("Frontend trajectories must have three dimensions.")
    lower, upper = np.quantile(trajectories, [0.10, 0.90], axis=0)
    return {
        "seed": seed,
        "times": list(range(trajectories.shape[2])),
        "generated_start": history_length,
        "trajectories": _plain(trajectories),
        "summary": {
            "mean": trajectories.mean(axis=0).tolist(),
            "std": trajectories.std(axis=0).tolist(),
            "p10": lower.tolist(),
            "p90": upper.tolist(),
            "minimum": trajectories.min(axis=(0, 2)).tolist(),
            "maximum": trajectories.max(axis=(0, 2)).tolist(),
        },
        **extra,
    }


@dataclass(frozen=True)
class TrajectoryRequest:
    count: int
    context_length: int
    prediction_length: int
    seed: int
    history: HistorySpec | RandomHistorySpec
    resample_innovations: bool


def parse_trajectory_request(data: Mapping[str, Any]) -> TrajectoryRequest:
    d = TRAJECTORY_DEFAULTS
    request = TrajectoryRequest(
        count=_number(data, d, "trajectory_count", int),
        context_length=_number(data, d, "context_length", int),
        prediction_length=_number(data, d, "prediction_length", int),
        seed=_number(data, d, "seed", int),
        history=build_history_spec(data),
        resample_innovations=_number(data, d, "resample_innovations", bool),
    )
    if not 1 <= request.count <= MAX_TRAJECTORIES:
        raise ValueError(f"O número de trajetórias deve estar entre 1 e {MAX_TRAJECTORIES}.")
    if request.context_length < 1 or request.prediction_length < 1:
        raise ValueError("Contexto e horizonte devem ser positivos.")
    if request.context_length + request.prediction_length > MAX_STEPS:
        raise ValueError(f"Contexto + horizonte não pode passar de {MAX_STEPS} passos.")
    if request.seed < 0:
        raise ValueError("A seed deve ser não negativa.")
    return request


def sample_studio_stream(system: MultiseriesSystem, request: TrajectoryRequest) -> TrainingBatchStream:
    """One window per trajectory covers exactly context + horizon, as in training.

    The stream re-samples the displayed system from its own config and seed, so
    its first draw equals ``system``; replacements follow the training policy.
    """

    width = request.context_length + request.prediction_length
    hybrid = getattr(system, "config", None) if isinstance(system, HybridSystem) else None
    if hybrid is not None:
        generator, config = generate_hybrid_training_batches, {"config": hybrid}
    else:
        generator, config = generate_training_batches, {"config": system.sampling_config}
    return generator(
        **config, num_systems=1, system_seeds=[system.seed],
        trajectories_per_system=request.count, seed=request.seed,
        trajectory=TrajectoryConfig(
            generated_steps=width, initialization=request.history,
            resample_innovations=request.resample_innovations),
        training=TrainingFormat(
            context_length=request.context_length, prediction_length=request.prediction_length,
            stride=width, batch_size=request.count, dtype="float32",
            layout="batch_channel_time"),
    )


def sample_studio_trajectories(
        system: MultiseriesSystem, data: Mapping[str, Any]) -> tuple[dict[str, Any], MultiseriesSystem]:
    """Return the payload and the system actually used (a replacement if one was drawn)."""

    request = parse_trajectory_request(data)
    stream = sample_studio_stream(system, request)
    batches = list(stream)
    if not batches:
        raise ValueError(
            "Nenhuma trajetória finita foi gerada; reduza o horizonte ou os coeficientes, "
            "ou mude a inicialização. Motivos: "
            + ", ".join(sorted({f['reason'] for f in stream.stats.failures})))
    batch = batches[0]
    final_system = batch.metadata[0]["system"]
    n = final_system.num_channels
    paths = np.concatenate((batch.context, batch.future), axis=2)[:, :n]
    histories = np.stack([m["initial_history"] for m in batch.metadata]).astype(np.float32)
    stats = stream.stats
    reasons: dict[str, int] = {}
    for failure in stats.failures:
        reasons[failure["reason"]] = reasons.get(failure["reason"], 0) + 1
    payload = trajectory_payload(
        paths, seed=request.seed, history_length=0,
        context_length=request.context_length,
        prediction_length=request.prediction_length,
        trajectory_indices=[m["trajectory_index"] for m in batch.metadata],
        initial_history=_plain(histories),
        initialization=request.history.kind,
        initialization_components=[m["initialization_components"] for m in batch.metadata],
        stats={"requested": request.count, "accepted": stats.accepted_trajectories,
               "skipped": stats.skipped_trajectories,
               "replaced_systems": stats.replaced_systems,
               "resampled_equations": stats.resampled_equations,
               "resampled_original_equations": stats.resampled_original_equations,
               "replacement_records": stats.replacement_records,
               "discarded_trajectories": stats.discarded_trajectories,
               "initialization_attempts": stats.initialization_attempts,
               "failure_reasons": reasons},
    )
    # The stream re-samples the displayed system from its seed; only a real
    # replacement should be reported as a different system.
    return payload, (system if stats.replaced_systems == 0 else final_system)
