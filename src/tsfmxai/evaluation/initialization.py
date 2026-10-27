"""Paired initialization experiments using unchanged generator mechanisms.

Histories are chronological; returned paths contain generated observations only.
All case comparisons reuse explicit innovations. Failed paths remain in counts.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from tsfmxai.evaluation.stability import StabilityDiagnosticConfig, diagnose_stability_prefixes
from tsfmxai.evaluation.tail_behavior import (
    TailDiagnosticConfig, classify_tail_windows, measure_tail_prefixes,
)
from tsfmxai.generators.contraction import ContractionTreeConfig, ContractiveMechanism, sample_contractive_mechanism
from tsfmxai.generators.heaps import HeapsVARConfig, HeapsVARMechanism, initialize_heaps_history, sample_heaps_var_mechanism
from tsfmxai.generators.random_expression import RandomTreeConfig, derive_sample_seed, sample_random_mechanism
from tsfmxai.generators.shao_wu import (
    AmplitudeMomentCertificate, ShaoWuConfig, ShaoWuMechanism, sample_shao_wu_mechanism,
)
from tsfmxai.generators.stochastic import simulate_innovation_path
from tsfmxai.generators.trajectory import required_multivariate_history_length


GROUPS = ("Original AST", "Jacobian", "Shao-Wu: factors", "Shao-Wu: output", "Heaps")


@dataclass(frozen=True)
class InitializationStudyConfig:
    mechanisms: int = 64
    trajectories: int = 3
    mechanism_seed: int = 2026102601
    trajectory_seed: int = 2026102602
    horizons: tuple[int, ...] = (96, 192, 384)
    innovation_std: float = .05
    explosion_limit: float = 1e6
    near_zero_radius: float = .15
    tail_steps: int = 48
    flat_tolerance: float = .15
    window_steps: int = 24
    constant_value: float = .5
    uniform_radius: float = 1.
    magnitude_scales: tuple[float, ...] = (0., .1, .5, 1., 2., 4.)
    forgetting_tolerance: float = 1e-6
    forgetting_tail: int = 24
    minimum_lag: int = 1
    maximum_lag: int = 12
    heaps_fixed_order: bool = False
    example_mechanisms: int = 3

    def __post_init__(self):
        if self.mechanisms < 1 or self.trajectories < 1:
            raise ValueError("Positive mechanism and trajectory counts required.")
        if not 1 <= self.minimum_lag <= self.maximum_lag:
            raise ValueError("Invalid lag range.")
        numeric = (self.innovation_std, self.constant_value, self.uniform_radius,
                   self.forgetting_tolerance, *self.magnitude_scales)
        if not all(np.isfinite(v) and v >= 0 for v in numeric):
            raise ValueError("Scales and tolerances must be finite and nonnegative.")
        if self.innovation_std <= 0 or self.uniform_radius <= 0:
            raise ValueError("Positive noise and uniform radius required.")
        if not 1 <= self.forgetting_tail <= min(self.horizons):
            raise ValueError("Invalid forgetting tail length.")
        self.tail_config.validate_horizons(self.horizons)

    @property
    def diagnostic_config(self):
        return StabilityDiagnosticConfig(self.horizons, self.explosion_limit,
                                         self.near_zero_radius, self.tail_steps)

    @property
    def tail_config(self):
        return TailDiagnosticConfig(self.window_steps, 2, self.flat_tolerance)


def generator_configs(config):
    tree = RandomTreeConfig(time_probability=0., parameter_probability=.55,
                            lag_probability=.45, add_probability=.8,
                            minimum_lag=config.minimum_lag, maximum_lag=config.maximum_lag,
                            maximum_parameter_magnitude=.8)
    return {
        "Original AST": tree,
        "Jacobian": ContractionTreeConfig(
            tree_config=replace(tree, coefficient_gate_lags=True),
            innovation_bound=np.sqrt(3.) * config.innovation_std),
        "Shao-Wu: factors": ShaoWuConfig(tree_config=tree, lag_transform="product_lags",
                                         innovation_std=config.innovation_std),
        "Shao-Wu: output": ShaoWuConfig(tree_config=tree, lag_transform="product_output",
                                        innovation_std=config.innovation_std),
        "Heaps": HeapsVARConfig(innovation_std=config.innovation_std,
                               minimum_order=config.maximum_lag if config.heaps_fixed_order else 1,
                               maximum_order=config.maximum_lag),
    }


def sample_study_mechanisms(group, config):
    """Reuse mechanism identities across all three notebooks and all cases."""
    settings = generator_configs(config)[group]
    sampler = (sample_random_mechanism if group == "Original AST" else
               sample_contractive_mechanism if group == "Jacobian" else
               sample_heaps_var_mechanism if group == "Heaps" else sample_shao_wu_mechanism)
    return [sampler(settings, seed=derive_sample_seed(config.mechanism_seed, i), sample_index=i)
            for i in range(config.mechanisms)]


def mechanism_parts(mechanism):
    if hasattr(mechanism, "expressions"):
        expressions, bindings = mechanism.expressions, mechanism.bindings
    else:
        expressions, bindings = (mechanism.expression,), (mechanism.binding,)
    order = (mechanism.order if isinstance(mechanism, HeapsVARMechanism)
             else required_multivariate_history_length(expressions, bindings))
    return expressions, bindings, max(1, order)


def paired_inputs(mechanism, config, mechanism_index, trajectory_index):
    """Case-independent local streams; stationary history uses a separate stream."""
    expressions, _, order = mechanism_parts(mechanism)
    init_seq, noise_seq, stationary_seq = np.random.SeedSequence(
        [config.trajectory_seed, mechanism_index, trajectory_index]
    ).spawn(3)
    shape = (len(expressions), order)
    unit_history = np.random.default_rng(init_seq).uniform(-1., 1., shape)
    rng = np.random.default_rng(noise_seq)
    noise_shape = (len(expressions), max(config.horizons))
    if isinstance(mechanism, ContractiveMechanism):
        bound = mechanism.certificate.innovation_bound
        innovations = rng.uniform(-bound, bound, noise_shape)
    elif isinstance(mechanism, HeapsVARMechanism):
        innovations = np.linalg.cholesky(mechanism.innovation_covariance) @ rng.standard_normal(noise_shape)
    else:
        std = mechanism.innovation_std if isinstance(mechanism, ShaoWuMechanism) else config.innovation_std
        innovations = rng.normal(0., std, noise_shape)
    stationary_seed = int(stationary_seq.generate_state(1, dtype=np.uint64)[0])
    return unit_history, innovations, stationary_seed


def history_admissible(mechanism, history):
    history = np.asarray(history)
    if not np.isfinite(history).all():
        return False
    return not isinstance(mechanism, ContractiveMechanism) or bool(
        np.max(np.abs(history)) <= mechanism.certificate.domain_radius + 1e-12)


def amplitude_bound_for_history(mechanism, history):
    """Recompute the output-mode bound for this fixed history, without rescaling F."""
    if not isinstance(mechanism, ShaoWuMechanism) or not isinstance(mechanism.certificate, AmplitudeMomentCertificate):
        return np.nan
    cert = mechanism.certificate
    initial = float(np.max(np.linalg.norm(history, axis=0)))
    return max(initial, (cert.bounded_term_norm + cert.innovation_l2) / (1 - cert.total_gain))


def simulate_from_history(mechanism, history, innovations, config):
    if not history_admissible(mechanism, history):
        raise ValueError("History lies outside the certified domain or is nonfinite.")
    expressions, bindings, _ = mechanism_parts(mechanism)
    path = simulate_innovation_path(expressions, bindings, history, innovations,
                                    stop_threshold=config.explosion_limit)
    if isinstance(mechanism, ContractiveMechanism):
        if not np.isfinite(path).all() or np.any(np.abs(path) > mechanism.certificate.domain_radius + 1e-10):
            raise AssertionError("Certified domain violated.")
    return path


def path_diagnostics(path, config):
    diagnostics = diagnose_stability_prefixes(np.max(np.abs(path), axis=0), config.diagnostic_config)
    tails = measure_tail_prefixes(path, config.diagnostic_config, config.tail_config)
    return [dict(horizon=d.horizon, exploded=d.exploded, near_zero=d.near_zero,
                 success=not d.exploded, explosion_step=d.explosion_step,
                 **classify_tail_windows(t, config.tail_config, near_zero_radius=config.near_zero_radius))
            for d, t in zip(diagnostics, tails)]


@dataclass(frozen=True)
class HistorySpec:
    """An explicit chronological startup law, separate from the recurrence.

    Cosine histories are prefixes of one cycle sampled at integer k. A fraction
    of 1/2 keeps the *first* half of that same cycle, without padding, resampling,
    or changing its period. Each prefix must cover the required lag history.
    """

    name: str
    kind: str
    amplitude: float = 1.
    offset: float = 0.
    period: int = 48
    fraction: float = 1.

    def __post_init__(self):
        if self.kind not in {"zero", "constant", "uniform", "stationary", "cosine"}:
            raise ValueError("Unknown history kind.")
        if not np.isfinite([self.amplitude, self.offset]).all() or self.amplitude < 0:
            raise ValueError("History amplitude must be finite and nonnegative; offset finite.")
        if self.kind == "cosine":
            if (not isinstance(self.period, int) or self.period < 4
                    or self.fraction not in (1., .5, .25)
                    or not float(self.period * self.fraction).is_integer()):
                raise ValueError("Cosine requires an integer period and a whole number of prefix samples.")


def make_initial_history(spec, mechanism, unit_history, stationary_seed):
    """Construct the declared history without clipping or changing the mechanism."""
    if spec.kind == "zero":
        return np.zeros_like(unit_history) + spec.offset
    if spec.kind == "constant":
        return np.full_like(unit_history, spec.offset)
    if spec.kind == "uniform":
        return spec.offset + spec.amplitude * unit_history
    if spec.kind == "stationary":
        if not isinstance(mechanism, HeapsVARMechanism):
            raise ValueError("Joint stationary initializer available only for Heaps.")
        return spec.offset + spec.amplitude * initialize_heaps_history(mechanism, seed=stationary_seed)
    length = int(spec.period * spec.fraction)
    if length < unit_history.shape[1]:
        raise ValueError("Cosine prefix is shorter than the required history.")
    values = spec.offset + spec.amplitude * np.cos(2 * np.pi * np.arange(length) / spec.period)
    return np.broadcast_to(values, (unit_history.shape[0], length)).copy()


def history_spec_unavailable(spec, mechanism):
    """Reject unsupported *laws* before observing a particular random draw."""
    if spec.kind == "stationary" and not isinstance(mechanism, HeapsVARMechanism):
        return "Joint stationary initializer unavailable"
    if spec.kind == "cosine" and spec.period * spec.fraction < mechanism_parts(mechanism)[2]:
        return "Cosine prefix shorter than required lag memory"
    if isinstance(mechanism, ContractiveMechanism):
        radius = mechanism.certificate.domain_radius
        if spec.kind in ("zero", "constant"):
            bound = abs(spec.offset)
        elif spec.kind == "uniform":
            bound = abs(spec.offset) + spec.amplitude
        elif spec.kind == "cosine":
            values = spec.offset + spec.amplitude * np.cos(
                2 * np.pi * np.arange(int(spec.period * spec.fraction)) / spec.period)
            bound = np.max(np.abs(values))
        else:
            return "Unbounded startup law outside certified domain"
        if bound > radius + 1e-12:
            return "Requested startup support exceeds certified box"
    return None


def run_history_case(group, mechanisms, config, spec):
    """Evaluate one startup across a fixed mechanism bank and paired noise streams."""
    rows, skipped, examples = [], [], []
    for i, mechanism in enumerate(mechanisms):
        reason = history_spec_unavailable(spec, mechanism)
        if reason:
            skipped.append(dict(method=group, case=spec.name, mechanism=i,
                                skipped_trajectories=config.trajectories, reason=reason))
            continue
        for r in range(config.trajectories):
            unit, noise, stationary_seed = paired_inputs(mechanism, config, i, r)
            start = perf_counter()
            history = make_initial_history(spec, mechanism, unit, stationary_seed)
            init_seconds = perf_counter() - start
            start = perf_counter()
            path = simulate_from_history(mechanism, history, noise, config)
            simulation_seconds = perf_counter() - start
            order = mechanism_parts(mechanism)[2]
            for outcome in path_diagnostics(path, config):
                rows.append(dict(method=group, case=spec.name, mechanism=i, trajectory=r,
                                 actual_order=order, history_length=history.shape[1],
                                 initial_max_abs=float(np.max(np.abs(history))),
                                 amplitude_l2_bound=amplitude_bound_for_history(mechanism, history),
                                 initialization_seconds=init_seconds, simulation_seconds=simulation_seconds,
                                 **outcome))
            if i < config.example_mechanisms:
                examples.append(dict(mechanism=i, trajectory=r, case=spec.name,
                                     history=history, path=path, actual_order=order,
                                     formula=mechanism.formulas if hasattr(mechanism, "formulas") else [mechanism.formula]))
    return dict(rows=rows, skipped=skipped, examples=examples)


def merge_study_results(results):
    """Combine condition blocks without dropping unsupported cases or failures."""
    return {key: [item for result in results for item in result.get(key, [])]
            for key in ("rows", "skipped", "examples", "curves")}


def run_initialization_cases(group, mechanisms, config, *, study):
    """Starting-value or initial-magnitude comparison; all attempts retained."""
    if study not in ("starting_values", "magnitude"):
        raise ValueError("Unknown study.")
    cases = (["zero", "positive constant", "negative constant", "uniform", "stationary joint"]
             if study == "starting_values" else list(config.magnitude_scales))
    rows, skipped, examples = [], [], []
    for case in cases:
        if case == "stationary joint" and group != "Heaps":
            skipped.append(dict(method=group, case=case, reason="Stationary initializer unavailable"))
            continue
        if study == "magnitude" and group == "Jacobian" and case > mechanisms[0].certificate.domain_radius:
            skipped.append(dict(method=group, case=case, reason="Requested support exceeds certified box"))
            continue
        for i, mechanism in enumerate(mechanisms):
            for r in range(config.trajectories):
                unit, noise, stationary_seed = paired_inputs(mechanism, config, i, r)
                if study == "magnitude":
                    history = case * unit
                elif case == "zero":
                    history = np.zeros_like(unit)
                elif case == "positive constant":
                    history = np.full_like(unit, config.constant_value)
                elif case == "negative constant":
                    history = np.full_like(unit, -config.constant_value)
                elif case == "uniform":
                    history = config.uniform_radius * unit
                else:
                    history = initialize_heaps_history(mechanism, seed=stationary_seed)
                path = simulate_from_history(mechanism, history, noise, config)
                for outcome in path_diagnostics(path, config):
                    rows.append(dict(method=group, case=str(case), mechanism=i, trajectory=r,
                                     initial_max_abs=float(np.max(np.abs(history))),
                                     amplitude_l2_bound=amplitude_bound_for_history(mechanism, history),
                                     **outcome))
                if i < 3 and r == 0:
                    examples.append(dict(mechanism=i, case=str(case), history=history,
                                         path=path, formula=(mechanism.formulas if hasattr(mechanism, "formulas") else [mechanism.formula])))
    return dict(rows=rows, skipped=skipped, examples=examples)


def pair_distances(history_a, history_b, path_a, path_b, horizon):
    """Output and augmented-state maximum gaps; failures are missing, never zero."""
    order = history_a.shape[1]
    output = np.full(horizon, np.nan)
    state = np.full(horizon + 1, np.nan)
    state[0] = np.max(np.abs(history_a - history_b))
    length = min(path_a.shape[1], path_b.shape[1], horizon)
    a = np.concatenate((history_a, path_a[:, :length]), axis=1)
    b = np.concatenate((history_b, path_b[:, :length]), axis=1)
    for step in range(length):
        if not np.isfinite(path_a[:, step]).all() or not np.isfinite(path_b[:, step]).all():
            break
        output[step] = np.max(np.abs(path_a[:, step] - path_b[:, step]))
        state[step + 1] = np.max(np.abs(a[:, step+1:step+order+1] - b[:, step+1:step+order+1]))
    return output, state


def jacobian_pair_verification(mechanism, a, b, pa, pb):
    """Check the weighted augmented-state inequality using chronological histories."""
    if not isinstance(mechanism, ContractiveMechanism):
        return None
    order = a.shape[1]
    weights = np.asarray(mechanism.certificate.state_weights)
    initial = np.max(np.abs(a[:, ::-1] - b[:, ::-1]) / weights)
    full_a, full_b = np.concatenate((a, pa), axis=1), np.concatenate((b, pb), axis=1)
    gamma = mechanism.certificate.lipschitz_upper_bound
    for step in range(1, min(pa.shape[1], pb.shape[1]) + 1):
        gap = np.max(np.abs(full_a[:, step:step+order][:, ::-1] - full_b[:, step:step+order][:, ::-1]) / weights)
        if gap > gamma**step * initial + 1e-10:
            return False
    return True


def run_forgetting_cases(group, mechanisms, config):
    rows, examples, curves = [], [], []
    horizon = max(config.horizons)
    for i, mechanism in enumerate(mechanisms):
        for r in range(config.trajectories):
            unit, noise, _ = paired_inputs(mechanism, config, i, r)
            pairs = {
                "opposite constants": (np.full_like(unit, config.constant_value), np.full_like(unit, -config.constant_value)),
                "opposite uniform": (unit * config.uniform_radius, -unit * config.uniform_radius),
                "zero vs uniform": (np.zeros_like(unit), unit * config.uniform_radius),
            }
            for case, (a, b) in pairs.items():
                pa = simulate_from_history(mechanism, a, noise, config)
                pb = simulate_from_history(mechanism, b, noise, config)
                da, db = path_diagnostics(pa, config), path_diagnostics(pb, config)
                # The threshold-crossing observation is a failure, not a valid distance.
                def valid_prefix(path):
                    bad = np.flatnonzero(np.any(~np.isfinite(path) | (np.abs(path) > config.explosion_limit), axis=0))
                    return path if not len(bad) else path[:, :bad[0]]
                distance, state = pair_distances(a, b, valid_prefix(pa), valid_prefix(pb), horizon)
                check = jacobian_pair_verification(mechanism, a, b, pa, pb)
                curves.append(dict(method=group, case=case, mechanism=i, trajectory=r,
                                   output_gap=distance, state_gap=state))
                for left, right in zip(da, db):
                    h = left["horizon"]
                    success = left["success"] and right["success"]
                    tail = state[h-config.forgetting_tail+1:h+1]
                    retained = bool(success and np.all(tail <= config.forgetting_tolerance))
                    last_above = np.flatnonzero(state[:h+1] > config.forgetting_tolerance)
                    settling = (int(last_above[-1]+1) if len(last_above) else 0) if retained else np.nan
                    rows.append(dict(method=group, case=case, mechanism=i, trajectory=r, horizon=h,
                                     pair_exploded=not success, pair_success=success,
                                     near_zero_a=left["near_zero"], near_zero_b=right["near_zero"],
                                     initial_gap=state[0], output_gap=distance[h-1], state_gap=state[h],
                                     persistent_forgetting=retained, observed_settling_step=settling,
                                     weighted_bound_passed=check))
                if i < 3 and r == 0:
                    examples.append(dict(mechanism=i, case=case, history=a, history_b=b,
                                         path=pa, path_b=pb))
    return dict(rows=rows, examples=examples, curves=curves, skipped=[])


def run_history_pair(group, mechanisms, config, left, right, *, name=None):
    """Compare two startup laws using exactly the same recurrence and innovations.

    State distances use only the p required lags, even when the displayed
    history contains a longer cosine prefix. Missing failed pairs stay missing.
    """
    rows, examples, curves, skipped = [], [], [], []
    horizon = max(config.horizons)
    name = name or f"{left.name} vs {right.name}"
    for i, mechanism in enumerate(mechanisms):
        reason = history_spec_unavailable(left, mechanism) or history_spec_unavailable(right, mechanism)
        if reason:
            skipped.append(dict(method=group, case=name, mechanism=i,
                                skipped_trajectories=config.trajectories, reason=reason))
            continue
        order = mechanism_parts(mechanism)[2]
        for r in range(config.trajectories):
            unit, noise, stationary_seed = paired_inputs(mechanism, config, i, r)
            a = make_initial_history(left, mechanism, unit, stationary_seed)
            b = make_initial_history(right, mechanism, unit, stationary_seed)
            pa = simulate_from_history(mechanism, a, noise, config)
            pb = simulate_from_history(mechanism, b, noise, config)

            def valid_prefix(path):
                bad = np.flatnonzero(np.any(~np.isfinite(path) | (np.abs(path) > config.explosion_limit), axis=0))
                return path if not len(bad) else path[:, :bad[0]]

            distance, state = pair_distances(a[:, -order:], b[:, -order:],
                                             valid_prefix(pa), valid_prefix(pb), horizon)
            check = jacobian_pair_verification(mechanism, a[:, -order:], b[:, -order:], pa, pb)
            curves.append(dict(method=group, case=name, mechanism=i, trajectory=r,
                               output_gap=distance, state_gap=state))
            for da, db in zip(path_diagnostics(pa, config), path_diagnostics(pb, config)):
                h = da["horizon"]
                success = da["success"] and db["success"]
                retained = bool(success and np.all(state[h-config.forgetting_tail+1:h+1]
                                                   <= config.forgetting_tolerance))
                above = np.flatnonzero(state[:h+1] > config.forgetting_tolerance)
                settling = (int(above[-1]+1) if len(above) else 0) if retained else np.nan
                rows.append(dict(method=group, case=name, mechanism=i, trajectory=r, horizon=h,
                                 actual_order=order, pair_exploded=not success, pair_success=success,
                                 near_zero_a=da["near_zero"], near_zero_b=db["near_zero"],
                                 initial_gap=state[0], output_gap=distance[h-1], state_gap=state[h],
                                 persistent_forgetting=retained, observed_settling_step=settling,
                                 weighted_bound_passed=check))
            if i < config.example_mechanisms:
                examples.append(dict(mechanism=i, trajectory=r, case=name, history=a, history_b=b,
                                     path=pa, path_b=pb, actual_order=order,
                                     formula=mechanism.formulas if hasattr(mechanism, "formulas") else [mechanism.formula]))
    return dict(rows=rows, examples=examples, curves=curves, skipped=skipped)
