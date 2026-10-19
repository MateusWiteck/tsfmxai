"""Matched finite-horizon generator diagnostics with stage-specific timings."""

from dataclasses import asdict, replace
from time import perf_counter
from typing import Sequence

import numpy as np

from tsfmxai.evaluation.stability import StabilityDiagnosticConfig, diagnose_stability_prefixes
from tsfmxai.evaluation.tail_behavior import (
    TailDiagnosticConfig, classify_tail_windows, measure_tail_prefixes,
)
from tsfmxai.generators.contraction import ContractiveMechanism
from tsfmxai.generators.heaps import HeapsVARMechanism, initialize_heaps_history
from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.generators.shao_wu import ShaoWuMechanism, initialize_shao_wu_history
from tsfmxai.generators.stochastic import simulate_innovation_path
from tsfmxai.generators.trajectory import required_history_length


def benchmark_stability_method(
    method: str,
    mechanisms: Sequence[SampledMechanism | ContractiveMechanism | ShaoWuMechanism | HeapsVARMechanism],
    *,
    root_seed: int,
    diagnostic_config: StabilityDiagnosticConfig,
    innovation_std: float = .05,
    sampling_seconds: float = 0.,
    keep_examples: int = 4,
    tail_config: TailDiagnosticConfig | None = None,
    tail_tolerances: tuple[float, ...] = (),
) -> dict:
    """Evaluate one realization per independent mechanism, with no rejection.

    The output includes every attempted mechanism in every horizon denominator.
    In multivariate experiments the magnitude envelope over all channels is
    classified: one exploding channel fails the system; a low-amplitude tail
    requires every channel to stay in the radius. Timing is serial wall-clock
    and includes failed attempts. Successful full trajectories/s additionally
    accounts for early stopping; values/s counts actual evaluated outputs.
    Optional tail diagnostics center each signed channel before measuring
    variation. Sensitivity reuses these measurements; its cost is included in
    diagnostic time. Generation and random streams are unchanged.
    """
    if not mechanisms:
        raise ValueError("At least one mechanism is required.")
    if not np.isfinite(innovation_std) or innovation_std < 0:
        raise ValueError("innovation_std must be finite and non-negative.")
    if not np.isfinite(sampling_seconds) or sampling_seconds < 0 or keep_examples < 0:
        raise ValueError("sampling_seconds and keep_examples must be non-negative.")
    if tail_tolerances and tail_config is None:
        raise ValueError("tail_tolerances requires tail_config.")
    sensitivity_configs = ()
    if tail_config is not None:
        tail_config.validate_horizons(diagnostic_config.horizons)
        if len(set(tail_tolerances)) != len(tail_tolerances):
            raise ValueError("tail_tolerances must be unique.")
        sensitivity_configs = tuple(replace(tail_config, flat_tolerance=t) for t in tail_tolerances)
    horizon = max(diagnostic_config.horizons)
    outcomes, examples, tail_sensitivity = [], [], []
    initialization_seconds = innovation_seconds = execution_seconds = diagnostic_seconds = 0.
    evaluated_values = evaluated_steps = burn_in_steps = successful = 0
    end_to_end_started = perf_counter()
    for position, mechanism in enumerate(mechanisms):
        index = position if mechanism.sample_index is None else mechanism.sample_index
        init_seq, noise_seq = np.random.SeedSequence([root_seed, int(index), 991]).spawn(2)
        init_seed = int(init_seq.generate_state(1, dtype=np.uint64)[0])
        started = perf_counter()
        if isinstance(mechanism, HeapsVARMechanism):
            expressions, bindings = mechanism.expressions, mechanism.bindings
            history = initialize_heaps_history(mechanism, seed=init_seed)
        elif isinstance(mechanism, ShaoWuMechanism):
            expressions, bindings = mechanism.expressions, mechanism.bindings
            history = initialize_shao_wu_history(mechanism, seed=init_seed)
            burn_in_steps += mechanism.certificate.burn_in_steps
        else:
            expressions, bindings = (mechanism.expression,), (mechanism.binding,)
            order = required_history_length(mechanism.expression, mechanism.binding)
            radius = mechanism.certificate.domain_radius if isinstance(mechanism, ContractiveMechanism) else 1.
            history = np.random.default_rng(init_seed).uniform(-radius, radius, (1, order))
        initialization_seconds += perf_counter() - started
        started = perf_counter()
        rng = np.random.default_rng(noise_seq)
        if isinstance(mechanism, HeapsVARMechanism):
            noise = np.linalg.cholesky(mechanism.innovation_covariance) @ rng.standard_normal(
                (mechanism.num_series, horizon))
        elif isinstance(mechanism, ContractiveMechanism):
            bound = mechanism.certificate.innovation_bound
            noise = rng.uniform(-bound, bound, (1, horizon))
        else:
            std = mechanism.innovation_std if isinstance(mechanism, ShaoWuMechanism) else innovation_std
            noise = rng.normal(0., std, (len(expressions), horizon))
        innovation_seconds += perf_counter() - started
        started = perf_counter()
        path = simulate_innovation_path(expressions, bindings, history, noise,
                                         stop_threshold=diagnostic_config.explosion_limit)
        execution_seconds += perf_counter() - started
        evaluated_values += path.size
        evaluated_steps += path.shape[1]
        started = perf_counter()
        envelope = np.max(np.abs(path), axis=0)
        diagnostics = diagnose_stability_prefixes(envelope, diagnostic_config)
        tail_statistics = (
            measure_tail_prefixes(path, diagnostic_config, tail_config)
            if tail_config is not None else None
        )
        # A failure at the final step is a failure even though len(path)==horizon.
        successful += int(not diagnostics[-1].exploded)
        if isinstance(mechanism, ContractiveMechanism):
            if not np.all(np.isfinite(path)) or np.any(np.abs(path) > radius + 1e-10):
                raise AssertionError("Certified domain was violated during execution.")
        for diagnostic_index, diagnostic in enumerate(diagnostics):
            prefix = path[:, :diagnostic.horizon]
            nonfinite = bool(np.any(~np.isfinite(prefix)))
            row = dict(
                method=method, sample_index=int(index), horizon=diagnostic.horizon,
                exploded=diagnostic.exploded, near_zero=diagnostic.near_zero,
                neither=diagnostic.neither, explosion_step=diagnostic.explosion_step,
                nonfinite=nonfinite,
                finite_threshold_crossing=bool(diagnostic.exploded and not nonfinite),
            )
            if tail_statistics is not None:
                statistics = tail_statistics[diagnostic_index]
                row.update(asdict(statistics))
                row.update(classify_tail_windows(statistics, tail_config,
                                                  near_zero_radius=diagnostic_config.near_zero_radius))
                for sensitivity_config in sensitivity_configs:
                    tail_sensitivity.append(dict(
                        method=method, sample_index=int(index), horizon=diagnostic.horizon,
                        flat_tolerance=sensitivity_config.flat_tolerance,
                        tail_eligible=statistics.tail_eligible,
                        **classify_tail_windows(statistics, sensitivity_config,
                                               near_zero_radius=diagnostic_config.near_zero_radius),
                    ))
            outcomes.append(row)
        diagnostic_seconds += perf_counter() - started
        if position < keep_examples:
            formulas = mechanism.formulas if hasattr(mechanism, "formulas") else (mechanism.formula,)
            examples.append(dict(sample_index=int(index), formulas=formulas,
                                 history=history, generated=path))
    elapsed = perf_counter() - end_to_end_started
    total = sampling_seconds + elapsed
    count = len(mechanisms)
    performance = dict(
        method=method, mechanisms=count, sampling_and_certificate_seconds=sampling_seconds,
        mechanisms_per_second=count / sampling_seconds if sampling_seconds else None,
        initialization_seconds=initialization_seconds, burn_in_steps=burn_in_steps,
        innovation_seconds=innovation_seconds, ast_execution_seconds=execution_seconds,
        diagnostic_seconds=diagnostic_seconds, evaluation_seconds=elapsed,
        total_seconds=total, attempted_trajectories_per_second=count / total,
        successful_full_trajectories_per_second=successful / total,
        full_horizon_success_rate=successful / count,
        evaluated_steps=evaluated_steps, evaluated_values=evaluated_values,
        ast_values_per_second=evaluated_values / execution_seconds,
        end_to_end_values_per_second=evaluated_values / total,
    )
    return dict(outcomes=outcomes, performance=performance, examples=examples,
                tail_sensitivity=tail_sensitivity)
