"""Adapters for the studio's existing generation families."""

from dataclasses import dataclass, replace, asdict
from typing import Any

import numpy as np

from tsfmxai.generators import (
    ContractionTreeConfig, HeapsVARConfig, ShaoWuConfig, StationaryVAR1Config,
    RandomTreeConfig, SampledMechanism, sample_random_mechanisms,
    sample_contractive_mechanisms, sample_heaps_var_mechanism,
    sample_shao_wu_mechanism, sample_stationary_var1_mechanism,
    sample_heaps_var_trajectories, sample_shao_wu_trajectories,
    sample_stationary_var1_trajectories, sample_multivariate_trajectories,
    required_multivariate_history_length, simulate_innovation_path,
)
from tsfmxai.representations import Add, Multiply, Lag, Tanh

METHODS = ("none", "coefficient_gate", "contraction", "shao_wu",
           "shao_wu_all_tanh", "shao_wu_product_lags", "shao_wu_product_output",
           "heaps", "stationary_var1")

SHAO_METHODS = ("shao_wu", "shao_wu_all_tanh", "shao_wu_product_lags", "shao_wu_product_output")


@dataclass(frozen=True)
class StudioSystem:
    method: str
    equations: tuple[SampledMechanism, ...]
    retained: Any
    tree_config: RandomTreeConfig
    innovation: float
    certificates: tuple[dict, ...]

    @property
    def history_length(self) -> int:
        return required_multivariate_history_length(
            [m.expression for m in self.equations], [m.binding for m in self.equations]
        )


def _remap_scalar(expression, target):
    if isinstance(expression, Lag):
        return Lag(target, expression.slot)
    if isinstance(expression, (Add, Multiply)):
        return type(expression)(_remap_scalar(expression.left, target),
                                _remap_scalar(expression.right, target))
    if isinstance(expression, Tanh):
        return Tanh(_remap_scalar(expression.operand, target))
    return expression


def sample_studio_system(config: RandomTreeConfig, *, method: str, seed: int,
                         innovation: float = .05, shao_wu_mode: str = "product_lags") -> StudioSystem:
    if method not in METHODS:
        raise ValueError("Método de controle de explosão desconhecido.")
    if config.num_series != 2:
        raise ValueError("O studio requer duas séries.")
    if not np.isfinite(innovation) or innovation < 0:
        raise ValueError("A inovação deve ser finita e não negativa.")
    if method in ("none", "coefficient_gate"):
        config = replace(config, coefficient_gate_lags=method == "coefficient_gate")
        equations = tuple(sample_random_mechanisms(config, root_seed=seed, count=2))
        return StudioSystem(method, equations, None, config, innovation, ())
    if method == "contraction" or method in SHAO_METHODS:
        config = replace(config, time_probability=0., require_time=False, require_lag=True)
    if method == "contraction":
        if innovation >= 1:
            raise ValueError("Na contração, o limite da inovação deve ser menor que 1.")
        retained = sample_contractive_mechanisms(
            ContractionTreeConfig(tree_config=replace(config, num_series=1),
                                  innovation_bound=innovation), root_seed=seed, count=2)
        equations = tuple(SampledMechanism(_remap_scalar(m.expression, i), m.binding,
                                          m.seed, m.sample_index) for i, m in enumerate(retained))
        certificates = tuple({**asdict(m.certificate), "holds": m.certificate.holds} for m in retained)
    else:
        if innovation <= 0:
            raise ValueError("Este método requer desvio de inovação maior que zero.")
        if method in SHAO_METHODS:
            if shao_wu_mode not in ("product_lags", "product_output"):
                raise ValueError("Escolha product_lags ou product_output para a saturação.")
            lag_transform = {"shao_wu": shao_wu_mode, "shao_wu_all_tanh": "tanh",
                             "shao_wu_product_lags": "product_lags",
                             "shao_wu_product_output": "product_output"}[method]
            retained = sample_shao_wu_mechanism(
                ShaoWuConfig(tree_config=config, innovation_std=innovation,
                             lag_transform=lag_transform), seed=seed)
        elif method == "heaps":
            retained = sample_heaps_var_mechanism(
                HeapsVARConfig(num_series=2, minimum_order=config.minimum_lag,
                               maximum_order=config.maximum_lag, innovation_std=innovation), seed=seed)
        else:
            retained = sample_stationary_var1_mechanism(
                StationaryVAR1Config(num_series=2, innovation_std_minimum=innovation,
                                     innovation_std_maximum=innovation), seed=seed)
        equations = tuple(SampledMechanism(e, b, retained.seed, i)
                          for i, (e, b) in enumerate(zip(retained.expressions, retained.bindings)))
        certificates = ({**asdict(retained.certificate), "holds": retained.certificate.holds,
                         "guarantee": getattr(retained.certificate, "guarantee", "stationary_var")},)
    return StudioSystem(method, equations, retained, config, innovation, certificates)


def sample_studio_trajectories(system: StudioSystem, *, trajectory_count: int,
                              generated_steps: int, root_seed: int,
                              initial_value_minimum: float = -1.,
                              initial_value_maximum: float = 1.,
                              innovation_std: float = 0.) -> np.ndarray:
    if trajectory_count < 1 or generated_steps < 1:
        raise ValueError("Número de trajetórias e passos devem ser positivos.")
    kwargs = dict(trajectory_count=trajectory_count, generated_steps=generated_steps, root_seed=root_seed)
    if system.method in ("none", "coefficient_gate"):
        return sample_multivariate_trajectories(
            [m.expression for m in system.equations], [m.binding for m in system.equations],
            initial_value_minimum=initial_value_minimum, initial_value_maximum=initial_value_maximum,
            innovation_std=innovation_std, maximum_absolute_value=None, **kwargs)
    if innovation_std != system.innovation:
        raise ValueError("A inovação mudou; gere um novo sistema para preservar o certificado.")
    if system.method == "heaps":
        return sample_heaps_var_trajectories(system.retained, **kwargs)
    if system.method in SHAO_METHODS:
        return sample_shao_wu_trajectories(system.retained, **kwargs)
    if system.method == "stationary_var1":
        return sample_stationary_var1_trajectories(system.retained, **kwargs)
    results = []
    for sequence in np.random.SeedSequence(root_seed).spawn(trajectory_count):
        init_seq, noise_seq = sequence.spawn(2)
        initial = np.random.default_rng(init_seq).uniform(-1., 1., (2, system.history_length))
        noise = np.random.default_rng(noise_seq).uniform(-system.innovation, system.innovation,
                                                       (2, generated_steps))
        generated = simulate_innovation_path(
            [m.expression for m in system.equations], [m.binding for m in system.equations], initial, noise)
        if not np.isfinite(generated).all() or np.any(np.abs(generated) > 1 + 1e-12):
            raise ArithmeticError("A trajetória saiu do domínio certificado.")
        results.append(np.concatenate((initial, generated), axis=1))
    return np.stack(results)
