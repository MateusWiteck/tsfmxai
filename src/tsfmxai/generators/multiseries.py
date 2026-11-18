"""Explicit and sampled multiseries mechanisms for equation-supervised training.

Topology describes allowed sources, not mandatory edges. Treatments belong to
equations; mixed systems make no claim of a joint stability certificate.
"""

from dataclasses import dataclass, field, replace
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

from tsfmxai.generators.combinatorial_expression import (
    CombinatorialTreeConfig, CombinatorialTreeSampler,
)
from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.generators.shao_wu import global_expression_bounds, saturate_product_lags
from tsfmxai.generators.trajectory import required_multivariate_history_length
from tsfmxai.representations import canonicalize_polynomial
from tsfmxai.representations.expression import (
    Add, Multiply, Parameter, Lag, Tanh, MechanismBinding, walk,
)


DEPENDENCIES = ("independent", "linked_chain", "previous_chain", "random")
TREATMENTS = ("original", "shao_wu", "jacobian", "heaps")
DEFAULT_TREATMENT_PROBS = {"original": 0., "shao_wu": .5, "jacobian": 1./6., "heaps": 1./3.}


@dataclass(frozen=True)
class LagSamplingConfig:
    """Terminal lag probability in original ASTs, conditional on tree constraints."""

    minimum: float = .25
    maximum: float = .75
    tau: float = 1.

    def __post_init__(self):
        if not np.isfinite([self.minimum, self.maximum, self.tau]).all():
            raise ValueError("Lag sampling settings must be finite.")
        if not 0 < self.minimum <= self.maximum <= 1 or self.tau <= 0:
            raise ValueError("Require 0 < minimum <= maximum <= 1 and tau > 0.")

    def probability(self, weights):
        q = np.asarray(weights, dtype=float)
        q = q / q.sum()
        effective = max(1., float(1. / np.sum(q * q)))
        return self.minimum + (self.maximum - self.minimum) * (effective - 1.) / (effective - 1. + self.tau)


def positive_integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def validate_probabilities(probabilities, *, choices=None):
    if not probabilities:
        raise ValueError("A probability mapping cannot be empty.")
    if choices is not None and set(probabilities) - set(choices):
        raise ValueError(f"Supported choices are {tuple(choices)}.")
    values = np.asarray(list(probabilities.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one.")


def draw_choice(rng, probabilities):
    keys = list(probabilities)
    weights = np.asarray(list(probabilities.values()), dtype=float)
    return keys[int(rng.choice(len(keys), p=weights / weights.sum()))]


def stream_seed(seed, *indices):
    """Stable identities independent of batching, rejection, and global RNG state."""
    return int(np.random.SeedSequence([seed, *indices]).generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True)
class InnovationSpec:
    family: str = "gaussian"
    std: float = 0.05

    def __post_init__(self):
        if self.family not in ("gaussian", "uniform"):
            raise ValueError("Innovation family must be gaussian or uniform.")
        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("Innovation std must be finite and nonnegative.")
        if self.family == "uniform" and not np.isfinite(np.sqrt(3.) * self.std):
            raise ValueError("Uniform innovation bound is not representable.")


@dataclass(frozen=True)
class NoiseConfig:
    std_range: tuple[float, float] = (0.01, 0.1)
    std_sampling: str = "uniform"
    family_policy: str = "by_treatment"

    def __post_init__(self):
        lo, hi = self.std_range
        if not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi:
            raise ValueError("Invalid noise std_range.")
        if self.std_sampling != "uniform" or self.family_policy != "by_treatment":
            raise ValueError("Currently supported: uniform std sampling, family by treatment.")


@dataclass(frozen=True)
class TreatmentSpec:
    method: str = "original"
    target_gain: float = 0.8
    safety_factor: float = 0.95
    domain_radius: float = 3.0

    def __post_init__(self):
        if not np.isfinite(self.domain_radius) or self.domain_radius <= 0:
            raise ValueError("domain_radius must be finite and positive.")
        if self.method not in TREATMENTS:
            raise ValueError("Unknown channel treatment.")
        if not 0 < self.target_gain < 1 or not 0 < self.safety_factor < 1:
            raise ValueError("Treatment gains must lie strictly between zero and one.")


@dataclass(frozen=True)
class MultiseriesConfig:
    # No scientific lag/degree defaults: supply the configuration selected by coverage.
    ast_config: CombinatorialTreeConfig
    channel_count_probs: Mapping[int, float]
    dependency_probs: Mapping[str, float]
    treatment_probs: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_TREATMENT_PROBS))
    max_external_sources: int = 2
    self_source_weight: float = 3.
    allow_no_lags: bool = True
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    target_gain: float = 0.8
    safety_factor: float = 0.95
    lag_sampling: LagSamplingConfig | None = field(default_factory=LagSamplingConfig)

    def __post_init__(self):
        validate_probabilities(self.channel_count_probs)
        for count in self.channel_count_probs:
            positive_integer(count, "channel count")
        validate_probabilities(self.dependency_probs, choices=DEPENDENCIES)
        validate_probabilities(self.treatment_probs, choices=TREATMENTS)
        positive_integer(self.max_external_sources, "max_external_sources", minimum=0)
        if not np.isfinite(self.self_source_weight) or self.self_source_weight <= 0:
            raise ValueError("self_source_weight must be finite and positive.")
        if type(self.allow_no_lags) is not bool:
            raise ValueError("allow_no_lags must be boolean.")
        TreatmentSpec("shao_wu", self.target_gain, self.safety_factor)
        if self.lag_sampling is not None and not isinstance(self.lag_sampling, LagSamplingConfig):
            raise ValueError("lag_sampling must be LagSamplingConfig or None.")


@dataclass(frozen=True)
class MultiseriesSystem:
    original_equations: tuple[SampledMechanism, ...]
    equations: tuple[SampledMechanism, ...]
    allowed_sources: tuple[tuple[int, ...], ...]
    treatments: tuple[TreatmentSpec, ...]
    innovations: tuple[InnovationSpec, ...]
    canonical: tuple
    treatment_scales: tuple[float, ...]
    dependency: str = "explicit"
    channel_order: tuple[int, ...] = ()
    seed: int | None = None
    sampling_config: MultiseriesConfig | None = None
    joint_certificate: None = None
    channel_certificates: tuple = ()

    @property
    def expressions(self):
        return tuple(e.expression for e in self.equations)

    @property
    def bindings(self):
        return tuple(e.binding for e in self.equations)

    @property
    def history_length(self):
        return required_multivariate_history_length(self.expressions, self.bindings)

    @property
    def num_channels(self):
        return len(self.equations)

    @property
    def effective_sources(self):
        return tuple(tuple(sorted({a[0] for term in p.skeleton for a in term}))
                     for p in self.canonical)


def build_multiseries_system(*, equations: Sequence[SampledMechanism],
                            allowed_sources: Mapping[int, Sequence[int]],
                            treatments: Sequence[TreatmentSpec],
                            innovations: Sequence[InnovationSpec], max_degree: int = 4):
    """Build without sampling; preserve raw ASTs and label the executed equations.

Shao-Wu uses the existing product-factor protection and derivative bounds.
Its per-equation scale uses the sum of absolute source/lag derivatives. This
is a conservative row sensitivity budget, NOT a joint mixed-system certificate.
"""
    equations, treatments, innovations = tuple(equations), tuple(treatments), tuple(innovations)
    n = len(equations)
    if not n or len(treatments) != n or len(innovations) != n or set(allowed_sources) != set(range(n)):
        raise ValueError("Provide one equation, treatment, innovation and source list per channel.")
    sources = tuple(tuple(sorted(set(allowed_sources[i]))) for i in range(n))
    for source_list in sources:
        if any(type(i) is not int or not 0 <= i < n for i in source_list):
            raise ValueError("Invalid source channel.")
    order = required_multivariate_history_length(
        [e.expression for e in equations], [e.binding for e in equations])
    final, polynomials, scales, certificates = [], [], [], []
    for i, (equation, treatment) in enumerate(zip(equations, treatments)):
        used = {node.variable for node in walk(equation.expression) if isinstance(node, Lag)}
        if not used <= set(sources[i]):
            raise ValueError(f"Channel {i} references a forbidden source.")
        expression, binding, scale = equation.expression, equation.binding, 1.
        certificate = None
        if treatment.method in ("jacobian", "heaps"):
            if used - {i}:
                raise ValueError("Jacobian and Heaps channels currently require self-only sources.")
            if treatment.method == "jacobian":
                from tsfmxai.generators.contraction import ContractionTreeConfig, build_contractive_mechanism
                if innovations[i].family != "uniform":
                    raise ValueError("Jacobian channels require bounded uniform innovations.")
                local = _assign_sources(expression, [0], [1.], np.random.default_rng(0))
                mechanism = build_contractive_mechanism(local, binding, ContractionTreeConfig(
                    domain_radius=treatment.domain_radius,
                    innovation_bound=np.sqrt(3.) * innovations[i].std,
                    lag_weight_growth=2. / treatment.target_gain,
                    target_lipschitz=treatment.target_gain,
                    scale_safety_factor=treatment.safety_factor), seed=equation.seed)
                expression = _assign_sources(mechanism.expression, [i], [1.], np.random.default_rng(0))
                binding, scale, certificate = mechanism.binding, mechanism.scale, mechanism.certificate
            else:
                from tsfmxai.generators.heaps import build_heaps_var_mechanism
                # Heaps maps unconstrained coefficients to a stable AR. Missing
                # intermediate lags can become nonzero under the reverse map.
                lag_order = max(binding.lag_values.values(), default=1)
                rng = np.random.default_rng(equation.seed)
                mechanism = build_heaps_var_mechanism(
                    rng.normal(0., .45, (lag_order, 1, 1)), np.eye(1), seed=equation.seed)
                if innovations[i].family != "gaussian":
                    raise ValueError("Heaps channels require Gaussian innovations.")
                expression = _assign_sources(mechanism.expressions[0], [i], [1.], rng)
                binding, certificate = mechanism.bindings[0], mechanism.certificate
        certificates.append(certificate)
        if treatment.method == "shao_wu":
            expression = saturate_product_lags(expression)
            _, derivatives = global_expression_bounds(expression, binding,
                                                       num_series=n, order=max(1, order))
            gain = float(derivatives.sum())
            if not np.isfinite(gain):
                raise ValueError("Protected equation has no finite sensitivity bound.")
            scale = treatment.safety_factor * min(1., treatment.target_gain / gain if gain else 1.)
            slot = "MULTISERIES_SCALE"
            while slot in binding.parameter_values:
                slot += "_"
            expression = Multiply(Parameter(slot), expression)
            binding = MechanismBinding({**binding.parameter_values, slot: scale}, dict(binding.lag_values))
        final.append(replace(equation, expression=expression, binding=binding))
        polynomials.append(canonicalize_polynomial(expression, binding, max_degree=max_degree))
        scales.append(scale)
    return MultiseriesSystem(equations, tuple(final), sources, treatments, innovations,
                            tuple(polynomials), tuple(scales), channel_certificates=tuple(certificates))


@lru_cache(maxsize=32)
def _sampler(config, lag_probability=None):
    return CombinatorialTreeSampler(config, lag_probability=lag_probability)


def _assign_sources(node, sources, weights, rng):
    if isinstance(node, Lag):
        return Lag(int(rng.choice(sources, p=weights)), node.slot)
    if isinstance(node, (Add, Multiply)):
        return type(node)(_assign_sources(node.left, sources, weights, rng),
                          _assign_sources(node.right, sources, weights, rng))
    if isinstance(node, Tanh):
        return Tanh(_assign_sources(node.operand, sources, weights, rng))
    return node


def sample_multiseries_system(config: MultiseriesConfig, *, seed: int):
    """Sample topology, ASTs and channel treatments using independent local streams.

AST structure is sampled with one placeholder source, then each lag occurrence
receives a permitted source with self preference. This is not uniform sampling
over all multivariate labelled ASTs. External subsets have uniformly sampled
cardinality from zero to the configured cap (linked chains allow the predecessor).
"""
    rng = np.random.default_rng(stream_seed(seed, 0))
    n = draw_choice(rng, config.channel_count_probs)
    dependency = draw_choice(rng, config.dependency_probs)
    order = tuple(int(i) for i in rng.permutation(n))
    allowed = {}
    for position, target in enumerate(order):
        candidates = (list(order[position-1:position]) if dependency == "linked_chain" else
                      list(order[:position]) if dependency == "previous_chain" else
                      [i for i in range(n) if i != target] if dependency == "random" else [])
        cap = min(len(candidates), config.max_external_sources)
        count = cap if dependency == "linked_chain" else int(rng.integers(cap + 1))
        external = rng.choice(candidates, size=count, replace=False).tolist() if count else []
        allowed[target] = sorted([target, *external])
    tree_config = replace(config.ast_config, num_series=1, require_lag=not config.allow_no_lags)
    sampler = _sampler(tree_config)
    equations, treatments, innovations = [], [], []
    for target in range(n):
        channel_seed = stream_seed(seed, 1, target)
        channel_rng = np.random.default_rng(stream_seed(seed, 2, target))
        method = draw_choice(channel_rng, config.treatment_probs)
        channel_sampler = (_sampler(replace(tree_config, require_lag=True))
                           if method == "jacobian" else sampler)
        if method in ("jacobian", "heaps"):
            allowed[target] = [target]
        sources = allowed[target]
        weights = np.array([config.self_source_weight if i == target else 1. for i in sources])
        weights /= weights.sum()
        if method == "original" and config.lag_sampling is not None:
            # Keep the existing operator-count prior, then draw a weighted tree
            # of that size. This avoids bias toward larger trees from label masses.
            import random
            sizes = sampler.size_counts()
            size = random.Random(stream_seed(seed, 3, target)).randrange(sum(sizes.values()))
            for operators, count in sizes.items():
                if size < count:
                    break
                size -= count
            fixed = replace(tree_config, minimum_operators=operators, maximum_operators=operators)
            channel_sampler = _sampler(fixed, config.lag_sampling.probability(weights))
        equation = channel_sampler.sample(seed=channel_seed, sample_index=target)
        expression = _assign_sources(equation.expression, sources, weights, channel_rng)
        equation = replace(equation, expression=expression)
        equations.append(equation)
        treatments.append(TreatmentSpec(method,
                                        config.target_gain, config.safety_factor))
        innovations.append(InnovationSpec("uniform" if method == "jacobian" else "gaussian",
                                          float(channel_rng.uniform(*config.noise.std_range))))
    system = build_multiseries_system(equations=equations, allowed_sources=allowed,
                                     treatments=treatments, innovations=innovations,
                                     max_degree=config.ast_config.max_degree)
    return replace(system, dependency=dependency, channel_order=order, seed=seed,
                   sampling_config=config)
