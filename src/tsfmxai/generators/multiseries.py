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
DEFAULT_SELF_ONLY_TREATMENT_PROBS = {"jacobian": .5, "heaps": .5}
DEFAULT_HEAPS_SUPPORT_PROBS = {"consecutive": .4, "isolated": .2, "periodic": .4}


@lru_cache(maxsize=32)
def heaps_lag_supports(maximum_lag: int) -> dict[str, tuple[tuple[int, int], ...]]:
    """Feasible (AR order, lag spacing) pairs, with disjoint support classes."""
    positive_integer(maximum_lag, "maximum_lag")
    return {
        "consecutive": tuple((p, 1) for p in range(1, maximum_lag + 1)),
        "isolated": tuple((1, s) for s in range(2, maximum_lag + 1)),
        "periodic": tuple((p, s) for p in range(2, maximum_lag + 1)
                          for s in range(2, maximum_lag // p + 1)),
    }


def draw_heaps_lag_support(rng, maximum_lag: int, probabilities=DEFAULT_HEAPS_SUPPORT_PROBS):
    """Draw a support class, then a pair uniformly within that class."""
    validate_probabilities(probabilities, choices=DEFAULT_HEAPS_SUPPORT_PROBS)
    options = heaps_lag_supports(maximum_lag)
    available = {name: weight for name, weight in probabilities.items()
                 if weight > 0 and options[name]}
    if not available:
        raise ValueError("No feasible Heaps lag supports have positive probability.")
    family = draw_choice(rng, available)
    pairs = options[family]
    return family, pairs[int(rng.integers(len(pairs)))]


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
    heaps_period: int = 1
    heaps_order: int | None = None
    heaps_support: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.domain_radius) or self.domain_radius <= 0:
            raise ValueError("domain_radius must be finite and positive.")
        if self.method not in TREATMENTS:
            raise ValueError("Unknown channel treatment.")
        positive_integer(self.heaps_period, "heaps_period")
        if self.heaps_order is not None:
            positive_integer(self.heaps_order, "heaps_order")
        if self.heaps_support is not None and self.heaps_support not in DEFAULT_HEAPS_SUPPORT_PROBS:
            raise ValueError("Unknown Heaps support class.")
        if not 0 < self.target_gain < 1 or not 0 < self.safety_factor < 1:
            raise ValueError("Treatment gains must lie strictly between zero and one.")


@dataclass(frozen=True)
class MultiseriesConfig:
    # No scientific lag/degree defaults: supply the configuration selected by coverage.
    ast_config: CombinatorialTreeConfig
    channel_count_probs: Mapping[int, float]
    dependency_probs: Mapping[str, float]
    # Channels with an external source use Shao-Wu; self-only channels draw here.
    self_only_treatment_probs: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SELF_ONLY_TREATMENT_PROBS))
    heaps_support_probs: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAPS_SUPPORT_PROBS))
    max_external_sources: int = 2
    self_source_weight: float = 3.
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    target_gain: float = 0.8
    safety_factor: float = 0.95

    def __post_init__(self):
        validate_probabilities(self.channel_count_probs)
        for count in self.channel_count_probs:
            positive_integer(count, "channel count")
        validate_probabilities(self.dependency_probs, choices=DEPENDENCIES)
        validate_probabilities(self.self_only_treatment_probs, choices=("jacobian", "heaps"))
        validate_probabilities(self.heaps_support_probs, choices=DEFAULT_HEAPS_SUPPORT_PROBS)
        positive_integer(self.max_external_sources, "max_external_sources", minimum=0)
        if not np.isfinite(self.self_source_weight) or self.self_source_weight <= 0:
            raise ValueError("self_source_weight must be finite and positive.")
        TreatmentSpec("shao_wu", self.target_gain, self.safety_factor)


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
                rng = np.random.default_rng(equation.seed)
                if treatment.heaps_order is None:
                    # Explicit legacy systems may still supply an AST-derived order.
                    candidate_lags = sorted(set(binding.lag_values.values()))
                    if not candidate_lags:
                        raise ValueError("Heaps requires heaps_order or a sampled lag.")
                    lag_order = int(rng.choice(candidate_lags))
                else:
                    lag_order = treatment.heaps_order
                mechanism = build_heaps_var_mechanism(
                    rng.normal(0., .45, (lag_order, 1, 1)), np.eye(1),
                    seed=equation.seed, lag_spacing=treatment.heaps_period)
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
def _sampler(config):
    return CombinatorialTreeSampler(config)


def _assign_sources(node, sources, weights, rng):
    if isinstance(node, Lag):
        return Lag(int(rng.choice(sources, p=weights)), node.slot)
    if isinstance(node, (Add, Multiply)):
        return type(node)(_assign_sources(node.left, sources, weights, rng),
                          _assign_sources(node.right, sources, weights, rng))
    if isinstance(node, Tanh):
        return Tanh(_assign_sources(node.operand, sources, weights, rng))
    return node


def _assign_sources_requiring_external(node, sources, weights, target, rng):
    """Assign sources while forcing one sampled lag to use an external channel."""
    external = [source for source in sources if source != target]
    lag_count = sum(isinstance(part, Lag) for part in walk(node))
    if not external or not lag_count:
        raise ValueError("Cross-series Shao-Wu needs an external source and a lag")
    required_index = int(rng.integers(lag_count))
    seen = 0

    def visit(part):
        nonlocal seen
        if isinstance(part, Lag):
            index = seen
            seen += 1
            source = (int(rng.choice(external)) if index == required_index else
                      int(rng.choice(sources, p=weights)))
            return Lag(source, part.slot)
        if isinstance(part, (Add, Multiply)):
            return type(part)(visit(part.left), visit(part.right))
        if isinstance(part, Tanh):
            return Tanh(visit(part.operand))
        return part

    return visit(node)


def sample_multiseries_system(config: MultiseriesConfig, *, seed: int,
                              exogenous_channels: frozenset[int] = frozenset()):
    """Sample topology, ASTs and channel treatments using independent local streams.

AST structure is sampled with one placeholder source, then each lag occurrence
receives a permitted source with self preference. This is not uniform sampling
over all multivariate labelled ASTs. External subsets have uniformly sampled
cardinality from zero to the configured cap (linked chains allow the predecessor).

Channels in ``exogenous_channels`` get no sampled tree: they hold a zero
placeholder whose values another generator supplies. Other channels may still
use them as sources. Per-channel streams keep every other channel unchanged.
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
    # Shao-Wu needs a lag to read an external source; Jacobian needs one for its
    # certificate. Heaps channels draw no tree.
    tree_config = replace(config.ast_config, num_series=1, require_lag=True)
    sampler = _sampler(tree_config)
    equations, treatments, innovations = [], [], []
    for target in range(n):
        channel_seed = stream_seed(seed, 1, target)
        channel_rng = np.random.default_rng(stream_seed(seed, 2, target))
        if target in exogenous_channels:
            allowed[target] = [target]
            equations.append(SampledMechanism(
                Parameter("EXOGENOUS"), MechanismBinding({"EXOGENOUS": 0.}, {}),
                channel_seed, target))
            treatments.append(TreatmentSpec("original", config.target_gain, config.safety_factor))
            innovations.append(InnovationSpec("gaussian", 0.))
            continue
        method = ("shao_wu" if len(allowed[target]) > 1 else
                  draw_choice(channel_rng, config.self_only_treatment_probs))
        if method in ("jacobian", "heaps"):
            allowed[target] = [target]
        sources = allowed[target]
        weights = np.array([config.self_source_weight if i == target else 1. for i in sources])
        weights /= weights.sum()
        if method == "heaps":
            support, (heaps_order, heaps_period) = draw_heaps_lag_support(
                channel_rng, config.ast_config.maximum_lag, config.heaps_support_probs)
            equation = SampledMechanism(
                Parameter("HEAPS_PLACEHOLDER"),
                MechanismBinding({"HEAPS_PLACEHOLDER": 0.}, {}),
                channel_seed, target)
        else:
            equation = sampler.sample(seed=channel_seed, sample_index=target)
            expression = (_assign_sources_requiring_external(
                equation.expression, sources, weights, target, channel_rng)
                if method == "shao_wu"
                else _assign_sources(equation.expression, sources, weights, channel_rng))
            equation = replace(equation, expression=expression)
        equations.append(equation)
        treatments.append(TreatmentSpec(method,
                                        config.target_gain, config.safety_factor,
                                        heaps_period=heaps_period if method == "heaps" else 1,
                                        heaps_order=heaps_order if method == "heaps" else None,
                                        heaps_support=support if method == "heaps" else None))
        innovations.append(InnovationSpec("uniform" if method == "jacobian" else "gaussian",
                                          float(channel_rng.uniform(*config.noise.std_range))))
    system = build_multiseries_system(equations=equations, allowed_sources=allowed,
                                     treatments=treatments, innovations=innovations,
                                     max_degree=config.ast_config.max_degree)
    return replace(system, dependency=dependency, channel_order=order, seed=seed,
                   sampling_config=config)
