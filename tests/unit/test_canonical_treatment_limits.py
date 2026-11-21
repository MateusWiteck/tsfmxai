"""Boundary cases for the production six-channel canonicalizer."""
import json
from pathlib import Path

import pytest

from tsfmxai.generators.multiseries import (
    InnovationSpec, TreatmentSpec, build_multiseries_system, heaps_lag_supports,
)
from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.representations.expression import Add, Lag, MechanismBinding, Multiply
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec, check_teacher_limits


CONFIG = json.loads((Path(__file__).resolve().parents[2] / "configs" /
                     "canonicalization_multiseries_six_series.json").read_text())
G = CONFIG["generator"]
CODEC = CanonicalCodec(G["num_series"], G["minimum_lag"], G["maximum_lag"], G["max_degree"])


def check_equation(method, expression, lags, **treatment_options):
    source = int(method == "shao_wu")
    equation = SampledMechanism(expression, MechanismBinding({}, lags), 17, 0)
    # The external channel is explicit so Shao-Wu reads a real neighbour.
    system = build_multiseries_system(
        equations=[equation] * (1 + source),
        allowed_sources={i: [source] for i in range(1 + source)},
        treatments=[TreatmentSpec(method, **treatment_options)] +
                   [TreatmentSpec("original")] * source,
        innovations=[InnovationSpec("uniform" if method == "jacobian" else "gaussian", .1)]
                    * (1 + source))
    encoded = CODEC.encode(system.equations[0])
    check_teacher_limits(encoded, CONFIG)
    decoded = CODEC.decode(encoded["target"][1:], encoded["coefficient_targets"][1:])
    assert decoded["valid"] and decoded["skeleton"] == encoded["skeleton"]
    return encoded


@pytest.mark.parametrize("family,order,spacing", [
    (family, order, spacing)
    for family, pairs in heaps_lag_supports(G["maximum_lag"]).items()
    for order, spacing in pairs
])
def test_heaps_support_roundtrip(family, order, spacing):
    encoded = check_equation("heaps", Lag(0, "L"), {"L": 1}, heaps_order=order,
                             heaps_period=spacing, heaps_support=family)
    if order == 16:
        assert len(encoded["prefix_ids"]) == 65
        with pytest.raises(ValueError, match="max_nodes"):
            check_teacher_limits(encoded, {**CONFIG, "model": {"max_nodes": 64}})


@pytest.mark.parametrize("method", ["jacobian", "shao_wu"])
@pytest.mark.parametrize("operators", range(1, 7))
def test_ast_sizes_roundtrip(method, operators):
    source = int(method == "shao_wu")
    lags = {str(i): lag for i, lag in enumerate([1, 16, 2, 15, 3, 14, 4][:operators + 1])}
    leaves = [Lag(source, slot) for slot in lags]
    expression = leaves[0]
    for index, leaf in enumerate(leaves[1:]):
        expression = (Multiply if index % 2 == 0 else Add)(expression, leaf)
    check_equation(method, expression, lags)


def test_decoder_capacity_for_every_binary_tree_shape():
    # Each leaf contributes at most one monomial with one atom. Ignoring
    # coefficient cancellations and repeated terms gives an upper bound.
    states = [{(1, 1)}]
    for operators in range(1, G["maximum_operators"] + 1):
        sizes = set()
        for left in range(operators):
            for n, a in states[left]:
                for m, b in states[operators - left - 1]:
                    sizes.add((n + m, a + b))
                    sizes.add((n * m, a * m + b * n))
        states.append(sizes)
    # One end token per term and one EOS. Scaling adds no lag atoms,
    # and the Shao-Wu tanh changes the atom type without adding a term.
    assert max(n + a + 1 for sizes in states for n, a in sizes) <= CONFIG["max_decode_steps"]
    # Binary tree, at most one tanh per leaf, then the scale and multiply.
    k = G["maximum_operators"]
    assert (2 * k + 1) + (k + 1) + 2 <= CONFIG["model"]["max_nodes"]
