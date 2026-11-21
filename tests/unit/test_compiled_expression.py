import numpy as np
import pytest

from tsfmxai.generators.multiseries import sample_multiseries_system
from tsfmxai.pipelines.prior_baselines import PriorProtocol, own_plan
from tsfmxai.generators.trajectory import compile_bound_expression, evaluate_bound_expression
from tsfmxai.representations import Add, Lag, MechanismBinding, Multiply, Parameter, Tanh


def test_compiled_expression_matches_recursive_evaluator_bitwise():
    # The training generator of the preliminary P25 run, without KernelSynth.
    protocol = PriorProtocol.read("configs/experiments/preliminary_hybrid_p25.json")
    config = own_plan(protocol, "independent", 1).stages[0].config.ast
    rng = np.random.default_rng(0)
    checked = set()
    for seed in range(40):
        system = sample_multiseries_system(config, seed=seed)
        length = system.history_length + 4
        histories = [list(rng.normal(scale=3.0, size=length)) for _ in range(system.num_channels)]
        for expression, binding, treatment in zip(system.expressions, system.bindings,
                                                  system.treatments):
            checked.add(treatment.method)
            compiled = compile_bound_expression(expression, binding)
            for time in range(system.history_length, length):
                expected = evaluate_bound_expression(expression, binding, time=time,
                                                     histories=histories)
                assert compiled(histories, time) == expected
    assert checked == {"shao_wu", "jacobian", "heaps"}


def test_compiled_expression_keeps_operation_order_and_tanh():
    # (P0 * x1(t-2)) + tanh(x0(t-1) * P1): negative literal and nested tanh.
    expression = Add(Multiply(Parameter("P0"), Lag(1, "L0")),
                     Tanh(Multiply(Lag(0, "L1"), Parameter("P1"))))
    binding = MechanismBinding(parameter_values={"P0": -0.1, "P1": 0.7},
                               lag_values={"L0": 2, "L1": 1})
    histories = [[0.3, -1.2, 2.5], [0.1, 0.4, -0.9]]
    compiled = compile_bound_expression(expression, binding)
    assert compiled(histories, 3) == evaluate_bound_expression(
        expression, binding, time=3, histories=histories)
    assert compiled(histories, 3) == pytest.approx(-0.1 * 0.4 + np.tanh(2.5 * 0.7))
