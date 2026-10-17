import numpy as np

from tsfmxai.evaluation.generator_tradeoff import benchmark_stability_method
from tsfmxai.evaluation.stability import StabilityDiagnosticConfig
from tsfmxai.generators import (
    SampledMechanism, HeapsVARConfig, ShaoWuConfig,
    sample_heaps_var_mechanisms, sample_shao_wu_mechanisms, simulate_innovation_path,
)
from tsfmxai.representations import Lag, Multiply, Parameter, MechanismBinding


def test_last_step_explosion_is_a_failed_full_trajectory():
    # Constant 101 crosses threshold on the first and final requested step.
    mechanism = SampledMechanism(Parameter("p"), MechanismBinding({"p": 101}, {}), 0, 0)
    result = benchmark_stability_method("original", [mechanism], root_seed=1,
        innovation_std=0, diagnostic_config=StabilityDiagnosticConfig((1,), 100, .15, 1))
    assert result["outcomes"][0]["exploded"]
    assert result["performance"]["full_horizon_success_rate"] == 0


def test_benchmark_new_methods_have_all_denominators_and_timing():
    cfg = StabilityDiagnosticConfig((12, 24), 100, .15, 8)
    samplers = [("shao-wu", sample_shao_wu_mechanisms, ShaoWuConfig()),
                ("heaps-var", sample_heaps_var_mechanisms, HeapsVARConfig(num_series=2))]
    for name, sampler, config in samplers:
        mechanisms = sampler(config, root_seed=18, count=3)
        result = benchmark_stability_method(name, mechanisms, root_seed=20,
                                            diagnostic_config=cfg, sampling_seconds=.1)
        assert len(result["outcomes"]) == 6
        assert result["performance"]["full_horizon_success_rate"] == 1
        assert result["performance"]["total_seconds"] >= .1
        assert result["performance"]["initialization_seconds"] > 0
        assert result["performance"]["evaluated_values"] == 3 * 24 * mechanisms[0].num_series


def test_synchronous_multivariate_execution_and_retained_failure():
    trees = (Lag(1, "l"), Multiply(Parameter("a"), Lag(0, "l")))
    bindings = (MechanismBinding({}, {"l": 1}), MechanismBinding({"a": 2.}, {"l": 1}))
    path = simulate_innovation_path(trees, bindings, np.array([[1.], [3.]]), np.zeros((2, 4)))
    np.testing.assert_allclose(path, [[3, 2, 6, 4], [2, 6, 4, 12]])
    stopped = simulate_innovation_path(trees, bindings, np.array([[1.], [3.]]),
                                       np.zeros((2, 4)), stop_threshold=5)
    assert stopped.shape == (2, 2)
    assert stopped[1, -1] == 6
