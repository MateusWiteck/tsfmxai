import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tsfmxai.explainability.forecast_gradients import input_attributions, support_scores


def test_attribution_of_scalar_forecast_on_original_input():
    context = torch.tensor([[3.0, -2.0], [2.0, 4.0]])
    prediction, gradient, weighted = input_attributions(
        lambda x: x[0, 1] ** 2 + 2 * x[1, 0], context
    )
    assert prediction.item() == 8
    torch.testing.assert_close(gradient, torch.tensor([[0.0, -4.0], [2.0, 0.0]]))
    torch.testing.assert_close(weighted, torch.tensor([[0.0, 8.0], [4.0, 0.0]]))
    assert context.grad is None


def test_support_scores_handle_sign_and_tied_boundary():
    metrics = support_scores([-4, 2, -2, 0], [True, False, True, False])
    assert metrics["recall_at_k"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["mass_on_support"] == pytest.approx(0.75)
    zero = support_scores([0, 0], [True, False])
    assert zero["recall_at_k"] == 0.5
    assert np.isnan(zero["mass_on_support"])
