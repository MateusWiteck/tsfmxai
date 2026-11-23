"""Input attribution and support recovery for scalar forecasts."""

import numpy as np
import torch


def input_attributions(forecast, context):
    """Return forecast, gradient and gradient times input on the original scale."""
    values = context.detach().requires_grad_(True)
    prediction = forecast(values)
    gradient, = torch.autograd.grad(prediction, values)
    return prediction.detach(), gradient.detach(), (gradient * values).detach()


def support_scores(attribution, support):
    """Rank absolute scores, averaging across ties at the top-k boundary."""
    scores = np.abs(np.asarray(attribution)).ravel()
    support = np.asarray(support, dtype=bool).ravel()
    k = int(support.sum())
    if scores.shape != support.shape or not k or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores and a matching, nonempty support.")
    cutoff = np.sort(scores)[-k]
    above = scores > cutoff
    tied = scores == cutoff
    recovered = support[above].sum() + (k - above.sum()) * support[tied].mean()
    precision_sum = 0.0
    for threshold in np.unique(scores[support]):
        selected = scores >= threshold
        precision_sum += ((scores == threshold) & support).sum() * support[selected].mean()
    return {
        "average_precision": float(precision_sum / k),
        "recall_at_k": float(recovered / k),
        "mass_on_support": float(scores[support].sum() / scores.sum()) if scores.any() else np.nan,
    }


def chronos_median(model, context, target):
    """Forecast one step for one target in a single multivariate group."""
    groups = torch.zeros(context.shape[0], device=context.device, dtype=torch.long)
    quantiles = list(model.chronos_config.quantiles)
    output = model(context=context, group_ids=groups, num_output_patches=1)
    return output.quantile_preds[target, quantiles.index(0.5), 0]
