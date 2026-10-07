import numpy as np

from tsfmxai.baselines import (
    difference_polynomial,
    fit_pooled_arma,
    multiplicative_lag_support,
    one_step_nmse,
    simulate_pooled_arma,
)


def test_difference_polynomial_combines_regular_and_seasonal_terms():
    result = difference_polynomial(d=1, D=1, seasonal_period=4)
    np.testing.assert_allclose(result, [1.0, -1.0, 0.0, 0.0, -1.0, 1.0])


def test_multiplicative_lag_support_contains_cross_lags():
    assert multiplicative_lag_support(2, 1, 12) == (1, 2, 12, 13, 14)


def test_scalar_pooled_arma_is_reproducible_and_finite():
    rng = np.random.default_rng(17)
    series = np.zeros((5, 96))
    noise = rng.normal(scale=0.2, size=series.shape)
    for time in range(2, series.shape[1]):
        series[:, time] = 0.6 * series[:, time - 1] - 0.15 * series[:, time - 2] + noise[:, time]
    fit = fit_pooled_arma(series, order=(2, 0, 1), css_iterations=3)
    first = simulate_pooled_arma(fit, series[:, :fit.start], length=96, rng=np.random.default_rng(91))
    second = simulate_pooled_arma(fit, series[:, :fit.start], length=96, rng=np.random.default_rng(91))
    np.testing.assert_array_equal(first, second)
    assert first.shape == series.shape
    assert np.isfinite(first).all()
    assert np.isfinite(one_step_nmse(fit, series))


def test_vector_seasonal_fit_preserves_shape():
    series = np.random.default_rng(23).normal(size=(4, 80, 2))
    fit = fit_pooled_arma(series, order=(1, 0, 1),
                          seasonal_order=(1, 0, 0, 6), css_iterations=2)
    generated = simulate_pooled_arma(fit, series[:, :fit.start], length=80,
                                      rng=np.random.default_rng(92))
    assert generated.shape == series.shape
    assert np.isfinite(generated).all()
