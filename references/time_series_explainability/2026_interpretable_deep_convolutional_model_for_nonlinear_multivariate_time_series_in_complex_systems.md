# Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems

## Basic Info

- **Title:** Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems
- **Model:** Deep Convolutional Interpreter for Time Series (DCIts)
- **Authors:** Domjan Baric and Davor Horvatic
- **Year:** 2026 journal version; the first arXiv version appeared in 2025
- **Venue:** Chaos, volume 36, article 063116
- **Paper:** https://arxiv.org/abs/2501.04339
- **PDF:** https://arxiv.org/pdf/2501.04339
- **DOI:** https://doi.org/10.1063/5.0325209
- **Code:** https://github.com/hc-xai/dcits
- **Relationship to the earlier benchmark:** This paper is a direct continuation of Baric et al. (2021), `Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions`. It reuses Datasets 1-8 from that benchmark, adjusts the switching process in Dataset 8, and adds controlled VAR(2) and cubic experiments.

## Problem Type

- **Task:** One-step-ahead multivariate time-series forecasting and recovery of the local transition mechanism.
- **Input:** A window `Q_t` with `N` time series and `L` lags, so `Q_t` has shape `N x L`.
- **Output:** The next multivariate observation `X_hat_(t+1)` with `N` values.
- **Explanation target:** For every target series and forecast example, identify the source series, lag, direction, coefficient magnitude, and realized contribution used in the prediction.
- **Explanation family:** Intrinsic, model-specific, local, lag-resolved coefficient explanation.

## Relevance To This Project

- **Forecasts future values:** Yes, one step ahead.
- **Produces explanations:** Yes, as part of the forecasting calculation.
- **Uses synthetic ground truth:** Yes. The known generator equations specify exact active source-lag pairs, coefficient signs, strengths, biases, regimes, and, in the cubic experiment, polynomial orders.
- **Main value:** It connects three objects explicitly: the generator coefficient, the learned local coefficient, and the actual numerical contribution to the forecast.
- **Relevance score:** 5/5.
- **Best role in this project:** A core reference for mechanism-recovery benchmarks, signed time-covariate ground truth, and an intrinsically interpretable baseline alongside post-hoc explanation methods.

## Central Idea In Simple Terms

DCIts learns a forecasting equation for every input window. For target series `n`, it predicts the next value by adding the effects of every source series `i` at every lag `l`:

```text
forecast[n] = sum over source i and lag l of
              coefficient[n, i, l] * observed_value[i, l]
```

In the paper's notation:

```text
X_hat[n, t+1] = sum_i sum_l alpha[t, n, i, l] * Q[t, i, l]
```

The learned tensor `alpha_t` has shape `N x N x L`:

```text
alpha_t[target, source, lag]
```

For one forecast, `alpha_t[3, 1, 5]` answers: how strongly does source series 1 at lag 5 enter the equation for target series 3?

## From Coefficients To Contributions

The paper separates a **coefficient** from a **realized contribution**:

```text
coefficient  = alpha_t[n, i, l]
input value  = Q_t[i, l]
contribution = alpha_t[n, i, l] * Q_t[i, l]
```

- `alpha_t[n, i, l]` describes the local slope or interaction strength.
- `alpha_t[n, i, l] * Q_t[i, l]` is the amount added to this particular forecast.
- A large coefficient can yield a small realized contribution when its input value is close to zero.
- A negative coefficient represents an inhibitory or inverse local effect. The sign of the realized contribution also depends on the sign of the observed input value.

This distinction matters for explanation evaluation. Recovering `alpha_t` tests whether the model learned the mechanism. Inspecting `alpha_t * Q_t` explains the numerical forecast for one example.

## Focuser And Modeler

DCIts factorizes the transition coefficient tensor into two learned tensors:

```text
F_t = sigmoid(Z_t)       # Focuser: soft source-lag selection in (0, 1)
C_t = Modeler(F_t * Q_t) # Modeler: signed local coefficients
alpha_t = C_t * F_t      # final signed transition coefficients
```

The two parts have different jobs:

1. **Focuser `F_t`:** gives a soft mask indicating which source-lag entries should remain active.
2. **Modeler `C_t`:** estimates the signed magnitude of each selected relation.
3. **Transition tensor `alpha_t`:** combines selection and signed strength. This is the main explanatory object.
4. **Contribution tensor `alpha_t * Q_t`:** contains the amount contributed by every source-lag entry to every target forecast.

Both branches use a bank of convolutional filters that view complementary structures in the input window: the whole `N x L` window, each series across time, all series at one time, and short lag windows of length 3 and 5. Fully connected bottleneck layers then produce `N * N * L` values, which are reshaped into the transition tensor.

## Local And Global Explanations

The tensors are computed separately for every input window, so the explanation can change between examples or regimes.

- **Local lag-resolved explanation:** inspect `alpha_t[n, i, l]` or `alpha_t[n, i, l] * Q_t[i, l]` for one forecast.
- **Global source-to-target summary:** aggregate absolute coefficients over lags:

```text
beta_tilde[n, i] = sum_l abs(alpha_t[n, i, l])
beta[n, i]       = beta_tilde[n, i] / sum_j beta_tilde[n, j]
```

`beta[n, i]` is the proportion of total absolute coefficient strength for target `n` assigned to source `i`. It gives an easy source-series ranking. Because it aggregates absolute values, `beta` keeps strength and removes direction. The full `alpha_t` tensor preserves lag and sign.

## Higher-Order Extension

The default experiments use a bias branch (`p = 0`) and a linear branch (`p = 1`). Optional branches use elementwise powers of each input:

```text
forecast[n] = bias[n]
              + sum_p sum_i sum_l alpha_t[p, n, i, l] * Q_t[i, l]**p
```

This allows the explanation to say that a source-lag relation is linear, quadratic, or cubic. Each branch remains directly decomposable into signed contributions. The implementation represents powers of individual source-lag entries. Products between different entries, such as `Q[i, l1] * Q[j, l2]`, are left as an extension for a richer basis.

## Dataset Details

### Reused Benchmark Family

The paper uses Datasets 1-8 from Baric et al. (2021):

- Dataset 1: noisy constant series.
- Dataset 2: independent autoregressive series with true lags 3 and 7 and coefficient `0.5` at each lag.
- Dataset 3: nonlinear autoregressive series with known lags.
- Dataset 4: two interdependent series with known cross-series lags.
- Dataset 5: one driver series generates the remaining series through known lagged linear coefficients.
- Dataset 6: nonlinear version of the driver-series process.
- Dataset 7: a custom signed vector-autoregressive process with biases and positive and negative cross-series interactions.
- Dataset 8: a switching process with two local source-lag regimes.

The authors corrected generator/code inconsistencies from the earlier description and adjusted Dataset 8 so that the first series carries the switching process.

### Additional Controlled Processes

- **VAR(2):** known lag-1 and lag-2 coefficient matrices provide a transparent signed coefficient-recovery test.
- **Cubic process:** known linear and cubic terms test whether higher-order DCIts recovers the active polynomial order, lag, sign, and magnitude.

### Experimental Protocol

- Contiguous chronological split: 60% train, 20% validation, 20% test.
- The split is performed before rolling windows are constructed, preventing a window from crossing a split boundary.
- Only training windows are shuffled.
- Each compared method sees the same fixed synthetic realization.
- Repeated training runs use different initialization and mini-batch seeds.
- Default training uses Adam, learning rate `1e-3`, batch size 64, MSE loss, at most 100 epochs, learning-rate scheduling, early stopping, and restoration of the best validation checkpoint.
- Selected coefficient-recovery experiments repeat training with MAE to assess loss-function sensitivity.

## Ground-Truth Explanation

The ground truth comes directly from the synthetic generator equations. If the generator contains

```text
X_2[t] = 1 - 1.0 * X_1[t-2] + noise
```

then the ground-truth explanation for target `X_2` contains:

```text
target = X_2
source = X_1
lag = 2
coefficient = -1.0
bias = +1.0
```

This gives several possible automatic labels:

- **Support mask:** whether `(target, source, lag)` is active.
- **Signed mask:** positive, negative, or zero.
- **Coefficient tensor:** the exact numerical generating coefficient.
- **Order tensor:** bias, linear, quadratic, cubic, and so on.
- **Regime-specific tensor:** the correct local mechanism for the current example.
- **Realized contribution:** ground-truth coefficient multiplied by the example's input value.

This is stronger than treating every nonzero term as equally important: it contains direction and strength as well as location.

## Explanation Evaluation Used In The Paper

The paper compares learned coefficients with generator coefficients in four main ways:

1. **Support recovery:** inspect whether stable nonzero `alpha` entries occur at the true source-lag locations.
2. **Sign recovery:** check whether excitatory/positive and inhibitory/negative coefficients have the correct direction.
3. **Magnitude recovery:** report learned coefficient mean plus or minus standard deviation beside the exact generator coefficient.
4. **Order and regime recovery:** test whether polynomial branches identify linear versus cubic terms and whether local Focuser patterns change with the switching regime.

Examples reported in the paper include:

- Dataset 7 coefficients with true values `-1`, `-2/7`, and `-5/7` are recovered with the correct source, lag, sign, and close magnitude.
- In the cubic process, a true linear coefficient of `-2.75` is recovered as approximately `-2.749 +/- 0.04` and `-2.7501 +/- 0.0006` for two series.
- The cubic branch recovers coefficients near `3.75`, while the quadratic branch remains zero.
- The VAR(2) experiment compares reconstructed lag matrices directly with the known matrices.

The published evaluation combines numerical coefficient comparisons, mean/standard-deviation reporting, and side-by-side heatmaps. It provides exact ground-truth tensors from which this project can additionally calculate aggregate metrics such as support precision/recall/F1, signed accuracy, coefficient MAE/RMSE, rank correlation, and contribution error.

## Stability Filter

Training is repeated `R` times with the same data and hyperparameters and different random seeds. For each coefficient, the paper calculates the across-run mean `mu` and standard deviation `sigma`, then keeps it in interpretation plots when:

```text
abs(mu[n, i, l]) > 1.95 * sigma[n, i, l]
```

The paper presents this as a conservative reproducibility filter. It suppresses coefficients that vary strongly across initializations and supports an empirical estimate of maximum useful lag: the largest lag with a stable non-negligible coefficient indicates the effective memory depth.

## Forecasting Evaluation

- **Main forecasting loss/metric:** MSE.
- **Sensitivity analysis:** MAE is also used in selected runs.
- **Main interpretable baseline:** IMV-LSTM, selected from the authors' earlier benchmark.
- **Additional forecasting robustness context:** DA-RNN, TCDF, and Seq2Graph.
- **Interpretation of forecasting accuracy:** It constrains the learned coefficient mechanism to remain predictive; coefficient recovery is evaluated separately against the generator.

## Relationship To Baric Et Al. (2021)

The 2021 paper contributes the transparent generator family and evaluates attention/causality patterns mostly through heatmaps and repeated-run summaries. The 2026 DCIts paper contributes a new forecasting architecture whose prediction equation exposes signed source-lag coefficients directly. It reuses the earlier generators so that learned coefficients can be compared numerically with the true coefficients.

```text
Baric et al. (2021):
known generator -> train attention models -> inspect whether attention highlights the known structure

DCIts (2026):
known generator -> train coefficient-producing model -> compare learned signed source-lag coefficients with generator coefficients
```

This makes DCIts especially relevant to the question of retrieving an underlying time-series mechanism from observations.

## Code Usability

- **Framework:** Python and PyTorch.
- **Repository:** https://github.com/hc-xai/dcits
- **Reproducibility statement:** The paper states that code, scripts, and notebooks for Datasets 1-8 are available in the repository.
- **Local execution status:** Pending in this project.
- **Adaptability:** High. The generators and exact coefficient tensors can become benchmark labels, while DCIts can serve as an intrinsic baseline against post-hoc explanations of foundation forecasters.

## Limitations For This Project

- DCIts explanations belong to its own forecasting equation, so comparison with Chronos or another black-box model requires a separate post-hoc explainer and shared ground-truth benchmark.
- The paper studies one-step multivariate forecasting; a multi-horizon benchmark needs one coefficient/contribution tensor per forecast horizon or a carefully defined aggregation.
- Higher-order branches use elementwise powers and keep the representation compact; mixed products between different variables or lags require added cross-term branches.
- The paper's coefficient-recovery evaluation emphasizes examples, heatmaps, and mean plus or minus standard deviation. A project-wide automatic benchmark should add aggregate tensor-level metrics.
- Synthetic coefficient ground truth describes the designed data-generating mechanism. Application to real observational data yields an estimated effective predictive mechanism whose scientific or causal interpretation depends on data coverage and assumptions.

## How It Helps This Project

1. Reuse its transparent generators for exact feature-time ground truth.
2. Store both generator coefficients and realized contributions for every example.
3. Evaluate explanation support, sign, magnitude, and stability separately.
4. Include DCIts as an intrinsically interpretable forecasting baseline.
5. Compare DCIts coefficient recovery with SHAPformer generator-SHAP ground truth:

```text
DCIts ground truth target: structural source-lag coefficients and contributions
SHAPformer ground truth target: Shapley values of feature coalitions evaluated on the generator
```

6. Extend the paper's visual comparisons with automatic metrics over the full target-source-lag tensor.

## Final Decision

**Core reference - analyze and adapt.** It provides one of the clearest links in the local collection between a known multivariate time-series generator, a forecast equation, a signed local explanation, and direct recovery of the underlying source-lag mechanism.
