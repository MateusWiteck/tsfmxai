# Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions

## Basic Info
- **Title:** Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions
- **Authors:** Domjan Barić, Petar Fumić, Davor Horvatić, Tomislav Lipić
- **Year:** 2021
- **Venue:** Entropy
- **Paper link:** https://www.mdpi.com/1099-4300/23/2/143
- **DOI:** https://doi.org/10.3390/e23020143
- **Code link:** Unknown from local preview
- **Dataset link:** Synthetic benchmark described in paper
- **Citation/BibTeX:** Entropy 2021, 23, 143
- **License:** MDPI article; code/data license unknown

## Problem Type
- **Task:** Multivariate time-series prediction / forecasting interpretability benchmark
- **Forecasting?** Yes, future values are predicted from historical multivariate time series
- **Input:** Multiple historical time series with interactions
- **Output:** Predicted future target values
- **Univariate/multivariate:** Multivariate

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** Yes, through attention-based interpretability
- **Ground-truth explanations?** Yes, via transparent synthetic generating processes
- **Black-box compatible?** Focuses on attention-based models, not fully black-box
- **Chronos/foundation forecasting relevance:** High as benchmark inspiration
- **Relevance score:** 5/5
- **Reason:** Closest local paper to forecasting + explanation correctness + synthetic known mechanisms.

## Dataset Details
- **Dataset names:** Ten synthetic datasets, numbered 1-10, with increasing complexity
- **Synthetic or real:** Synthetic
- **Input shape:** Multivariate time-series history
- **Output shape:** Forecast/prediction target
- **Forecast target:** Future time-series values
- **Covariates/exogenous variables:** Multiple interacting time series
- **Generation mechanism:** Transparent statistical/mechanistic time-series interactions. Datasets 1-8 are handcrafted dynamical systems; dataset 9 is based on a 2D Ising model; dataset 10 is a logistic-map-inspired system.
- **Detailed synthetic generators:**
  - Dataset 1: constant time series with Gaussian noise.
  - Dataset 2: independent autoregressive series using lags 3 and 7.
  - Dataset 3: nonlinear autoregressive series using `tanh` and lags 3, 7, and 9.
  - Dataset 4: two interdependent time series without autoregression.
  - Dataset 5: one autoregressive driver series, with the other series generated from lagged values of the first series.
  - Dataset 6: nonlinear version of dataset 5 using `tanh`.
  - Dataset 7: custom vector autoregressive system with explicit cross-series lagged dependencies.
  - Dataset 8: switching system where the active dependency rule changes depending on the value of the first series.
  - Dataset 9: first-order 2D square-lattice Ising model with a 10 x 10 lattice and temperatures including the phase-transition temperature.
  - Dataset 10: logistic-map-inspired multivariate system with different `r` values, including chaotic regimes.
- **Generation parameters:** For datasets 1-8, initial points are sampled from `N(0, 1)`, values are generated sequentially, and Gaussian noise is injected with probability `f = 0.3`, `mu_noise = 0`, and `sigma_noise = 0.1`. Most datasets use `N = 5` time series, except dataset 4 with `N = 2` and datasets 7-8 with `N = 4`.
- **Length/split:** Each series has 20,000 generated points, the first 1,000 are discarded, and the last 2,000 points are used as the test set.
- **Code:** The paper says the generation code is available at `https://github.com/hc-xai/mts-interpretability-benchmark`.

## Ground-Truth Explanation
- **Known mask?** Yes, through known generating process
- **Mask shape:** Feature/time interaction relevance, depending on dataset
- **Mask generation:** From the known lagged equations: autoregressive lags define temporal importance, cross-series equations define feature/covariate importance, and switch/statistical/mechanistic rules define harder cases.
- **Best use for this project:** Strong reference for generating `history -> future` samples where the true relevant lags and variables are known. The mask can be represented as temporal masks, feature masks, or feature-time masks depending on the dataset.
- **Ambiguity:** Autocorrelation and cross-correlation may complicate attribution.
- **Synthetic-generation references cited:** The paper explicitly contrasts its diagnostic generators with CauseMe and M4, arguing those are not ideal for this benchmark because CauseMe focuses on causal association processes and M4 masks real-world origins. It also uses the logistic map and the Ising model as known statistical/mechanistic process references.

## Model Details
- **Architectures:** Attention-based deep models, including IMV-LSTM
- **Black-box/interpretable:** Intrinsic attention interpretability
- **Required interface:** Model attention/importance outputs

## Explanation Method
- **Type:** Intrinsic attention-based interpretability
- **Target:** Relevant variables and temporal interactions
- **Family:** Attention mechanisms
- **Attribution scope:** Temporal Attribution, Covariate Attribution, and Time-Covariate Attribution. The synthetic equations define both which lag matters and which time series/variable drives each target, so explanations can be evaluated as lag-only, variable-only, or variable-at-lag masks.

## Evaluation
- **Forecasting metrics:** Mean squared error (MSE), reported per synthetic dataset and summarized across repeated experiments. The paper also treats prediction stability as the standard deviation of MSE divided by mean MSE.
- **Explanation metrics:** No single automatic explanation-correctness metric is defined. The paper has known synthetic ground truth, then compares attention/causality heatmaps with the expected lag/source-series patterns mostly by visual/manual inspection. It also reports mean and standard deviation of attention coefficients and TCDF retrieved-causality percentages across runs.
- **Quantitative or visual:** Forecasting/stability metrics are quantitative; explainability correctness is mostly qualitative/manual heatmap comparison against known synthetic mechanisms.
- **Evaluation summary check:** Important correction: this paper has ground-truth explanations, but it does not turn them into a formal automatic score such as F1, AUPRC, IoU, or rank correlation. It frames evaluation with the Performance-Explainability Framework and uses MSE/stability for performance, while explanation correctness is judged by whether learned attention patterns visually match the known equations.
- **Evaluation references cited:** Uses the Performance-Explainability Framework by Fauvel, Masson, and Fromont (2020) as the high-level benchmark structure, with six dimensions: performance, comprehensibility, granularity, information, faithfulness, and user. It also cites Lundberg and Lee (2017) for SHAP as an important model-agnostic reference point, although SHAP is not the main evaluated method.

## Explainability Calculus From The Paper
- **Key paper text segments:** The paper states that attention can be read as "the model's feature's importance". For seq2graph and IMV-LSTM, alpha coefficients "model autocorrelation" and beta coefficients "model crosscorrelation".
- **Ground-truth basis:** The synthetic equations define which time series and which lags should matter. For a target series `y_i`, the expected explanation is built from the nonzero terms in the data generator: relevant source series become feature/covariate ground truth, and relevant lag indices become temporal ground truth.
- **Alpha coefficients (`alpha`):** These are temporal attention weights. For each target/source time series and input window of size `w`, `alpha` has `w` values and the weights sum to 1. The paper uses mean `alpha` heatmaps to see whether high temporal attention falls on the true generating lags, e.g. lags 3 and 7 in dataset 2.
- **Beta coefficients (`beta`):** These are cross-series attention weights. For a target series, `beta` gives one importance value per candidate source series and the weights sum to 1. The paper compares the mean `beta` matrix with the known dependency matrix from the synthetic generator. For dataset 4, high values should appear on the anti-diagonal because each of the two series is generated from the other. For dataset 2, high values should appear on the main diagonal because each series is autoregressive.
- **IMV-LSTM calculation path:** IMV-LSTM preserves a variable-wise hidden-state matrix. The hidden-state matrix is used to calculate `alpha`. Then `alpha * hidden_state` is concatenated with the hidden state, written in the paper as `[alpha * h, h]`, and this is used to calculate `beta`. In implementation terms, `alpha` gives lag saliency first, then `beta` gives variable/source saliency using both the hidden state and the lag-weighted hidden state.
- **seq2graph calculation path:** The dual-purpose RNN produces `alpha` coefficients for autocorrelation over the input window. The decoder produces `beta` coefficients for cross-correlation across series. The paper evaluates these by plotting the mean coefficients against the dependency pattern implied by the generator.
- **TCDF calculation path:** TCDF uses an attention vector over the `N` input time series, then applies semi-binarization to approximate hard attention. The paper does not treat the resulting values as normalized attention coefficients. Instead, it counts, across repeated experiments, the percentage of runs in which a causal relation is retrieved for each `(target series, source series)` pair. Rows do not have to sum to 1 because a run may retrieve no cause for a target.
- **DA-RNN calculation path:** DA-RNN attention requires aggregation before comparison. Aggregated input attention corresponds to `beta`-like cross-series importance. Aggregated temporal attention corresponds to `alpha`-like temporal importance, but the paper warns that this temporal attention maps to encoder outputs rather than directly to input series, so it cannot produce full feature-lag explanations.
- **Confidence calculation:** The paper uses repeated experiments to compute the mean and standard deviation of attention coefficients. A nearly uniform mean with high standard deviation is interpreted as low-confidence interpretability. For example, with `N = 5`, a no-information `beta` distribution is about `1 / N = 0.2`; mean values around `0.18-0.22` with standard deviation around `0.05` are treated as not meaningfully different from uniform.
- **Prediction stability calculation:** For model-performance stability, the paper plots `std(MSE) / mean(MSE)` across experiments for each dataset.
- **Sensitivity calculation:** For sensitivity to generator hyperparameters, the paper uses normalized error: `MSE / max(MSE)_model`, where the denominator is the maximum error observed for that model in the corresponding sensitivity sweep.
- **Important limitation:** The paper mostly evaluates interpretability visually/qualitatively against known ground truth rather than defining one universal explanation score. To reproduce this benchmark quantitatively in this project, convert the paper's heatmap comparisons into explicit metrics such as top-k source accuracy, lag-hit rate, or AUPRC against the generator-derived feature-time mask.

## Code Usability
- **Code available?** Unknown from local PDF
- **Runs locally?** Not checked
- **Adaptability:** High for designing synthetic forecasting-XAI generators.

## Limitations
- Attention-based, not model-agnostic
- May not directly apply to black-box Chronos
- Need extract exact generator definitions

## How It Helps This Project
- Strong benchmark template for `history -> future` with known drivers.
- Should be studied deeply and possibly replicated/adapted.

## Final Decision
**Adapt idea / inspect deeply** as a core reference.
