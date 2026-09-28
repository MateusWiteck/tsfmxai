# Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting

## Basic Info
- **Title:** Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Authors:** Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister
- **Year:** 2020
- **Venue:** International Journal of Forecasting / arXiv
- **Paper link:** https://arxiv.org/abs/1912.09363
- **Code link:** https://github.com/google-research/google-research/tree/master/tft
- **Dataset link:** Uses multiple real-world forecasting datasets
- **Citation/BibTeX:** TFT paper
- **License:** Check code repository

## Problem Type
- **Task:** Multi-horizon time-series forecasting
- **Forecasting?** Yes
- **Input:** Static covariates, known future inputs, observed historical inputs
- **Output:** Multi-step future forecasts, often quantiles
- **Univariate/multivariate:** Multi-input, target-specific forecasting

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** Yes, through variable selection and interpretable attention
- **Ground-truth explanations?** No synthetic ground-truth mask as central benchmark
- **Black-box compatible?** No, interpretability is architecture-specific
- **Chronos/foundation forecasting relevance:** Medium
- **Relevance score:** 3/5
- **Reason:** Forecasting + interpretability, but not model-agnostic and not ground-truth attribution benchmark.

## Dataset Details
- **Dataset names:** Real-world multi-horizon datasets
- **Synthetic or real:** Mostly real
- **Input shape:** Historical target/covariates + known future inputs + static features
- **Output shape:** Forecast horizon, often probabilistic quantiles
- **Generation mechanism:** None

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** Not applicable
- **Ambiguity:** Attention/selection weights require validation.

## Model Details
- **Architecture:** Temporal Fusion Transformer
- **Components:** Variable selection networks, gating layers, recurrent layers, interpretable self-attention
- **Black-box/interpretable:** Intrinsically interpretable architecture
- **Requires gradients/probabilities/samples/attention?** Uses model internals/attention/selection weights

## Explanation Method
- **Type:** Intrinsic
- **Target:** Static variable importance, temporal variable importance, attention over time
- **Family:** Attention + variable selection

## Evaluation
- **Forecasting metrics:** Quantile loss and forecasting benchmarks
- **Explanation metrics:** Interpretability case studies, not ground-truth mask scores
- **Quantitative or visual:** Mostly visual/case-study interpretability

## Code Usability
- **Code available?** Yes
- **Runs locally?** Likely, with setup work
- **Adaptability:** Useful as a forecasting interpretability baseline, not as a synthetic benchmark.

## Limitations
- Architecture-specific
- Attention is not guaranteed faithful
- No known historical-driver mask

## How It Helps This Project
- Provides model-side ideas for interpretable forecasting.
- Useful comparison point against post-hoc explanations for Chronos-like models.

## Final Decision
**Use as citation / baseline model**, not as the main benchmark.

