# LoMEF: A Framework to Produce Local Explanations for Global Model Time Series Forecasts

## Basic Info
- **Title:** LoMEF: A framework to produce local explanations for global model time series forecasts
- **Authors:** Dilini Rajapaksha, Christoph Bergmeir, Rob J. Hyndman
- **Year:** 2023
- **Venue:** International Journal of Forecasting, 39(3), 1424-1447
- **Paper link:** https://arxiv.org/abs/2111.07001
- **Publisher/source:** https://robjhyndman.com/publications/lomef/
- **Code link:** Unknown from searched sources
- **Dataset link:** Benchmark datasets used in experiments; exact list requires paper extraction
- **Local PDF:** Missing

## Problem Type
- **Task:** Local explanation of global forecasting models
- **Forecasting?** Yes
- **Input:** A time series whose forecast from a global model needs explanation
- **Output:** Forecast from the global black-box model and local surrogate explanation

## Relevance To This Project
- **Predicts future values?** The explained model does
- **Explains forecasts?** Yes
- **Ground-truth explanations?** No synthetic mask focus
- **Black-box compatible?** Yes, model-agnostic
- **Chronos/foundation forecasting relevance:** High conceptually
- **Relevance score:** 4/5
- **Reason:** Directly addresses local explanations for global forecasting models, which is close to explaining Chronos-style foundation forecasters.

## Dataset Details
- **Dataset names:** Forecasting benchmark datasets; exact list needs extraction
- **Synthetic or real:** Likely real benchmark datasets
- **Input shape:** Individual time series plus neighborhood samples
- **Output shape:** Forecast horizon / one-step-ahead forecasts depending on setup

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** Not applicable
- **Evaluation target:** Fidelity/stability/comprehensibility of surrogate explanations

## Model Details
- **Architecture:** Any global forecasting model
- **Black-box/interpretable:** Black-box global model explained by local interpretable surrogate
- **Surrogates:** Simpler univariate forecasting models such as ETS

## Explanation Method
- **Type:** Post-hoc, local, model-agnostic
- **Family:** Local surrogate for forecasting
- **Explanation target:** Trend, seasonality, coefficients, interpretable local model components
- **Neighborhood generation:** Bootstrapping or one-step-ahead global model forecasts

## Evaluation
- **Forecasting metrics:** Accuracy of local surrogate forecasts
- **Explanation metrics:** Fidelity, stability, comprehensibility, qualitative assessment
- **Evaluation summary check:** The summary agrees with the paper abstract and publisher/source snippets. LoMEF evaluates local surrogate explainers in quantitative dimensions of accuracy, fidelity, and stability/consistency, then uses qualitative examples to judge comprehensibility.
- **Evaluation references cited:** Local PDF is missing, so exact reference-list verification is still pending. From the available source pages, the evaluation appears grounded in forecasting benchmark practice and local surrogate-model explanation logic rather than synthetic attribution masks. Once the PDF is added, extract its formal citations for fidelity, stability, and surrogate-local-explanation evaluation.

## Code Usability
- **Code available?** Unknown
- **Runs locally?** Not checked
- **Adaptability:** High for black-box forecast explanation workflow.

## Limitations
- Does not produce temporal/feature-time ground-truth masks
- Explanation is through interpretable surrogate components, not direct attribution heatmaps
- Local surrogate choice may constrain explanation type

## How It Helps This Project
- Strong candidate for explaining Chronos-2 as a global black-box forecaster.
- Provides evaluation dimensions beyond mask overlap: fidelity, accuracy, stability, comprehensibility.

## Final Decision
**Adapt idea** as a model-agnostic local explanation method for foundation forecasters.
