# XForecast: Evaluating Natural Language Explanations for Time Series Forecasting

## Basic Info
- **Title:** XForecast: Evaluating Natural Language Explanations for Time Series Forecasting
- **Authors:** Taha Aksu, Chenghao Liu, Amrita Saha, Sarah Tan, Caiming Xiong, Doyen Sahoo
- **Year:** 2024
- **Venue:** arXiv
- **Paper link:** https://arxiv.org/abs/2410.14180
- **Code link:** CatalyzeX indicates code/notebook links; exact repository not collected
- **Dataset link:** Need extraction from paper
- **Local PDF:** Missing

## Problem Type
- **Task:** Evaluation of natural-language explanations for time-series forecasting
- **Forecasting?** Yes
- **Input:** Historical time series and forecast
- **Output:** Natural-language explanation plus simulatability evaluation

## Relevance To This Project
- **Predicts future values?** Forecasting model does
- **Explains forecasts?** Yes, in natural language
- **Ground-truth explanations?** Not attribution-mask focused
- **Black-box compatible?** Yes, explanations can be generated for black-box forecasts
- **Chronos/foundation forecasting relevance:** Medium, especially if adding LLM explanations
- **Relevance score:** 3/5
- **Reason:** Useful for human-facing explanation evaluation, not synthetic attribution masks.

## Dataset Details
- **Dataset names:** Need extraction from paper
- **Synthetic or real:** Likely forecasting examples/tasks for NLE evaluation
- **Input shape:** Historical time series plus forecast
- **Output shape:** Natural-language explanation and simulated forecast from surrogate/human proxy

## Ground-Truth Explanation
- **Known mask?** No direct feature-time mask
- **Evaluation target:** Whether an explanation helps a surrogate/human predict the model forecast

## Model Details
- **Architecture:** Forecasting model plus LLM explanation generator/evaluator
- **Black-box/interpretable:** Black-box forecasts explained in natural language
- **Required interface:** Forecast outputs and text generation/evaluation

## Explanation Method
- **Type:** Natural-language explanation
- **Family:** Simulatability-based evaluation
- **Explanation target:** Human-understandable rationale for forecast
- **Attribution scope:** Not directly an attribution-mask method. It evaluates natural-language explanations rather than Temporal, Covariate, or Time-Covariate attribution maps.

## Evaluation
- **Forecasting metrics:** Forecast comparison through simulatability
- **Explanation metrics:** Direct simulatability and synthetic simulatability
- **Human alignment:** Metrics align with human judgments according to abstract
- **Evaluation summary check:** The summary agrees with the paper abstract. XForecast evaluates whether a natural-language explanation helps a human-surrogate reproduce or understand a forecasting model's output; it is not an attribution-mask or causal-driver benchmark.
- **Evaluation references cited:** Local PDF is missing, so exact reference-list verification is pending. The core evaluation reference concept is simulatability from XAI: if an explanation is useful, an evaluator should better predict the model's behavior from the input, forecast, and explanation. The paper introduces direct simulatability and synthetic simulatability for time-series forecasting explanations.

## Code Usability
- **Code available?** Possibly; exact link not collected
- **Runs locally?** Not checked
- **Adaptability:** Good for later user-interface layer.

## Limitations
- Not attribution-mask evaluation
- LLM-generated rationales may not be faithful unless tested
- More suitable after quantitative XAI benchmark exists

## How It Helps This Project
- Useful if project evolves toward natural-language explanations for Chronos forecasts.
- Simulatability could complement mask metrics in user studies.

## Final Decision
**Use later** for natural-language explanation evaluation, not core benchmark.
