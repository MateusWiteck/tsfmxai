# Why Should I Trust You? Explaining the Predictions of Any Classifier

## Basic Info
- **Title:** "Why Should I Trust You?" Explaining the Predictions of Any Classifier
- **Authors:** Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
- **Year:** 2016
- **Venue:** KDD 2016 / arXiv
- **Paper link:** https://arxiv.org/abs/1602.04938
- **Code link:** https://github.com/marcotcr/lime
- **Dataset link:** Not central to this project
- **Citation/BibTeX:** LIME paper
- **License:** Check code repository

## Problem Type
- **Task:** General model-agnostic explanation for classifiers/regressors
- **Forecasting?** Not specifically
- **Input:** Any interpretable representation
- **Output:** Classifier or regressor prediction

## Relevance To This Project
- **Predicts future values?** Not the paper focus
- **Explains forecasts?** Can explain regressors, but not time-series forecasting-specific
- **Ground-truth explanations?** Uses simulated/human evaluation, not forecasting masks
- **Black-box compatible?** Yes
- **Chronos/foundation forecasting relevance:** Medium as a generic perturbation/surrogate method
- **Relevance score:** 2/5
- **Reason:** Useful methodologically, but not a forecasting-XAI benchmark.

## Dataset Details
- **Dataset names:** Text/image/tabular examples in paper
- **Synthetic or real:** Mixed
- **Input shape:** Depends on task
- **Output shape:** Class/regression prediction
- **Forecast target:** None

## Ground-Truth Explanation
- **Known mask?** Not for forecasting
- **Mask shape:** Depends on interpretable components
- **Mask generation:** Perturbed interpretable components, not causal forecasting masks

## Model Details
- **Architecture:** Any black-box classifier/regressor
- **Black-box/interpretable:** Black-box explained by local sparse surrogate
- **Required interface:** Prediction function over perturbed inputs

## Explanation Method
- **Type:** Post-hoc, local, model-agnostic
- **Family:** Local surrogate / perturbation
- **Explanation target:** Interpretable components
- **Cost:** Many perturbation calls per explained instance

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Fidelity, user trust, simulated tasks
- **Quantitative or visual:** Both

## Code Usability
- **Code available?** Yes
- **Runs locally?** Likely
- **Adaptability:** Can wrap a forecasting model if an interpretable perturbation space is designed.

## Limitations
- Not time-series-specific
- Perturbations can be unrealistic for autocorrelated series
- No native lag/window ground truth

## How It Helps This Project
- Useful baseline idea: local surrogate over historical windows or covariates.
- Need time-series-aware perturbation to avoid unrealistic histories.

## Final Decision
**Adapt idea** as a generic black-box explanation baseline, not as a benchmark.

