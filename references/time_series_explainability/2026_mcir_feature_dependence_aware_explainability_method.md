# MCIR: A Feature Dependence-Aware Explainability Method with Reliability Guarantees

## Basic Info
- **Title:** MCIR: A Feature Dependence-Aware Explainability Method with Reliability Guarantees
- **Authors:** Anonymous authors
- **Year:** Under review; local PDF
- **Venue:** TMLR submission
- **Paper link:** Local PDF: `Refferences/2026_mcir_feature_dependence_aware_explainability_method.pdf`
- **Code link:** Unknown
- **Dataset link:** Unknown
- **Citation/BibTeX:** Not available
- **License:** Unknown

## Problem Type
- **Task:** Feature dependence-aware global explainability
- **Forecasting?** Not specifically
- **Input:** Tabular/data-rich ML features
- **Output:** Model predictions

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** Not directly
- **Ground-truth explanations?** Uses synthetic household-energy dataset and UCI HAR according to abstract
- **Black-box compatible?** Global feature importance method
- **Chronos/foundation forecasting relevance:** Medium for correlated lag/covariate problem
- **Relevance score:** 3/5
- **Reason:** Not forecasting-specific, but directly addresses feature dependence and collinearity in attribution.

## Dataset Details
- **Dataset names:** Synthetic household-energy dataset, UCI HAR
- **Synthetic or real:** Mixed
- **Input/output:** General ML prediction tasks
- **Forecast target:** Not clear

## Ground-Truth Explanation
- **Known mask?** Synthetic benchmark likely has controlled feature dependence
- **Mask shape:** Feature-level, not time/window-level
- **Ambiguity:** Central issue is multicollinearity/redundancy.

## Model Details
- **Architecture:** Model-agnostic global importance method
- **Black-box/interpretable:** Explains black-box or complex models globally
- **Required interface:** Data and model outputs / dependence estimates

## Explanation Method
- **Type:** Global, dependence-aware
- **Family:** Conditional information / correlation impact ratio
- **Target:** Features
- **Cost:** Claimed lightweight compared with some baselines
- **Attribution scope:** Covariate Attribution. It can be applied to lagged covariates if each lag is encoded as a feature, but it is not natively a temporal or feature-time attribution method.

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Ranking stability using Kendall tau / `Delta_rank = 1 - tau`, Spearman correlation, exact Jaccard@K, Group-Jaccard@K, allocation stability, Family Share / Original Mass for redundant feature groups, deletion/insertion or perturbation faithfulness curves, deletion AUC, top-K overlap, and runtime.
- **Quantitative or visual:** Quantitative cross-domain experiments plus rank overlays, redundancy stress tests, and deletion curves.
- **Evaluation summary check:** The old summary was correct but incomplete. MCIR is especially relevant to this project because it argues that exact top-feature overlap can be misleading when lags or covariates are correlated; group-level agreement and deletion-curve behavior can be better evaluation signals.
- **Evaluation references cited:** Compares against SHAP/TreeSHAP (Lundberg and Lee, 2017), SAGE (Covert, Lundberg, and Lee, 2020), permutation feature importance, MI/CMI/HSIC-style dependence measures, and CIR-family methods. It discusses ROAR (Hooker et al., 2019) and deletion tests (Samek et al., 2017) as faithfulness-evaluation references, plus Covert and Lee work on explaining by removing and Shapley stability under dependence.

## Code Usability
- **Code available?** Unknown
- **Adaptability:** Useful for collinear lag/covariate ranking ideas.

## Limitations
- Global feature-level method
- Not temporal/forecast-specific
- Under review / anonymous

## How It Helps This Project
- Useful for the core problem that correlated lags can make attribution unstable.
- Could inspire dependence-aware evaluation or attribution post-processing.

## Final Decision
**Adapt idea** for correlated lag/covariate analysis.
