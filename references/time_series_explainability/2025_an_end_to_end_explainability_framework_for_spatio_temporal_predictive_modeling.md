# An End-to-End Explainability Framework for Spatio-Temporal Predictive Modeling

## Basic Info
- **Title:** An end-to-end explainability framework for spatio-temporal predictive modeling
- **Authors:** Massimiliano Altieri, Michelangelo Ceci, Roberto Corizzo
- **Year:** 2025
- **Venue:** Machine Learning
- **DOI:** https://doi.org/10.1007/s10994-024-06733-6
- **Paper link:** Local PDF: `Refferences/2025_an_end_to_end_explainability_framework_for_spatio_temporal_predictive_modeling.pdf`
- **Code link:** Unknown from local PDF preview
- **Dataset link:** Uses real-world forecasting datasets; exact dataset links need deeper extraction
- **License:** Springer Nature article; local PDF states author copyright, terms apply

## Problem Type
- **Task:** Spatio-temporal predictive modeling with explanations
- **Forecasting?** Yes, includes real-world forecasting datasets
- **Input:** Sensor-network observations collected at regular intervals and locations
- **Output:** Predictions/forecasts from deep spatio-temporal models
- **Univariate/multivariate:** Multivariate and spatial/node-based

## Relevance To This Project
- **Predicts future values?** Yes, in forecasting settings
- **Explains forecasts?** Yes
- **Ground-truth explanations?** Not clearly; evaluation appears qualitative and quantitative but not synthetic mask-first
- **Black-box compatible?** Yes, described as model-agnostic masking meta-optimization
- **Chronos/foundation forecasting relevance:** Medium
- **Relevance score:** 3.5/5
- **Reason:** Strong model-agnostic XAI framing for spatio-temporal forecasts, but less directly aligned with univariate/multivariate Chronos-style synthetic mask benchmarks.

## Dataset Details
- **Dataset names:** Real-world forecasting datasets; exact names need extraction from full paper
- **Synthetic or real:** Real-world
- **Input shape:** Time x nodes/locations x features
- **Output shape:** Forecast/prediction target over nodes/time
- **Covariates/exogenous variables:** Sensor features and node/location dimensions
- **Data generation mechanism:** None; real data

## Ground-Truth Explanation
- **Known mask?** Not central / unclear
- **Mask shape:** Explanations can cover features, timesteps, and node locations
- **Mask generation:** Learned by masking/meta-optimization, not generated as known causal truth
- **Ambiguity:** Without synthetic ground truth, explanation faithfulness depends on perturbation/evaluation protocol.

## Model Details
- **Architecture:** Deep learning spatio-temporal predictive models
- **Black-box/interpretable:** Black-box models explained by post-hoc masking framework
- **Required interface:** Model prediction calls under masked inputs

## Explanation Method
- **Type:** Post-hoc, model-agnostic, global salient-factor extraction
- **Family:** Masking / meta-optimization
- **Explanation targets:** Features, timesteps, node locations
- **Cost:** Likely higher than simple attribution because mask optimization requires repeated inference/training
- **Attribution scope:** Temporal Attribution, Covariate Attribution, and Time-Covariate Attribution, with an additional spatial/node axis. In project terms, this is a feature-time-location masking method.

## Evaluation
- **Forecasting metrics:** The paper evaluates explanations on trained spatio-temporal predictive models; the XAI comparison focuses less on forecasting leaderboard metrics and more on how predictions change under masked inputs.
- **Explanation metrics:** Model-level Fidelity+ and Fidelity-, phenomenon-level Fidelity+ and Fidelity-, normalized variants of those fidelity metrics, sparsity, and execution/time-complexity analysis.
- **Baselines:** Random Forest feature importance, LIME, Perturbation, GNNExplainer, PGExplainer, and GraphLIME depending on dataset/model setting.
- **Predictive models tested:** LSTM, GRU, Bi-LSTM, Attention-LSTM, SVD-LSTM, CNN-LSTM, and GCN-LSTM.
- **Datasets:** Beijing Multisite Air Quality, Lightsource/PV Italy, and Wind NREL appear in the experimental tables.
- **Quantitative or visual:** Both; quantitative fidelity/sparsity/time tables plus qualitative spatio-temporal explanations.
- **Evaluation summary check:** The earlier summary was accurate at a high level but needed the actual metrics. The important correction is that this paper is not synthetic-mask evaluation; it is perturbation/masking fidelity evaluation over feature-time-location explanations.
- **Evaluation references cited:** Uses Fidelity+ from Pope et al. (2019), graph-XAI evaluation taxonomy from Yuan et al. (2022), LIME from Ribeiro et al. (2016), SHAP from Lundberg and Lee (2017), GNNExplainer (Ying et al., 2019), PGExplainer (Luo et al., 2020), GraphLIME (Huang et al., 2022), and ShapTime as a time-series XAI reference.

## Code Usability
- **Code available?** Unknown
- **Runs locally?** Not checked
- **Adaptability:** Good conceptual fit if project expands to sensor networks or graph/spatio-temporal forecasting.

## Limitations
- Spatio-temporal graph/sensor setting may be broader than current project
- Not a synthetic ground-truth mask benchmark
- No synthetic ground-truth mask validation; the exact evaluation is perturbation/masking fidelity, sparsity, and runtime

## How It Helps This Project
- Useful reference for explaining across three axes: features, time, and locations.
- Supports a future extension from feature-time masks to feature-time-node masks.
- Offers a model-agnostic masking pattern that could be adapted to Chronos-style black-box forecasting.

## Final Decision
**Adapt idea** for model-agnostic masking and multi-axis explanation design; not a core synthetic benchmark.
