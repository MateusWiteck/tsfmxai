# TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data

## Basic Info
- **Title:** TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data
- **Authors:** Ege Onur Taga, M. Emrullah Ildiz, Samet Oymak
- **Year:** 2025
- **Venue:** arXiv
- **Paper link:** https://arxiv.org/abs/2502.16294
- **Code link:** Unknown from local preview
- **Dataset link:** Synthetic training process described by paper
- **Citation/BibTeX:** Not collected
- **License:** Unknown

## Problem Type
- **Task:** Multivariate time-series forecasting
- **Forecasting?** Yes
- **Input:** Historical multivariate context
- **Output:** Future values
- **Univariate/multivariate:** Multivariate

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** No, not primarily
- **Ground-truth explanations?** No XAI mask indicated from preview
- **Black-box compatible?** Forecasting model, not explainer
- **Chronos/foundation forecasting relevance:** Medium for synthetic forecasting generation
- **Relevance score:** 2/5
- **Reason:** Useful for synthetic forecasting data/model ideas, but not explainability.

## Dataset Details
- **Dataset names:** `LMC-Synth` synthetic multivariate forecasting training scheme
- **Synthetic or real:** Synthetic for training/evaluation plus real benchmarks
- **Input shape:** Multivariate history
- **Output shape:** Forecast horizon
- **Generation mechanism:** Synthetic data generation for zero-shot/few-shot forecasting. The generator creates multivariate time series from Gaussian-process latent functions composed with `KernelSynth`, then mixes those latent functions across channels using a Linear Model of Coregionalization (LMC).
- **Core idea:** First generate latent univariate functions using Gaussian-process kernel compositions that create trends, periodic patterns, locally periodic patterns, and other realistic motifs. Then form each output channel as a convex combination of latent functions, so different channels can share structure and become correlated.
- **LMC-Synth algorithm:** Sample the number of latent functions from a Weibull distribution, generate each latent function with `KernelSynth`, sample channel-specific mixing weights from a Dirichlet distribution, and compute each channel as `C_i(t) = sum_j alpha_ij * l_j(t)`.
- **Independent-channel variant:** The paper also generates data where each channel is independent, `C_i(t) = l_i(t)`, because some multivariate datasets have weak or no cross-channel dependence. This variant is used in ablation/curriculum-style training.
- **Training corpus:** The paper reports generating 15,000 synthetic datasets with length 1,024 and channel size 160, sampled into windows of length 192: 96 input steps and 96 output steps, giving roughly 1.5 million synthetic training points.
- **Training target:** Forecast the future multivariate output from the input context using MSE loss.

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** None
- **Ambiguity:** This is synthetic forecasting pretraining data, not synthetic XAI benchmark data. It does not provide known temporal/covariate attribution masks or ground-truth explanations.
- **Synthetic-generation references cited:** The generator builds on KernelSynth from Chronos/Ansari et al. (2024), compositional kernel search from Duvenaud et al. (2013), Linear Model of Coregionalization from geostatistics/Journel and Huijbregts, Gaussian-process multi-output modeling/Alvarez et al. (2012), and Prior-data Fitted Networks from Mueller et al. (2022), with ForecastPFN as a related synthetic forecasting prior.

## Model Details
- **Architecture:** Transformer-based forecasting model
- **Black-box/interpretable:** Forecasting model, not explanation method
- **Required interface:** Forecasting inputs/outputs

## Explanation Method
- **Type:** None
- **Target:** Not applicable

## Evaluation
- **Forecasting metrics:** Forecasting benchmark metrics
- **Explanation metrics:** None

## Code Usability
- **Code available?** Unknown
- **Adaptability:** Synthetic generation ideas may be useful.

## Limitations
- Forecasting-only
- No XAI evaluation
- No known attribution masks

## How It Helps This Project
- Could inspire robust synthetic forecasting generators, but not explanation benchmark design.

## Final Decision
**Use only as citation** for synthetic forecasting, not XAI.
