# Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation

## Basic Info
- **Title:** Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation
- **Authors:** Tryambak Gangopadhyay, Sin Yong Tan, Zhanhong Jiang, Rui Meng, Soumik Sarkar
- **Year:** 2020 arXiv; ICASSP 2021 publication
- **Venue:** ICASSP 2021
- **Paper link:** https://arxiv.org/abs/2008.04882
- **DOI:** https://doi.org/10.1109/ICASSP39728.2021.9413914
- **Code link:** Unknown
- **Dataset link:** Two public datasets and one domain-specific dataset; exact names need extraction
- **Local PDF:** Missing

## Problem Type
- **Task:** Multivariate time-series prediction with interpretation
- **Forecasting?** Yes / prediction of future values from past inputs
- **Input:** Multivariate time series
- **Output:** Time-series predictions
- **Univariate/multivariate:** Multivariate

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts/predictions?** Yes, through spatiotemporal attention
- **Ground-truth explanations?** No clear synthetic mask; attention validated from domain knowledge
- **Black-box compatible?** No, model-specific
- **Chronos/foundation forecasting relevance:** Medium
- **Relevance score:** 3/5
- **Reason:** Forecasting/prediction + feature/time interpretability, but intrinsic attention and no known ground-truth mask.

## Dataset Details
- **Dataset names:** Two public datasets and one domain-specific dataset; extract later
- **Synthetic or real:** Real
- **Input shape:** Variables over time
- **Output shape:** Prediction target
- **Generation mechanism:** None

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** Attention over time and variables
- **Validation:** Domain-knowledge plausibility

## Model Details
- **Architecture:** Spatiotemporal Attention Mechanism (STAM)
- **Properties:** Causal, scalable, learns important timesteps and variables
- **Black-box/interpretable:** Intrinsic architecture

## Explanation Method
- **Type:** Intrinsic
- **Family:** Spatial + temporal attention
- **Explanation target:** Important variables and important timesteps

## Evaluation
- **Forecasting metrics:** Prediction accuracy against baselines
- **Explanation metrics:** Domain-knowledge validation of learned attention weights

## Code Usability
- **Code available?** Unknown
- **Runs locally?** Not checked
- **Adaptability:** Useful for feature-time attention baseline.

## Limitations
- No synthetic ground-truth masks
- Attention explanations need independent validation
- Model-specific

## How It Helps This Project
- Provides architecture and terminology for simultaneous temporal and covariate attribution.
- Useful as baseline/citation when discussing feature-time explanations.

## Final Decision
**Use as model-oriented citation**, not core benchmark.

