# Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks

## Basic Info
- **Title:** Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks
- **Authors:** Leonardos Pantiskas, C. Verstoep, Henri Bal
- **Year:** 2021 conference publication from SSCI 2020 proceedings
- **Venue:** IEEE conference publication
- **Paper link:** https://ieeexplore.ieee.org/document/9308570
- **Alternative PDF:** https://research.vu.nl/ws/files/121454922/tacn_vu.pdf
- **Code link:** Unknown
- **Dataset link:** Unknown
- **Local PDF:** Missing

## Problem Type
- **Task:** Interpretable multivariate time-series forecasting
- **Forecasting?** Yes
- **Input:** Multivariate historical time series
- **Output:** Future outputs / forecasts
- **Univariate/multivariate:** Multivariate

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** Yes, through temporal attention
- **Ground-truth explanations?** Unknown
- **Black-box compatible?** No, architecture-specific
- **Chronos/foundation forecasting relevance:** Medium
- **Relevance score:** 3/5
- **Reason:** Forecasting + interpretability, but likely attention-internal and not ground-truth synthetic-mask benchmark.

## Dataset Details
- **Dataset names:** Need PDF extraction
- **Synthetic or real:** Unknown
- **Input/output:** Historical MTS to future forecasts
- **Generation mechanism:** Unknown

## Ground-Truth Explanation
- **Known mask?** Unknown
- **Mask shape:** Attention over timesteps
- **Mask generation:** If no synthetic ground truth, attention is only an intrinsic explanation signal

## Model Details
- **Architecture:** Temporal Attention Convolutional Neural Network
- **Components:** 1D CNN / dilated causal convolution / attention
- **Black-box/interpretable:** Intrinsically interpretable architecture
- **Required interface:** Model attention weights

## Explanation Method
- **Type:** Intrinsic
- **Family:** Temporal attention
- **Explanation target:** Influential timesteps in input for future outputs

## Evaluation
- **Forecasting metrics:** Forecasting accuracy metrics
- **Explanation metrics:** No automatic explanation-correctness metric identified from the local note/source preview. The method presents temporal attention over influential input timesteps, so explainability appears to be attention visualization/inspection rather than ground-truth mask scoring.
- **Quantitative or visual:** Forecasting evaluation is quantitative; explainability is visual/attention-based unless the full PDF reveals an additional metric.

## Code Usability
- **Code available?** Unknown
- **Runs locally?** Not checked
- **Adaptability:** Useful model baseline, less useful for black-box explanation.

## Limitations
- Attention-specific
- May lack ground-truth explanation validation
- Code details and full evaluation details still need extraction if this becomes a core reference

## How It Helps This Project
- Useful as a model-oriented comparison for temporal attention interpretability.
- Could inspire temporal attribution visualization and baseline architecture.

## Final Decision
**Use as model-oriented citation**, not core synthetic benchmark.
