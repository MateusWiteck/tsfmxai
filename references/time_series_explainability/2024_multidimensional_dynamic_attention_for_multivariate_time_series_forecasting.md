# Multidimensional Dynamic Attention for Multivariate Time Series Forecasting

## Basic Info
- **Title:** Multidimensional Dynamic Attention for Multivariate Time Series Forecasting
- **Authors:** Sarah Almaghrabi, Mashud Rana, Margaret Hamilton, Mohammad Saiedur Rahaman
- **Year:** 2024
- **Venue:** Applied Soft Computing
- **DOI:** https://doi.org/10.1016/j.asoc.2024.112350
- **Paper link:** Local PDF: `Refferences/2024_multidimensional_dynamic_attention_for_multivariate_time_series_forecasting.pdf`
- **Code link:** The PDF says code will be publicly available on GitHub after acceptance; exact URL not visible in local preview
- **Dataset link:** Synthetic and real-world datasets; exact links need deeper extraction
- **License:** Open access according to ScienceDirect search snippet; verify publisher license if reused

## Problem Type
- **Task:** Multivariate time-series forecasting
- **Forecasting?** Yes
- **Input:** Historical multivariate time series with lagged variables
- **Output:** Multi-step / sequence-to-sequence future predictions
- **Univariate/multivariate:** Multivariate, heterogeneous MTS

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** Yes, through dynamic lagged-variable attention/importance
- **Ground-truth explanations?** Yes for synthetic datasets, via known important lagged variables; evaluation is mainly visual/qualitative rather than a formal mask-overlap score
- **Black-box compatible?** No, attention is architecture-specific
- **Chronos/foundation forecasting relevance:** Medium
- **Relevance score:** 4/5
- **Reason:** Directly about forecasting and identifying important lagged variables, but explanation is intrinsic to the MDA architecture.

## Dataset Details
- **Dataset names:** `Toy1`, `Toy2`, plus real-world solar and pollution datasets
- **Synthetic or real:** Both
- **Input shape:** Time x variables, with lagged-variable relationships
- **Output shape:** Sequence-to-sequence forecast horizon
- **Covariates/exogenous variables:** Multiple variables, heterogeneous seasonal/irregular patterns
- **Data generation mechanism:** Two nonlinear synthetic systems with known lagged-variable importance are used to test whether attention-based models identify the correct feature-time drivers.
- **Toy1 generator:** Six random input variables are sampled from fixed uniform ranges: `x1 in [2, 5]`, `x2 in [10, 30]`, `x3 in [5, 10]`, `x4 in [30, 70]`, `x5 in [100, 200]`, and `x6 in [65, 85]`. The target is generated from lagged values: `x1(t-1) * x2(t-2) + x3(t-5) + x4(t-1) + x4(t-4) + x4(t-5) + x4(t-7) + x4(t-8) + epsilon`, where `epsilon` is Gaussian white noise with mean 0 and standard deviation 0.1. The paper generates 20,000 samples.
- **Toy2 generator:** Uses four independent Mackey-Glass chaotic time series with different `(beta, gamma, tau)` parameter settings. The target is generated as `x1(t-2) * x2(t-6) + x3(t-4) + x4(t-1) * x4(t-2) + epsilon`.
- **Known important variables:** In `Toy1`, `x4` is repeatedly important at several lags; `x1`, `x2`, and `x3` also contribute; `x5` and `x6` are irrelevant. In `Toy2`, `x4` is important at `t-1` and `t-2`, with additional drivers `x1(t-2)`, `x2(t-6)`, and `x3(t-4)`.
- **Split:** The paper uses 60% train, 20% validation, and 20% test for the datasets.

## Ground-Truth Explanation
- **Known mask?** Yes for the synthetic datasets
- **Mask shape:** Lagged variable importance across variables, time, and forecast steps
- **Mask generation:** Directly from the target equations. Each term in the target equation defines an important variable-lag pair; variables absent from the equation are irrelevant.
- **Ambiguity:** Attention weights are not automatically faithful; synthetic validation helps but should be separated from forecast accuracy.
- **Best use for this project:** Excellent reference for feature-time masks in multivariate forecasting. It is especially useful for building cases where the future target depends on products/interactions between lagged covariates.
- **Synthetic-generation references cited:** `Toy1` is taken from Cao et al. (2021), “A multiattention-based supervised feature selection method for multivariate time series.” `Toy2` uses the Mackey-Glass chaotic time-series benchmark; the paper cites the Mackey-Glass-style setup through its synthetic-data reference list.

## Model Details
- **Architecture:** Multidimensional Dynamic Attention model
- **Components:** Dynamic representation learner and multiple attention calculations
- **Black-box/interpretable:** Intrinsically interpretable attention model
- **Required interface:** Model internals/attention weights

## Explanation Method
- **Type:** Intrinsic
- **Family:** Dynamic attention
- **Explanation target:** Important lagged variables for each prediction step
- **Cost:** Model-specific training/inference

## Evaluation
- **Forecasting metrics:** Sequence-to-sequence forecasting metrics
- **Explanation metrics:** No named automatic explanation metric is defined. The synthetic datasets identify the true important variable-lag pairs, and the paper visually compares lagged-variable-importance heatmaps/overview importance plots against those expected drivers. It also discusses whether irrelevant variables receive nonzero importance.
- **Quantitative or visual:** Forecasting metrics are quantitative; explanation evaluation is mostly visual/qualitative heatmap comparison on synthetic data plus domain-plausibility analysis on real data.
- **Evaluation summary check:** Important correction: the paper has known synthetic feature-time ground truth, but the local summary should not imply an extracted automatic explainability score. Its interpretability evidence is primarily whether the plotted importance patterns highlight the expected variables/lags such as `x4` in `Toy1`/`Toy2` and avoid irrelevant variables.

## Code Usability
- **Code available?** Promised, exact link unknown
- **Runs locally?** Not checked
- **Adaptability:** High for synthetic lagged-variable generator design; lower for black-box Chronos explanations.

## Limitations
- Architecture-specific
- Attention-based explanation must be validated
- Explanation evaluation is not a formal automatic mask metric; convert the known synthetic variable-lag equations into explicit metrics if adapting this paper for benchmarking

## How It Helps This Project
- Strong source for designing multivariate forecasting generators with known important lagged variables.
- Useful for comparing black-box post-hoc attribution against intrinsic attention models.

## Final Decision
**Adapt synthetic dataset ideas** and cite as forecasting interpretability reference.
