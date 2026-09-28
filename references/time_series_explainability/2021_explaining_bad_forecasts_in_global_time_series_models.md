# Explaining Bad Forecasts in Global Time Series Models

## Basic Info
- **Title:** Explaining Bad Forecasts in Global Time Series Models
- **Authors:** Jože Rožanec, Elena Trajkova, Klemen Kenda, Blaž Fortuna, Dunja Mladenić
- **Year:** 2021
- **Venue:** Applied Sciences
- **Paper link:** https://doi.org/10.3390/app11199243
- **Local PDF:** `Refferences/2021_explaining_bad_forecasts_in_global_time_series_models.pdf`
- **Code link:** Unknown from local preview
- **Dataset link:** Two public real-world datasets are used, details require deeper extraction
- **Citation/BibTeX:** Applied Sciences 2021, 11, 9243
- **License:** CC BY 4.0 article

## Problem Type
- **Task:** Explainability for global time-series forecasting models, especially bad forecasts
- **Forecasting?** Yes
- **Input history length:** Depends on dataset/model
- **Forecast horizon:** Depends on dataset/model
- **Univariate or multivariate:** Global forecasting over multiple series with feature vectors
- **Direct or recursive forecasting:** Need deeper extraction
- **Point or probabilistic forecast:** Need deeper extraction

## Relevance To This Project
- **Does it predict future values?** Yes
- **Does it explain the forecast?** Yes
- **Does it provide ground-truth explanations?** No known synthetic ground-truth mask
- **Does it work with black-box models?** Yes, it uses XAI around forecasting models
- **Could it apply to Chronos/foundation forecasting models?** Conceptually yes for diagnosing bad forecasts, though implementation may need adaptation
- **Relevance score:** 3/5
- **Reason:** Forecasting-XAI and black-box diagnosis are relevant, but it does not appear to provide synthetic ground-truth attribution masks.

## Dataset Details
- **Dataset names:** Two public real-world datasets, not identified in first-page preview
- **Synthetic or real:** Real
- **Input shape:** Feature vectors/time-series history for global forecasting
- **Output shape:** Forecast values
- **Forecast target:** Future time-series values
- **Covariates/exogenous variables:** Uses feature relevance for forecasts
- **Train/validation/test split:** Need deeper extraction
- **Data generation mechanism:** None
- **Known historical drivers:** No ground-truth drivers

## Ground-Truth Explanation
- **Known explanation mask?** No
- **Mask shape:** Not applicable
- **How mask is generated:** Not applicable
- **Ambiguity:** Explanations are diagnostic rather than ground-truth validated

## Model Details
- **Model architecture:** Global time-series forecasting models; exact models require deeper extraction
- **Black-box or interpretable model:** Black-box/global models explained post-hoc
- **Forecasting model or classifier:** Forecasting model
- **Required training setup:** Train global forecasting model
- **Inputs expected by the model:** Multiple time-series histories/features
- **Outputs produced by the model:** Forecasts
- **Foundation model support:** Not direct, but the diagnostic workflow may transfer
- **Requires gradients/probabilities/samples/internal attention?** Uses XAI, anomaly detection, influence-like training sample analysis, and counterfactuals

## Explanation Method
- **Post-hoc or intrinsic:** Post-hoc
- **Local or global:** Local forecast diagnostics
- **Model-specific or model-agnostic:** Likely model-agnostic/modular
- **Explanation target:** Feature relevance, influential training samples, outlier status, counterfactual examples
- **Method family:** XAI dashboard, anomaly detection, counterfactual explanations
- **Computational cost:** Depends on XAI modules and counterfactual generation
- **Attribution scope:** Primarily Covariate Attribution plus diagnostic instance/counterfactual explanations. It is not a ground-truth Temporal or Time-Covariate attribution benchmark.

## Evaluation
- **Forecasting metrics:** Mean Absolute Scaled Error (MASE), including before/after comparisons when removing noisy or anomalous training examples.
- **Explanation metrics:** Number of outliers detected in the test set, number of explained target instances (ETI), qualitative dashboard usefulness, local feature relevance, influential/similar training examples, and counterfactual examples. It does not use a synthetic mask-overlap metric.
- **Quantitative or visual:** Dashboard plus quantitative validation of the bad-forecast diagnosis workflow.
- **Evaluation summary check:** The old summary was mostly right but should be read as a diagnostic-evaluation paper, not a ground-truth-attribution benchmark. It evaluates whether the workflow can identify bad forecasts, anomalous data, relevant features, and actionable counterfactual changes; it does not test whether a saliency mask matches a known causal mask.
- **Evaluation references cited:** Uses MASE from Hyndman (2006) for forecasting error; LIME from Ribeiro, Singh, and Guestrin (2016) for feature relevance; SHAP/Lundberg and Lee (2017), TimeSHAP/Bento et al., Anchors, DLIME, and LIMEtree as related XAI references; Schlegel et al., “Towards a rigorous evaluation of XAI methods on time series,” as an evaluation reference; Hooker et al. and Samek/Mueller are cited in the broader XAI evaluation discussion; COPOD and ABOD are cited for anomaly/outlier detection.

## Code Usability
- **Code available?** Unknown from local preview
- **Can run locally?** Not checked
- **Dependencies:** Unknown
- **GPU required?** Unknown
- **Includes notebooks/trained models/evaluation scripts?** Unknown
- **Adaptation difficulty:** Medium

## Limitations
- No synthetic ground-truth explanation mask
- Focuses on bad forecast diagnosis, not general faithful attribution benchmark
- Real-data explanations are harder to validate objectively

## How It Helps This Project
- Useful for thinking about user-facing explanations: feature relevance, influential examples, outlier warning, and counterfactuals.
- Less useful for the synthetic benchmark core, but useful for the eventual interface around bad Chronos forecasts.

## Final Decision
**Use as citation / adapt diagnostic ideas**, not as the main synthetic benchmark.
