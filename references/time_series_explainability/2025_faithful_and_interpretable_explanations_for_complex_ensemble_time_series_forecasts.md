# Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts Using Surrogate Models and Forecastability Analysis

## Basic Info
- **Title:** Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis
- **Authors:** Yikai Zhao, Jiekai Ma / Meredith Ma appears in Amazon Science listing; verify final author metadata
- **Year:** 2025
- **Venue:** KDD 2025 Workshop on AI for Supply Chain / arXiv
- **Paper link:** https://arxiv.org/abs/2510.08739
- **Amazon Science page:** https://www.amazon.science/publications/faithful-and-interpretable-explanations-for-complex-ensemble-time-series-forecasts-using-surrogate-models-and-forecastability-analysis
- **Code link:** Unknown
- **Dataset link:** M5 dataset plus feature injection experiments
- **Local PDF:** Missing

## Problem Type
- **Task:** Explain complex ensemble time-series forecasts
- **Forecasting?** Yes
- **Input:** Time-series features used by an AutoML/global forecasting ensemble
- **Output:** Forecasts, surrogate predictions, SHAP explanations, forecastability score

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** Yes
- **Ground-truth explanations?** Yes for feature injection experiments with known effects
- **Black-box compatible?** Yes, via surrogate model that mimics AutoGluon forecasts
- **Chronos/foundation forecasting relevance:** High
- **Relevance score:** 4/5
- **Reason:** Very relevant for surrogate-based explanations and known-effect validation, though centered on ensemble/AutoGluon rather than foundation models.

## Dataset Details
- **Dataset names:** M5 dataset; feature injection experiments
- **Synthetic or real:** Real M5 demand data plus controlled synthetic feature injection
- **Input shape:** Forecasting feature table / time-series features
- **Output shape:** Forecasts from original ensemble and surrogate model
- **Synthetic-generation mechanism:** This paper does not generate a full synthetic time-series dataset from scratch. Instead, it injects a synthetic feature with a predefined effect into a real forecasting dataset, retrains the surrogate on the modified data, and checks whether SHAP recovers the known injected effect.
- **Feature injection example:** The paper illustrates a synthetic price feature with known demand effects. For example, increasing price can define a negative demand effect, and the modified demand becomes `old demand + known price effect`.
- **Purpose of injection:** The injected feature creates a partial ground truth for explanation faithfulness. It is a sanity check for whether the surrogate + SHAP pipeline can recover one known effect in an otherwise real forecasting setting.

## Ground-Truth Explanation
- **Known mask?** Controlled feature injection gives known ground-truth effects
- **Mask shape:** Feature-level / engineered-feature attribution, not necessarily raw time-feature mask
- **Mask generation:** The known injected effect is the ground truth. The paper compares denormalized SHAP values for the injected feature against the known effect and reports high Pearson correlation.
- **Ambiguity:** Real M5 series heterogeneity and scale differences require normalization
- **Important caveat:** This is not a full ground-truth mask for all original features or lags. It validates whether the explanation method recovers the main effect of the injected feature, but it does not prove faithfulness for all interactions among real features.
- **Synthetic-generation references cited:** The paper cites M4, a unified XAI benchmark for faithfulness evaluation using known/synthetic truths; Angelov et al. (2023) on evaluating anomaly explanations with generated ground truth; Adebayo et al. (2018) for saliency-map sanity checks; Doshi-Velez and Kim (2017) for rigorous interpretability evaluation; and Molnar/Yeh et al. for fidelity and sensitivity concepts. It also uses SHAP/TreeSHAP as the attribution machinery.

## Model Details
- **Architecture:** AutoGluon time-series ensemble as black box, LightGBM surrogate
- **Black-box/interpretable:** Black-box ensemble explained by surrogate
- **Required interface:** Forecasts from original model and training data for surrogate

## Explanation Method
- **Type:** Post-hoc surrogate explanation
- **Family:** LightGBM surrogate + SHAP
- **Explanation target:** Forecast features / injected drivers
- **Reliability component:** Spectral predictability / forecastability analysis
- **Attribution scope:** Covariate Attribution. The validation focuses on engineered or injected feature effects, not raw timestamp-level temporal masks.

## Evaluation
- **Forecasting metrics:** Forecast accuracy and surrogate fidelity
- **Explanation metrics:** Agreement between SHAP values and known ground-truth injected effects
- **Additional metric:** Forecastability as confidence/reliability indicator
- **Evaluation summary check:** The summary agrees with the arXiv/Amazon Science abstract. The key evaluation design is feature injection: known artificial effects are added, SHAP values from the LightGBM surrogate are checked against those known effects, and surrogate fidelity is checked against the AutoGluon ensemble. Forecastability is evaluated by comparing spectral predictability with pure-noise benchmarks and correlating it with forecast accuracy and surrogate fidelity.
- **Evaluation references cited:** Local PDF is missing, so exact reference-list verification is pending. Based on the abstract/source listing, the important evaluation anchors are SHAP-style feature attribution, surrogate fidelity, controlled feature-injection validation, and spectral predictability/forecastability analysis.

## Code Usability
- **Code available?** Unknown
- **Runs locally?** Not checked
- **Adaptability:** High if project uses tabularized features or a surrogate around Chronos outputs.

## Limitations
- Surrogate explains the black-box approximation, not necessarily the original model internals
- Feature-level explanations may not map directly to raw time-point masks
- Need PDF/code for details

## How It Helps This Project
- Excellent source for surrogate explanation evaluation and feature-injection validation.
- Forecastability score could become a confidence measure for explanation reliability.

## Final Decision
**Adapt idea** for surrogate explanations and explanation reliability scoring.
