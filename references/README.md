# Scientific Reference Library

This file connects each paper in `references/` to the project roadmap in
`ROADMAP.md`. Papers are organized by their primary role:

- `generator/`: scalable synthetic-data or forecasting-prior generators;
- `benchmarking/`: work centered on controlled evaluation protocols;
- `time_series_explainability/`: forecasting and time-series XAI;
- `synthetic_symbolic_regression/`: symbolic mechanism generation and recovery.

PDFs are retained locally for research use and ignored by Git. Reviewed
Markdown summaries and `library.bib` form the versioned scientific record.

Project focus:

```text
history -> future values
post-hoc explanations for time-series foundation models
temporal, covariate, and time-covariate attribution
synthetic benchmarks with known ground-truth masks
forecast and explanation evaluation
```

## How To Read The Table

- **Plan contribution** says where the paper helps in the roadmap.
- **Benchmark value** says whether it helps build synthetic data with known
  attribution masks.
- **Explanation value** says whether it helps implement or evaluate explainers.
- **Use decision** says how directly the paper should influence the project.

## Main Comparison Table

| Year | Paper | Local files | Forecasting? | Explainability? | Ground-truth synthetic masks? | Main idea | Plan contribution | How it can collaborate with the plan | Limitations for this project | Priority | Use decision |
|---:|---|---|---|---|---|---|---|---|---|---:|---|
| 2016 | Why Should I Trust You? Explaining the Predictions of Any Classifier | `2016_why_should_i_trust_you_explaining_predictions_of_any_classifier.md` / `.pdf` | Generic regressor/classifier, not time-series forecasting-specific | Yes | No forecasting masks | LIME explains black-box predictions with a local interpretable surrogate trained on perturbed samples. | Phase 4 explanation methods; Phase 5 faithfulness checks | Provides a baseline pattern for model-agnostic local explanations. For this project, perturb historical windows or covariates, query Chronos-2, fit a local surrogate, and compare surrogate importance to synthetic masks. | Vanilla LIME perturbations can create unrealistic time series because they ignore autocorrelation, seasonality, and covariate dependence. Needs time-series-aware perturbation. | 2 | Adapt idea as a baseline, not as benchmark |
| 2017 | Local Interpretable Model-Agnostic Explanations for Music Content Analysis | `2017_local_interpretable_model_agnostic_explanations_for_music_content_analysis.md` / `.pdf` | No | Yes, for audio classification | No | Adapts LIME to temporal, frequency, and time-frequency audio segments. | Phase 4 perturbation utilities; visualization ideas | Useful for thinking about segment-level perturbations. The analogy is to perturb contiguous history windows rather than individual points. | Audio classification, not forecasting. No future values and no ground-truth forecasting attribution. | 1 | Use only as citation |
| 2019 | Attention Is Not Explanation | `2019_attention_is_not_explanation.md` / `.pdf` | No | Yes, critique of attention explanations | No | Shows attention weights can fail as faithful explanations. | Phase 5 explanation evaluation; literature motivation | Supports the argument that attention/saliency must be validated against ground-truth masks, not trusted visually. Helps justify metrics such as Precision@K, Recall@K, AUPRC, and faithfulness. | NLP classification, not forecasting. Does not give generators or metrics for time-series forecasting. | 2 | Use as cautionary citation |
| 2020 | Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting | `2020_temporal_fusion_transformers_for_interpretable_multi_horizon_time_series_forecasting.md` / `.pdf` | Yes | Yes, intrinsic interpretability | No | TFT combines multi-horizon forecasting with variable selection, gating, recurrent layers, and interpretable attention. | Phase 3 baselines; Phase 4 explanation modes; Phase 6 real datasets | Gives a forecasting-specific model baseline and useful explanation targets: static covariate importance, dynamic covariate importance, and temporal attention. Its decomposition maps well to temporal, covariate, and time-covariate attribution modes. | Architecture-specific. Attention/selection weights are not guaranteed faithful. No known ground-truth mask benchmark. | 3 | Use as forecasting interpretability baseline/citation |
| 2020 | Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks | `2020_interpretable_multivariate_time_series_forecasting_with_temporal_attention_convolutional_neural_networks.md` / `.pdf` | Yes | Yes, intrinsic temporal attention | Unknown | Uses a temporal attention convolutional neural network to forecast multivariate time series while indicating influential input timesteps. | Phase 3 model baselines; Phase 4 temporal attribution vocabulary | Helps define a model-oriented baseline for temporal attention and shows how attention can be attached to convolutional forecasting models. Useful for comparing post-hoc explanations against intrinsic temporal attention. | Architecture-specific and likely attention-based rather than model-agnostic. Ground-truth mask status unknown. | 3 | Use as model-oriented citation |
| 2020 | Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation | `2020_spatiotemporal_attention_for_multivariate_time_series_prediction_and_interpretation.md` / `.pdf` | Yes | Yes, intrinsic spatiotemporal attention | No clear synthetic masks | STAM learns both important timesteps and important variables for multivariate time-series prediction. | Phase 3 baselines; Phase 4 feature-time attribution; Phase 6 real datasets | Useful for defining simultaneous temporal and covariate attribution. It supports the idea that explanations should identify both when and which variable influenced the prediction. | Model-specific attention. Attention validated mainly through domain knowledge, not known synthetic masks. | 3 | Use as feature-time attribution citation |
| 2021 | Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions | `2021_benchmarking_attention_based_interpretability_of_deep_learning_in_multivariate_time_series_predictions.md` / `.pdf` | Yes | Yes | Yes, through transparent synthetic generating processes | Builds synthetic multivariate forecasting/prediction datasets with known interactions and evaluates prediction performance, interpretability correctness, and sensitivity. | Phase 1 synthetic benchmark generation; Phase 5 explanation evaluation | This is the strongest benchmark reference. It supports designing synthetic generators where the data-generating process defines the true explanatory drivers. It directly informs temporal/covariate/time-covariate masks and the need to evaluate explanation correctness separately from forecast accuracy. | Focuses on attention-based models, not fully black-box foundation models. Need to adapt from attention correctness to post-hoc attribution for Chronos-2. | 5 | Core reference; inspect deeply and adapt |
| 2021 | Explaining Bad Forecasts in Global Time Series Models | `2021_explaining_bad_forecasts_in_global_time_series_models.md` / `.pdf` | Yes | Yes | No | Uses XAI, anomaly detection, influential training examples, and counterfactual examples to explain bad forecasts in global forecasting models. | Phase 5 faithfulness/stability; Phase 6 real datasets; future user-facing diagnostics | Useful for the eventual user perspective: when a forecast is bad, show feature relevance, similar/influential training samples, anomaly status, and counterfactual changes. Complements mask-based synthetic evaluation with real-world debugging tools. | No synthetic ground-truth attribution masks. More diagnostic/dashboard oriented than benchmark oriented. | 3 | Use as diagnostic design reference |
| 2021 | Transformer Interpretability Beyond Attention Visualization | `2021_transformer_interpretability_beyond_attention_visualization.md` / `.pdf` | No | Yes, Transformer-specific | No | Propagates relevance through Transformer layers beyond raw attention visualization. | Phase 4 explanation methods; possible Transformer internals | Useful if the project later accesses internals of Transformer-based forecasting models. Provides a stronger alternative to raw attention maps. | Vision/text focus. Requires model internals, so it may not apply to Chronos-2 as a black box. No forecasting masks. | 2 | Possible method citation |
| 2022 | Explainable AI for Time Series Classification: A Review, Taxonomy and Research Directions | `2022_explainable_ai_for_time_series_classification_review_taxonomy_research_directions.md` / `.pdf` | No | Yes | No | Reviews XAI methods for time-series classification and organizes explanations into time-point, subsequence, and instance-based categories. | Literature review; explanation mode definitions | Helps define terminology and explains why classification-XAI is not enough for this project. Useful contrast when motivating the need for forecasting-specific benchmarks. | Classification-only. Does not solve `history -> future` explanation. | 2 | Use as taxonomy citation |
| 2023 | LoMEF: A Framework to Produce Local Explanations for Global Model Time Series Forecasts | `2023_lomef_framework_to_produce_local_explanations_for_global_model_time_series_forecasts.md` / `.pdf` | Yes | Yes, model-agnostic local surrogate | No | Explains forecasts from global forecasting models by training simpler local univariate surrogate models such as ETS in a neighborhood around the series being explained. | Phase 4 explanation methods; Phase 5 fidelity/stability evaluation; Phase 6 real datasets | Very relevant to explaining Chronos-2 as a black-box/global forecaster. Provides a concrete model-agnostic surrogate pattern and evaluation dimensions: accuracy, fidelity, stability, and comprehensibility. | Does not provide ground-truth attribution masks. Explanation is through surrogate forecasting components rather than direct temporal/covariate heatmaps. | 4 | Adapt as black-box local surrogate method |
| 2024 | Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features | `2024_explaining_speech_classification_models_via_word_level_audio_segments_and_paralinguistic_features.md` / `.pdf` | No | Yes | No | Explains speech classification using word-level audio segments and paralinguistic features. | Minor visualization/segmentation inspiration | Only loosely useful for human-facing segment explanations. Could inspire showing explanations over meaningful temporal chunks. | Speech classification, not forecasting. No future values, no mask benchmark. | 1 | Reject for core plan |
| 2024 | Evaluating Anomaly Explanations Using Ground Truth | `2024_evaluating_anomaly_explanations_using_ground_truth.md` / `.pdf` | No | Yes | Yes, but tabular anomaly/circuit masks rather than forecasting masks | Builds a digital-circuit anomaly benchmark with local ground-truth feature explanations and evaluates local explainers with correctness and robustness metrics. | Phase 5 explanation evaluation; robustness under irrelevant/noise features | Strong reference for automatic explanation evaluation: compare ranked local explanations against known local ground-truth feature sets using MRR, MAP, and R-Precision, then test robustness with ELA under added noise features. | Not time series and not forecasting. Ground truth is binary feature-level, not temporal or feature-time. | 3 | Use as evaluation-metrics reference |
| 2024 | Multidimensional Dynamic Attention for Multivariate Time Series Forecasting | `2024_multidimensional_dynamic_attention_for_multivariate_time_series_forecasting.md` / `.pdf` | Yes | Yes, dynamic attention over lagged variables | Yes, Toy1 and Toy2 have known lagged-variable drivers | Proposes MDA, a dynamic attention model that computes lagged-variable importance for multistep heterogeneous multivariate forecasting. | Phase 1 synthetic lagged-variable generators; Phase 3 model baselines; Phase 4 feature-time explanations | Useful for designing synthetic datasets where important lagged variables are known and for comparing black-box post-hoc explanations with intrinsic attention models. Toy1 uses explicit lagged nonlinear terms; Toy2 uses Mackey-Glass chaotic series and known lagged interactions. | Architecture-specific. Attention weights still need validation. Synthetic ground truth covers selected lagged variables, not every real-data feature. | 4 | Adapt synthetic lagged-variable ideas |
| 2024 | XForecast: Evaluating Natural Language Explanations for Time Series Forecasting | `2024_xforecast_evaluating_natural_language_explanations_for_time_series_forecasting.md` / `.pdf` | Yes | Yes, natural-language explanations | No attribution masks | Introduces simulatability-based metrics for evaluating natural-language explanations of time-series forecasts. | Later user-facing explanation layer; Phase 5 explanation evaluation extensions | Useful if the project evolves from attribution maps to natural-language explanations. Simulatability can complement mask metrics by testing whether an explanation helps a surrogate/human recover the forecast. | Not an attribution-mask benchmark. LLM rationales may be unfaithful without separate validation. | 3 | Use later for NLP explanation evaluation |
| 2025 | Chronos-2: From Univariate to Universal Forecasting | `2025_chronos_2_from_univariate_to_universal_forecasting.md` / `.pdf` | Yes | No | No | Universal/foundation time-series forecasting model. | Phase 3 ForecastProvider and Chronos-2 provider | This is the main target model family. The project should wrap Chronos-2 behind `ForecastProvider`, generate forecasts on synthetic benchmarks, then run post-hoc explanation methods on the input history/covariates. | Does not provide XAI methods or ground-truth masks. Needs external benchmark and explainer stack. | 5 for model context | Use directly as target forecasting model |
| 2025 | An End-to-End Explainability Framework for Spatio-Temporal Predictive Modeling | `2025_an_end_to_end_explainability_framework_for_spatio_temporal_predictive_modeling.md` / `.pdf` | Yes, in spatio-temporal settings | Yes, model-agnostic masking | No clear synthetic masks | Proposes a model-agnostic masking meta-optimization method that explains spatio-temporal predictions across features, timesteps, and node locations. | Phase 4 masking explainers; Phase 6 spatio-temporal extension | Useful if the project expands from feature-time masks to feature-time-node masks. Provides a multi-axis explanation design and model-agnostic masking strategy for sensor-network forecasting. | Broader graph/sensor setting than current Chronos focus. Not a synthetic ground-truth mask benchmark. | 3 | Adapt masking and multi-axis explanation ideas |
| 2025 | Explainable Artificial Intelligence for Economic Time Series: Review and Taxonomy | `2025_explainable_artificial_intelligence_for_economic_time_series_review.md` / `.pdf` | Review, often forecasting-related | Yes | No | Surveys XAI for economic time series and discusses propagation, perturbation/game-theoretic, and global function-based tools. | Literature review; Phase 4 method taxonomy; Phase 6 real/economic datasets | Useful for framing time-series-specific pitfalls: autocorrelation, non-stationarity, seasonality, mixed frequencies, and regime shifts. Supports why standard tabular XAI must be adapted. | Review only. No code, generator, or direct benchmark. | 3 | Use as taxonomy/motivation citation |
| 2025 | Explainable Time-Series Forecasting with Sampling-Free SHAP for Transformers | `2025_explainable_time_series_forecasting_with_sampling_free_shap_for_transformers.md` / `.pdf` | Yes | Yes | Yes, synthetic load data has ground-truth SHAP values | Proposes SHAP-style explanations for Transformer forecasting without expensive sampling. | Phase 4 explainers; Phase 5 explanation evaluation; possible synthetic benchmark inspiration | Highly relevant for forecasting-XAI. Its synthetic load generator creates `168 history hours -> 168 future hours` with calendar, holiday, temperature, multiplier, and noise covariates; ground truth explanations are SHAP values computed on the generator. | Transformer-specific. Ground truth is SHAP-value based and “true to data,” not a binary causal mask. May not directly work for Chronos-2 if internals are inaccessible. | 4 | Inspect deeply and adapt |
| 2025 | Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis | `2025_faithful_and_interpretable_explanations_for_complex_ensemble_time_series_forecasts.md` / `.pdf` | Yes | Yes, surrogate + SHAP | Yes, through feature injection experiments | Trains a LightGBM surrogate to mimic AutoGluon ensemble forecasts, explains the surrogate with SHAP, and uses forecastability analysis as a reliability signal. | Phase 4 surrogate explainers; Phase 5 explanation faithfulness/reliability | Strong reference for surrogate explanations around black-box forecasters. Feature injection experiments are useful for validating that explanations recover known effects. Forecastability metrics can become confidence scores for explanations. | Feature-level surrogate explanations may not map directly to raw history masks. Missing local code. | 4 | Adapt surrogate and reliability evaluation ideas |
| 2025 | Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention | `2025_learning_temporal_saliency_for_time_series_forecasting_with_cross_scale_attention.md` / `.pdf` | Yes | Yes | Yes, synthetic saliency ground truth | CrossScaleNet learns temporal saliency through cross-scale attention and validates on synthetic datasets with known saliency. | Phase 1 temporal generator; Phase 4 temporal attribution; Phase 5 temporal saliency metrics | Directly supports temporal attribution. It justifies generating known temporal saliency masks and evaluating whether models identify important history regions. Useful for Phase 1.2 and temporal explanation metrics. | Intrinsic architecture, not black-box post-hoc explanation. Need adapt saliency benchmark to Chronos-2. | 4 | Adapt temporal-saliency benchmark ideas |
| 2025 | TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data | `2025_timepfn_effective_multivariate_time_series_forecasting_with_synthetic_data.md` / `.pdf` | Yes | No | No XAI masks | Trains/evaluates multivariate forecasters using synthetic data and few-shot learning ideas. | Phase 1 synthetic data realism; Phase 3 baselines | Useful for synthetic forecasting generation philosophy and model baseline context. Can inspire how to make generated series diverse enough for forecasting models. | Forecasting accuracy only. Does not address explanations or attribution masks. | 2 | Use only as synthetic forecasting citation |
| 2026 | Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems | `2026_interpretable_deep_convolutional_model_for_nonlinear_multivariate_time_series_in_complex_systems.md` / `.pdf` | Yes, one step ahead | Yes, intrinsic local coefficients | Yes, exact source-lag-sign coefficients and controlled regimes/orders | DCIts factorizes a sample-specific transition tensor into a sparse Focuser and signed Modeler, then uses the same tensor to calculate the forecast. | Phase 1 signed generator labels; Phase 3 intrinsic baseline; Phase 4 mechanism explanation; Phase 5 coefficient recovery | Direct continuation of Baric et al. (2021). It turns known generator equations into exact support, sign, magnitude, order, and contribution targets and evaluates whether a forecasting model recovers them. | Model-specific and one-step. Higher-order branches use elementwise powers; a shared multi-horizon black-box benchmark needs adaptation and aggregate tensor metrics. | 5 | Core reference; adapt generators, labels, and coefficient-recovery evaluation |
| 2026 | MCIR: A Feature Dependence-Aware Explainability Method with Reliability Guarantees | `2026_mcir_feature_dependence_aware_explainability_method.md` / `.pdf` | Not specifically | Yes | Synthetic feature-dependence tests, not time-series masks | Global feature importance method designed for dependent/collinear features. | Phase 1 difficulty levels; Phase 4 covariate attribution; Phase 5 stability/faithfulness | Important for correlated covariates/lags. It can guide difficulty levels with redundant variables and help evaluate whether explanations incorrectly split importance across correlated features. | Global feature-level method, not forecasting-specific. No temporal mask by default. | 3 | Adapt dependence-aware ideas |
| 2026 | Recursive Language Models | `2026_recursive_language_models.md` / `.pdf` | No | No | No | Recursive inference paradigm for long-context LLMs. | None for core plan | Minimal relevance. Could maybe inspire long-context document processing, but not forecasting-XAI. | Off-topic. | 1 | Reject |
| 2026 | Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs | `2026_time_series_forecasting_as_reasoning_slow_thinking_approach_with_reinforced_llms.md` / `.pdf` | Yes | Weak/implicit reasoning | No | Uses reinforced LLM reasoning for forecasting. | Phase 3 future model context; possible narrative explanations | Useful as a future reference if the project explores language-rationale forecasting. Less useful for attribution masks. | Reasoning traces are not necessarily faithful explanations. No synthetic mask evaluation. | 2 | Use only as peripheral citation |

## Contribution By Plan Phase

| Plan phase | Most useful papers | What to take from them |
|---|---|---|
| **Phase 1: Synthetic benchmark generation** | `2021_benchmarking_attention_based_interpretability...`, `2026_interpretable_deep_convolutional_model...`, `2025_learning_temporal_saliency...`, `2025_explainable_time_series_forecasting_with_sampling_free_shap...`, `2024_multidimensional_dynamic_attention...`, `2026_mcir...`, `2025_timepfn...` | Transparent data-generating processes, exact signed source-lag coefficients, known temporal saliency, synthetic mask evaluation, correlated/redundant covariates, and diverse synthetic forecasting signals. |
| **Phase 1.1: Benchmark data contract** | `2021_benchmarking_attention_based_interpretability...`, `2026_interpretable_deep_convolutional_model...`, `2025_learning_temporal_saliency...` | Store history, future, metadata, masks, signed generator coefficients, and realized contribution tensors explicitly. Separate forecast target from explanation target. |
| **Phase 1.2: Temporal generator** | `2025_learning_temporal_saliency...`, `2021_benchmarking_attention_based_interpretability...` | Generate known important historical regions and evaluate temporal saliency. |
| **Phase 1.3: Covariate generator** | `2020_temporal_fusion_transformers...`, `2026_interpretable_deep_convolutional_model...`, `2024_multidimensional_dynamic_attention...`, `2026_mcir...`, `2021_benchmarking_attention_based_interpretability...` | Define informative, weak, irrelevant, lagged, signed, and correlated variables. Test whether explanation methods separate true drivers from redundant covariates and recover direction. |
| **Phase 1.4: Time-covariate generator** | `2021_benchmarking_attention_based_interpretability...`, `2026_interpretable_deep_convolutional_model...`, `2024_multidimensional_dynamic_attention...`, `2025_explainable_time_series_forecasting_with_sampling_free_shap...`, `2020_spatiotemporal_attention...` | Use multivariate interactions where a specific variable at a specific lag/window affects the forecast. Include dynamic signed coefficients, regime-dependent structure, and feature-time patterns. |
| **Phase 1.5: Attribution mask utilities** | `2021_benchmarking_attention_based_interpretability...`, `2026_interpretable_deep_convolutional_model...`, `2025_learning_temporal_saliency...`, `2024_multidimensional_dynamic_attention...`, `2025_an_end_to_end_explainability_framework...`, `2026_mcir...` | Support temporal masks, feature masks, feature-time masks, signed coefficient tensors, realized contribution tensors, weighted masks, and aggregation from lag-resolved tensors to temporal/covariate summaries. |
| **Phase 1.6: Visualization** | `2020_temporal_fusion_transformers...`, `2021_explaining_bad_forecasts...`, `2025_learning_temporal_saliency...`, `2024_xforecast...` | Build plots that show history, future, saliency/attention/mask overlays, user-facing bad-forecast diagnostics, and later natural-language explanations. |
| **Phase 2: Data infrastructure** | `2025_chronos_2...`, `2021_benchmarking_attention_based_interpretability...` | Define model-ready windowing and consistent sample schema for foundation forecasting. |
| **Phase 3: Forecasting** | `2025_chronos_2...`, `2026_interpretable_deep_convolutional_model...`, `2020_temporal_fusion_transformers...`, `2024_multidimensional_dynamic_attention...`, `2020_interpretable_multivariate_time_series_forecasting...`, `2025_timepfn...`, `2026_time_series_forecasting_as_reasoning...` | Implement Chronos-2 provider, add DCIts and other interpretable forecasting baselines, and keep synthetic forecasting generation realistic. |
| **Phase 4: Explanation methods** | `2016_why_should_i_trust_you...`, `2023_lomef...`, `2025_faithful_and_interpretable_explanations...`, `2025_explainable_time_series_forecasting_with_sampling_free_shap...`, `2026_interpretable_deep_convolutional_model...`, `2025_learning_temporal_saliency...`, `2025_an_end_to_end_explainability_framework...`, `2026_mcir...`, `2021_transformer_interpretability...` | Implement perturbation/LIME-like, local surrogate, SHAP-like, intrinsic signed-transition, temporal saliency, masking, dependence-aware, and possible Transformer-internal methods. |
| **Phase 5: Explanation evaluation** | `2021_benchmarking_attention_based_interpretability...`, `2026_interpretable_deep_convolutional_model...`, `2024_evaluating_anomaly_explanations_using_ground_truth...`, `2019_attention_is_not_explanation`, `2025_faithful_and_interpretable_explanations...`, `2024_xforecast...`, `2025_learning_temporal_saliency...`, `2026_mcir...` | Evaluate explanation correctness separately from forecasting accuracy. Add support, sign, coefficient, contribution, ranking, fidelity, stability, robustness, comprehensibility, forecastability, and simulatability metrics where relevant. |
| **Phase 6: Real datasets** | `2021_explaining_bad_forecasts...`, `2023_lomef...`, `2025_faithful_and_interpretable_explanations...`, `2025_an_end_to_end_explainability_framework...`, `2020_temporal_fusion_transformers...`, `2025_explainable_artificial_intelligence_for_economic_time_series_review...` | Transfer methods to real forecasting datasets where ground truth is unavailable, using diagnostics, local surrogates, stability, forecastability, counterfactuals, masking, and plausibility rather than mask scores. |

## Recommended Reading Order

1. **2021 Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions**  
   Start here because it is closest to synthetic forecasting-XAI with known mechanisms.

2. **2026 Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems**  
   Read this next because it continues the 2021 benchmark and directly recovers signed source-lag coefficients from the known generators.

3. **2025 Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention**  
   Use this for temporal attribution and saliency-specific benchmark ideas.

4. **2025 Explainable Time-Series Forecasting with Sampling-Free SHAP for Transformers**  
   Use this for SHAP-style forecasting explanations and possible synthetic explanation evaluation.

5. **2025 Chronos-2**  
   Use this to design the `ForecastProvider` and black-box model interface.

6. **2024 Multidimensional Dynamic Attention for Multivariate Time Series Forecasting**  
   Use this for lagged-variable importance and multistep multivariate attention ideas.

7. **2023 LoMEF**  
   Use this for black-box local surrogate explanations of global/foundation forecasters.

8. **2025 Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts**  
   Use this for surrogate SHAP, feature-injection validation, and forecastability-based reliability.

9. **2026 MCIR**  
   Use this before adding correlated covariate difficulty levels.

10. **2020 TFT**  
   Use this as an interpretable forecasting baseline and vocabulary for covariate importance.

11. **2021 Explaining Bad Forecasts**  
   Use this later for user-facing diagnostics and real-data explanation workflows.

12. **2024 XForecast**  
   Use this later if the project adds natural-language explanations or user-facing explanation evaluation.

13. **2025 End-to-End Explainability Framework for Spatio-Temporal Predictive Modeling**  
   Use this only if the project expands to node/location-aware or sensor-network forecasting.

## Category Index

This section groups the references by how they can help the project. Some papers
belong to more than one category. Local files are linked when they exist; papers
without local PDFs are marked as citation-only for now.

### Papers That Evaluate Explainable AI

These papers are most useful for deciding whether an explanation is faithful,
stable, useful, or correct.

- [2021 Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions](2021_benchmarking_attention_based_interpretability_of_deep_learning_in_multivariate_time_series_predictions.md)  
  Evaluates MSE and MSE stability, then checks attention/causality patterns against known synthetic mechanisms using the Performance-Explainability Framework.
  **Explainability evaluation type:** **Ground truth present, but mostly qualitative/manual.** For each synthetic dataset, the equations tell which source series and lags should matter. The model produces attention/causality scores over lags or source series, and the paper plots heatmaps of those scores. Correctness is judged by whether the high-score regions visually match the expected equation terms, not by a formal automatic score such as F1, AUPRC, IoU, or rank correlation. Repeated runs provide mean/std attention and TCDF retrieval percentages, which help judge confidence, but they are not a universal explanation-correctness metric.

- [2026 Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems](2026_interpretable_deep_convolutional_model_for_nonlinear_multivariate_time_series_in_complex_systems.md)  
  Continues the Baric et al. benchmark with DCIts, an intrinsic model whose sample-specific transition tensor contains signed coefficients for every target, source, and lag.
  **Explainability evaluation type:** **Exact structural and coefficient ground truth.** The generator supplies the active source-lag support, sign, coefficient magnitude, bias, regime, and polynomial order. The paper compares learned `alpha` tensors with these exact values using side-by-side heatmaps, numerical coefficient recovery, and mean plus or minus standard deviation over repeated runs. Entries are retained in interpretation plots when `abs(mean) > 1.95 * standard_deviation`. This enables automatic support F1, signed accuracy, coefficient MAE/RMSE, contribution error, and stability metrics in this project, although the paper itself emphasizes coefficient examples and plots rather than one aggregate explanation score.

- [2025 Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention](2025_learning_temporal_saliency_for_time_series_forecasting_with_cross_scale_attention.md)  
  Evaluates MSE/MAE, temporal saliency against known synthetic ground truth, and real-data saliency with comprehensiveness/sufficiency and ShapTime comparisons.
  **Automatic explainability metrics:** **Temporal Attribution + Time-Covariate Attribution.** Each synthetic dataset defines a binary ground-truth map: important lags such as `1-15` or `71-77`, and important feature indices such as `{0, 1}`. The model outputs a saliency/attention map over the input history; threshold or rank the predicted saliency map and compare it with known important lag-feature positions.
  **Sufficiency/comprehensiveness:** These metrics are not proposed by this paper; they are adopted from broader rationale/XAI faithfulness evaluation, especially ERASER / DeYoung et al. (arXiv:1911.03429). For real data, where no ground-truth mask exists, rank timesteps by saliency and choose the top `r%`. Sufficiency keeps only those timesteps and compares the new forecast with the original: `Sufficiency(r) = distance(f(x), f(mask_keep(x, top_r_saliency)))`; lower is better. Comprehensiveness removes those timesteps and compares the new forecast with the original: `Comprehensiveness(r) = distance(f(x), f(mask_remove(x, top_r_saliency)))`; higher is better. The CrossScaleNet PDF reports ratios such as `10%`, `20%`, and `50%`, but does not clearly specify the exact masking baseline or forecast-distance function.

- [2025 Explainable Time-Series Forecasting with Sampling-Free SHAP for Transformers](2025_explainable_time_series_forecasting_with_sampling_free_shap_for_transformers.md)  
  Evaluates RMSE/MAE/MSE/MAPE, runtime, and SHAP-value agreement against synthetic ground-truth SHAP values from the data-generating process.
  **Automatic explainability metrics:** **Covariate Attribution + coarse Temporal Attribution; partial Time-Covariate Attribution through feature groups.** The synthetic generator is treated as the reference function. For each test example, SHAP is run on that generator to compute ground-truth SHAP values for feature groups such as past-load days, hour, day-of-week, month, holiday, temperature, multiplier, and noise features. SHAPformer also outputs SHAP values for the same groups. The practical comparison is group-by-group: do the predicted SHAP values have the same sign, magnitude, and ranking as the generator SHAP values? Global feature importance is calculated by summing absolute SHAP values over many samples and comparing the resulting feature-importance bars with the ground truth. Dependence plots are checked by comparing whether learned SHAP-vs-feature curves reproduce known generator effects, such as holidays shrinking load or temperature multiplying load.

- [2025 Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis](2025_faithful_and_interpretable_explanations_for_complex_ensemble_time_series_forecasts.md)  
  Citation-only local note. Evaluates surrogate SHAP explanations using feature injection and forecastability/reliability analysis.
  **Automatic explainability metrics:** **Covariate Attribution.** The paper injects a synthetic feature with a known effect into a real forecasting problem. After training the surrogate, SHAP gives an attribution value for that injected feature. The practical check is: when the injected feature is designed to increase or decrease demand by a known amount, does the SHAP value move in the same direction and roughly the same relative strength? They also compute surrogate fidelity by comparing the surrogate forecast vector with the black-box ensemble forecast vector. High fidelity means the SHAP explanation is at least explaining a model that behaves like the original forecaster.

- [2024 XForecast: Evaluating Natural Language Explanations for Time Series Forecasting](2024_xforecast_evaluating_natural_language_explanations_for_time_series_forecasting.md)  
  Citation-only local note. Evaluates natural-language explanations using simulatability metrics.
  **Automatic explainability metrics:** **Not directly Temporal/Covariate/Time-Covariate Attribution.** The explanation is text, so the calculation is not mask overlap. In direct simulatability, an evaluator model receives the time-series context and the natural-language explanation, then tries to predict or select the original model's forecast. The score is the improvement in reproducing the forecast when the explanation is included versus absent. In synthetic simulatability, the task is controlled so the expected forecast logic is known; the text explanation is useful if it helps the evaluator recover that expected behavior.

- [2021 Explaining Bad Forecasts in Global Time Series Models](2021_explaining_bad_forecasts_in_global_time_series_models.md)  
  Evaluates bad-forecast diagnosis with MASE, detected outliers, explained target instances, feature relevance, counterfactuals, influential examples, and anomaly detection.
  **Automatic explainability metrics:** **Primarily Covariate Attribution plus diagnostic/counterfactual explanations.** The system first detects bad or anomalous forecasts. Feature relevance methods assign scores to input variables for that forecast. Counterfactual validity is then checked by changing selected relevant features and rerunning the diagnostic: if the modified input would no longer produce an anomalous/bad forecast, the counterfactual is counted as useful. Explained target instances are counted as the number of bad forecasts for which the workflow produces such diagnostics. There is no ground-truth mask; the automatic target is whether the explanation workflow changes the anomaly/bad-forecast status in a coherent way.

- [2019 Attention Is Not Explanation](2019_attention_is_not_explanation.md)  
  Not forecasting-specific, but evaluates attention faithfulness with Kendall tau against gradients/erasure importance and adversarial attention tests.
  **Automatic explainability metrics:** **Sequence/Token Attribution; analogous to Temporal Attribution for time series.** For each input token, the model has an attention weight. The paper computes independent importance scores for the same tokens using gradients or leave-one-out/erasure: remove or mask a token and measure how much the prediction changes. Then it ranks tokens by attention and by the proxy importance score, and computes Kendall tau between the two rankings. Low correlation means attention is not matching the proxy importance. The adversarial test searches for a different attention vector that is far from the original attention vector but gives almost the same prediction; if this exists, attention is not a faithful explanation.

- [2026 MCIR: A Feature Dependence-Aware Explainability Method with Reliability Guarantees](2026_mcir_feature_dependence_aware_explainability_method.md)  
  Evaluates rank stability, group-level agreement, Jaccard@K, allocation stability, Family Share, deletion/insertion faithfulness, and runtime under feature dependence.
  **Automatic explainability metrics:** **Covariate Attribution.** Each method outputs a ranked list or score vector over features. Rank stability compares two rankings, for example full data versus resampled data, using Kendall tau or Spearman correlation. Jaccard@K compares the set of top-K features from two runs. When the data contain duplicated or correlated features, exact feature overlap can be unfair, so Group-Jaccard@K checks whether the top features belong to the same correlated feature family. Family Share measures how much attribution mass stays inside the true correlated group. Deletion faithfulness removes top-ranked features from the input and checks whether model performance drops faster than when removing low-ranked features.

- [2025 An End-to-End Explainability Framework for Spatio-Temporal Predictive Modeling](2025_an_end_to_end_explainability_framework_for_spatio_temporal_predictive_modeling.md)  
  Evaluates model-agnostic masking with Fidelity+/Fidelity-, normalized fidelity, sparsity, and time-complexity across feature-time-location explanations.
  **Automatic explainability metrics:** **Temporal Attribution + Covariate Attribution + Time-Covariate Attribution, with an additional spatial/node axis.** The explainer learns a mask over features, timesteps, and locations. Fidelity+ applies the complement of the mask: remove the parts marked important and measure how much the model prediction changes. A good explanation should cause a large change when its important parts are removed. Fidelity- keeps only the important parts and removes the rest; a good explanation should preserve much of the original prediction. Normalized fidelity divides these effects by a baseline effect from a zero/empty input. Sparsity is the percentage of the input removed or kept, measuring whether the explanation is compact instead of selecting everything.

### Papers That Generate Synthetic Datasets For Time-Series Forecasting

These papers are most useful for designing data where the expected output is a
future value or future trajectory, not a class label.

- [2021 Benchmarking Attention-Based Interpretability of Deep Learning in Multivariate Time Series Predictions](2021_benchmarking_attention_based_interpretability_of_deep_learning_in_multivariate_time_series_predictions.md)  
  Core reference. Generates ten transparent multivariate forecasting/prediction datasets: constant, autoregressive, nonlinear autoregressive, cross-series, driver-series, VAR, switching-rule, Ising, and logistic-map-inspired systems. Useful for known temporal, feature, and feature-time masks.
  **Ground-truth explainability:** Yes. The known equations define true relevant lags, variables, and cross-series dependencies.

- [2026 Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems](2026_interpretable_deep_convolutional_model_for_nonlinear_multivariate_time_series_in_complex_systems.md)  
  Reuses Datasets 1-8 from Baric et al. (2021), adjusts the switching dataset, and adds transparent VAR(2) and cubic processes. The generator equations define exact target-source-lag coefficients and order-specific terms.
  **Ground-truth explainability:** Yes. Ground truth supports binary support, sign, numerical coefficient, regime, polynomial order, and per-example realized-contribution labels. Its main methodological contribution is the DCIts intrinsic forecasting architecture and direct coefficient-recovery analysis.

- [2025 Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention](2025_learning_temporal_saliency_for_time_series_forecasting_with_cross_scale_attention.md)  
  Generates `SYN1`-`SYN8`, each with predefined important lags, important feature sets, and target noise. Very useful for testing whether an explainer detects recent versus distant important history windows.
  **Ground-truth explainability:** Yes. Each dataset has predefined important lags and important feature indices, which can be stored as temporal, feature, or feature-time masks.

- [2024 Multidimensional Dynamic Attention for Multivariate Time Series Forecasting](2024_multidimensional_dynamic_attention_for_multivariate_time_series_forecasting.md)  
  Uses `Toy1` and `Toy2` with explicit lagged-variable target equations. `Toy1` has random covariates and known irrelevant variables; `Toy2` uses Mackey-Glass chaotic series. Strong source for feature-time mask design.
  **Ground-truth explainability:** Yes. The target equations specify the exact important lagged variables; irrelevant variables are also known.

- [2025 Explainable Time-Series Forecasting with Sampling-Free SHAP for Transformers](2025_explainable_time_series_forecasting_with_sampling_free_shap_for_transformers.md)  
  Generates a synthetic hourly load dataset with calendar, holiday, temperature, multiplier, and noise covariates. Ground truth explanations are SHAP values computed on the known data-generation process, so it is best for SHAP-value evaluation rather than binary mask scoring.
  **Ground-truth explainability:** Yes, but as ground-truth SHAP values rather than binary masks. Best for evaluating attribution values and feature-dependence plots.

- [2025 TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data](2025_timepfn_effective_multivariate_time_series_forecasting_with_synthetic_data.md)  
  Generates large-scale synthetic multivariate forecasting pretraining data using Gaussian-process KernelSynth plus Linear Model of Coregionalization. Useful for realistic synthetic forecasting signals and cross-channel dependence, but it does not generate XAI masks.
  **Ground-truth explainability:** No. It generates synthetic forecasting data for pretraining, not explanation labels or masks.

- [2025 Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis](2025_faithful_and_interpretable_explanations_for_complex_ensemble_time_series_forecasts.md)  
  Uses controlled feature injection on real forecasting data rather than a full synthetic dataset. Useful for validating that explanations recover known injected effects, but not enough by itself for full temporal/covariate mask benchmarks.
  **Ground-truth explainability:** Partial. The injected feature has a known effect, but the full original time series does not have complete ground-truth masks for all drivers.

### Papers That Use Methods For Explainability

These papers are most useful for choosing or adapting explanation methods.

- [2016 Why Should I Trust You? Explaining the Predictions of Any Classifier](2016_why_should_i_trust_you_explaining_predictions_of_any_classifier.md)  
  LIME: model-agnostic local surrogate explanations.

- [2023 LoMEF: A Framework to Produce Local Explanations for Global Model Time Series Forecasts](2023_lomef_framework_to_produce_local_explanations_for_global_model_time_series_forecasts.md)  
  Citation-only local note. Model-agnostic local surrogate explanations for global forecasting models.

- [2025 Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis](2025_faithful_and_interpretable_explanations_for_complex_ensemble_time_series_forecasts.md)  
  Citation-only local note. LightGBM surrogate + SHAP for complex ensemble forecasts.

- [2025 Explainable Time-Series Forecasting with Sampling-Free SHAP for Transformers](2025_explainable_time_series_forecasting_with_sampling_free_shap_for_transformers.md)  
  SHAP-style explanation method for Transformer forecasting.

- [2026 Interpretable Deep Convolutional Model for Nonlinear Multivariate Time Series in Complex Systems](2026_interpretable_deep_convolutional_model_for_nonlinear_multivariate_time_series_in_complex_systems.md)  
  DCIts: an intrinsic local explanation in which a Focuser selects source-lag entries, a Modeler assigns signed coefficients, and their product forms the transition tensor used directly in the one-step forecast.

- [2025 Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention](2025_learning_temporal_saliency_for_time_series_forecasting_with_cross_scale_attention.md)  
  Temporal saliency through cross-scale attention.

- [2024 Multidimensional Dynamic Attention for Multivariate Time Series Forecasting](2024_multidimensional_dynamic_attention_for_multivariate_time_series_forecasting.md)  
  Dynamic attention over lagged variables for multistep forecasting.

- [2020 Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting](2020_temporal_fusion_transformers_for_interpretable_multi_horizon_time_series_forecasting.md)  
  Intrinsic interpretability through variable selection and attention.

- [2020 Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks](2020_interpretable_multivariate_time_series_forecasting_with_temporal_attention_convolutional_neural_networks.md)  
  Citation-only local note. Temporal attention convolutional forecasting model.

- [2020 Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation](2020_spatiotemporal_attention_for_multivariate_time_series_prediction_and_interpretation.md)  
  Citation-only local note. Spatiotemporal attention over variables and timesteps.

- [2025 An End-to-End Explainability Framework for Spatio-Temporal Predictive Modeling](2025_an_end_to_end_explainability_framework_for_spatio_temporal_predictive_modeling.md)  
  Model-agnostic masking/meta-optimization for features, timesteps, and node locations.

- [2026 MCIR: A Feature Dependence-Aware Explainability Method with Reliability Guarantees](2026_mcir_feature_dependence_aware_explainability_method.md)  
  Dependence-aware global feature importance.

- [2021 Transformer Interpretability Beyond Attention Visualization](2021_transformer_interpretability_beyond_attention_visualization.md)  
  Transformer-specific relevance propagation beyond raw attention.

- [2024 XForecast: Evaluating Natural Language Explanations for Time Series Forecasting](2024_xforecast_evaluating_natural_language_explanations_for_time_series_forecasting.md)  
  Citation-only local note. Natural-language explanations for forecasts.

## Core Takeaways For The Plan

1. The project should not depend on classification-XAI benchmarks.
2. The strongest benchmark design should be:

```text
history + covariates -> future forecast
known temporal/covariate/time-covariate rule -> ground-truth mask
post-hoc explanation -> predicted attribution
predicted attribution vs ground truth -> explanation metrics
```

3. Attention-based papers are useful, but attention must be validated because
   attention is not automatically explanation.
4. Correlated lags and covariates must become explicit difficulty levels because
   they can make attribution unstable.
5. Real-data explanation should be treated differently from synthetic evaluation:
   on real data, there is usually no ground-truth mask, so use diagnostics,
   stability, counterfactuals, and plausibility checks.
6. Local surrogate papers such as LoMEF and the ensemble-surrogate work are
   especially relevant for Chronos-2 because they treat the forecaster as a
   black box.
7. Natural-language explanation work such as XForecast should come after the
   quantitative attribution benchmark, not before it.
