Regime-switching theory

    James Hamilton

    "Regime Switching Models"

Event-driven forecasting


# Generating time series mechanisms
for univariate or multi agent interaction systems

- lag-driven
- state-driven (hidden states also)
- event-driven (regime-driven could be represented inside)



Dynamical Systems Theory → "How does a system evolve over time according to governing rules?"
State-Space / Control Theory → "What hidden states and inputs explain the observed behavior?"
Complex Systems → "How do interactions among many entities create collective temporal patterns?"
Point Process Theory → "How and when do events occur and trigger future events?"
Econometrics / Classical Time Series → "How can observed temporal patterns be decomposed and predicted?"
Stochastic Processes → "What probabilistic rules govern the evolution of randomness over time?"
Chaos / Nonlinear Time Series → "Can complex or chaotic dynamics be reconstructed from observations?"


generative-causal ontology question

# My notes about the papers

2021_benchmarking_attention
    - Very important for the data generation
    - Uses attention mechanism and show that it work
    - The evaluation of explanations is only visual 

2025_learning_temporal_saliency
    - quantitative real-data explanations metrics: sufficiency and comprehensiveness.
    - synthetic Datsets are simpler
    - evaluate different metrics 
        - Sufficiency(r) = distance(f(x), f(mask_keep(x, top_r_saliency)))
        - Comprehensiveness(r) = distance(f(x), f(mask_remove(x, top_r_saliency)))

2025_explainable_time_series_forecasting_with_sampling_free_shap_for_transformers (Nature)
    - Synthetic data generation with ground truth explanations using SHAP in the generating of ground truth (this concerns me I need to see the implementation) I asked for the code to know exactly what they do 
    - They propose SHAPformer for explainability, we could test this model

2025_faithful_and_interpretable_explanations_for_complex_ensemble_time_series_forecasts
    - Generate Gound Truths and evaluate the correlation between the generated ones and the actual one
    - Covariate attribution: They create only a partial ground truth by injecting a synthetic feature with a known effect into real M5 demand data, then checking whether SHAP recovers that known injected effect.
    - Do not provide the code for this creation

2025_an_end_to_end_explainability_framework_for_spatio_temporal_predictive_modeling
    - do not generate synthetical data or ground truth labels for explainability 
    - They evaluate fidelity by masking/removing the explanation-selected input parts and measuring how much the trained model’s prediction changes, using the model’s original prediction as the reference.

2026_mcir_feature_dependence_aware_explainability_method
    - They evaluate explainability indirectly: MCIR-M produces feature-importance rankings, then checks whether those rankings are stable, non-redundant, and faithful, using metrics like Jaccard@K, Kendall τ, deletion/insertion curves, redundancy collapse, and ERI, instead of comparing to a binary/weighted ground-truth explanation.
    - They propose their model that they say that can deal with highly correlated inputs
    - TODO: see if the implementation could be used for time-variant attributions
    - not focused on time-series forecasting; in its experiments it mainly performs covariate/feature attribution, not native time or time-covariate attribution




# Things to relate my paper with:


- Causal discovery: Infers cause-and-effect relationships from observational time-series data.

- Temporal causal discovery: Identifies causal relationships while considering time order and lagged effects.

- Causal graph: Represents variables as nodes and causal influences as directed edges.

- Causal inference: Estimates how changing one variable would affect another.

- Causal sufficiency: Assumes that all relevant common causes have been measured.

- Causal associations: Statistical relationships interpreted as direct causal links after controlling for other factors.






