# Time Series Foundation Model Explainable AI

Repository for Explainable AI in Time Series Foundation Models, initially focused on Chronos-2.

Main objectives:

1. Forecasting with multiple covariate availability settings
2. Temporal explainability
3. Covariate explainability
4. Time-Covariate explainability
5. Ground-truth synthetic benchmarks for explanation evaluation
6. Forecast and explanation evaluation

---

# 1. Research Questions

## RQ1

Can post-hoc methods explain Time Series Foundation Models?

## RQ2

Which explanation mode is more informative?

- Temporal Attribution
- Covariate Attribution
- Time-Covariate Attribution

## RQ3

Are generated explanations faithful to model behavior?

## RQ4

How does covariate availability affect explanation quality?

## RQ5

Can synthetic datasets with known attribution patterns be used to evaluate explanations?

---

# 2. Explanation Modes

## Temporal Attribution

Question:

```text
Which historical timestamps influenced the forecast?
```

Output:

```text
importance[time]
```

Example:

```text
t-7 strongly affected prediction
```

---

## Covariate Attribution

Question:

```text
Which variables influenced the forecast?
```

Output:

```text
importance[covariate]
```

Example:

```text
temperature was the most important variable
```

---

## Time-Covariate Attribution

Question:

```text
Which variable at which timestamp influenced prediction?
```

Output:

```text
importance[time,covariate]
```

Example:

```text
temperature at t−3 was highly important
```

---

# 3. Synthetic Benchmark

Synthetic data is required because real datasets rarely contain explanation ground truth.

Synthetic datasets will explicitly generate:

```text
signal
+ hidden attribution rule
+ target
+ attribution mask
```

The attribution mask becomes the ground truth.

---

# 4. Synthetic Dataset Families

## 4.1 Temporal Benchmark

Purpose:

```text
evaluate temporal attribution
```

Generation:

```text
background signal
+ temporal event
```

Example:

```text
signal = seasonality + trend + noise

if mean(signal[t=40:50]) > threshold:
      future_target += c
```

Ground truth:

```text
important_times=[40:50]
```

Expected:

```text
temporal explanation highlights t=40:50
```

---

## 4.2 Covariate Benchmark

Purpose:

```text
evaluate variable attribution
```

Generate:

```text
temperature
humidity
wind
pressure
noise variables
```

Rule:

```text
target=
0.7*temperature
+
0.3*wind
+
noise
```

Ground truth:

```text
temperature=.7
wind=.3
others=0
```

Expected:

```text
temperature > wind >>> others
```

---

## 4.3 Time-Covariate Benchmark

Purpose:

```text
evaluate variable+time attribution
```

Rule:

```text
only temperature at t−5:t−2
affects prediction
```

Ground truth:

```text
[(temperature,t−5),
(temperature,t−4),
(temperature,t−3),
(temperature,t−2)]
```

Expected:

heatmap:

```text
             temp wind holiday
t−1            0     0     0
t−2            1     0     0
t−3            1     0     0
t−4            1     0     0
```

---

# 5. Difficulty Levels

Synthetic generators should support increasing complexity.

Level 1:

```text
single variable
single temporal region
```

Level 2:

```text
multiple regions
```

Level 3:

```text
lag dependencies
```

Example:

```text
temperature at t−10
```

Level 4:

```text
variable interactions
```

Example:

```text
temperature matters only if holiday=1
```

Level 5:

```text
nonlinear rules
```

Example:

```text
if temperature>20 and wind<5:
      increase target
```

---

# 6. Repository Structure

The canonical current organization is documented in `README.md`. The tree
below describes the intended software modules as they are implemented under
`src/tsfmxai/`; exploratory prototypes remain under the ignored
`experiments_local/` directory until they are promoted into reusable code or a
tracked scientific experiment.

```text
time-series-foundation-model-xai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── src/
│
│   ├── synthetic/
│   │   ├── temporal_generator.py
│   │   ├── covariate_generator.py
│   │   ├── time_covariate_generator.py
│   │   ├── attribution_masks.py
│   │   └── benchmark_configs.py
│   │
│   ├── datasets/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   └── windowing.py
│   │
│   ├── forecasting/
│   │   ├── base.py
│   │   ├── chronos2_provider.py
│   │   └── baselines.py
│   │
│   ├── explainability/
│   │   ├── temporal_explainer.py
│   │   ├── covariate_explainer.py
│   │   ├── time_covariate_explainer.py
│   │   └── perturbation_utils.py
│   │
│   ├── evaluation/
│   │   ├── forecast_metrics.py
│   │   ├── explanation_metrics.py
│   │   └── benchmark_metrics.py
│   │
│   └── visualization/
│       ├── heatmaps.py
│       └── attribution_plots.py
│
├── notebooks/
│
│   ├── 01_generate_synthetic.ipynb
│   ├── 02_temporal_benchmark.ipynb
│   ├── 03_covariate_benchmark.ipynb
│   ├── 04_time_covariate_benchmark.ipynb
│   ├── 05_chronos_forecasting.ipynb
│   └── 06_explanations.ipynb
│
├── ROADMAP.md
└── README.md
```

---

# 7. Development Roadmap

# Phase 1

Synthetic benchmark generation

Goal:

```text
Build inspectable synthetic datasets where the data-generating rule and the
ground-truth explanation mask are known before any model or explainer is used.
```

Phase 1 should establish the benchmark contract used by later forecasting,
explanation, and evaluation phases.

## Phase 1.1 Benchmark Data Contract

Define a shared output format for every synthetic generator.

Each generated benchmark should return:

```text
series_id
timestamp
target history
future target
dynamic covariates
static covariates, if needed
ground_truth_temporal_mask
ground_truth_covariate_mask
ground_truth_time_covariate_mask
generation_metadata
```

Minimum metadata:

```text
benchmark_family
difficulty_level
context_length
prediction_length
covariate_names
important_times
important_covariates
rule_parameters
random_seed
```

Expected files:

```text
src/synthetic/benchmark_configs.py
src/synthetic/attribution_masks.py
src/datasets/schema.py
```

Acceptance criteria:

```text
All generators emit the same schema.
Masks align exactly with generated context windows and covariate columns.
Generation is reproducible from a seed.
```

## Phase 1.2 Temporal Synthetic Generator

Create a generator for rules where specific historical timestamps influence the
forecast.

Implementation target:

```text
src/synthetic/temporal_generator.py
```

Required capabilities:

```text
Generate trend, seasonality, noise, and local temporal events.
Support one or more important temporal regions.
Produce a temporal attribution mask with shape [context_length].
Produce optional time-covariate masks when covariates are present.
Expose difficulty levels 1-3 initially.
```

Example rule:

```text
future_target += c * mean(target[t_start:t_end])
ground_truth_temporal_mask[t_start:t_end] = 1
```

Acceptance criteria:

```text
Important time regions are visible in the returned mask.
Changing the important region changes the generated future target.
Noise variables or irrelevant periods are not marked important.
```

## Phase 1.3 Covariate Synthetic Generator

Create a generator for rules where specific variables influence the forecast.

Implementation target:

```text
src/synthetic/covariate_generator.py
```

Required capabilities:

```text
Generate multiple dynamic covariates.
Include informative, weakly informative, and irrelevant covariates.
Support linear weighted rules first.
Produce a covariate attribution mask with shape [num_covariates].
Produce optional time-covariate masks by broadcasting important variables over
the relevant context window.
```

Example rule:

```text
future_target = 0.7 * temperature + 0.3 * wind + noise
ground_truth_covariate_mask = {temperature: 0.7, wind: 0.3, others: 0}
```

Acceptance criteria:

```text
Known informative covariates receive nonzero ground-truth importance.
Irrelevant covariates receive zero ground-truth importance.
Weights are recoverable from generation metadata.
```

## Phase 1.4 Time-Covariate Synthetic Generator

Create a generator for rules where a specific variable at a specific historical
time range influences the forecast.

Implementation target:

```text
src/synthetic/time_covariate_generator.py
```

Required capabilities:

```text
Generate several covariates across the context window.
Apply rules to selected covariates at selected lags.
Produce a time-covariate attribution mask with shape
[context_length, num_covariates].
Derive temporal and covariate summaries from the time-covariate mask.
Support lag-window rules such as t-5:t-2.
```

Example rule:

```text
future_target += c * mean(temperature[t-5:t-2])
ground_truth_time_covariate_mask[t-5:t-2, temperature] = 1
```

Acceptance criteria:

```text
The mask identifies both the correct covariate and correct lag window.
Temporal and covariate masks derived from the heatmap are consistent.
Unrelated covariates and timestamps remain zero.
```

## Phase 1.5 Attribution Mask Utilities

Implement reusable helpers for creating, validating, normalizing, and aggregating
masks.

Implementation target:

```text
src/synthetic/attribution_masks.py
```

Required utilities:

```text
create_temporal_mask(context_length, important_ranges)
create_covariate_mask(covariate_names, weights)
create_time_covariate_mask(context_length, covariate_names, rules)
aggregate_time_covariate_to_temporal(mask)
aggregate_time_covariate_to_covariate(mask)
normalize_mask(mask)
validate_mask_shape(mask, expected_shape)
```

Acceptance criteria:

```text
Mask utilities are tested independently from generators.
Aggregated masks preserve the expected important regions and variables.
Invalid mask shapes fail early with clear errors.
```

## Phase 1.6 Visualization For Inspection

Build simple visual diagnostics for generated data and ground-truth masks.

Implementation targets:

```text
src/visualization/attribution_plots.py
src/visualization/heatmaps.py
notebooks/01_generate_synthetic.ipynb
```

Required plots:

```text
Target history and future target.
Temporal mask overlaid on historical context.
Covariate importance bar chart.
Time-covariate attribution heatmap.
Small benchmark report figure for each synthetic family.
```

Acceptance criteria:

```text
Each synthetic family has at least one saved example plot.
Plots make the known attribution pattern visually obvious.
Visualization code works from generated benchmark objects without special cases.
```

## Phase 1.7 Tests And Reproducibility

Add focused tests for generation behavior.

Suggested tests:

```text
Same seed produces identical data and masks.
Different seeds produce different samples.
All generated outputs match the shared schema.
Masks have expected shapes.
Ground-truth masks match configured rules.
Generated targets respond to the intended important signal.
Irrelevant covariates do not appear in masks.
```

Expected files:

```text
tests/synthetic/test_temporal_generator.py
tests/synthetic/test_covariate_generator.py
tests/synthetic/test_time_covariate_generator.py
tests/synthetic/test_attribution_masks.py
```

## Phase 1 Deliverables

By the end of Phase 1, the repository should contain:

```text
Three synthetic benchmark generators.
A shared benchmark config and output schema.
Ground-truth temporal, covariate, and time-covariate masks.
Mask validation and aggregation utilities.
Example generated datasets under data/synthetic/.
Inspection plots for all benchmark families.
Focused unit tests for generators and masks.
```

Done means:

```text
A later explainer can consume a generated benchmark and compare its predicted
importance directly against the known mask, without needing to infer or repair
the benchmark format.
```

Implementation order:

```text
1. Define config objects and output schema.
2. Implement attribution mask utilities.
3. Implement temporal generator.
4. Implement covariate generator.
5. Implement time-covariate generator.
6. Add plots for inspection.
7. Add tests and fixed example outputs.
```

Original Phase 1 checklist:

```text
Step 1: Create temporal synthetic generator
Step 2: Create covariate synthetic generator
Step 3: Create time-covariate synthetic generator
Step 4: Generate attribution masks
Step 5: Implement Phase 1 visualizations for the synthetic benchmarks
```

Deliverable:

```text
Synthetic datasets with known explanation ground truth.
Implemented visualizations for Phase 1 benchmark inspection and reporting.
```

---

# Phase 2

Data infrastructure

Step 6:

```text
Create schema validator
```

Step 7:

```text
Implement dataset loader
```

Step 8:

```text
Implement window generation
```

---

# Phase 3

Forecasting

Step 9:

```text
Implement ForecastProvider
```

Step 10:

```text
Implement Chronos2 provider
```

Step 11:

```text
Add baselines
```

---

# Phase 4

Explanation methods

Step 12:

```text
Temporal attribution
```

Step 13:

```text
Covariate attribution
```

Step 14:

```text
Time-covariate attribution
```

---

# Phase 5

Explanation evaluation

Step 15:

Compare:

```text
predicted importance

vs

ground-truth mask
```

Metrics:

```text
IOU
Precision@K
Recall@K
AUPRC
Faithfulness
Stability
```

---

# Phase 6

Real datasets

Run on:

```text
ETT
Electricity
Weather
Traffic
M4
UCR
custom datasets
```

Compare:

```text
synthetic performance
vs
real performance
```

---

# Final Pipeline

```text
Generate synthetic data
-> generate ground-truth masks
-> forecast
-> explain
-> compare explanations to masks
-> benchmark metrics
-> real datasets
-> final evaluation
```
