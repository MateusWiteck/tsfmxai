# Evaluating Anomaly Explanations Using Ground Truth

## Basic Info
- **Title:** Evaluating Anomaly Explanations Using Ground Truth
- **User search phrase / alternate framing:** Evaluating Correctness and Robustness of Local Anomaly Explanations Using Ground Truth
- **Authors:** Liat Antwarg Friedman, Chen Galed, Lior Rokach, Bracha Shapira
- **Year:** 2024
- **Venue:** AI
- **Paper link:** https://www.mdpi.com/2673-2688/5/4/117
- **PDF:** `Refferences/2024_evaluating_anomaly_explanations_using_ground_truth.pdf`
- **DOI:** https://doi.org/10.3390/ai5040117
- **Code link:** https://github.com/XAI-Lab/CREM
- **Dataset link:** https://doi.org/10.7910/DVN/W4FPPN
- **Citation/BibTeX:** AI 2024, 5(4), 2375-2392
- **License:** Open-access MDPI article

## Problem Type
- **Task:** Local anomaly explanation evaluation
- **Forecasting?** No
- **Input:** Binary tabular instances from digital-circuit truth tables, with original and anomalous circuit behavior
- **Output:** Anomaly labels plus local ground-truth feature explanations for anomalous outputs
- **Univariate/multivariate:** Multivariate/tabular

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No, explains anomalies
- **Ground-truth explanations?** Yes, local feature sets for anomalous circuit outputs
- **Black-box compatible?** Evaluation is used with model-agnostic explainers such as Kernel SHAP, Sampling SHAP, and LIME
- **Chronos/foundation forecasting relevance:** Indirect
- **Relevance score:** 3/5
- **Reason:** Not time-series forecasting, but highly relevant to the core question of evaluating local explanations against ground truth and separating correctness from robustness under noise.

## Dataset Details
- **Dataset names:** Digital-circuit anomaly benchmark based on ISCAS '85 and 74x-series circuits: C17, 74283, 74182, and 74181.
- **Synthetic or real:** Structured synthetic/benchmark data derived from digital circuit truth tables.
- **Input shape:** Binary feature vectors representing circuit inputs, optionally padded with noise attributes.
- **Output shape:** Binary circuit outputs; an instance is anomalous when the modified/anomalous circuit output differs from the original circuit output.
- **Generation mechanism:** Start from circuit logic. Create anomalous versions by replacing one logic operator at a time with its negated operator. Generate truth tables for both original and anomalous circuits. Add controlled attribute noise levels for robustness evaluation.
- **Noise:** Adds zero to six redundant/noise features depending on the experiment.
- **Dataset/code availability:** Dataset and local ground-truth explanations are published on Harvard Dataverse; evaluation code is on GitHub.

## Ground-Truth Explanation
- **Known mask?** Yes
- **Mask shape:** Local binary feature set for each anomalous output/instance.
- **Mask generation:** Ground truth is generated from the anomalous circuit diagram using a Boolean influence/backtracking algorithm. For a given anomalous output and instance, the method backtracks through the circuit graph. At each logic operator, it tests minimal subsets of input nodes by flipping their values and checking whether the operator output changes. Influential input nodes become the local ground-truth explanation.
- **Important distinction:** This is not SHAP-as-ground-truth. The ground truth is derived from the known symbolic Boolean mechanism and instance-specific circuit path.
- **Best use for this project:** Good reference for designing explanation labels from a known mechanism and for evaluating ranked explanations against a local ground-truth set.
- **Ambiguity:** It is not a time-series or forecasting setup. Direct adaptation would require replacing circuit graphs with known temporal/covariate causal generators.

## Model Details
- **Anomaly detector:** A simplified autoencoder-like anomaly detector implemented with scikit-learn style APIs. It maps original truth-table behavior to anomalous truth-table behavior and explains outputs with reconstruction differences.
- **Explainers evaluated:** Kernel SHAP, Sampling SHAP, and LIME.
- **Black-box/interpretable:** Explainers are model-agnostic; the benchmark generator itself is transparent.

## Explanation Method
- **Type:** Post-hoc local feature-importance explanations evaluated against known local ground truth
- **Family:** LIME/SHAP-style model-agnostic explanations
- **Target:** Features responsible for anomalous outputs
- **Attribution scope:** Covariate Attribution. No temporal dimension.

## Evaluation
- **Forecasting metrics:** None
- **Explanation correctness metrics:** Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Mean R-Precision / MR-Precision. The produced explanation is sorted by absolute feature-importance score and compared with the local ground-truth feature set.
- **Explanation robustness metrics:** Equalized Loss of Accuracy (ELA), adapted to use R-precision instead of accuracy when noise attributes are added.
- **Quantitative or visual:** Quantitative correctness and robustness metrics.
- **Evaluation summary check:** This is exactly the kind of paper that makes explainability evaluation automatic: it has local ground-truth feature labels and computes ranking-based correctness metrics. It is not forecasting-specific, but its metric design is useful for Phase 5 explanation evaluation.
- **Evaluation references cited:** Uses SHAP/Lundberg and Lee, LIME/Ribeiro et al., Boolean influence/power-index ideas, and robustness via Equalized Loss of Accuracy.

## Code Usability
- **Code available?** Yes
- **Runs locally?** Not checked
- **Adaptability:** Medium. The metric implementation and benchmark-contract idea are useful; the circuit-specific generator is less directly useful.

## Limitations
- Not time series
- Not forecasting
- Binary/tabular circuit data may be too far from Chronos-style history-to-future forecasting
- Ground-truth explanations are local feature sets, not temporal masks or feature-time masks

## How It Helps This Project
- Strong citation for automatic explanation-correctness evaluation with local ground truth.
- Useful metrics: MRR, MAP, R-Precision, and robustness under added noise.
- Useful conceptual distinction: explanation correctness should be evaluated against local ground truth, while robustness should verify that explanations do not elevate irrelevant/noise features.

## Final Decision
**Use as evaluation-metrics reference**, not as a forecasting benchmark.
