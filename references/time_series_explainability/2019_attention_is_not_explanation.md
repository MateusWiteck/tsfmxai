# Attention Is Not Explanation

## Basic Info
- **Title:** Attention is not Explanation
- **Authors:** Sarthak Jain, Byron C. Wallace
- **Year:** 2019
- **Venue:** NAACL / arXiv
- **Paper link:** https://arxiv.org/abs/1902.10186
- **Code link:** Unknown from local preview
- **Dataset link:** Not central
- **Citation/BibTeX:** Standard attention interpretability critique
- **License:** Unknown

## Problem Type
- **Task:** NLP classification interpretability critique
- **Forecasting?** No
- **Input:** Text sequences
- **Output:** Class labels

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No
- **Ground-truth explanations?** No forecasting masks
- **Black-box compatible?** Conceptual critique
- **Chronos/foundation forecasting relevance:** Medium as a warning against treating attention as explanation
- **Relevance score:** 2/5
- **Reason:** Important interpretability caution, but not a forecasting benchmark.

## Dataset Details
- **Dataset names:** NLP datasets
- **Synthetic or real:** Real text tasks
- **Input/output:** Text to class label
- **Forecast target:** None

## Ground-Truth Explanation
- **Known mask?** No forecasting mask
- **Mask shape:** Not applicable

## Model Details
- **Architecture:** Attention-based neural NLP models
- **Black-box/interpretable:** Studies whether attention weights are faithful explanations
- **Required interface:** Model attention and prediction outputs

## Explanation Method
- **Type:** Intrinsic attention interpretation critique
- **Main claim:** Attention weights can be weakly correlated with other importance measures and can be adversarially changed while preserving predictions.
- **Attribution scope:** Sequence/Token Attribution in the original NLP setting. For this project, it is most analogous to Temporal Attribution because attention over sequence positions is being tested as an explanation.

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Kendall tau correlation between attention weights and gradient-based feature importance; Kendall tau correlation between attention weights and leave-one-out/erasure importance; adversarial/counterfactual attention distributions that keep predictions nearly unchanged while changing attention substantially. The paper also uses output-distance and attention-distance ideas such as total variation distance and Jensen-Shannon divergence.
- **Evaluation summary check:** The existing summary agrees with the paper, but the precise point is stronger: attention is tested as an explanation by comparing it to independent importance proxies and by showing that very different attention maps can yield similar predictions. For this project, that supports treating attention maps from forecasting Transformers as hypotheses that still require validation.
- **Evaluation references cited:** Uses Ross et al. (2017) for faithful/gradient-based explanations, Li et al. (2016) for erasure/leave-one-out feature importance, and Sundararajan et al. (2017) for Integrated Gradients as a related attribution method.

## Code Usability
- **Code available?** Unknown
- **Adaptability:** Useful for evaluating attention-based forecasting models skeptically.

## Limitations
- NLP/classification focus
- Does not propose forecasting-XAI data

## How It Helps This Project
- Supports not relying only on attention maps from forecasting Transformers.
- Useful for the evaluation argument: attention should be validated against ground truth.

## Final Decision
**Use as citation** for attention-is-not-necessarily-explanation.
