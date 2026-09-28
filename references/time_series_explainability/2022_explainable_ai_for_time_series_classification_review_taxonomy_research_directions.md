# Explainable AI for Time Series Classification: A Review, Taxonomy and Research Directions

## Basic Info
- **Title:** Explainable AI for Time Series Classification: A Review, Taxonomy and Research Directions
- **Authors:** Andreas Theissler, Francesco Spinnato, Udo Schlegel, Riccardo Guidotti
- **Year:** 2022
- **Venue:** IEEE Access
- **Paper link:** https://ieeexplore.ieee.org/document/9895252
- **DOI:** 10.1109/ACCESS.2022.3207765
- **Code link:** Not applicable
- **Dataset link:** Not applicable
- **Citation/BibTeX:** IEEE Access review
- **License:** Check publisher

## Problem Type
- **Task:** Review of XAI for time-series classification
- **Forecasting?** No
- **Input:** Time series
- **Output:** Class labels

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No
- **Ground-truth explanations?** Review, not benchmark
- **Black-box compatible?** Surveys many methods
- **Chronos/foundation forecasting relevance:** Low to medium as taxonomy only
- **Relevance score:** 2/5
- **Reason:** Classification-focused, but useful taxonomy for explanation types.

## Dataset Details
- **Dataset names:** Many reviewed classification datasets
- **Synthetic or real:** Mixed in literature
- **Forecast target:** None

## Ground-Truth Explanation
- **Known mask?** Not applicable
- **Mask shape:** Classification explanation taxonomy

## Model Details
- **Architectures:** Surveys time-series classification models
- **Black-box/interpretable:** Surveys both

## Explanation Method
- **Families:** Time-point, subsequence, instance-based explanations and related taxonomies

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Surveyed for classification

## Code Usability
- **Code available?** Not applicable
- **Adaptability:** Useful for organizing explanation terminology, not benchmark creation.

## Limitations
- Classification-only
- Does not solve `history -> future` explanation

## How It Helps This Project
- Helps write background and distinguish classification-XAI from forecasting-XAI.

## Final Decision
**Use as citation** for taxonomy; do not use as benchmark.

