# Local Interpretable Model-Agnostic Explanations for Music Content Analysis

## Basic Info
- **Title:** Local Interpretable Model-Agnostic Explanations for Music Content Analysis
- **Authors:** Saumitra Mishra, Bob L. Sturm, Simon Dixon
- **Year:** Unknown from local preview
- **Venue:** Unknown from local preview
- **Paper link:** Local PDF: `Refferences/2017_local_interpretable_model_agnostic_explanations_for_music_content_analysis.pdf`
- **Code link:** The abstract says experimental code/results are available online, but the local PDF preview does not show the URL.
- **Dataset link:** Unknown
- **Citation/BibTeX:** Not collected
- **License:** Unknown

## Problem Type
- **Task:** Music content analysis / audio classification explanation
- **Forecasting?** No
- **Input:** Audio-derived representations, e.g. temporal, frequency, or time-frequency segments
- **Output:** Class label, such as singing voice detection

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No
- **Ground-truth explanations?** No clear ground-truth mask
- **Black-box compatible?** Yes, through LIME-style model-agnostic explanations
- **Chronos/foundation forecasting relevance:** Low
- **Relevance score:** 1/5
- **Reason:** Useful only as general LIME background for segmented temporal/audio inputs, not forecasting-XAI.

## Dataset Details
- **Dataset names:** Singing voice detection datasets, not fully identified from preview
- **Synthetic or real:** Real/audio
- **Input shape:** Audio segments or time-frequency representations
- **Output shape:** Classification label
- **Forecast target:** None
- **Covariates/exogenous variables:** Not applicable
- **Splits:** Unknown
- **Generation mechanism:** None

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** Not applicable
- **Mask generation:** Not applicable
- **Ambiguity:** Explanations are qualitative/local, not ground-truth attribution.

## Model Details
- **Architectures:** Decision tree, random forest, convolutional neural network
- **Black-box/interpretable:** Mix of transparent and black-box models
- **Requires gradients/probabilities/samples/attention?** LIME-style perturbation requires model outputs under perturbed samples.

## Explanation Method
- **Type:** Post-hoc, local, model-agnostic
- **Family:** LIME
- **Target:** Temporal/audio/frequency segments
- **Cost:** Requires repeated perturbation and model inference.

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Mostly qualitative agreement/diagnosis
- **Quantitative or visual:** Mostly qualitative/local explanation analysis

## Code Usability
- **Code available?** Claimed by paper, link not visible in preview
- **Runs locally?** Not checked
- **Adaptability:** Low for forecasting; moderate for general segmented perturbation.

## Limitations
- Classification/audio-specific
- No future-value forecasting
- No known forecasting attribution mask

## How It Helps This Project
- Provides background for adapting LIME to temporally segmented inputs.
- Not useful as a forecasting-XAI benchmark.

## Final Decision
**Use only as citation** for model-agnostic local explanation background.

