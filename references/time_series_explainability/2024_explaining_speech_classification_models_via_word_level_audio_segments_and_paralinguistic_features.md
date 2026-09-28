# Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features

## Basic Info
- **Title:** Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features
- **Authors:** Eliana Pastor, Alkis Koudounas, Giuseppe Attanasio, Dirk Hovy, Elena Baralis
- **Year:** 2024
- **Venue:** EACL 2024
- **Paper link:** Local PDF: `Refferences/2024_explaining_speech_classification_models_via_word_level_audio_segments_and_paralinguistic_features.pdf`
- **Code link:** Unknown from local preview
- **Dataset link:** Unknown
- **Citation/BibTeX:** Not collected
- **License:** Unknown

## Problem Type
- **Task:** Speech classification explanation
- **Forecasting?** No
- **Input:** Speech/audio segments and paralinguistic features
- **Output:** Class labels

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No
- **Ground-truth explanations?** Not forecasting-related
- **Black-box compatible?** Likely post-hoc explanation framing
- **Chronos/foundation forecasting relevance:** Low
- **Relevance score:** 1/5
- **Reason:** Speech classification, not forecasting.

## Dataset Details
- **Dataset names:** Spoken language understanding / speech datasets
- **Synthetic or real:** Real
- **Input/output:** Audio to class label
- **Forecast target:** None

## Ground-Truth Explanation
- **Known mask?** Not relevant for forecasting
- **Mask shape:** Audio/word-level attribution, not time-series forecast mask

## Model Details
- **Architecture:** Speech classification models
- **Black-box/interpretable:** Explained as predictive black boxes
- **Required interface:** Model predictions and audio/feature segmentation

## Explanation Method
- **Type:** Post-hoc explanation
- **Target:** Word-level audio segments and paralinguistic features

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Speech classification explanation evaluation

## Code Usability
- **Code available?** Unknown
- **Adaptability:** Low

## Limitations
- Classification-only
- Domain-specific to speech
- No future values or temporal forecasting drivers

## How It Helps This Project
- Minimal. Could inspire human-readable segment explanations, but not benchmark design.

## Final Decision
**Reject for core project**, keep only as general XAI reference.

