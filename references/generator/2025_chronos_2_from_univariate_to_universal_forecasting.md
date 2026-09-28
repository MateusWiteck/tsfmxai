# Chronos-2: From Univariate to Universal Forecasting

## Basic Info
- **Title:** Chronos-2: From Univariate to Universal Forecasting
- **Authors:** Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, Michael Bohlke-Schneider
- **Year:** 2025
- **Venue:** Technical report / arXiv
- **Paper link:** https://arxiv.org/abs/2510.15821
- **Code link:** https://github.com/amazon-science/chronos-forecasting
- **Dataset link:** See paper/code
- **Citation/BibTeX:** Not collected
- **License:** Check code repository

## Problem Type
- **Task:** Universal time-series forecasting
- **Forecasting?** Yes
- **Input:** Time-series history, potentially with broader universal forecasting setup
- **Output:** Future values / forecast distribution

## Relevance To This Project
- **Predicts future values?** Yes
- **Explains forecasts?** No, not primarily
- **Ground-truth explanations?** No
- **Black-box compatible?** This is the black-box/foundation model target
- **Chronos/foundation forecasting relevance:** Very high
- **Relevance score:** 4/5 for model context, 1/5 for XAI benchmark
- **Reason:** Core forecasting model reference, but not explanation dataset.

## Dataset Details
- **Dataset names:** See Chronos-2 report
- **Synthetic or real:** Broad forecasting training/evaluation data
- **Input/output:** History to future
- **Generation mechanism:** Not an XAI generator

## Ground-Truth Explanation
- **Known mask?** No
- **Mask shape:** None

## Model Details
- **Architecture:** Foundation forecasting model
- **Black-box/interpretable:** Black-box for this project
- **Required interface:** Forecast API/model inference

## Explanation Method
- **Type:** None
- **Target:** Not applicable

## Evaluation
- **Forecasting metrics:** Forecasting benchmark metrics
- **Explanation metrics:** None

## Code Usability
- **Code available?** Yes
- **Adaptability:** High as target model; not as benchmark.

## Limitations
- No built-in XAI ground truth
- Need external explanation and evaluation framework

## How It Helps This Project
- Provides the foundation forecasting model to explain.
- Synthetic benchmark should be built around this model's `history -> future` behavior.

## Final Decision
**Use directly** as target forecasting model context, not as XAI benchmark.

