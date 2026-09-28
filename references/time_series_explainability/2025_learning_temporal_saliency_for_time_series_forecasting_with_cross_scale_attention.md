# Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention

## Basic Info
- **Title:** Learning Temporal Saliency for Time Series Forecasting with Cross-Scale Attention
- **Authors:** Ibrahim Delibasoglu, Fredrik Heintz
- **Year:** 2025
- **Venue:** arXiv
- **Paper link:** https://arxiv.org/abs/2509.22839
- **Local PDF:** `Refferences/2025_learning_temporal_saliency_for_time_series_forecasting_with_cross_scale_attention.pdf`
- **Code link:** Unknown from local preview
- **Dataset link:** Synthetic datasets with known saliency ground truth are described in the paper
- **Citation/BibTeX:** Not collected
- **License:** Unknown

## Problem Type
- **Task:** Time-series forecasting with intrinsic temporal saliency
- **Forecasting?** Yes
- **Input history length:** Depends on dataset/configuration
- **Forecast horizon:** Depends on dataset/configuration
- **Univariate or multivariate:** Time-series forecasting; exact settings need deeper extraction
- **Direct or recursive forecasting:** Unknown from local preview
- **Point or probabilistic forecast:** Unknown from local preview

## Relevance To This Project
- **Does it predict future values?** Yes
- **Does it explain the forecast?** Yes, through learned temporal saliency
- **Does it provide ground-truth explanations?** Yes, synthetic datasets with known saliency ground truth
- **Does it work with black-box models?** No, it proposes an intrinsic architecture, CrossScaleNet
- **Could it apply to Chronos/foundation forecasting models?** Indirectly; useful as benchmark/evaluation inspiration, not directly as a black-box explainer
- **Relevance score:** 4/5
- **Reason:** Very relevant because it is forecasting-specific and evaluates temporal saliency against known synthetic ground truth, but it is architecture-specific rather than black-box model-agnostic.

## Dataset Details
- **Dataset names:** Eight synthetic saliency datasets `SYN1`-`SYN8`, plus public forecasting benchmarks
- **Synthetic or real:** Both
- **Input shape:** Historical time-series context
- **Output shape:** Future forecast values
- **Forecast target:** Future values
- **Covariates/exogenous variables:** Multivariate synthetic features. Each synthetic dataset marks a subset of features as important and the rest as non-important.
- **Train/validation/test split:** Not specified in the extracted synthetic-generation section; check code for exact split before reproducing.
- **Data generation mechanism:** The target variable is constructed from current values of important features and from lagged values at predefined important lags. Gaussian noise is added to the target. The generator is designed to produce known temporal saliency maps.
- **Known lag/window/feature drivers:** Yes. Important lags and important feature sets are defined per dataset.
- **Synthetic dataset configurations:**
  - `SYN1`: important lags `1-15`; important features `{0, 1}`; noise `0.01`.
  - `SYN2`: important lags `1-5, 9-10, 15-16, 18, 20, 25-26, 35-36, 50-52, 91-95`; important features `{0, 2}`; noise `0.05`.
  - `SYN3`: important lags `9-10, 15-16, 18, 20-25, 31, 34, 60-65`; important features `{1, 2}`; noise `0.08`.
  - `SYN4`: important lags `9-10, 15-16, 18-21, 41-42, 45-46`; important features `{1, 2}`; noise `0.10`.
  - `SYN5`: important lags `71-77`; important features `{1, 2}`; noise `0.06`.
  - `SYN6`: important lags `48-57`; important features `{0, 2}`; noise `0.05`.
  - `SYN7`: important lags `60, 62-69`; important features `{0, 1}`; noise `0.02`.
  - `SYN8`: mixture of `SYN5` and `SYN6`; important features `{0, 1, 2}`; noise `0.11`.
- **Temporal-difficulty design:** `SYN1`-`SYN4` emphasize recent historical points, while `SYN5`-`SYN8` emphasize older/distant historical regions. This is directly useful for testing whether a forecasting explainer can find non-recent drivers.

## Ground-Truth Explanation
- **Known explanation mask?** Yes for synthetic datasets
- **Mask shape:** Temporal saliency over input history, plus feature-level importance because each dataset defines important feature indices.
- **How mask is generated:** Important lags define the temporal mask and important feature indices define the feature mask. Together they can be represented as a feature-time mask.
- **Binary or weighted mask:** The paper describes known saliency maps from the predefined lag/feature structure; implementation can store binary masks from important lag-feature pairs and optionally weighted masks if the generator coefficients are exposed in code.
- **Explains whole forecast horizon or one step?** The paper evaluates forecasting outputs and input-history saliency; for implementation, store masks over the input history and attach them to the forecast sample/horizon used by the model.
- **Ambiguity due to autocorrelation/collinearity:** Possible, but unknown from preview
- **Synthetic-generation references cited:** The paper cites time-series forecasting architectures such as PatchTST, TimeMixer, LMSAutoTSF, ETSformer, and ShapTime/Integrated Gradients as context, but the `SYN1`-`SYN8` generator itself appears to be introduced by this paper rather than copied from a prior benchmark.

## Model Details
- **Model architecture:** CrossScaleNet
- **Black-box or interpretable model:** Intrinsically interpretable forecasting model
- **Forecasting model or classifier:** Forecasting model
- **Required training setup:** Train CrossScaleNet on forecasting data
- **Inputs expected by the model:** Historical time-series windows
- **Outputs produced by the model:** Forecast values and temporal saliency information
- **Foundation model support:** Not direct
- **Requires gradients/probabilities/samples/internal attention?** Uses internal cross-scale attention/saliency

## Explanation Method
- **Post-hoc or intrinsic:** Intrinsic
- **Local or global:** Local temporal saliency for forecasts
- **Model-specific or model-agnostic:** Model-specific
- **Explanation target:** Temporal saliency over input history
- **Method family:** Cross-scale attention / intrinsic saliency
- **Computational cost:** Intended to be cheaper than post-hoc ablation for temporal saliency
- **Attribution scope:** Temporal Attribution and Time-Covariate Attribution. The paper is primarily temporal saliency, but the synthetic datasets also define important feature sets, so important lag-feature pairs can be treated as feature-time masks.

## Evaluation
- **Forecasting metrics:** Mean squared error (MSE) and mean absolute error (MAE) on synthetic and public forecasting datasets.
- **Explanation metrics:** Synthetic datasets are evaluated against known temporal saliency ground truth; real datasets are evaluated with comprehensiveness and sufficiency metrics, and attention/saliency maps are compared visually with ShapTime outputs.
- **Quantitative or visual:** Quantitative forecasting tables, saliency-ground-truth comparisons, comprehensiveness/sufficiency scores, and visual saliency maps.
- **Evaluation summary check:** The old summary was correct that this is a forecasting-specific saliency paper, but it missed the exact metrics. For this project, the most important point is that CrossScaleNet is evaluated both as a forecaster and as an explainer: accuracy is measured by MSE/MAE, while saliency quality is checked through synthetic ground truth and perturbation-style sufficiency/comprehensiveness on real data.
- **Sufficiency/comprehensiveness origin:** The paper does not propose these metrics. It adopts the standard rationale/XAI faithfulness idea used in papers such as ERASER: A Benchmark to Evaluate Rationalized NLP Models by DeYoung et al. (arXiv:1911.03429). The idea is to test whether the selected explanation evidence is enough to preserve the model decision, and whether removing that evidence damages the decision.
- **Sufficiency calculation adapted to forecasting:** Rank input timesteps by saliency, choose the top `r%` timesteps, keep only those selected timesteps, replace the rest with a baseline, rerun the forecasting model, and measure the distance between the original forecast and the masked-input forecast. In formula form: `Sufficiency(r) = distance(f(x), f(mask_keep(x, top_r_saliency)))`. Lower is better because the selected salient timesteps should be sufficient to reproduce the original forecast.
- **Comprehensiveness calculation adapted to forecasting:** Rank input timesteps by saliency, choose the top `r%` timesteps, remove or baseline those selected timesteps, keep the rest, rerun the forecasting model, and measure the distance between the original forecast and the perturbed forecast. In formula form: `Comprehensiveness(r) = distance(f(x), f(mask_remove(x, top_r_saliency)))`. Higher is better because removing truly important timesteps should change the forecast substantially.
- **Distance and masking caveat:** The CrossScaleNet paper reports sufficiency/comprehensiveness at ratios such as `10%`, `20%`, and `50%`, but the local PDF text does not clearly specify the exact distance function or masking baseline. For reproduction, define these explicitly, e.g. MAE or MSE between forecast trajectories and a baseline such as zero, mean value, interpolation, or sampled replacement.
- **Evaluation references cited:** Uses ShapTime as the main post-hoc time-series XAI comparison/reference for visual explanation comparison. It also mentions Integrated Gradients as an alternative feature attribution approach, mainly to contrast computational cost and model dependence. For sufficiency/comprehensiveness provenance, cite ERASER / DeYoung et al. (2019) even though the CrossScaleNet paper does not clearly spell out that lineage.

## Code Usability
- **Code available?** Unknown from local preview
- **Can run locally?** Not checked
- **Dependencies:** Unknown
- **GPU required?** Likely useful for model training
- **Includes notebooks/trained models/evaluation scripts?** Unknown
- **Adaptation difficulty:** Medium; conceptually useful even if architecture-specific

## Limitations
- Architecture-specific, not black-box
- The saliency explanation may not transfer directly to Chronos
- Need deeper reading to extract exact synthetic generator and mask format

## How It Helps This Project
- Strong reference for forecasting-specific temporal saliency with known synthetic ground truth.
- Useful to design the first temporal-mask benchmark where `history -> future` and the important historical timestamps are known.
- Good citation to justify that temporal saliency in forecasting is distinct from feature importance in classification.

## Final Decision
**Adapt idea** and inspect deeply.
