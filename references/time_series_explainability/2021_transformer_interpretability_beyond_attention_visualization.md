# Transformer Interpretability Beyond Attention Visualization

## Basic Info
- **Title:** Transformer Interpretability Beyond Attention Visualization
- **Authors:** Hila Chefer, Shir Gur, Lior Wolf
- **Year:** 2021
- **Venue:** CVPR 2021
- **Paper link:** https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html
- **Code link:** https://github.com/hila-chefer/Transformer-Explainability
- **Dataset link:** Vision/text benchmarks in paper
- **Citation/BibTeX:** CVPR 2021 paper
- **License:** Check code repository

## Problem Type
- **Task:** Transformer explanation
- **Forecasting?** No
- **Input:** Images/text for Transformer models
- **Output:** Class predictions

## Relevance To This Project
- **Predicts future values?** No
- **Explains forecasts?** No
- **Ground-truth explanations?** Not forecasting masks
- **Black-box compatible?** No, uses Transformer internals
- **Chronos/foundation forecasting relevance:** Medium if explaining Transformer internals
- **Relevance score:** 2/5
- **Reason:** Useful Transformer attribution method, but not time-series forecasting benchmark.

## Dataset Details
- **Dataset names:** Vision/text datasets
- **Synthetic or real:** Real
- **Forecast target:** None

## Ground-Truth Explanation
- **Known mask?** Not for forecasting
- **Mask shape:** Image/text relevance

## Model Details
- **Architecture:** Transformer models
- **Black-box/interpretable:** Model-specific explanation using attention/relevance propagation
- **Required interface:** Transformer internals and gradients/relevance propagation

## Explanation Method
- **Type:** Post-hoc/model-specific
- **Family:** Relevance propagation beyond attention visualization
- **Target:** Tokens/patches

## Explanation Calculation
This paper argues that **attention alone is not an explanation**. A high attention value only says that one token attends to another token. It does not say whether that connection helped the chosen class, hurt it, or was irrelevant.

Chefer et al. combine three signals:
- attention maps,
- gradients of the chosen class with respect to those attention maps,
- relevance scores propagated through the Transformer using LRP / Deep Taylor-style rules.

The calculation can be summarized as:

```text
class score -> relevance through Transformer layers -> token/patch heatmap
```

1. **Choose the class to explain.**

Pick a target class `t`, usually the predicted class, and explain the class score:

```text
y_t
```

This makes the explanation class-specific. The same input can produce different maps for different classes.

2. **Take the attention matrices from each Transformer block.**

For each Transformer block `b`, the model has an attention tensor:

```text
A_b shape = heads x tokens x tokens
```

In ViT, tokens are image patches plus a `[CLS]` token. In text models, tokens are words/subwords plus special tokens.

3. **Compute class gradients through attention.**

The method asks how the target class score would change if an attention connection changed:

```text
G_b = d y_t / d A_b
```

This is what makes the attention signal class-sensitive.

4. **Propagate relevance through the Transformer.**

The paper propagates relevance from the selected class backward through the Transformer. For the attention softmax in block `b`, call this relevance:

```text
R_b
```

This is the "beyond attention visualization" part. The method does not only inspect attention matrices; it also propagates decision relevance through attention, skip connections, matrix multiplication, and other Transformer operations.

5. **Build a class-specific attention relevance matrix for each block.**

For each block, combine the class gradient and propagated relevance:

```text
Abar_b = I + mean_heads((G_b * R_b)_positive)
```

where:

```text
I             = identity matrix for residual/skip connection
*             = element-wise multiplication
mean_heads    = average over attention heads
(.)_positive  = keep only positive class-supporting contributions
```

In words:

```text
keep attention paths that are relevant and positively support class t
```

Negative contributions are removed because the heatmap is meant to show positive evidence for the selected class.

6. **Propagate relevance across all Transformer blocks.**

The block-level matrices are multiplied through the Transformer:

```text
C = Abar_1 @ Abar_2 @ ... @ Abar_B
```

where `@` is matrix multiplication and `B` is the number of Transformer blocks. The result is:

```text
C shape = tokens x tokens
```

`C` estimates how class-supporting relevance flows between tokens through the stack of Transformer blocks.

7. **Extract the final image/text explanation.**

For classification Transformers, the `[CLS]` token is used for the final decision. The explanation takes the `[CLS]` row:

```text
relevance_scores = C[CLS]
```

For ViT:

```text
remove special tokens
reshape patch scores into the image patch grid
upsample to the original image size
```

The output is a heatmap showing which patches positively contributed to class `t`.

## Difference From Attention Rollout
Standard attention rollout multiplies attention matrices across blocks:

```text
Ahat_b = I + mean_heads(A_b)
rollout = Ahat_1 @ Ahat_2 @ ... @ Ahat_B
```

Rollout uses only attention, so it is not class-specific. Chefer et al. replace raw attention with class-specific weighted relevance:

```text
Chefer block signal = attention relevance * class gradient
```

Short version:

```text
attention rollout:
attention only

Chefer et al.:
attention + class gradients + propagated relevance
```

So rollout can highlight tokens that receive attention but do not support the chosen class, while Chefer et al. tries to keep only positive relevance paths for that class.

## Difference From LIME Image Segmentation
This is **not** superpixel segmentation.

LIME does:

```text
image -> superpixels -> perturb superpixels -> model queries -> local surrogate
```

Chefer et al. does:

```text
image/text -> Transformer internals -> gradients + relevance propagation -> token/patch heatmap
```

So LIME is black-box and perturbation-based, while this method is model-specific and requires access to Transformer internals.

## Evaluation
- **Forecasting metrics:** None
- **Explanation metrics:** Vision/text explanation benchmarks

## Code Usability
- **Code available?** Yes
- **Adaptability:** Possibly useful if Chronos internals are accessible, but not black-box.

## Limitations
- Not time-series-specific
- Not forecasting-specific
- Requires model internals

## How It Helps This Project
- Useful caution and method source for Transformer relevance if the forecasting model exposes internals.

## Final Decision
**Use only as citation / possible method reference**.
