---
title: Asymptotic Spectral Theory for Nonlinear Time Series
authors: Xiaofeng Shao; Wei Biao Wu
year: 2007
doi: 10.1214/009053606000001479
url: https://arxiv.org/pdf/math/0611029
category: explosion_treatment
tags: [stationarity, geometric-moment-contraction, nonlinear-autoregression]
status: reviewed
---

# Scientific summary

The paper develops asymptotic spectral results for nonlinear causal processes.
Section 5 provides the useful generator foundation: a moment contraction
condition admitting a stationary solution, including vector-valued states and
multiple lags. Theorem 5.1 extends the stated Lipschitz condition to moment
orders below one; its proof builds on earlier random-recursion theory.
The benchmark's AST sampling and explanation extraction are project proposals.

All page numbers below refer to the 30-page arXiv electronic reprint,
`math/0611029v2`, whose pagination differs from the journal's pp. 1773-1801.
The local PDF has this note's basename.

# Mathematical extraction

Write the paper's recurrence as

\[
X_{t+1}=R(X_t,\ldots,X_{t-p+1};\varepsilon_{t+1}),\quad X_t\in\mathbb R^d.
\]

| Result | Equation and hypotheses | Evidence | Project use |
|---|---|---|---|
| GMC definition | \(\mathbb E\|X_t-X'_t\|^\alpha\le C\rho^t\), \(0<\rho<1\); independent innovation histories through time 0, identical future innovations | Sec. 3, Eq. (3.1), PDF p. 5 | Report the moment order and the coupling used. |
| Stationary solution and GMC | I.i.d. innovations; measurable, time-homogeneous \(R\); \(R(y_0;\varepsilon)\in L^\alpha\) at some history; global inequality below | Sec. 5, Thm. 5.1, Eqs. (5.1)-(5.3), PDF pp. 11-12 | Essential certificate, including the anchor-moment check. |
| Random Lipschitz factors | \(\sum_j\|H_j(\varepsilon)\|_{L^\alpha}^{\alpha_*}<1\) bounds the coordinate effects | Thm. 5.1, PDF p. 11 | Allows innovation-dependent gains with controlled moments. |
| Stable filtering | GMC input and stable ARMA recursion preserve GMC | Thm. 5.2, PDF p. 12 | Complementary extension for longer memory. |
| Nonlinear examples | EXPar, AR(2)-ARCH(2), bilinear, asymmetric GARCH, signed volatility | Examples 5.1-5.5, PDF pp. 12-15 | Additional operator/noise families require their own moment conditions. |

Precisely, for \(\alpha>0\), \(\alpha_*=\min(1,\alpha)\), and
\(y=(x_1,\ldots,x_p)\), Theorem 5.1 requires, for every pair of histories,

\[
\|R(y;\varepsilon)-R(y';\varepsilon)\|_{L^\alpha}^{\alpha_*}
\le\sum_{j=1}^p a_j\|x_j-x'_j\|_2^{\alpha_*},
\qquad a_j\ge0,\quad \sum_j a_j<1.
\]

The conclusion is a causal strictly stationary solution with the declared
moment and GMC property. The thesis uses the explicit specialization
\(\alpha=2\). Theorem 5.2's roots are transition roots *inside* the unit circle;
Heaps uses the reciprocal lag-polynomial roots *outside* it.

# Proposed AST adaptation and limits

Our first implementation would allocate a global derivative budget across lags,
with additive Gaussian innovations and an explicit finite second-moment check.
For vector outputs, each lag budget bounds a whole derivative block in a common
vector norm; per-output checks alone would miss feedback between variables.

An illustrative project formula is
\(x_{t+1}=0.3\tanh(x_t)\tanh(x_{t-1})+\varepsilon_{t+1}\).
Each absolute partial derivative is at most 0.3, so the budget is 0.6; at the
zero history, the output is the innovation. This verifies the second-moment
specialization for Gaussian innovations. The formula is our illustration.

The proposed vocabulary admits weighted sums, bounded smooth unary functions,
and products whose factors and derivatives have certified bounds. General
polynomial feedback has unbounded global derivatives; it may instead belong
to the current bounded-domain strategy. These sufficient tests select a
restricted family of stationary mechanisms. Burn-in is an approximation to
stationary initialization, requiring an error bound or explicit residual
diagnostic. Gaussian tails also permit crossings of a finite amplitude threshold.

The current AST supports sums and products. Bounded unary operators, global
derivative propagation, and GMC certification remain implementation work.

See [the joint research decision](../../docs/research/stationarity_generator_strategies.md)
for initialization, explanation targets, and the four-method comparison.
