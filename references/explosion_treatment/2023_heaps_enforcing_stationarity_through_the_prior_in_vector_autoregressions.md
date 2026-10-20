---
title: Enforcing Stationarity through the Prior in Vector Autoregressions
authors: Sarah E. Heaps
year: 2023
doi: 10.1080/10618600.2022.2079648
url: https://arxiv.org/pdf/2004.09455
category: explosion_treatment
tags: [stationarity, VAR, partial-autocorrelation, reparameterization]
status: reviewed
---

# Scientific summary

Heaps develops a stationary VAR parameterization and Bayesian priors with
interpretable symmetry/shrinkage properties. Earlier unconstrained mappings
are attributed to Ansley and Kohn (1986). Symmetric square roots preserve
singular-vector orientation and support orthogonally compatible priors; a
new backward covariance recursion makes the inverse mapping possible.

The benchmark samples these parameters, reconstructs the equations, and
simulates them. The generator is a project adaptation of the published
parameterization; explanation extraction remains a separate benchmark task.
Page numbers refer to the 15-page arXiv version `2004.09455v3` (May 2022),
associated with the 2023 journal article, 32(1), 74-83. The local PDF has this
note's basename.

# Mathematical extraction

| Result | Equation and hypotheses | Evidence | Project use |
|---|---|---|---|
| Causal stationary VAR | \(x_t=\sum_{\ell=1}^p C_\ell x_{t-\ell}+\varepsilon_t\), Gaussian white innovations with covariance \(\Sigma_\varepsilon\succ0\); roots of \(\det(I-\sum C_\ell z^\ell)=0\) outside unit circle | Introduction, PDF pp. 1-2 | Equivalent companion condition \(\rho(B)<1\). |
| Partial-autocorrelation coordinates | All singular values of each \(P_\ell\) below 1; covariance retained | Sec. 2.1, PDF pp. 3-4 | Encodes the stationary parameter region. |
| Unconstrained coordinates | \(P_\ell=(I+U_\ell U_\ell^\top)^{-1/2}U_\ell\), symmetric root | Sec. 2.2, Eqs. (2)-(3), PDF pp. 4-5 | Draw free matrices then map singular values \(u\mapsto u/\sqrt{1+u^2}\). |
| Reconstruction | Backward recursion computes marginal stationary covariance; forward/backward prediction-error recursions recover \(C_\ell\) | Appendix, reverse mapping, PDF p. 12 | Essential implementation steps. |
| Sparsity and prior effects | Intermediate zeros have a different meaning from final coefficient zeros; prior scale changes partial-autocorrelation distributions | Secs. 3.3-3.4, p. 6; Sec. 3.6, p. 7 | Measure final support and persistence after reconstruction. |
| Joint stationary history | Initial observations have a stationary joint Gaussian distribution; companion covariance follows a discrete Lyapunov equation | Sec. 4, PDF p. 8; Appendix, p. 11 | Sample the entire correlated lag history. |

# Implemented adaptation and limits

The implementation draws \(U_1,\ldots,U_p\) and positive-definite
\(\Sigma_\varepsilon\). It maps to partial autocorrelations, uses the Appendix's
two recursions, verifies the companion radius, and converts the final
coefficients to one linear AST per target. It preserves the specified symmetric
square-root convention and matrix multiplication order.

For the full lag state, compute
\(V=BVB^\top+Q\), where \(Q\) inserts the innovation covariance in the newest
state block. Draw the joint Gaussian history with covariance \(V\), independently
of future innovations. This gives exact stationary initialization in real
arithmetic, with numerical tolerances recorded in the implementation.

Final coefficients yield signed variable-lag effects. Record structural support
from those coefficients, with a declared numerical threshold for floating-point
zeros. Arbitrarily masking final coefficients requires renewed certification.
Sparse sampled intermediate matrices alone provide insufficient control over
the final interaction graph. Coordinatewise nonlinear products belong to the
separate nonlinear strategy; VAR represents linear cross-series interactions.

Alongside the earlier scalar AR-PACF and VAR(1) samplers in `stationary.py`,
`src/tsfmxai/generators/heaps.py` implements the symmetric VAR(p) inverse
mapping and caches a factor of the joint stationary covariance. Trajectories
start from that Gaussian history and immediately execute the recurrence;
no preliminary path is simulated or discarded. Scaling the covariance
calculations to larger lag-state dimensions remains future work.

Essential prior reference: Ansley and Kohn (1986), *A Note on Reparameterizing a
Vector Autoregressive Moving Average Model to Enforce Stationarity*,
[DOI](https://doi.org/10.1080/00949658608810893). Metadata and scope were checked
against the publisher abstract; detailed attribution follows Heaps, Secs. 2.1-2.2
and Appendix. Heaps's contribution builds on that earlier mapping.

See [the joint research decision](../../docs/research/stationarity_generator_strategies.md)
for the implemented design and evaluation protocol.
