---
title: Convergence Properties for Discrete-Time Nonlinear Systems
authors: Duc N. Tran; Björn S. Rüffer; Christopher M. Kellett
year: 2019
doi: 10.1109/TAC.2018.2879951
url: https://doi.org/10.1109/TAC.2018.2879951
category: explosion_treatment
tags: [discrete-time, contraction, incremental-stability, jacobian]
status: reviewed
---

# Relevance to the thesis

Provides the discrete-time nonlinear contraction result used to interpret a
uniform Jacobian-norm bound as exponential loss of dependence on initial state.

# Method and main result

The paper compares incremental stability, convergent dynamics, and contraction
for differentiable discrete-time systems. Contraction is expressed through the
Jacobian of the state transition in a bounded state-dependent metric. For smooth
dynamics it is equivalent to uniform exponential incremental stability.

# Relation to our approach

The thesis uses a more restrictive sufficient case: a single induced norm and a
uniform Jacobian bound below one on a certified domain. The paper also makes the
positive-invariance assumption explicit, motivating a separate domain check.

# Limitations

The paper develops a general stability theory rather than an AST sampler or a
synthetic time-series generator. The weighted infinity-norm certificate and
interval implementation are specializations made in this thesis.

# Important evidence

- PDF pp. 5--6: definitions of exponential incremental stability and contraction,
  including the transformed Jacobian.
- PDF p. 11: Theorem 15 equates global contraction with global exponential
  incremental stability for continuously differentiable dynamics.
- PDF p. 12: Theorem 16 connects contraction on a compact positively invariant
  set with exponential convergence.
