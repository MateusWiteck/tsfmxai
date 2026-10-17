---
title: Contraction Analysis of Discrete-Time Stochastic Systems
authors: Yu Kawano; Yohei Hosoe
year: 2024
doi: 10.1109/TAC.2023.3283678
url: https://doi.org/10.1109/TAC.2023.3283678
category: explosion_treatment
tags: [discrete-time, stochastic-systems, contraction, jacobian]
status: reviewed
---

# Relevance to the thesis

Supports applying contraction reasoning when every recursive step also receives
a random innovation.

# Method and main result

The paper defines uniform incremental exponential stability in the first and
second moments for discrete-time stochastic nonlinear systems. It derives
Jacobian-based conditions in Riemannian and Euclidean metrics and specializes
them to independent and identically distributed and Markov processes.

# Relation to our approach

The current generator imposes a stronger uniform pathwise condition for every
allowed innovation. Two trajectories coupled by the same innovation sequence
therefore forget their initial histories. A separate invariant-domain test
keeps all bounded-noise transitions inside the certified region.

# Limitations

The paper's moment conditions do not directly provide the AST interval bounds
or the particular coefficient-rescaling rule implemented by the generator.

# Important evidence

- PDF p. 3: stochastic variational system and definition of moment uniform
  incremental exponential stability.
- PDF pp. 3--4: Theorems 4.1 and 4.3 give Jacobian/metric conditions for first-
  and second-moment incremental stability.
- PDF p. 5: specialization to temporally independent and i.i.d. processes.
