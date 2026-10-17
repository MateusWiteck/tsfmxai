---
title: Guide on Set Invariance for Delay Difference Equations
authors: Mohammed-Tahar Laraba; Sorin Olaru; Silviu-Iulian Niculescu; Franco Blanchini; Giulia Giordano; Daniele Casagrande; Stefano Miani
year: 2016
doi: 10.1016/j.arcontrol.2016.04.020
url: https://doi.org/10.1016/j.arcontrol.2016.04.020
category: explosion_treatment
tags: [delay-difference-equations, augmented-state, invariant-sets]
status: reviewed
---

# Relevance to the thesis

Grounds the representation of a finite-lag recurrence as a first-order system in
an augmented state and the requirement that the chosen state domain remain
invariant under recursion.

# Method and main result

For linear delay-difference equations, the paper presents two complementary
views: stack delayed values into an extended state, or reason directly with a
set containing every delayed value. It reviews necessary and sufficient
conditions and constructions for delay-invariant sets.

# Relation to our approach

The AST generator adopts the extended lag state but applies direct interval
bounds to a nonlinear transition. A robust version of the inclusion test also
accounts for every innovation inside a bounded noise set.

# Limitations

The detailed algebraic conditions in the paper concern linear delay-difference
equations. The benchmark's nonlinear AST certificate uses the same state and
invariance concepts with a different verification procedure.

# Important evidence

- PDF p. 2 (journal p. 14): construction of the extended state and companion
  state-transition matrix.
- PDF p. 3 (journal p. 15): definitions of positive invariance, contraction, and
  delay invariance.
- PDF p. 4 (journal p. 16): properties and conditions for delay-invariant sets.
