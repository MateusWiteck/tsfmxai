# Coefficient-gated AST generator: pseudocode

This proposal preserves the original random topology and numerical execution.
Its implementation is the `coefficient_gate_lags` branch of
[`random_expression.py`](../../../src/tsfmxai/generators/random_expression.py).

**Reading rule:** black lines are inherited from the original AST generator;
<span style="color:#c62828"><strong>red lines are additions or replacements.</strong></span>

## Mechanism sampling

<pre>
ALGORITHM COEFFICIENT_GATED_AST(config C, root_seed r, sample_index i)
 1. s ← DERIVE_SAMPLE_SEED(r, i)
 2. rng ← LOCAL_RANDOM_GENERATOR(s)
 3. n ← RANDOM_INTEGER(C.minimum_operators, C.maximum_operators)
 4. leaf_kinds ← SAMPLE n + 1 TYPES FROM {PARAMETER, TIME, LAG}
 5. ENFORCE the required terminal types
 6. terminals ← CREATE_TERMINALS(leaf_kinds)
 7. tree ← SAMPLE_FULL_BINARY_TREE_WITH_CATALAN_WEIGHTS(n, terminals)
 8. tree ← CANONICALIZE_AND_RELABEL(tree)
 9. binding ← SAMPLE_PARAMETER_VALUES_AND_INTEGER_LAGS(tree, C, rng)
<span style="color:#c62828;font-weight:700">10. FOR every lag leaf LAG(xj, Lk) in tree:</span>
<span style="color:#c62828;font-weight:700">11.   create a new parameter slot Pg</span>
<span style="color:#c62828;font-weight:700">12.   binding[Pg] ← SAMPLE_SIGNED_MAGNITUDE(C.parameter_range)</span>
<span style="color:#c62828;font-weight:700">13.   REPLACE LAG(xj, Lk) BY MULTIPLY(PARAMETER(Pg), LAG(xj, Lk))</span>
<span style="color:#c62828;font-weight:700">14. tree, binding ← CANONICALIZE_RELABEL_AND_REBIND(tree, binding)</span>
15. RETURN mechanism(tree, binding, seed=s, sample_index=i)
</pre>

The key transformation is

<p style="color:#c62828;font-weight:700;text-align:center">
LAG(x<sub>j</sub>, L<sub>k</sub>) →
MULTIPLY(PARAMETER(P<sub>g</sub>), LAG(x<sub>j</sub>, L<sub>k</sub>)).
</p>

Consequently, a lag no longer enters the recurrence with an implicit gain of
one. Each occurrence receives its own sampled signed coefficient. Independent
coefficients reduce the typical gain but do not constrain their combined
effect through additions and multiplications, so this is a mitigation rather
than a mathematical stability certificate.

## Trajectory sampling

The trajectory algorithm is unchanged: sample an initial history, evaluate the
bound AST recursively, add Gaussian innovation, and stop with
`DivergentTrajectoryError` if a numerical limit is crossed. The proposal acts
on the mechanism before execution; it does not reject trajectories after a
pilot simulation.
