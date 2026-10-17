# Contraction-certified AST generator: pseudocode

This proposal retains nonlinear `ADD`/`MULTIPLY` lagged ASTs, then scales and
certifies them over a bounded augmented-state domain. The implementation is in
[`contraction.py`](../../../../src/tsfmxai/generators/contraction.py).

**Reading rule:** black lines come from the original AST workflow;
<span style="color:#c62828"><strong>red lines implement coefficient gating,
the contraction certificate, invariant-domain control, or bounded-noise
execution.</strong></span>

## Certified mechanism sampling

<pre>
ALGORITHM CONTRACTION_CERTIFIED_AST(config C, root_seed r, sample_index i)
 1. s ← DERIVE_SAMPLE_SEED(r, i)
 2. rng ← LOCAL_RANDOM_GENERATOR(s)
 3. base_tree ← SAMPLE_ORIGINAL_FULL_BINARY_AST(C.tree_config, rng)
 4. base_binding ← SAMPLE_PARAMETER_VALUES_AND_INTEGER_LAGS(base_tree, C, rng)
<span style="color:#c62828;font-weight:700"> 5. IF C.tree_config.coefficient_gate_lags, FOR every LAG leaf:</span>
<span style="color:#c62828;font-weight:700">      REPLACE LAG(x0, Lk) BY MULTIPLY(PARAMETER(Pg), LAG(x0, Lk))</span>
<span style="color:#c62828;font-weight:700">      and sample the signed gate Pg</span>
<span style="color:#c62828;font-weight:700"> 6. REQUIRE a univariate tree containing no TIME terminal</span>
<span style="color:#c62828;font-weight:700"> 7. L ← LARGEST_BOUND_LAG(base_tree)</span>
<span style="color:#c62828;font-weight:700"> 8. D ← [-C.domain_radius, C.domain_radius]^L</span>
<span style="color:#c62828;font-weight:700"> 9. value_bound, derivative_bounds ← INTERVAL_PROPAGATION(base_tree, D)</span>
<span style="color:#c62828;font-weight:700">10. weights[j] ← C.lag_weight_growth^j,  j = 0, ..., L - 1</span>
<span style="color:#c62828;font-weight:700">11. base_top_gain ← Σ[j] derivative_bounds[j] × weights[j]</span>
<span style="color:#c62828;font-weight:700">12. contraction_scale_limit ← C.target_lipschitz / base_top_gain</span>
<span style="color:#c62828;font-weight:700">13. domain_scale_limit ← (C.domain_radius - C.innovation_bound)</span>
<span style="color:#c62828;font-weight:700">                           / value_bound.maximum_absolute_value</span>
<span style="color:#c62828;font-weight:700">14. scale ← C.scale_safety_factor ×</span>
<span style="color:#c62828;font-weight:700">            MIN(1, contraction_scale_limit, domain_scale_limit)</span>
<span style="color:#c62828;font-weight:700">15. tree ← MULTIPLY(PARAMETER(CONTRACTION_SCALE), base_tree)</span>
<span style="color:#c62828;font-weight:700">16. certificate ← CERTIFY(tree, binding, D, innovation_bound, weights)</span>
<span style="color:#c62828;font-weight:700">17. REQUIRE certificate.Lipschitz_bound &lt; 1</span>
<span style="color:#c62828;font-weight:700">18. REQUIRE certificate.output_bound + innovation_bound ≤ domain_radius</span>
19. RETURN mechanism(tree, binding, scale, certificate, seed=s, sample_index=i)
</pre>

The augmented state is
`s(t) = [x(t), x(t-1), ..., x(t-L+1)]`. Its transition first computes the new
AST output and then shifts the remaining lag coordinates. With weights
`w[j] = q^j`, the certificate bounds the first Jacobian row by the weighted
derivative sum and the shift rows by `1/q`. The reported Lipschitz upper bound
is the maximum of these two values.

## Certified trajectory sampling

<pre>
ALGORITHM CONTRACTION_CERTIFIED_TRAJECTORIES(mechanism M, K, H, root_seed r)
 1. child_seeds ← SPAWN_K_DETERMINISTIC_SUBSTREAMS(r, K)
 2. FOR each trajectory k = 1, ..., K:
<span style="color:#c62828;font-weight:700"> 3.   history ← SAMPLE_UNIFORM([-M.domain_radius, M.domain_radius], length=L)</span>
 4.   FOR each future step t = 1, ..., H:
 5.     deterministic_value ← EVALUATE_AST(M.tree, M.binding, t, history)
<span style="color:#c62828;font-weight:700"> 6.     innovation ← SAMPLE_UNIFORM(-M.innovation_bound, M.innovation_bound)</span>
 7.     x[t] ← deterministic_value + innovation
<span style="color:#c62828;font-weight:700"> 8.     REQUIRE |x[t]| ≤ M.domain_radius; otherwise raise an internal error</span>
 9.     APPEND x[t] TO history
10. RETURN all K trajectories
</pre>

Bounded innovation is part of the guarantee: Gaussian noise has unbounded
support and therefore cannot provide deterministic invariance of a finite
domain. Monte Carlo validation additionally evaluates local Jacobian norms on
sampled states, but it complements rather than creates the analytic
certificate.
