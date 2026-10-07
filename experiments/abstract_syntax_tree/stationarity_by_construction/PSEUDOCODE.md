# Stationary-by-construction generator: pseudocode

This proposal replaces unrestricted random AST topology with linear AR or
VAR(1) equations sampled from parameterizations that satisfy their
stationarity conditions. The implementation is in
[`stationary.py`](../../../src/tsfmxai/generators/stationary.py).

**Reading rule:** black lines are shared infrastructure;
<span style="color:#c62828"><strong>red lines replace the original AST sampling
or trajectory initialization.</strong></span>

## Univariate AR mechanism

<pre>
ALGORITHM STATIONARY_AR(config C, root_seed r, sample_index i)
 1. s ← DERIVE_SAMPLE_SEED(r, i)
 2. rng ← LOCAL_RANDOM_GENERATOR(s)
<span style="color:#c62828;font-weight:700"> 3. p ← RANDOM_INTEGER(C.minimum_order, C.maximum_order)</span>
<span style="color:#c62828;font-weight:700"> 4. κ[1:p] ← SAMPLE_UNIFORM(C.pacf_minimum, C.pacf_maximum)</span>
<span style="color:#c62828;font-weight:700">                 with every |κ[l]| &lt; 1</span>
<span style="color:#c62828;font-weight:700"> 5. φ[1:p] ← LEVINSON_RECURSION_FROM_PACF(κ)</span>
<span style="color:#c62828;font-weight:700"> 6. roots ← ROOTS_OF(1 - Σ[l=1..p] φ[l] z^l)</span>
<span style="color:#c62828;font-weight:700"> 7. REQUIRE minimum(|roots|) &gt; 1; store the stationarity certificate</span>
<span style="color:#c62828;font-weight:700"> 8. μ ← SAMPLE_STATIONARY_MEAN(C); σ ← SAMPLE_INNOVATION_STD(C)</span>
<span style="color:#c62828;font-weight:700"> 9. intercept ← μ (1 - Σ[l=1..p] φ[l])</span>
<span style="color:#c62828;font-weight:700">10. tree ← AST(intercept + Σ[l=1..p] φ[l] LAG(x0, l))</span>
<span style="color:#c62828;font-weight:700">11. F ← AR_COMPANION_MATRIX(φ)</span>
<span style="color:#c62828;font-weight:700">12. P ← SOLVE_DISCRETE_LYAPUNOV(P = F P Fᵀ + Q)</span>
13. tree ← CANONICALIZE_AND_RELABEL(tree)
14. RETURN mechanism(tree, binding, stationary_covariance=P, certificate)
</pre>

## Multivariate VAR(1) mechanism

<pre>
ALGORITHM STATIONARY_VAR1(config C, root_seed r, sample_index i)
 1. s ← DERIVE_SAMPLE_SEED(r, i)
 2. rng ← LOCAL_RANDOM_GENERATOR(s)
<span style="color:#c62828;font-weight:700"> 3. A_raw ← SAMPLE_GAUSSIAN_MATRIX(C.num_series, C.num_series)</span>
<span style="color:#c62828;font-weight:700"> 4. APPLY_RANDOM_SPARSITY_MASK(A_raw, C.edge_probability)</span>
<span style="color:#c62828;font-weight:700"> 5. ρ_target ← SAMPLE_UNIFORM(C.spectral_radius_minimum,</span>
<span style="color:#c62828;font-weight:700">                              C.spectral_radius_maximum)</span>
<span style="color:#c62828;font-weight:700"> 6. A ← A_raw × ρ_target / SPECTRAL_RADIUS(A_raw)</span>
<span style="color:#c62828;font-weight:700"> 7. REQUIRE SPECTRAL_RADIUS(A) &lt; 1; store the stationarity certificate</span>
<span style="color:#c62828;font-weight:700"> 8. SAMPLE stationary mean μ and diagonal innovation covariance Q</span>
<span style="color:#c62828;font-weight:700"> 9. FOR target series k:</span>
<span style="color:#c62828;font-weight:700">      tree[k] ← AST(((I - A)μ)[k] + Σ[j] A[k,j] LAG(xj, 1))</span>
<span style="color:#c62828;font-weight:700">10. P ← SOLVE_DISCRETE_LYAPUNOV(P = A P Aᵀ + Q)</span>
11. CANONICALIZE_AND_RELABEL every tree
12. RETURN mechanism(trees, bindings, stationary_covariance=P, certificate)
</pre>

## Trajectory sampling

<pre>
ALGORITHM STATIONARY_TRAJECTORIES(mechanism M, K, H, root_seed r)
 1. child_seeds ← SPAWN_K_DETERMINISTIC_SUBSTREAMS(r, K)
 2. FOR each trajectory k = 1, ..., K:
<span style="color:#c62828;font-weight:700"> 3.   initial_state ← SAMPLE_MULTIVARIATE_NORMAL(M.stationary_mean,</span>
<span style="color:#c62828;font-weight:700">                                                M.stationary_covariance)</span>
 4.   FOR each future step t = 1, ..., H:
 5.     deterministic_value ← EVALUATE_LINEAR_AST(M, history)</n+<span style="color:#c62828;font-weight:700"> 6.     innovation ← SAMPLE_GAUSSIAN(0, M.innovation_covariance)</span>
 7.     APPEND deterministic_value + innovation TO history
 8. RETURN all K trajectories
</pre>

The exact stationary initialization removes the arbitrary burn-in used by the
original AST trajectory sampler. For the implemented linear Gaussian families,
the marginal distribution is therefore stationary from the first returned
sample.
