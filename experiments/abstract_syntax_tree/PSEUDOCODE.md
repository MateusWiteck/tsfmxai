# Original AST generator: reference pseudocode

This file is the reference against which the generators in the subdirectories
are compared. The implementation is in
[`random_expression.py`](../../src/tsfmxai/generators/random_expression.py) and
[`trajectory.py`](../../src/tsfmxai/generators/trajectory.py).

The generator separates the symbolic structure from its numerical binding. An
AST contains operators and slots such as `PARAMETER(P0)` and `LAG(x0, L0)`;
the binding stores the numerical value of `P0` and the integer lag represented
by `L0`.

## Mechanism sampling

<pre>
ALGORITHM ORIGINAL_AST_MECHANISM(config C, root_seed r, sample_index i)
 1. s ← DERIVE_SAMPLE_SEED(r, i)
 2. rng ← LOCAL_RANDOM_GENERATOR(s)
 3. n ← RANDOM_INTEGER(C.minimum_operators, C.maximum_operators)
 4. leaf_kinds ← SAMPLE n + 1 TYPES FROM {PARAMETER, TIME, LAG}
                  using the probabilities in C
 5. ENFORCE C.require_lag and C.require_time in leaf_kinds
 6. terminals ← CREATE_TERMINALS(leaf_kinds)
                  LAG also samples a source-series index
 7. tree ← SAMPLE_FULL_BINARY_TREE(n, terminals)
                  choose each split with its Catalan shape weight
                  label each internal node as ADD or MULTIPLY
 8. tree ← CANONICALIZE_COMMUTATIVE_CHILDREN(tree)
 9. tree ← RELABEL_SLOTS(tree, parameters=P0..., lags=L0...)
10. FOR each parameter slot Pj in tree:
      binding[Pj] ← SAMPLE_SIGNED_MAGNITUDE(C.parameter_range)
11. FOR each lag slot Lj in tree:
      binding[Lj] ← RANDOM_INTEGER(C.minimum_lag, C.maximum_lag)
12. RETURN mechanism(tree, binding, seed=s, sample_index=i)
</pre>

The number of binary operators is `n`, so a full binary tree has `n + 1`
leaves. The Catalan weights make every full-binary-tree shape with `n`
operators equally likely, rather than making every root split equally likely.

## Trajectory sampling from one fixed mechanism

<pre>
ALGORITHM ORIGINAL_AST_TRAJECTORIES(mechanism M, trajectory_count K,
                                    generated_steps H, root_seed r)
 1. L ← LARGEST_BOUND_LAG(M)
 2. child_seeds ← SPAWN_K_DETERMINISTIC_SUBSTREAMS(r, K)
 3. FOR each trajectory k = 1, ..., K:
 4.   initialization_seed, innovation_seed ← SPLIT(child_seeds[k])
 5.   history ← SAMPLE_UNIFORM_HISTORY(length=L, initialization_seed)
 6.   FOR t = L, ..., L + H - 1:
 7.     deterministic_value ← EVALUATE_AST(M.tree, M.binding, t, history)
 8.     innovation ← SAMPLE_GAUSSIAN(0, innovation_std, innovation_seed)
 9.     x[t] ← deterministic_value + innovation
10.     IF x[t] is non-finite OR exceeds the configured execution limit:
          RAISE DivergentTrajectoryError
11.     APPEND x[t] TO history
12. RETURN all K trajectories
</pre>

The execution limit detects a trajectory after it becomes numerically unsafe;
it does not constrain the sampled mechanism or provide a stability guarantee.
The subdirectory pseudocodes highlight in red the lines that change this
reference procedure.
