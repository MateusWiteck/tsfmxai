# Four generator strategies: research decision

Reviewed on 2026-10-11. This record accompanies report Section 1.4 and the
paper-specific notes in `references/explosion_treatment/`. It defines the
comparison and distinguishes implemented specializations from future extensions.

## Current implementation

| Component | Observed source | Status |
|---|---|---|
| Random AST topology, parameter/lag binding, deterministic indexed seeds | `src/tsfmxai/generators/random_expression.py` | Implemented; current operators are addition/multiplication. |
| AST representation and evaluation | `src/tsfmxai/representations/expression.py`, `src/tsfmxai/generators/trajectory.py` | Reusable code, including TANH formatting, binding validation, execution and derivatives. |
| Bounded-domain contraction | `src/tsfmxai/generators/contraction.py` | Implemented for one series; interval bounds, weighted augmented-state norm, output scaling, uniform bounded innovations, and a Monte Carlo diagnostic. |
| Gaussian stationary linear mechanisms | `src/tsfmxai/generators/stationary.py` | Scalar AR through PACF and VAR(1) through spectral rescaling; reusable initialization/covariance infrastructure. |
| Shao-Wu global/moment certificate | `src/tsfmxai/generators/shao_wu.py` | Implemented second-moment specialization with globally bounded TANH interactions, Gaussian innovations and a certified L2 burn-in tolerance. |
| Heaps symmetric VAR(p) reconstruction | `src/tsfmxai/generators/heaps.py` | Implemented symmetric recursions, retained linear ASTs, independent covariance checks and joint stationary Gaussian initialization. |

The curated AST directory includes `contraction_by_construction/`,
`shao_wu_gmc/` and `heaps_stationary_var/`. Their complete experiment records are
inside executed notebooks. Notebook 6 reruns 5,000 univariate mechanisms per
method using one shared AST execution backend and native initialization/noise
laws; the new method-specific notebooks also evaluate two-variable systems.
Notebook 7 shows the four current algorithms. Previous scalar stationary AR and
coefficient-gating alternatives remain available as library functionality;
their separate rows have been replaced in the current comparison.

## Decision matrix

| Method | Mathematical claim | Diversity and interactions | Sparsity / GT | Incremental implementation cost |
|---|---|---|---|---|
| Original random AST | Empirical baseline; each sampled recurrence has its own dynamical properties | Broad sums/products, lags and optional time terms; multivariate evaluation infrastructure exists | Exact retained formulas; simplify cancellations before functional-support labels | Baseline available |
| Current Jacobian method | Uniform common-noise contraction on an invariant box; bounded trajectories for the stored noise law | Nonlinear sums/products restricted to a certified box; scalar implementation | Topology retained under nonzero global scaling, with final coefficients retained | Available; multivariate block certificate would be additional work |
| Shao-Wu adaptation | Causal strict stationarity and GMC under Theorem 5.1; implemented at moment order 2 | Globally bounded TANH interactions, Gaussian innovations, scalar/vector states | Direct variable-lag selection with lag-block bounds accounting for cross-variable feedback | Available; each retained history has an explicit finite-burn-in L2 error bound |
| Heaps adaptation | Stationary causal Gaussian VAR(p) with joint invariant initialization | Linear interactions between many variables and lags, potentially oscillatory dynamics | Signed final coefficients; intermediate sparsity generally changes during reconstruction | Available; matrix recursions and cached joint covariance factor |

The last two are adaptations of published mathematics to this benchmark.
Shao-Wu and the current generator share contraction theory. Their experimental
distinction is global/moment certification with admissible unbounded noise
versus pathwise certification and invariance on a bounded box. Neither family
is claimed to contain the other for all operator/noise choices.

## Implemented pipelines and extension points

### Shao-Wu

1. Fix moment order 2 initially; sample lag/source support and a globally
   Lipschitz expression using weighted sums and bounded nonlinear factors.
2. Retain global derivative bounds at each AST node. For multiple variables,
   propagate operator norms for each lag block using one declared state norm.
3. Scale/allocate lag gains to meet the coordinate budget, and verify the
   finite-moment condition at a reference history. Store the certificate and
   innovation law with the mechanism. State-dependent innovation amplitudes
   need additional moment bounds and are a later extension.
4. Generate independent trajectories using per-mechanism/per-trajectory seeds.
   Use exact invariant sampling when available; otherwise declare a finite
   burn-in and bound its initialization error. Common-noise coupling is an
   additional empirical diagnostic.
5. Retain the final expression, numerical parameters, state and innovations for
   compatible explanation references. Promote reusable code under `src/tsfmxai/`
   with tests for analytic bounds, seeds, domain handling and propagation.

For the tested example `0.3*tanh(x[t-1])*tanh(x[t-2]) + epsilon[t]`, both
absolute derivative bounds are 0.3. Their sum is 0.6; at a zero reference history
the output equals the innovation. This verifies Theorem 5.1 at moment order 2
with Gaussian noise. The globally bounded nonlinear drift also simplifies
moment control. This example is a project derivation, independent of simulation.

### Heaps

1. Choose series count/order and distributions for unconstrained matrices and
   a positive-definite innovation covariance. Keep these choices in provenance:
   they determine the mechanism distribution.
2. Map singular values into the unit interval using symmetric square roots.
3. Follow the Appendix's backward covariance recursion and forward/backward
   coefficient recursions; independently check the companion radius and matrix
   residuals. Replacing noncommuting matrix products by scalar-style shortcuts
   would change the mapping.
4. Construct the augmented-state stationary covariance and draw its joint
   Gaussian history, independently of future innovations. Preserve the sampled
   final coefficients as the executed ASTs.
5. Read source-lag support from final coefficients. Re-certify every imposed
   structural modification. Expose the method through the same mechanism and
   trajectory interfaces used by other samplers.

Measured costs separate mechanism sampling/certification, initialization,
innovations, AST execution and diagnostics. Heaps's implementation obtains the
joint covariance directly from the reconstructed autocovariances and caches
its factor. Independent Lyapunov solvers validate that covariance in tests;
the benchmark avoids solving a Kronecker system per trajectory. Larger
multivariate and parallel-throughput studies remain further experiments.

## Explanation and mathematical validation contract

Keep four objects explicit: the retained generating mechanism, the statistical
dependence of its observations, the forecasting function learned from data,
and the explanation returned for that learned function.

- Functional variable-lag support comes from the final equations, accounting
  for exact zero coefficients and cancellations. A directed support graph needs
  additional assumptions for an intervention-based causal interpretation.
- Signed coefficients are directly available for linear terms. Realized term
  contributions additionally depend on the input values.
- Jacobians give local sensitivity. At later horizons, differentiate the whole
  recursive unrolling, including all paths through previously generated values.
- Declare whether innovations are fixed (realized trajectory) or integrated
  out (an expected forecast). For nonlinear recurrences, recursively setting
  innovations to zero can differ from the multi-step conditional expectation.
- SHAP also requires its coalition definition and background/resampling law.
  Coefficients, derivatives and Shapley contributions answer different questions.

Validation has three levels: verify theorem hypotheses; check the numerical
implementation against analytic examples and residuals; then use simulations
to measure behaviour and robustness. Stationarity describes time-invariant
joint laws (and finite, time-invariant covariances when second moments exist).
An invariant law plus arbitrary initialization gives a transient; drawing the
initial state from that law gives stationarity from the beginning.

An explosion threshold and near-zero threshold are operational metrics. Their
values, windows, and denominators must be stated. Finite samples below those
thresholds supply empirical evidence. Gaussian stationary processes can exceed
a fixed amplitude threshold, so report threshold crossings separately from
nonfinite numerical outputs and long-term growth diagnostics.

## Up to three priority experiments

1. **Guarantees and initialization.** Use analytic scalar and multivariate
   reference cases, identical-innovation couplings, and varied initial histories.
   Hypothesis: the certified decay and covariance relations hold; finite burn-in
   leaves a quantifiable transient. Measure certificate residuals, decay ratios,
   initial-to-late distribution differences, explosion rate and near-zero rate.
2. **Diversity versus computational cost.** Compare the four methods across
   variable counts, lag orders and expression complexity. Match dimensions,
   lengths, number of trajectories, seeds and innovation variance where possible;
   report differences in noise support and mechanism family. Measure proposal
   acceptance, retained functional support, coefficient/persistence distribution,
   explosion/near-zero rates and sampling, certification, initialization and
   simulation time separately. Report total wall time, trajectories/s and
   values/s, then repeat matched work with CPU workers and feasible GPU batches.
3. **Usefulness of the explanation references.** On matched linear and bounded
   nonlinear subsets, compare generator derivatives with finite differences and
   linear multi-step effects with companion-matrix powers. Evaluate a fitted
   forecaster and a compatible explainer separately: forecast error, support
   precision/recall, coefficient error when explicitly recoverable, and local
   sensitivity/attribution agreement with the matching target. Stratify by
   operations, interaction order and horizon.

## Thesis scope and indispensable references

**Essential:** complete certificate assumptions, vector/lag interpretation,
innovation law, initialization, strict versus second-order stationarity, final
coefficient sparsity, and the distinction between published theorems and our
generator/ground-truth adaptation. These appear in the paper-specific report
files and the four-strategy synthesis.

**Complementary:** Shao-Wu's stable ARMA filtering and heteroscedastic examples;
Heaps's exchangeable priors and their influence on the sampling distribution.

**Outside the initial implementation:** reproducing spectral-bootstrap theory,
Bayesian posterior inference/HMC, and an unrestricted stationarity decision
procedure for arbitrary nonlinear ASTs.

The existing Tran/Kawano/Diaconis/Laraba references ground the current method.
Ansley and Kohn (1986) is the necessary additional attribution for the earlier
VAR reparameterization; Heaps explains the precise extension. No further
mathematical citation is required for the proposed two initial specializations.
