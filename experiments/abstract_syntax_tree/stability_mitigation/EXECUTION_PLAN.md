# Execution plan: reducing explosive and near-zero AST trajectories

## 1. Objective

Develop and compare alternative sampling strategies for the recurrent AST
generator. Each strategy will have one executed notebook. The final comparison
will identify the Pareto trade-off among:

1. explosion frequency;
2. near-zero-tail frequency;
3. generation speed per accepted mechanism;
4. retained structural diversity.

The objective is not merely to suppress large values. A sampler that avoids
explosions by producing mostly trivial or near-zero series is also undesirable.

## 2. Baseline that must remain unchanged

Notebook `../4_large_scale_stability_and_equivalence.ipynb` is the frozen
baseline. At horizon 192 it generated the following results for 5,000 ASTs:

| Initial-value bound | Explosions | Near-zero tails |
|---:|---:|---:|
| 0.0 | 2,366 (47.32%) | 563 (11.26%) |
| 0.1 | 2,369 (47.38%) | 563 (11.26%) |
| 0.5 | 2,401 (48.02%) | 562 (11.24%) |
| 1.0 | 2,487 (49.74%) | 522 (10.44%) |
| 2.0 | 2,713 (54.26%) | 384 (7.68%) |

The evidence in that notebook indicates that bare lag leaves are the first
mechanism to address. A term such as `LAG(0, L0)` has implicit coefficient one,
so additions of several bare lags can create autoregressive gain greater than
one. Random multiplication often introduces an explicit parameter with
magnitude below one, which explains why multiplication count alone did not
predict explosion risk as originally expected.

The executed preliminary experiment
`00_lag_count_explosion_evidence.ipynb` stratifies this baseline by the number
of `Lag` nodes. At horizon 192 and zero initial input, the observed explosion
rate rises from 3.84% with one lag to 68.22% with two lags. This is strong
associational evidence, while the planned paired coefficient-gating experiment
is required to isolate the effect from tree size and operator composition.

The baseline notebook and its recorded results must not be edited. New methods
will reuse its evaluation protocol through common library code.

## 3. Locked evaluation protocol

Every method must be evaluated with the same mechanism count, trajectory
seeds, initial-value panels, innovations and horizons. The required settings
are:

- 5,000 accepted mechanisms;
- horizons 96, 192 and 384;
- initial absolute-value limits 0, 0.1, 0.5, 1 and 2;
- Gaussian innovation standard deviation 0.05;
- explosion when a value is non-finite or has absolute value above 100;
- near-zero tail when all final 24 values remain within three innovation
  standard deviations of zero, namely `[-0.15, 0.15]`;
- separate mechanism, screening and evaluation seeds;
- the same evaluation innovations and initial values for paired mechanisms;
- Wilson 95% intervals for explosion and near-zero rates.

Screening data and evaluation data must use different seeds. Otherwise a
rejection method would be evaluated on the same trajectories used to accept
its mechanisms and its result would be optimistically biased.

### Primary outcome metrics

For every method, horizon and initial-value bound, record:

- explosion count and rate;
- near-zero count and rate over all mechanisms;
- near-zero rate among non-exploded mechanisms;
- neither-exploded-nor-near-zero count and rate;
- median explosion step and survival curve;
- Wilson 95% intervals.

The phrase *near-zero* denotes the finite-window rule above. It does not prove
asymptotic convergence.

### Generation-speed metrics

Timing must separate the following stages:

1. candidate AST sampling;
2. coefficient transformation or stability projection;
3. pilot screening, when present;
4. accepted trajectory generation;
5. complete wall-clock time.

Report:

- proposed mechanisms per second;
- accepted mechanisms per second;
- generated time points per second;
- median and 95th-percentile latency per accepted mechanism;
- proposal-to-acceptance ratio;
- time spent in screening or projection;
- peak resident memory when practical.

The mandatory comparison is single-process CPU execution, after one discarded
warm-up and over at least five timed repetitions. A secondary scaling test may
use 1, 2, 4 and 8 workers. GPU or vectorized-batch generation should be treated
as a separate implementation experiment so that algorithmic and backend
speedups are not mixed.

### Diversity safeguards

Explosion and near-zero rates can be improved artificially by generating only
simple noise-like or constant processes. Therefore every method must also
report:

- unique canonical skeleton fraction;
- operator-count and tree-depth distributions;
- lag coverage;
- addition and multiplication counts;
- coefficient-magnitude distribution;
- trajectory variance and lag-one autocorrelation;
- fraction of generated mechanisms removed or altered by the method.

These are diagnostic constraints rather than the main optimization metrics.

## 4. Proposed methods and notebooks

Reusable implementations belong in `src/tsfmxai/`; notebooks only configure,
execute and explain each experiment.

### 01 — Configuration-only risk reduction

Planned notebook: `01_configuration_only.ipynb`

Change only the current `RandomTreeConfig`: test smaller maximum tree sizes,
lower addition probability, higher parameter probability and narrower
coefficient magnitudes. This establishes how far the present implementation
can improve without changing the AST grammar.

Expected cost: negligible. Expected limitation: naked lags still have
coefficient one, so this method cannot directly control total autoregressive
gain. It may also reduce structural complexity instead of solving the cause.

### 02 — Coefficient-gated lag leaves

Implemented notebook: `02_coefficient_gated_lags.ipynb`

Replace each sampled bare lag atom with a coefficient-bearing atom:

```text
LAG(j, l)  ->  MULTIPLY(P, LAG(j, l))
```

The topology and numeric bindings remain separate, but every temporal
dependency receives an explicit sampled strength. Constants should be sampled
as intercept terms rather than being accidentally mixed with lag coefficients.

Expected cost: very small. Expected benefit: removal of the implicit unit-gain
path identified in Notebook 4. Expected limitation: the sum of several
explicit coefficients can still exceed one.

Executed result at horizon 192 and zero initial input: explosion frequency
fell from 47.32% to 8.68%, while the near-zero-tail rate rose from 11.26% to
18.40%. Sampling was about 17.6% slower in this run. Complete simulation took
more wall time because the gated mechanisms both contain more AST nodes and
survive for substantially more evaluated steps.

### 03 — L1 dynamic-gain normalization

Planned notebook: `03_l1_gain_normalization.ipynb`

For expressions that are affine in lagged variables, extract the effective lag
coefficients and rescale them to a sampled budget `rho < 1`:

```text
a_k <- rho * a_k / max(sum_j |a_j|, epsilon)
```

This gives a cheap sufficient contraction condition for the linear additive
part. Evaluate several `rho` ranges close to one; very small values may reduce
explosions while increasing near-zero tails.

Expected cost: linear in the number of lag terms. Expected limitation: the L1
condition is conservative and does not certify arbitrary multiplicative
nonlinear terms.

### 04 — Spectral-radius projection

Planned notebook: `04_spectral_radius_projection.ipynb`

For linear AR mechanisms, build the companion matrix and project unstable
coefficient vectors to a target spectral radius below one. Use deterministic
iterative shrinkage with explicit verification of the resulting radius; an
optional safeguarded search can recover part of the last shrinkage interval.
Do not assume that the radius scales linearly or monotonically with all AR
coefficients. The multivariate extension will use the VAR companion matrix.

Expected cost: eigenvalue computation plus projection iterations. Expected
benefit: less conservative mechanisms than L1 normalization. Expected
limitation: applies only to the linear subset of the present grammar.

### 05 — Held-out pilot rejection filter

Planned notebook: `05_held_out_pilot_filter.ipynb`

Sample with the existing generator, simulate each candidate on a small stress
panel, and reject candidates that explode or collapse near zero. Record every
proposal and rejection reason. Evaluate accepted mechanisms with separate
initial conditions and innovation seeds.

Expected benefit: works with nonlinear trees without deriving a certificate.
Expected cost: potentially high because many candidates require simulations
before rejection. Expected limitation: empirical filtering does not guarantee
behavior outside the pilot panel and can distort the mechanism distribution.

### 06 — Bounded nonlinear atoms

Planned notebook: `06_bounded_nonlinear_atoms.ipynb`

Test a more structured nonlinear grammar in which lagged inputs first enter
bounded atoms and the outer dynamic weights obey a gain budget. Avoid wrapping
an unrestricted lag product in a bounded output without checking its
derivative; bounded output alone does not imply a bounded Jacobian. A candidate
design is a weighted sum of `tanh`-transformed linear lag combinations and
products of already bounded atoms.

Expected benefit: retains nonlinear behavior while making amplitude control
meaningful. Expected cost: new AST operators, derivatives, serialization and
tests. This is the most invasive individual method and should follow the
linear experiments.

### 07 — Hybrid sampler

Planned notebook: `07_hybrid_sampler.ipynb`

Combine coefficient-gated lag atoms with L1 or spectral projection when
applicable, followed by a small held-out pilot filter for the remaining
nonlinear cases. The pilot should be lighter than in Method 05 because most
obvious failures have already been removed structurally.

Expected outcome: the strongest explosion reduction with better accepted
throughput than pure rejection. The experiment must retain ablation labels so
the contribution and runtime of each stage remain visible.

### 08 — Final Pareto comparison

Planned notebook: `08_stability_speed_tradeoff.ipynb`

Aggregate the executed records of the baseline and Methods 01–07. Show:

- explosion rate versus near-zero rate;
- explosion rate versus accepted mechanisms per second;
- near-zero rate versus accepted mechanisms per second;
- a three-objective Pareto table;
- results stratified by horizon, initial-value bound and structural complexity;
- bootstrap or paired confidence intervals for differences from the baseline.

Do not hide the trade-off inside one arbitrary score. First identify methods
that are Pareto dominated. If one operating point must be selected, apply an
explicit admissibility rule to explosion and near-zero rates and then choose
the fastest remaining method.

## 5. Required library changes before the method notebooks

1. Extract the Notebook 4 stability simulation and metric calculation into a
   reusable module under `src/tsfmxai/evaluation/`.
2. Add tests proving that the extracted evaluator reproduces the frozen
   Notebook 4 definitions.
3. Split candidate proposal, stabilization and acceptance into explicit
   stages. Preserve the original candidate AST and binding in the experiment
   record whenever a method transforms them.
4. Add stable seeds for proposal index, screening trajectory and evaluation
   trajectory so worker count cannot change the accepted dataset.
5. Return rejection reason, proposal count, transformation time and screening
   time from the sampler API.
6. Implement every method behind an explicit strategy/configuration object;
   do not silently change the behavior of the existing baseline sampler.
7. Add unit tests for coefficient extraction, L1 normalization, companion
   matrices, spectral projection, deterministic rejection and timing-field
   presence.

## 6. Execution order and dependencies

```text
freeze protocol and extract evaluator
                |
                v
      reproduce baseline metrics
                |
       +--------+-------------------+------------------+
       |                            |                  |
       v                            v                  v
configuration tuning     coefficient-gated lags   pilot rejection
                                    |
                          +---------+---------+
                          |                   |
                          v                   v
                 L1 normalization    spectral projection
                          |                   |
                          +---------+---------+
                                    |
                                    v
                         bounded nonlinear atoms
                                    |
                                    v
                              hybrid sampler
                                    |
                                    v
                         final Pareto comparison
```

Configuration tuning, coefficient gating and pure pilot rejection can be
implemented in parallel after the common evaluator is validated. L1 and
spectral methods depend on reliable coefficient extraction. The hybrid depends
on results and implementations from the earlier methods.

## 7. Decision rule

A method improves the baseline only when it reduces explosion frequency on
held-out evaluation trajectories while its near-zero frequency remains inside
a pre-registered non-inferiority margin and structural coverage is preserved.
Uncertainty intervals, rather than point estimates alone, must support this
decision. Among scientifically acceptable methods, accepted mechanisms per
second determines the preferred operating point.

If no method dominates on all three primary objectives, retain the Pareto
frontier and expose more than one generator profile, for example a faster
profile and a stricter-stability profile.
