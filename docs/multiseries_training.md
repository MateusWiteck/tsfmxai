# Multiseries training batches

The default `TrainingFormat()` follows the prospective Chronos-2 comparison:
96 context steps, 96 forecast steps, stride 96, and 16 multivariate examples
per batch.

## Hybrid AST and KernelSynth channels (experimental)

`HybridConfig(ast=..., family_probs={"ast": 0.75, "kernel_synth": 0.25})`
selects the **generator family** independently for each channel. This choice is
separate from the AST treatment distribution. KernelSynth channels are
univariate Gaussian-process draws with temporal covariance; they do not use a
cross-channel covariance matrix. AST channels may refer to lagged KernelSynth
channels through the ordinary multiseries topology. KernelSynth channels are
exogenous: their values never depend on AST channels.

`generate_hybrid_training_batches` and `OnlineBatchPlan` retain the same
forecasting shapes. `channel_mask` stays true for KernelSynth channels, while
their `equation_mask` and `noise_targets['mask']` are false. Their zero-filled
target storage is padding, not ground truth. The equation loss and innovation
scale loss must use these masks; the forecast loss includes every channel.
KernelSynth's GP scale is not an AST innovation standard deviation.

Families are drawn before any tree. KernelSynth channels receive a zero
placeholder equation instead of a sampled AST, so no unused tree is sampled or
certified, and replacement statistics count AST equations only. AST channels
are identical to those of the plain multiseries sampler with the same seed.
Each KernelSynth path, including the prefix used as lag history, is drawn
before the AST recurrence runs. The family choice is kept distinct so another
channel generator can be added without treating it as an AST postprocessing
method. The batch-speed record in
`experiments/generator_model_evaluation/generator_selection/kernel_synth_hybrid_assessment/` predates this
change and includes the former unused-tree overhead.

AST treatments are not adjusted for KernelSynth inputs. This is deliberate:
KernelSynth paths are standardized GP draws and are assumed stable enough as
exogenous drivers. A Shao-Wu channel's sensitivity bound therefore does not
certify the response to these inputs; non-finite paths are still rejected and
retried by the stream.

In `noise_targets['family']`, −1 means "no family target". It marks both padded
and KernelSynth channels; `channel_mask` separates the two.

An experimental timing runner is `scripts/benchmark_hybrid_batches.py`. It
reports delivered-batch latency with fixed workers and records the realized
KernelSynth fraction. Its timings are generation-only; they do not measure
training throughput or GPU stalls.
It samples approximately 208-point paths for the 96/96 configuration and is
not directly comparable to the TimePFN LMC benchmark, which uses 1,024-point
cached trajectories and includes model optimization in its cold-update timing.

The public entry points are `sample_multiseries_system`,
`build_multiseries_system` (in `tsfmxai.generators.multiseries`), and
`generate_training_batches` (in `tsfmxai.datasets`). Generation is streaming:
consume the returned iterator to execute it. The iterator has a fixed `.codec`
and live `.stats`. It is single-pass; create another iterator to replay a run.

## Example

These small numbers are an executable example, **not** coverage-selected defaults.
The coverage experiment must determine the reference AST lag/degree configuration.

```python
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import HistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig, NoiseConfig
from tsfmxai.datasets import (
    TrajectoryConfig, TrainingFormat, generate_training_batches,
)

config = MultiseriesConfig(
    ast_config=CombinatorialTreeConfig(
        minimum_operators=0, maximum_operators=3,
        minimum_lag=1, maximum_lag=16, max_degree=2,
    ),
    channel_count_probs={6: 1.0},
    dependency_probs={
        "independent": .25, "linked_chain": .25,
        "previous_chain": .25, "random": .25,
    },
    max_external_sources=2,
    self_source_weight=3.,
    noise=NoiseConfig(std_range=(.01, .1)),
)
stream = generate_training_batches(
    config=config, num_systems=8, trajectories_per_system=2, seed=42,
    trajectory=TrajectoryConfig(
        generated_steps=128,
        initialization=HistorySpec("uniform radius 1", "uniform", amplitude=1.),
    ),
    training=TrainingFormat(
        context_length=32, prediction_length=16, stride=16, batch_size=8,
        dtype="float32", noise_target="scale_and_family",
    ),
)
for batch in stream:
    print(batch.context.shape, batch.future.shape, batch.equation_targets.shape)
print(stream.stats)
```

The prospective model-training protocol fixes six active channels in every
example. Its optional curriculum varies AST operation count only. The library
can express other channel counts for separate generator studies, but those do
not enter this comparison. All windows/trajectories of a system should stay in the
same evaluation split when testing generalization to new mechanisms. This API
does not create train/test splits automatically.

## Topology and AST sampling

Each system draws a channel permutation. `linked_chain` permits the immediate
predecessor; `previous_chain` permits sources earlier in that permutation;
`random` permits any other channel (including delayed cycles); `independent`
permits no external sources. For previous-chain/random, external subset size is
uniform from zero to `min(max_external_sources, candidates)`. The linked chain
permits its predecessor whenever the cap is positive. These are permissions,
not required edges. The final canonical support determines effective sources.

The existing combinatorial sampler draws a scalar-source AST structure. Each lag
occurrence is then assigned a source from its allowed set: self has relative
weight `self_source_weight`, each external source weight one. This is deliberately
not uniform sampling over all multivariate labelled ASTs. `ast_config.num_series`
is overridden by this two-stage procedure. Every sampled tree requires at least one lag: a Shao-Wu channel must read an
external source, and a Jacobian channel needs a lag for its certificate. Heaps
channels draw no tree.

## Explicit systems and treatment semantics

`build_multiseries_system(equations=..., allowed_sources=..., treatments=...,
innovations=..., max_degree=...)` accepts existing `SampledMechanism` objects,
one `TreatmentSpec` and `InnovationSpec` per channel, and a source mapping keyed
by every target channel. It performs no sampling. Original equations and final
equations are both retained. `sample_multiseries_system(config, seed=...)` samples
the inputs and calls that builder.

This release implements per-channel `original` and `shao_wu` treatments.
Shao-Wu reuses `saturate_product_lags` and `global_expression_bounds` from the
existing implementation. Once source connections are assigned, let `g` be the
sum of the absolute derivative bounds over all sources and lags for one equation.
Its scale is `safety_factor * min(1, target_gain / g)` (ratio replaced by one for
zero gain). This conservative per-equation row budget differs from the existing
joint block spectral-norm scaling. We store the applied scale and do **not**
claim a joint certificate for the mixed system. External sources are not
automatically saturated unless the selected transformation calls for it.

Heaps is a joint VAR construction and bounded-domain contraction currently has a
scalar constructor. Their per-channel adaptations remain unresolved; requesting
them fails explicitly rather than silently substituting another method.

Sampled innovations are independent Gaussian draws across channels and times,
with one uniformly sampled std per channel fixed for all its trajectories.
The explicit builder also accepts `InnovationSpec("uniform", std)` using bounds
`+/-sqrt(3)*std`. No noise is silently replaced. Allowed families and std are
retained regardless of the selected training target.

## Initialization and retries

`HistorySpec` and `make_initial_history` are shared with the
`time_series_initialization` experiments. They now live in
`generators/initialization.py`; the old evaluation imports still work.
Supported mixed-system laws are zero, constant, uniform, and cosine prefixes.
Amplitude, offset, cosine period and prefix fraction keep their study semantics.
A cosine prefix must cover the required lag history and is used at its actual
length. A zero-lag system can start with an empty history. There is no hidden
burn-in: `generated_steps` excludes startup observations, and windows contain
generated observations only. The Heaps stationary initializer remains available
to its original users, but is rejected for arbitrary mixed systems.

### Random startup families (`RandomHistorySpec`)

The current general generator default is `RandomHistorySpec`; the separate
prospective prior comparison still pins a uniform `HistorySpec`. For **every
channel**, each of three families appears independently with its own
probability; the parameters of the present families are drawn and the
components are summed. With no family present the channel history is zero.

| Family | Component | Parameters drawn (defaults) |
|---|---|---|
| constant | `s * m` | sign `s = ±1`, `m ~ U(constant_magnitude)` = U(0, 1) |
| noise | i.i.d. `U(-r, r)` per lag | `r ~ U(noise_radius)` = U(0.1, 1) |
| cosine | `A cos(2πk/T + φ)` | `A ~ U(cosine_amplitude)` = U(0.1, 1), integer `T ~ U{cosine_period}` = U{4..48}, `φ ~ U(0, 2π)` |

All probabilities default to 0.5. The cosine is evaluated directly on the
required history length (random phase instead of a cycle prefix), so it never
fails the lag-coverage check. Draw order is fixed per channel (presence of all
families, then all parameters), so one family's parameters do not depend on
which others appear. Every retry draws families and parameters again, so
retries differ even when the drawn family is a constant. Window metadata records
`initialization_components`: for every channel, the list of present families and
their parameters (`None` for a fixed `HistorySpec`).

```python
from tsfmxai.generators.initialization import RandomHistorySpec
trajectory = TrajectoryConfig(
    generated_steps=128,
    initialization=RandomHistorySpec(constant_probability=.5, noise_probability=.5,
                                     cosine_probability=.5),
)
```

The constant ranges follow the exploratory constant sweep (2026-11-16, same bank
as `time_series_initialization/final_methods`): explosions occur only for the
original AST and depend mostly on the tree (50% even with a zero history,
rising to 62% at ±16); Jacobian, Shao–Wu and Heaps did not explode for any
tested constant. The Heaps stationary law is not a family because it exists only
for Heaps VAR mechanisms.

Possible future families, not implemented or studied: Gaussian white noise,
linear trend, random walk / AR(1), step or impulse, and Gaussian-process draws
from the KernelSynth bank; components shared across channels; products of
components; a cap on the summed magnitude.

### Failure policy

Failure policy (fixed, not configurable): one initial attempt plus exactly 3 retries.
By default only the history seed changes: innovations are identical between attempts.
`resample_innovations=True` explicitly changes both. Deterministic initializers
produce identical histories on retries.

If all 4 attempts of any trajectory go to inf/NaN, the system sampled from a config is
discarded and a new tree is drawn: replacement `r >= 1` of system `i` uses seed
`stream_seed(seed, 11, i, r)`, so output stays independent of batching and workers.
All trajectories of the discarded system are dropped and regenerated with the
replacement. This repeats until a system works (a defensive limit of 10 000
replacements raises `RuntimeError`). Explicit `systems=` cannot be resampled, so their
failing trajectories are skipped and counted. `stats.replaced_systems` and
`stats.discarded_trajectories` count the discards, and each window's metadata records
`system_replacements`.

Simulation is float64 without a magnitude threshold, clipping, or normalization.
Nonfinite inputs, nonfinite execution, arithmetic overflow, and nonfinite output
conversion are recorded separately. Float32 conversion is checked for the whole
generated trajectory and noise std targets. Finite large values are accepted.
No invalid prefix is silently salvaged. Configuration/grammar errors propagate
instead of being treated as random trajectory failures. Failure records retain
system/trajectory indices and both seeds; they can accumulate in long runs.

## Batch contract

The default layout is `[batch, time, channel]`; `batch_channel_time` is also
available. Channel padding is fixed to the maximum possible count in the sampling
configuration, or maximum count among explicit systems. The last batch may be
smaller than `batch_size`. No model-specific normalization or tokenization is
applied to the numeric series. Arrays are NumPy float32/float64, not torch tensors.

| Field | Shape / meaning |
|---|---|
| `context`, `future` | Numeric windows in the selected layout |
| `channel_mask` | `[batch, channel]`, true for actual channels |
| `equation_targets` | `[batch, channel, token]`, BOS/factors/END_TERM/EOS using the existing canonical codec |
| `equation_mask` | Same shape, true for non-padding tokens |
| `noise_targets['std']` | `[batch, channel]`, original-units innovation std |
| `noise_targets['family']` | Optional: Gaussian=0, uniform=1, padding=-1 |
| `metadata` | Per-window system, indices, actual startup history, seeds, configs and simulation dtype |

The skeleton and noise refer to the executed mechanism in original coordinates.
The head does not target coefficients or realized random innovations. Use
`noise_target="scale"` to omit family supervision without discarding its metadata.
Use `stream.codec` for a stable vocabulary; token padding length varies per batch.
For PyTorch, `torch.from_numpy(batch.context)` creates the input tensor. A model
adapter must explicitly handle any further normalization and target conventions.
This is a generic numeric batch contract, not a claim of drop-in compatibility
with every foundation-model training loop.

Window metadata holds references to in-memory mechanism/configuration objects,
not a standalone JSON manifest. Retain these objects or serialize them explicitly
when persisting a corpus; this API does not write datasets or sidecar artifacts.


### Four-family sampling

The prospective treatment policy follows the sampled dependency topology:
a channel with at least one permitted external source uses Shao-Wu and assigns
at least one lag occurrence to an external channel. Otherwise it draws
Jacobian or Heaps with equal probability. An exogenous KernelSynth channel
counts as an external source. For six channels, random topology, and at most
two external sources, the expected treatment shares are approximately
two thirds Shao-Wu and one sixth each Jacobian and Heaps; realized shares
depend on topology and hybrid family draws. Jacobian and
Heaps use self-only sources. There is no joint mixed-system
stability certificate. Jacobian retains its bounded-domain certificate (radius
3 by default), uses uniform innovations with the requested standard deviation,
and rejects incompatible initial histories without clipping. Heaps constructs
a stable scalar AR via the existing Heaps reverse map, using a unit reference
covariance to determine coefficients; its actual innovations use the sampled
Gaussian standard deviation. Random startup does not imply stationarity.
Heaps samples its lag support directly, without building a preliminary
AST. Conditional on choosing a Heaps channel, the default support-class
probabilities are 40% consecutive (`1..p`), 20% isolated (`{s}` for
`s=2..16`), and 40% periodic with at least two lags (`{s,2s,...,ps}` for
`p>=2`, `s>=2`, and `p*s<=16`). Within a class, the feasible `(p,s)` pairs
are equiprobable. At maximum lag 16, the classes contain 16, 15, and 19
pairs, respectively: 50 disjoint support patterns. This allocates explicit
probability to single-lag cases such as `{8}` and `{9}`, without calling
them recurring seasonal patterns. The code records class, order, and period
in the channel treatment. `original_equations` contains a marked placeholder
for a Heaps channel, not an unused sampled AST; the executed equation in
`equations` is the only valid skeleton target.

The reconstructed scalar AR has additive lag terms, so it has no
multiplicative lag interactions or cross-channel dependencies. The standalone
Heaps VAR implementation can represent linear cross-channel dependencies,
but this mixed-channel sampler does not use that coupled form or its joint
stationary initializer. The standalone constructor accepts a fixed
`lag_spacing` and builds the stationary lag-state covariance for its
expanded order. In a 100-system, six-channel stability sweep with 96/96
windows, all 600 Heaps channel certificates held and all generated windows
were finite; this is an empirical numerical check, not a bound on Gaussian
sample amplitudes.

After the initial history attempt and three retries, an exhausted trajectory
causes the whole config-sampled system to be replaced, including its accepted
trajectories. This policy applies to all families. Explicit systems are not
replaced. `replaced_systems` counts replacements; `resampled_equations` counts
equations in discarded systems, and `resampled_original_equations` counts the
original subset. `replacement_records` retains discarded seeds and reasons.
These counters include numerical output-conversion failures as well as
nonfinite recurrence outputs; consult failure reasons for the distinction.
