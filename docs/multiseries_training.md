# Multiseries training batches

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
    channel_count_probs={1: .25, 2: .25, 4: .5},
    dependency_probs={
        "independent": .25, "linked_chain": .25,
        "previous_chain": .25, "random": .25,
    },
    treatment_probs={"original": .5, "shao_wu": .5},
    max_external_sources=2,
    self_source_weight=3.,
    allow_no_lags=True,
    noise=NoiseConfig(std_range=(.01, .1)),
)
stream = generate_training_batches(
    config=config, num_systems=8, trajectories_per_system=2, seed=42,
    trajectory=TrajectoryConfig(
        generated_steps=128,
        initialization=HistorySpec("uniform radius 1", "uniform", amplitude=1.),
        max_initialization_retries=3,
        on_retry_exhausted="skip",
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

Changing `channel_count_probs` between calls supports curriculum learning. A fixed
channel count is `{count: 1.0}`. Probabilities must sum to one; draws do not enforce
exact dataset proportions. All windows/trajectories of a system should stay in the
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
is overridden by this two-stage procedure. `allow_no_lags` controls the sampler's
`require_lag`, permitting constant-plus-noise mechanisms. It does not prescribe a
probability of such mechanisms or guarantee that memory is absent.

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

One initial attempt plus `max_initialization_retries` is allowed. By default
only the history seed changes: innovations are identical between attempts.
`resample_innovations=True` explicitly changes both. Deterministic initializers
produce identical histories on retries. Exhaustion either skips the trajectory
or raises, according to `on_retry_exhausted`. No replacement system is sampled.
Failure-policy benchmarking remains necessary before choosing a production policy.

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
