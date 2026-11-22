# Mechanism Studio

Local interface to preview the **multivariate training generator** before
spending time on training. It runs the same code the training pipeline uses:
`sample_multiseries_system`, or `sample_hybrid_system` when the KernelSynth
probability is positive, for the system, and the corresponding training batch
stream for the trajectories. The adapter lives in `src/tsfmxai/visualization/studio_multiseries.py`;
the frontend reimplements no scientific behavior.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe frontend\app.py
```

Then open <http://127.0.0.1:8000>. Default values are served by
`/api/defaults` and mirror the six-series canonical training configuration.

## What it shows

- **Parameters** (sidebar), grouped as system, equation structure, treatment and
  noise, and trajectories. Hover or focus the `i` icon beside any parameter for
  a description. Five starting points are provided: the two preliminary training
  candidates P25 and P50 (25% or 50% KernelSynth channels), didactic, linked chain
  and univariate.
- **Channel families and treatments**, as in training: each channel is drawn as
  a KernelSynth series with the chosen probability, otherwise it gets an
  equation. An equation channel whose topology permits an external source uses
  Shao–Wu and reads at least one external series; a channel without external
  sources uses Jacobian or Heaps (the Jacobian share is configurable). Heaps
  channels show their lag support (consecutive, isolated or periodic), order and
  period; KernelSynth channels show their kernel composition.
- **Topology graph** and **channel-lag map**: which channels may and do read
  which, and at which lags. Dashed edges are permitted but unused.
- **One equation per channel**: executed formula, pre-treatment formula for
  Shao–Wu channels, canonical polynomial (the form the equation-head teacher
  encodes), tree, and bindings.
- **Trajectory player**: one realization at a time, every channel aligned, with
  the context and horizon of the training window marked, optional initial
  history, hover readout, linear or signed-log scale, state-space view and an
  ensemble heatmap.
- **Random startup families** (default, as in training): for every channel the
  page draws which families appear (constant, white noise, cosine) with the chosen
  probabilities, draws their parameters and sums them. The player lists the drawn
  components of each channel for the selected realization. The fixed laws (uniform,
  zero, constant, cosine) remain available for comparison.
- **Acceptance rate and system replacement**: fixed policy, identical to training. A
  realization that goes to inf/NaN gets 3 new initial histories; if all 3 fail, the
  system is replaced by a newly sampled tree. There is no option to change or disable
  this. The page shows the accepted system and how many were discarded.

A window is `context + horizon` generated steps with no burn-in; the initial
history only starts the recurrence and is never part of a window. Trajectories
use float32 output as in training.

Each system is kept in memory (last 64) so that new trajectories can be drawn
from the same system. Request limits: 12 channels, 64 realizations, 1024 steps.
