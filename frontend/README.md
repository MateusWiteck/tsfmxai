# Mechanism Studio

Local interface for inspecting a sampled two-equation system and drawing
multiple multivariate trajectories from the same bound recurrences.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe frontend\app.py
```

Then open <http://127.0.0.1:8000>.

The server uses only Python's standard library plus the project dependencies.
It imports the reusable sampler and trajectory executor from `src/tsfmxai/`;
the frontend does not reimplement their scientific behavior.

## Current scope

The executable system has `num_series=2` and retains one sampled AST for each
target, `x0(t)` and `x1(t)`. Either equation may read lagged values from either
series. At each time step both right-hand sides are evaluated against the same
past before either new value is appended, so the system is synchronous and has
no implicit contemporaneous ordering.

The trajectory count represents independent realizations of that same
two-equation system. Each realization receives independently sampled initial
conditions and, when enabled, Gaussian innovations, while formulas, parameters,
and lag bindings remain fixed.

The frontend applies no finite magnitude threshold to generated values. It
stops only when numerical execution produces a non-finite value (`NaN` or
infinity), which cannot be represented by the JSON response or plotted
meaningfully.

The interface currently shows:

- both canonical AST topologies through an equation selector;
- both skeleton and numerically bound formulas;
- parameter and lag bindings for the selected target;
- an explicit target-by-source lag-support matrix;
- aligned panels showing one complete multivariate trajectory at a time;
- play/pause, previous/next, direct trajectory selection, and playback speed;
- independent scaling for each selected trajectory, with linear or signed-log axes;
- an optional, collapsed ensemble overview with one heatmap per variable;
- a synchronized state-space view `x0(t) x x1(t)` for the selected trajectory;
- the boundary between initial conditions and recursively generated values.

Candidate next views are horizon-wise dependency propagation, local
contribution flow between the two equations, a stability map over parameters
and initial conditions, and a comparison between sampled and canonicalized
systems.

Playback runs entirely in the browser, advancing through all saved trajectories
once and stopping at the last one. It does not resample or alter values. The
signed-log view uses `sign(x) * log10(1 + abs(x))` and labels ticks in original
units. Each trajectory gets its own scale, so compare magnitudes using tick
labels rather than apparent line height. New generation pauses playback;
manual navigation also pauses it.

The same playback can work on static hosting with exported trajectory data.
The current GitHub Pages site renders experiment notebooks; the studio still
uses the local Python API to sample new mechanisms and trajectories.

## Explosion-control selector

Choose a method before clicking **Gerar duas equações**. The studio retains the
actual sampled formulas and certificates and uses each family's existing
trajectory generator; it does not clip trajectories to simulate stability.

| Method | Behavior |
|---|---|
| None | Original unrestricted, coupled AST sampler. |
| Coefficient-gated lags | Existing heuristic; no stability certificate. |
| Jacobian contraction | Two independently certified scalar recurrences on `[-1, 1]`, with bounded uniform innovations. |
| Shao–Wu | Coupled, time-homogeneous `tanh` recurrences, Gaussian innovations, and certified finite burn-in. |
| Heaps VAR(p) | Coupled linear stationary VAR; lag bounds select the order and history is drawn from its joint stationary law. |
| Stationary VAR(1) | Coupled linear VAR with spectral radius below one and stationary Gaussian initialization. |

The innovation control sets a uniform bound for Jacobian contraction and a
Gaussian standard deviation for the other methods. Certified Gaussian methods
require a positive standard deviation. Changing the innovation setting requires
resampling the certified mechanism. Manual initial bounds apply to the two
uncertified methods; the others retain their prescribed initialization law.
Tree/operator controls apply to AST families, not linear VAR families. The
certificate status is displayed above the equations. Gaussian stationarity and
GMC do not impose a hard bound on every realized amplitude.
