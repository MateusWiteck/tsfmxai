# Mechanism Studio

Local interface for inspecting a sampled AST and drawing multiple trajectories
from the same bound recurrence.

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

The executable recurrence has `num_series=1`. The trajectory count represents
independent realizations of that same mechanism, each with independently
sampled initial conditions and, when enabled, Gaussian innovations. A future
multivariate generator should sample and retain one target mechanism per output
series before this control is expanded beyond one.

The frontend applies no finite magnitude threshold to generated values. It
stops only when numerical execution produces a non-finite value (`NaN` or
infinity), which cannot be represented by the JSON response or plotted
meaningfully.

The interface currently shows:

- canonical AST topology and node types;
- skeleton and numerically bound formulas;
- parameter and lag bindings;
- overlaid trajectories and their mean;
- a trajectory-by-time heatmap;
- the boundary between initial conditions and recursively generated values.

Candidate next views are a variable-lag ground-truth map, horizon-wise
contribution propagation, a stability map over parameters and initial
conditions, and a comparison between sampled and canonicalized trees.
