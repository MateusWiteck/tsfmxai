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
- aligned trajectory panels and their per-series means;
- one trajectory-by-time heatmap per series;
- a two-dimensional state-space view `x0(t) x x1(t)`;
- the boundary between initial conditions and recursively generated values.

Candidate next views are horizon-wise dependency propagation, local
contribution flow between the two equations, a stability map over parameters
and initial conditions, and a comparison between sampled and canonicalized
systems.
