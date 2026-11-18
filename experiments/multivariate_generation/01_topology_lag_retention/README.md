# Multivariate generation: topology and lag retention

Question: does topology-dependent lag retention reduce retries and generation cost?
Hypothesis: reducing lag retention for concentrated source distributions reduces
nonfinite trajectories, especially for independent channels. This is an exploratory
prior change, not a controlled attribution of failures to original channels.

Configuration is explicit in run.py and the serialized dataclass defaults in the
source revision accompanying this record: 500 accepted systems per topology, six
channels, two trajectories per system, 128 generated steps, at most four operators,
lags 1--16, degree at most four. Treatment probabilities are original .4, Shao-Wu
.3, Jacobian .1, Heaps .2; Gaussian noise std is sampled in [.01,.1], with bounded
uniform noise for Jacobian. RandomHistorySpec sums independently present constant,
uniform-noise and cosine components (probability .5 each; default intervals).
Context/future/stride are 64/16/16, batch size 32, output float32, simulation float64.
Each run yields 4,000 windows. Failed trajectories receive three new histories,
then the whole sampled system is replaced. RNG streams are keyed by the explicit
root seed, system, trajectory and attempt; innovations are fixed across retries.

Retention-on uses minimum=.5, maximum=1, tau=1 on original channels only. Retention
depends on inverse source concentration; rejected lag leaves become parameters.
It changes lag and polynomial-degree coverage and is not a stability certificate.

record.json preserves the observed outputs, including three seeds without retention
and one with retention. These are session observations rather than a preregistered
study. Timings use perf_counter and include retries and batch assembly. Brief test
execution overlapped parts of the retention run; process load, hardware and exact
historical source snapshot were not recorded. Avoid precise speedup claims.
Original-equation discard counts do not identify which channel caused failure.

Reproduce from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe experiments/multivariate_generation/01_topology_lag_retention/run.py --retention off --seed 314159
.\.venv\Scripts\python.exe experiments/multivariate_generation/01_topology_lag_retention/run.py --retention on --seed 314159
```

Repeat with seeds 271828 and 161803 for the other recorded baseline configurations.
New executions require a new experiment directory; do not overwrite these results.
No raw trajectories are retained. The measurements support fewer replacements
under retention for seed 314159; they do not isolate a causal effect of topology
or certify coverage preservation. Eight-worker and interrupted larger runs are
outside this experiment record.
