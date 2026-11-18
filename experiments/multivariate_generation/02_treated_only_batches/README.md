# Treated-only batch generation

Question: how fast is generation with untreated AST probability zero?
Weights are renormalized to Shao-Wu 1/2, Jacobian 1/6, Heaps 1/3.
The full configuration and seed 314159 are explicit in run.py. Results and
runtime provenance are in results.json. Eight persistent spawned DataLoader
workers use one numerical thread each and prefetch factor two. Twenty batches
warm up the pipeline; the next 100 measure aggregate delivery throughput.
Every batch has 32 windows from four six-channel systems and two trajectories
per system. Trajectories contain 128 generated steps, context/future/stride
64/16/16. ASTs use 1--4 operators, lags 1--16, degree at most four. Startup
uses RandomHistorySpec defaults and output is float32. No original treatments
are allowed in consumed batches. Jacobian and Heaps channels remain self-only.
This is a fixed curriculum stage, with no training or GPU transfer timed.

Reproduce from the repository root with PYTHONPATH=src and the project Python,
executing experiments/multivariate_generation/02_treated_only_batches/run.py.
Use a new directory for later runs to preserve recorded results.

Observed throughput is roughly 47--53 ms per batch after warmup, with about
8--10 seconds of startup. Earlier mixture timings were 59.7, 56.0 and 59.5 ms
for independent, random and chain. Comparisons are descriptive, with one run
per topology and no confidence intervals. No quality or joint stability claim
follows from throughput. Source changes were uncommitted during measurement.
