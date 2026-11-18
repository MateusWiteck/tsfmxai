# Terminal sampling repeated timing study

Question: does corrected terminal sampling reduce resampling and generation time?
Hypothesis: lower lag probability reduces finite-trajectory failures.
Each run uses 500 accepted systems, six channels, two 128-step trajectories,
operators 1--4, lags 1--16 and degree at most 4. The diagnostic mixture is
original .4, Shao-Wu .3, Jacobian .1, Heaps .2. Context/future/stride are 64/16/16,
batch size 32, yielding 4000 windows per run. Noise and RandomHistorySpec use
source defaults. Computation is serial CPU float64, output float32. Three
seeds are 314159, 271828 and 161803. The two policies use identical root seeds.
Baseline is the uniform combinatorial prior. Corrected sampling chooses
terminal types during construction, minimum .25, maximum .75, tau 1.
Independent, random and predecessor-chain topologies are tested in that order.
Original-only terminal sampling is conditional on structural constraints.

results.json records every run, time, system replacements and history attempts.
Mean and sample standard deviation (ddof=1) across seeds describe both runtime
and sampled-mechanism variation. They are not uncertainty intervals for a fixed
workload. Run order is fixed, and OS process load can affect timing. The paired
notebook has a separate complete experiment record inside its own cell.

Reproduce timing from repository root with PYTHONPATH=src using project Python:
python experiments/multivariate_generation/05_terminal_sampling_repeated/run.py
The paired notebook is executed with jupyter nbconvert --execute --inplace.
Raw arrays remain in memory. New measurements should use a new directory.
