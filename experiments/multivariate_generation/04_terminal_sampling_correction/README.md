# Corrected terminal sampling

Supersedes post-sampling leaf replacement as an implementation of the requested
lag probability. Weighted combinatorial terminal construction is active on
original channels, with minimum .25, maximum .75 and tau 1. This subtracts
25 percentage points from the initial .5--1 curve. Sources use self weight 3.

run.py records 500 accepted systems per topology, seed 314159, six channels,
two 128-step trajectories, four operators, lag cap 16, degree cap 4 and
context/future/stride 64/16/16, batch size 32. Its explicit diagnostic mixture
is original .4, Shao-Wu .3, Jacobian .1, Heaps .2. results.json contains timing,
replacements and initialization attempts. The output is 4,000 windows per case.

batches.py uses the treated-only training mixture and eight persistent workers.
The first 20 batches warm up; the next 100 measure delivery throughput, including
prefetch and serialization. No training or device transfer is included.
Its results are in batch_results.json. Since originals have weight zero,
terminal probability has no active original channels in this measurement.

The paired notebook is a separate complete experiment with its own in-cell
record. It compares 300 original candidates under each sampling setting, with
up to four histories and paired treated trajectories.

Run scripts from repository root with PYTHONPATH=src using the project Python.
Timing comparisons are exploratory and use one seed. Earlier experiment files
remain historical evidence and are not results for this corrected implementation.
