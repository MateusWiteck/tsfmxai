# Default lag range

Newly sampled trees use lags 1 through 16 by default. The random and
combinatorial samplers, canonical codec, initialization study and studio share
this upper limit. The stationary VAR generator uses maximum order 16.
Explicit experimental ranges remain possible.

The project decision is based on the following coverage results supplied by
the researcher on 2026-11-10:

| Maximum order | Reproduced functions |
|---:|---:|
| 1 | 15 |
| 2 | 21 |
| 4 | 32 |
| 8 | 37 |
| 12 | 39 |
| 16 | 42 |
| 24–128 | 43 |
| 256 | 45 |

Order 16 improves coverage from 39 to 42 functions relative to order 12.
Increasing to 24–128 adds only one function in this experiment. This motivates
the default; it is not a universal guarantee of coverage for arbitrary series.
The supplied table does not specify the reconstruction-error criterion or
experiment artifact, so those details are not inferred here.

Current RAW/TANH training uses this default (`canonicalization_5m_raw_tanh.json`).
The 12-lag runs and their checkpoints were removed because they are
incompatible with the 16-lag codec.

Changing the lag vocabulary changes the token embeddings and output dimension.
Train new checkpoints for the 16-lag configuration; the earlier checkpoints
remain tied to their original codec.
