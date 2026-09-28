# Architecture

The target library separates mechanism generation, observation generation,
model execution, explanation retrieval, and evaluation:

```text
mechanism sampler -> canonical AST -> trajectory generator
                                      |
                                      v
                               dataset windows
                                      |
                                      v
model adapter -> forecasts / recovered mechanism / explanations
                                      |
                                      v
             canonical normalization -> ground-truth metrics
```

The canonical mechanism representation is the source of truth. It must retain
operators, variables, lags, coefficients, regimes, and the innovation law.
Dataset generation executes that representation. Explanation ground truths are
derived from the same retained mechanism for every forecast horizon.

Code ownership follows the package directories:

- `generators/`: sample mechanisms and trajectories;
- `representations/`: AST, grammar, operators, parsing, and serialization;
- `datasets/`: windows, batches, and data contracts;
- `models/` and `baselines/`: adapters around evaluated forecasters;
- `symbolic_regression/`: mechanism-recovery methods;
- `explainability/`: post-hoc and explanation-aware interfaces;
- `evaluation/`: forecasting, mechanism, and explanation metrics;
- `pipelines/`: reproducible orchestration;
- `visualization/`: diagnostic plots and benchmark reports.

Architecture decisions that become stable should be recorded under
`docs/decisions/`.
