# No Train No Gain: Revisiting Efficient Training Algorithms for Transformer-based Language Models

- Authors: Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt J. Kusner.
- Venue: NeurIPS 2023.
- Source: https://proceedings.neurips.cc/paper_files/paper/2023/file/51f3d6252706100325ddc435ba0ade0e-Paper-Conference.pdf
- Local PDF: `2023_no_train_no_gain.pdf`.
- Citation key: `kaddour2023notrainnogain`.

## Relevance

The paper evaluates training methods under fixed time budgets and explains why the learning-rate schedule must be compatible with the unit used to define the budget. It is a methodological precedent, not a study of synthetic time-series generators.

## Important evidence

- PDF p. 1, abstract and introduction: compares methods under fixed computation budgets and reports downstream performance at 6, 12, and 24 hours in a reference system.
- PDF p. 3, Section 2.1: a baseline evaluated before its learning rate has fully decayed can appear unfairly worse; the schedule should decay in the same unit that defines the budget.
- PDF p. 4, Section 3, `Learning-rate schedule`: schedules are adjusted to elapsed time under reference-system-time budgets. The authors distinguish that from raw wall-clock time used in other work.

## Limitation for our study

Using elapsed machine time including online data generation and cache behavior is our adaptation. The cited paper does not prescribe our stage division, OneCycle parameters, or generator policy.
