# tsfmxai

Research software and thesis material for generating synthetic discrete-time
series with known mechanisms and explanation ground truths. The project is
intended to support both synthetic pretraining priors and reproducible
benchmarks for forecasting, mechanism recovery, and explainability.

## Repository map

| Path | Purpose | Git policy |
|---|---|---|
| `src/tsfmxai/` | Reusable library code | tracked |
| `tests/` | Unit and integration tests | tracked |
| `configs/` | Generator, dataset, model, and benchmark configurations | tracked |
| `references/` | Paper summaries and canonical bibliography | summaries tracked; PDFs local |
| `docs/` | Research, architecture, and methodology documentation | tracked |
| `report/` | Thesis manuscript in LaTeX | sources tracked; `build/` ignored |
| `experiments/` | Curated scientific evidence | tracked |
| `experiments_local/` | Exploratory notebooks, cloned repositories, and temporary analyses | ignored |
| `data/` | Data lifecycle and small samples | generated/full data ignored |
| `outputs/` | Generated reports and artifacts | ignored |

## Start here

1. Read `docs/research/problem_definition.md` and
   `docs/research/research_questions.md`.
2. Read `ARCHITECTURE.md` for the intended software boundaries.
3. Read `REPRODUCIBILITY.md` before generating data or publishing an
   experiment.
4. Use `ROADMAP.md` for the planned development phases.

The reusable library is currently a scaffold. Existing exploratory analyses
have deliberately remained in `experiments_local/` until their interfaces and
tests are stable enough to promote into `src/`.
