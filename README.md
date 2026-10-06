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
| `frontend/` | Local mechanism and trajectory visualization | tracked |

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

## Published experiment notebooks

The Quarto website publishes saved outputs from every notebook under
`experiments/`, including nested directories. New notebooks are discovered
automatically. `experiments_local/` and `notebooks/` are outside the site.
Rendering never executes notebook cells or requires the experiment dependencies.

Install [Quarto](https://quarto.org/docs/get-started/) (the workflow pins 1.8.27),
then run from the repository root:

```powershell
quarto preview
# Build only, without starting a server:
quarto render --no-execute
```

Generated HTML is written to ignored `site/`; Quarto state in `.quarto/` is also
ignored. Keep experiment records and outputs inside the executed notebook as
described in `REPRODUCIBILITY.md`.

To enable publication, select **Settings > Pages > Build and deployment >
Source > GitHub Actions** in the GitHub repository. Commit and push the site
configuration and notebooks to `main`. The `Publish experiment notebooks`
workflow builds and deploys updates automatically; pull requests build without
deploying. It can also be run manually from the Actions tab.

The published URL is <https://mateuswiteck.github.io/tsfmxai/> after the first
successful deployment. Save executed outputs before committing a notebook;
the website displays exactly those saved results. Images or downloads linked
from notebook Markdown must also be committed alongside the notebook.

## Local visualization

Run the mechanism studio from the repository root:

```powershell
.\.venv\Scripts\python.exe frontend\app.py
```

Then open <http://127.0.0.1:8000>. The interface samples a canonical mechanism
and generates multiple reproducible trajectories while keeping that mechanism
fixed.
