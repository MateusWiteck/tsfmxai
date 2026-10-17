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

## Stationary-by-construction generation

The library can sample Gaussian AR and VAR(1) mechanisms whose stationarity is
certified before trajectory generation. AR coefficients are obtained from
partial autocorrelations strictly inside `(-1, 1)`. VAR(1) coefficient matrices
are sampled with spectral radius below one. Both generators draw their initial
state from the stationary covariance, so the returned series does not begin
with an arbitrary initialization transient.

```python
from tsfmxai.generators import (
    StationaryARConfig,
    sample_stationary_ar_mechanism,
    sample_stationary_ar_trajectories,
)

mechanism = sample_stationary_ar_mechanism(
    StationaryARConfig(minimum_order=3, maximum_order=3),
    seed=7,
)
series = sample_stationary_ar_trajectories(
    mechanism,
    trajectory_count=32,
    generated_steps=256,
    root_seed=11,
)

assert mechanism.certificate.holds
print(mechanism.formula)
```

## Contraction-certified nonlinear generation

Nonlinear lagged ASTs can be scaled and certified on a bounded augmented-state
domain. The certificate uses interval derivative bounds and a weighted infinity
norm so both the recurrence row and lag-shift rows have gain below one. Hard
domain invariance requires the bounded innovation law stored by the mechanism.

```python
from tsfmxai.generators import (
    ContractionTreeConfig,
    sample_contractive_mechanism,
    sample_contractive_trajectories,
)

mechanism = sample_contractive_mechanism(
    ContractionTreeConfig(),
    seed=7,
)
series = sample_contractive_trajectories(
    mechanism,
    trajectory_count=32,
    generated_steps=256,
    root_seed=11,
)

assert mechanism.certificate.holds
assert abs(series).max() <= mechanism.certificate.domain_radius + 1e-12
```

## Published experiments and report

The Quarto website publishes saved outputs from every notebook under
`experiments/`, including nested directories. New notebooks are discovered
automatically. `experiments_local/` and `notebooks/` are outside the site.
Rendering never executes notebook cells or requires the experiment dependencies.
The **Relatório** tab embeds the PDF compiled from `report/main.tex` and offers
open/download links, preserving the manuscript's equations, figures, tables,
and bibliography. The same PDF is staged under ignored `outputs/report/` and
copied into the website; the LaTeX source remains the single source of truth.

Install [Quarto](https://quarto.org/docs/get-started/) (the workflow pins 1.8.27)
and TeX Live or MiKTeX with `latexmk`, then run from the repository root:

```powershell
python scripts/prepare_notebook_site.py
python scripts/prepare_report_site.py
quarto preview
# Build only, without starting a server:
quarto render --no-execute
```

Run the preparation command again after adding, moving, or renaming notebooks.
Run `prepare_report_site.py` again after editing the report or bibliography to
refresh the local preview. It stops on compilation errors or unresolved
references, so a failed build cannot silently publish an older report.
The homepage and sidebar preserve the exact directory names and notebook
filenames from `experiments/`. Navigation is generated under ignored `outputs/`;
the preparation command never writes to the experiment directories. Only
committed notebooks are available to the GitHub Actions build.

Generated HTML is written to ignored `site/`; Quarto state in `.quarto/` is also
ignored. Keep experiment records and outputs inside the executed notebook as
described in `REPRODUCIBILITY.md`.

To enable publication, select **Settings > Pages > Build and deployment >
Source > GitHub Actions** in the GitHub repository. Commit and push the site
configuration, notebooks, and all report sources to `main`. The
`Publish experiments and report` workflow compiles the report and deploys the
whole site on every push to `main`; pull requests build without deploying.
It can also be run manually from the Actions tab. Local commits become public
after they are pushed and the workflow completes successfully.

The published URL is <https://mateuswiteck.github.io/tsfmxai/> after the first
successful deployment. Save executed outputs before committing a notebook;
the website displays exactly those saved results. Images or downloads linked
from notebook Markdown must also be committed alongside the notebook.
The report page is <https://mateuswiteck.github.io/tsfmxai/relatorio.html>.
Commit new LaTeX chapter/figure sources and bibliography edits together; the
PDFs and LaTeX build files remain ignored. The embedded reader uses the browser's
PDF support, with direct links available on mobile and unsupported viewers.

## Local mechanism studio

Run the mechanism studio from the repository root:

```powershell
.\.venv\Scripts\python.exe frontend\app.py
```

Then open <http://127.0.0.1:8000>. The interface samples a canonical two-series
system, retains one equation per target, and generates multiple reproducible
multivariate trajectories while keeping both mechanisms fixed.
