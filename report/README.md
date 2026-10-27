# Thesis report

`english/` contains the existing thesis. `english/main.tex` is its manuscript
entry point. It contains the preamble and assembles
the front matter and chapters with `\input`. Manuscript content is organized as
follows:

- `english/frontmatter/`: notation and other material before the numbered chapters;
- `english/chapters/`: research goal, research plan, and the related-work index;
- `english/chapters/related_work/`: one file per reviewed paper plus synthesis files;
- `english/chapters/implemented_generator/`: Chapter 4, covering vocabulary and ASTs,
  mechanism construction, trajectory execution, seeds, and parallel batches;
- `english/chapters/stability_control/`: Chapter 5, covering desired properties,
  diagnostics, the original baseline, Jacobian contraction, both nonlinear
  saturation modes, Heaps' VAR construction, and the guarantee comparison;
- `english/chapters/06_time_series_initialization.tex`: Chapter 6, covering startup
  choices, generator-specific restrictions, and controlled comparisons;
- `english/chapters/experimental_design/`: generator study configurations and timing
  protocols, assembled by Chapter 9;
- `english/chapters/results/`: recorded AR-coverage and generator-trade-off measurements,
  assembled by Chapter 10;
- `english/figures/manual/`: manually maintained LaTeX figures;
- `english/figures/generated/`: figures produced by scripts;
- `english/tables/`: standalone tables when separating them improves reuse or clarity.

The canonical bibliography is `../references/library.bib`; `english/latexmkrc` adds
that directory to BibTeX's search path without duplicating the bibliography.
Generated LaTeX artifacts belong under `english/build/` and are ignored by Git.
`english/latexmkrc` sets this output directory even for manual `latexmk -pdf main.tex`
commands. Keep temporary previews, screenshots, and review logs outside this
folder, under the repository's ignored `outputs/` directory.
Chapters 7--11 cover explanation ground truths, the benchmark API, experimental
design, results, and conclusions. Refer to chapters, sections, equations, and
tables through their LaTeX labels so numbering follows reorganizations.

The stable PDF to open or share is **`outputs/report/relatorio.pdf`**, relative
to the repository root. Refresh that same file after manuscript changes with:

```text
python scripts/prepare_report_site.py
```

Run this command from the repository root. It is also the website build entry
point. `report/english/build/main.pdf` is the internal compiler output; the preparation
script updates the stable PDF only after a successful, validated build.

For a compiler-only build from `report/english/`:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

## Website publication

The website's **Relatório** tab displays the PDF produced from this entry point.
From the repository root, `python scripts/prepare_report_site.py` compiles the
report and stages an identical copy in `outputs/report/relatorio.pdf` for Quarto.
The HTML wrapper lives in `relatorio.qmd`; the current report content is
maintained in English LaTeX. Re-run the preparation script after edits when
previewing locally.

GitHub Actions performs these steps on every push to `main`, then publishes
the report together with the saved notebook outputs. Compilation errors or
unresolved references stop publication. Commit all referenced chapter and
figure sources and `references/library.bib`; generated PDFs remain ignored.
