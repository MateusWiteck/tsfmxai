# Thesis report

`main.tex` is the manuscript entry point. It contains the preamble and assembles
the front matter and chapters with `\input`. Manuscript content is organized as
follows:

- `frontmatter/`: notation and other material before the numbered chapters;
- `chapters/`: research goal, research plan, and the related-work index;
- `chapters/related_work/`: one file per reviewed paper plus synthesis files;
- `figures/manual/`: manually maintained LaTeX figures;
- `figures/generated/`: figures produced by scripts;
- `tables/`: standalone tables when separating them improves reuse or clarity.

The canonical bibliography is `../references/library.bib`; `latexmkrc` adds
that directory to BibTeX's search path without duplicating the bibliography.
Generated LaTeX artifacts belong under `build/` and are ignored by Git.

From this directory, the current manuscript can be compiled with:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

## Website publication

The website's **Relatório** tab displays the PDF produced from this entry point.
From the repository root, `python scripts/prepare_report_site.py` compiles the
report and stages an identical copy in `outputs/report/relatorio.pdf` for Quarto.
The HTML wrapper lives in `relatorio.qmd`; report content is maintained only in
LaTeX. Re-run the preparation script after edits when previewing locally.

GitHub Actions performs these steps on every push to `main`, then publishes
the report together with the saved notebook outputs. Compilation errors or
unresolved references stop publication. Commit all referenced chapter and
figure sources and `references/library.bib`; generated PDFs remain ignored.
