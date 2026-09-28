# Thesis report

`main.tex` is the current manuscript. The canonical bibliography is
`../references/library.bib`; `latexmkrc` adds that directory to BibTeX's search
path without duplicating the bibliography. Generated LaTeX artifacts belong
under `build/` and are ignored by Git.

From this directory, the current manuscript can be compiled with:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```
