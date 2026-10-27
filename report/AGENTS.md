# Report instructions

Scientific claims require entries in `../references/library.bib` and support
from the cited source. The English manuscript lives under `english/`; keep
its generated artifacts under `english/build/`. Do not edit
generated figures or tables when their producing script can be updated instead.

Keep the delivered PDF at `../outputs/report/relatorio.pdf`. Refresh it through
`python scripts/prepare_report_site.py` from the repository root; do not create
dated or renamed report PDF copies. Store temporary visual-review images and
review logs under `../outputs/`, outside the manuscript and compiler cache.

Keep `english/main.tex` as the assembly point rather than adding manuscript prose to it.
Edit each reviewed paper in its corresponding file under
`english/chapters/related_work/`, and keep cross-paper comparisons in the relevant
`synthesis.tex`. Use `\input` for manuscript composition and preserve citation
keys, labels, and the established notation when moving content.
