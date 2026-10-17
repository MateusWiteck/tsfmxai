"""Compile the canonical LaTeX report and stage its PDF for the Quarto site."""

from pathlib import Path
import shutil
import subprocess


REPORT_RESOURCE = Path("outputs/report/relatorio.pdf")


def prepare_report_site(repository: Path) -> Path:
    """Build the report before copying it, failing the site build on any error.

    Only LaTeX build files and the ignored website resource are written. The
    manuscript and bibliography remain the single source of report content.
    A failed build never silently publishes a previously compiled PDF.
    """
    repository = Path(repository).resolve()
    report = repository / "report"
    for required in (report / "main.tex", repository / "references/library.bib"):
        if not required.is_file():
            raise FileNotFoundError(f"Required report source is missing: {required}")

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required. Install TeX Live or MiKTeX with latexmk.")

    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-outdir=build", "main.tex"],
        cwd=report,
        check=True,
    )

    log = report / "build/main.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if "There were undefined references" in log_text or "There were undefined citations" in log_text:
        raise RuntimeError(f"The report has unresolved references or citations. Inspect {log}")

    pdf = report / "build/main.pdf"
    with pdf.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise RuntimeError(f"LaTeX did not produce a valid PDF header: {pdf}")

    destination = repository / REPORT_RESOURCE
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, destination)
    return destination
