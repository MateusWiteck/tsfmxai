from pathlib import Path
import re
import subprocess
from unittest.mock import Mock

import pytest

from tsfmxai.visualization import report_site


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "report/english/build").mkdir(parents=True)
    (tmp_path / "references").mkdir()
    (tmp_path / "report/english/main.tex").write_text("canonical LaTeX")
    (tmp_path / "references/library.bib").write_text("canonical bibliography")
    return tmp_path


def fake_build(repository, pdf=b"%PDF-1.5\ncompiled report", log="Build completed"):
    (repository / "report/english/build/main.pdf").write_bytes(pdf)
    (repository / "report/english/build/main.log").write_text(log)


def test_builds_from_report_directory_and_copies_exact_pdf(repository, monkeypatch):
    monkeypatch.setattr(report_site.shutil, "which", lambda name: "/tools/latexmk")
    run = Mock(side_effect=lambda *args, **kwargs: fake_build(repository))
    monkeypatch.setattr(report_site.subprocess, "run", run)

    result = report_site.prepare_report_site(repository)

    run.assert_called_once_with(
        ["/tools/latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-outdir=build", "main.tex"],
        cwd=repository / "report/english", check=True,
    )
    assert result == repository / report_site.REPORT_RESOURCE
    assert result.read_bytes() == (repository / "report/english/build/main.pdf").read_bytes()
    assert (repository / "report/english/main.tex").read_text() == "canonical LaTeX"
    assert (repository / "references/library.bib").read_text() == "canonical bibliography"


def test_build_failure_propagates_instead_of_staging_stale_pdf(repository, monkeypatch):
    fake_build(repository)
    monkeypatch.setattr(report_site.shutil, "which", lambda name: "/tools/latexmk")
    monkeypatch.setattr(report_site.subprocess, "run", Mock(
        side_effect=subprocess.CalledProcessError(1, ["latexmk"])))
    with pytest.raises(subprocess.CalledProcessError):
        report_site.prepare_report_site(repository)
    assert not (repository / report_site.REPORT_RESOURCE).exists()


def test_rebuild_updates_the_same_public_pdf(repository, monkeypatch):
    monkeypatch.setattr(report_site.shutil, "which", lambda name: "/tools/latexmk")
    versions = iter((b"%PDF-1.5\nfirst", b"%PDF-1.5\nsecond"))
    monkeypatch.setattr(report_site.subprocess, "run", Mock(
        side_effect=lambda *args, **kwargs: fake_build(repository, pdf=next(versions))))
    first = report_site.prepare_report_site(repository)
    second = report_site.prepare_report_site(repository)
    assert first == second == repository / "outputs/report/relatorio.pdf"
    assert second.read_bytes() == b"%PDF-1.5\nsecond"
    assert list(second.parent.glob("*.pdf")) == [second]


def test_manual_latexmk_builds_keep_artifacts_in_build_directory():
    root = Path(__file__).resolve().parents[2]
    config = (root / "report/english/latexmkrc").read_text(encoding="utf-8")
    assert re.search(r"\$out_dir\s*=\s*['\"]build['\"]\s*;", config)


def test_missing_tool_has_actionable_error(repository, monkeypatch):
    monkeypatch.setattr(report_site.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Install TeX Live or MiKTeX"):
        report_site.prepare_report_site(repository)


@pytest.mark.parametrize("missing", ["report/english/main.tex", "references/library.bib"])
def test_requires_canonical_sources(repository, missing):
    (repository / missing).unlink()
    with pytest.raises(FileNotFoundError, match="Required report source"):
        report_site.prepare_report_site(repository)


@pytest.mark.parametrize("warning", [
    "LaTeX Warning: There were undefined references.",
    "Package natbib Warning: There were undefined citations.",
])
def test_rejects_unresolved_references(repository, monkeypatch, warning):
    fake_build(repository, log=warning)
    monkeypatch.setattr(report_site.shutil, "which", lambda name: "/tools/latexmk")
    monkeypatch.setattr(report_site.subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="unresolved references"):
        report_site.prepare_report_site(repository)
    assert not (repository / report_site.REPORT_RESOURCE).exists()


def test_rejects_invalid_pdf(repository, monkeypatch):
    fake_build(repository, pdf=b"error page")
    monkeypatch.setattr(report_site.shutil, "which", lambda name: "/tools/latexmk")
    monkeypatch.setattr(report_site.subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="valid PDF header"):
        report_site.prepare_report_site(repository)


def test_site_wires_report_and_builds_it_for_every_main_push():
    root = Path(__file__).resolve().parents[2]
    config = (root / "_quarto.yml").read_text(encoding="utf-8")
    page = (root / "relatorio.qmd").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/publish-notebooks.yml").read_text(encoding="utf-8")
    resource = report_site.REPORT_RESOURCE.as_posix()
    assert "- relatorio.qmd" in config and "href: relatorio.qmd" in config
    assert resource in config and resource in page
    assert 'type="application/pdf"' in page and "sidebar: false" in page
    assert "download=" in page and 'target="_blank"' in page
    assert "paths:" not in workflow and "branches: [main]" in workflow
    assert workflow.index("python scripts/prepare_report_site.py") < workflow.index("quarto render")
    assert "cmp report/english/build/main.pdf site/" + resource in workflow


def test_manuscript_split_preserves_inputs_labels_and_chapter_order():
    report = Path(__file__).resolve().parents[2] / "report" / "english"
    visited = set()

    def expand(relative):
        path = report / relative
        assert path.is_file(), f"Missing LaTeX input: {relative}"
        assert path not in visited, f"Repeated LaTeX input: {relative}"
        visited.add(path)
        source = path.read_text(encoding="utf-8")
        source = re.sub(r"(?<!\\)%[^\n]*", "", source)
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda match: expand(match[1] + ".tex"),
            source,
        )

    manuscript = expand("main.tex")
    chapters = re.findall(r"\\section\{([^{}]+)\}", manuscript)
    assert chapters == [
        "Related Work", "Research Goal and Research Question", "Research Plan",
        "Synthetic Time-Series Generator",
        "Stability and Quality Control of Generated Series",
        "Different Initializations for Time Series",
        "Explanation Ground-Truth Generation",
        "Benchmark API and Evaluation Methodology", "Experimental Design",
        "Results and Discussion", "Conclusion and Future Work",
    ]
    labels = re.findall(r"\\label\{([^}]+)\}", manuscript)
    for options in re.findall(r"\\begin\{lstlisting\}\[([^\]]*)\]", manuscript):
        labels.extend(re.findall(r"label\s*=\s*\{?([^,}\s]+)", options))
    assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
    references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", manuscript))
    assert references <= set(labels), f"Undefined labels: {references - set(labels)}"
    generator = manuscript.split(r"\section{Synthetic Time-Series Generator}", 1)[1]
    generator, remainder = generator.split(
        r"\section{Stability and Quality Control of Generated Series}", 1)
    stability = remainder.split(r"\section{Different Initializations for Time Series}", 1)[0]
    results = manuscript.split(r"\section{Results and Discussion}", 1)[1]
    assert len(re.findall(r"\\subsection\{", generator)) == 6
    assert len(re.findall(r"\\subsection\{", stability)) == 7
    for label in ("tab:implemented-ar-evidence", "tab:generator-outcomes",
                  "tab:generator-speed", "tab:generator-tail-outcomes"):
        assert rf"\label{{{label}}}" in results
        assert rf"\label{{{label}}}" not in generator + stability
