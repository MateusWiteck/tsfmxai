"""Build website navigation without writing to experiment directories."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def notebook_navigation(repository: Path) -> list[dict]:
    """Return a directory tree with exact folder and notebook filenames."""
    experiments = repository / "experiments"

    def visit(directory: Path) -> list[dict]:
        entries = []
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or path.is_symlink():
                continue
            if path.is_dir():
                children = visit(path)
                if children:
                    entries.append({"section": path.name, "contents": children})
            elif path.suffix == ".ipynb":
                entries.append({"text": path.name, "href": path.relative_to(repository).as_posix()})
        return entries

    return [{"section": "experiments", "contents": visit(experiments)}]


def navigation_html(entries: list[dict]) -> str:
    """Render the same tree as accessible, expandable HTML."""
    lines = ['<ul class="notebook-tree">']
    for entry in entries:
        if "section" in entry:
            lines.append('<li><details open><summary><code>' + escape(entry["section"]) + '/</code></summary>')
            lines.append(navigation_html(entry["contents"]))
            lines.append('</details></li>')
        else:
            target = quote(str(Path(entry["href"]).with_suffix(".html")).replace("\\", "/"), safe="/")
            lines.append(f'<li><a href="{target}"><code>{escape(entry["text"])}</code></a></li>')
    lines.append('</ul>')
    return "\n".join(lines)


def prepare_notebook_site(repository: Path) -> None:
    """Write generated navigation only under ignored outputs/."""
    entries = notebook_navigation(repository)
    output = repository / "outputs"
    output.mkdir(exist_ok=True)
    metadata = {"website": {"sidebar": {"style": "docked", "collapse-level": 2,
                "contents": [{"href": "index.qmd", "text": "Experimentos"}, *entries]}}}
    (output / "notebook-navigation.yml").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "notebook-tree.html").write_text(navigation_html(entries) + "\n", encoding="utf-8")
