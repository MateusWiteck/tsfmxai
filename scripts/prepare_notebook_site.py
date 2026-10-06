"""Prepare Quarto navigation using only the Python standard library."""

from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository / "src"))

from tsfmxai.visualization.notebook_site import prepare_notebook_site

prepare_notebook_site(repository)
