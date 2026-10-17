"""Build and stage the report using the same LaTeX entry point as local writing."""

from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository / "src"))

from tsfmxai.visualization.report_site import prepare_report_site

if __name__ == "__main__":
    pdf = prepare_report_site(repository)
    print(f"Report ready: {pdf.relative_to(repository)} ({pdf.stat().st_size:,} bytes)")
