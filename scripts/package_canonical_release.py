"""Package a completed canonicalization run for Hugging Face (models + dataset).

    python scripts/package_canonical_release.py \
        --run-dir experiments/canonicalization/models/outputs/raw_tanh \
        --output release/canonical_ast
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.symbolic_regression.canonical_release import package_release  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="experiments/canonicalization/models/outputs/raw_tanh")
    parser.add_argument("--output", default="release/canonical_ast")
    args = parser.parse_args()
    release = package_release(ROOT / args.run_dir, ROOT / args.output, repo_root=ROOT)
    total = sum(item["bytes"] for item in release["files"].values())
    print(f"wrote {len(release['files'])} files, {total / 1e9:.2f} GB, to {args.output}")


if __name__ == "__main__":
    main()
