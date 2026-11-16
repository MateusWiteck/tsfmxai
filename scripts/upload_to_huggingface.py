"""Upload a packaged release (see package_canonical_release.py) to Hugging Face.

Run it on the machine that holds your Hugging Face token (`huggingface-cli login`):

    python scripts/upload_to_huggingface.py --release experiments/canonicalization/outputs/release/six_series \
        --models-repo USER/tsfmxai-canonical-encoder-decoder \
        --dataset-repo USER/tsfmxai-canonical-asts

Repositories are created private unless --public is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="experiments/canonicalization/outputs/release/six_series")
    parser.add_argument("--models-repo", required=True,
                        help="one model repository; each variant goes in its own subfolder")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--replace", action="store_true", help="Replace existing main snapshots, deleting stale files in the same commit")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show replacement plan without publishing")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    from tsfmxai.symbolic_regression.canonical_publish import (
        validate_release, replacement_plan, publish_replacement, configure_rate_limit_retries,
    )
    configure_rate_limit_retries()
    api, release = HfApi(), Path(args.release)
    manifest = validate_release(release)
    if args.replace or args.dry_run:
        plans = [replacement_plan(api, repo, kind, release / part) for repo, kind, part in (
            (args.models_repo, "model", "models"), (args.dataset_repo, "dataset", "dataset"))]
        for plan in plans:
            print(plan["repo_id"], "parent:", plan["parent_commit"],
                  "files:", len(plan["files"]), "remove:", plan["delete"], flush=True)
        if args.dry_run:
            return
        for plan in plans:
            result = publish_replacement(api, plan,
                "Replace with " + manifest["provenance"]["experiment"])
            print("published", result.commit_url, flush=True)
        return
    api.create_repo(args.models_repo, repo_type="model", private=not args.public, exist_ok=True)
    api.create_repo(args.dataset_repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.models_repo, repo_type="model", folder_path=release / "models",
                      commit_message="Upload canonical encoder-decoder models")
    api.upload_folder(repo_id=args.dataset_repo, repo_type="dataset", folder_path=release / "dataset",
                      commit_message="Upload canonical AST corpus")
    print("uploaded", args.models_repo, "and", args.dataset_repo)


if __name__ == "__main__":
    main()
