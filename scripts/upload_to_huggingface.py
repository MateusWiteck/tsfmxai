"""Upload a packaged release (see package_canonical_release.py) to Hugging Face.

Run it on the machine that holds your Hugging Face token (`huggingface-cli login`):

    python scripts/upload_to_huggingface.py --release release/canonical_ast \
        --models-repo USER/tsfmxai-canonical-encoder-decoder \
        --dataset-repo USER/tsfmxai-canonical-asts

Repositories are created private unless --public is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="release/canonical_ast")
    parser.add_argument("--models-repo", required=True,
                        help="one model repository; each variant goes in its own subfolder")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api, release = HfApi(), Path(args.release)
    api.create_repo(args.models_repo, repo_type="model", private=not args.public, exist_ok=True)
    api.create_repo(args.dataset_repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.models_repo, repo_type="model", folder_path=release / "models",
                      commit_message="Upload canonical encoder-decoder models")
    api.upload_folder(repo_id=args.dataset_repo, repo_type="dataset", folder_path=release / "dataset",
                      commit_message="Upload canonical AST corpus")
    print("uploaded", args.models_repo, "and", args.dataset_repo)


if __name__ == "__main__":
    main()
