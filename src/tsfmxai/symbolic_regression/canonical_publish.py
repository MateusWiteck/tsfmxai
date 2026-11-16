"""Publish an exact release snapshot without leaving stale corpus shards on main."""
from pathlib import Path
import hashlib
import json


def validate_release(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    expected = set(manifest["files"])
    actual = {p.relative_to(root).as_posix() for part in ("models", "dataset")
              for p in (root / part).rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Release files differ from the manifest")
    for name, info in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Release path escapes its directory")
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest


def replacement_plan(api, repo_id, repo_type, folder):
    """Pin the parent so concurrent remote edits cannot be silently overwritten."""
    folder = Path(folder)
    local = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
    if "README.md" not in local:
        raise ValueError("Release requires a root README.md")
    head = api.repo_info(repo_id=repo_id, repo_type=repo_type).sha
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=head))
    stale = sorted(remote - local.keys() - {".gitattributes"})
    return dict(repo_id=repo_id, repo_type=repo_type, parent_commit=head,
                files=local, delete=stale)


def publish_replacement(api, plan, message):
    """Add/replace and delete in one commit; retain the repository's history."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                  for name, path in sorted(plan["files"].items())]
    operations += [CommitOperationDelete(path_in_repo=name) for name in plan["delete"]]
    return api.create_commit(repo_id=plan["repo_id"], repo_type=plan["repo_type"],
                             parent_commit=plan["parent_commit"], operations=operations,
                             commit_message=message)
