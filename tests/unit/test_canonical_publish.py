import hashlib
import json
from types import SimpleNamespace

import pytest

from tsfmxai.symbolic_regression.canonical_publish import (
    replacement_plan, publish_replacement, validate_release,
)


def test_replacement_removes_old_shards_in_same_commit_and_pins_parent(tmp_path):
    (tmp_path / "README.md").write_text("new corpus")
    (tmp_path / "new.parquet").write_bytes(b"new")
    class API:
        def repo_info(self, **kw): return SimpleNamespace(sha="previous")
        def list_repo_files(self, **kw):
            assert kw["revision"] == "previous"
            return [".gitattributes", "README.md", "old.parquet", "obsolete/config.json"]
        def create_commit(self, **kw): self.commit = kw; return "done"
    api = API()
    plan = replacement_plan(api, "user/corpus", "dataset", tmp_path)
    assert plan["delete"] == ["obsolete/config.json", "old.parquet"]
    assert publish_replacement(api, plan, "new release") == "done"
    assert api.commit["parent_commit"] == "previous"
    assert len(api.commit["operations"]) == 4


def test_upload_validation_rejects_corruption_and_extra_files(tmp_path):
    (tmp_path / "dataset").mkdir()
    p = tmp_path / "dataset/README.md"
    p.write_bytes(b"good")
    manifest = {"files": {"dataset/README.md": {"bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}}}
    (tmp_path / "release_manifest.json").write_text(json.dumps(manifest))
    assert validate_release(tmp_path) == manifest
    p.write_bytes(b"oops")
    with pytest.raises(ValueError, match="Checksum"):
        validate_release(tmp_path)
    (tmp_path / "dataset/stale.parquet").write_bytes(b"old")
    with pytest.raises(ValueError, match="manifest"):
        validate_release(tmp_path)
