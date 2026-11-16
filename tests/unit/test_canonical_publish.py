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


def test_rate_limited_verification_is_retried_without_sleeping_in_test(monkeypatch):
    import requests
    import huggingface_hub
    import time
    from tsfmxai.symbolic_regression.canonical_publish import configure_rate_limit_retries
    factories, sleeps, calls = [], [], []
    responses = [requests.Response(), requests.Response()]
    responses[0].status_code = 429
    responses[0].headers["Retry-After"] = "2"
    responses[0]._content = b""
    responses[0]._content_consumed = True
    responses[1].status_code = 200
    def request(self, *args, **kwargs):
        calls.append((args, kwargs))
        return responses[len(calls)-1]
    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(huggingface_hub, "configure_http_backend", lambda backend_factory: factories.append(backend_factory))
    configure_rate_limit_retries()
    response = factories[0]().request("POST", "https://huggingface.co/example/verify", data=b"verification")
    assert response.status_code == 200
    assert sleeps == [2.]
    assert calls[0] == calls[1]
