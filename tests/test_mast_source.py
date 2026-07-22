from __future__ import annotations

import pytest

from fetchr import mast_source
from fetchr.mast_source import ArvyoDataNotFoundError


def _make_checkout(tmp_path):
    root = tmp_path / "arvyo-data"
    (root / "scripts").mkdir(parents=True)
    (root / "manifest.csv").write_text("tic_id,label\n1,planet\n")
    return root


def test_resolve_via_manifest_path(tmp_path):
    root = _make_checkout(tmp_path)
    resolved = mast_source.resolve_arvyo_data_root(manifest_path=root / "manifest.csv")
    assert resolved == root


def test_resolve_via_explicit_arvyo_data_path(tmp_path):
    root = _make_checkout(tmp_path)
    resolved = mast_source.resolve_arvyo_data_root(arvyo_data_path=root)
    assert resolved == root


def test_resolve_via_env_var(tmp_path, monkeypatch):
    root = _make_checkout(tmp_path)
    monkeypatch.setenv("ARVYO_DATA_PATH", str(root))
    resolved = mast_source.resolve_arvyo_data_root()
    assert resolved == root


def test_resolve_raises_without_any_candidate(monkeypatch):
    monkeypatch.delenv("ARVYO_DATA_PATH", raising=False)
    with pytest.raises(ArvyoDataNotFoundError, match="Could not locate"):
        mast_source.resolve_arvyo_data_root()


def test_resolve_raises_when_not_a_real_checkout(tmp_path):
    # manifest.csv sitting alone, no sibling scripts/ dir -- e.g. someone
    # copied just the manifest onto a fresh machine.
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("tic_id,label\n1,planet\n")
    with pytest.raises(ArvyoDataNotFoundError, match="does not look like"):
        mast_source.resolve_arvyo_data_root(manifest_path=manifest_path)


def test_rebuild_one_rejects_unknown_schema_version(tmp_path):
    root = _make_checkout(tmp_path)
    with pytest.raises(ValueError, match="schema_version"):
        mast_source.rebuild_one(
            1, tmp_path / "out", label="planet", mission="tess",
            meta={}, manifest_path=root / "manifest.csv", schema_version="9.9",
        )
