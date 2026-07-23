from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from fetchr import _worker

# see test_verify.py for why this needs importlib rather than `import
# fetchr.verify as verify` (fetchr/__init__.py shadows the submodule name).
verify = importlib.import_module("fetchr.verify")


def _write_config(tmp_path, output_dir, rows, kaggle_index, rebuild_missing=False):
    config = {
        "output_dir": str(output_dir),
        "rebuild_missing": rebuild_missing,
        "manifest_path": str(tmp_path / "manifest.csv"),
        "arvyo_data_path": None,
        "kaggle_index": kaggle_index,
        "rows": rows,
    }
    config_path = tmp_path / "sync_worker_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture(autouse=True)
def _reset_worker_cache(monkeypatch, tmp_path):
    _worker.reset_config_cache()
    yield
    _worker.reset_config_cache()


def _write_source_npz(path, tic_id, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1000
    with open(path, "wb") as f:
        np.savez(
            f, time=np.linspace(0, 1, n), flux=np.ones(n), flux_err=np.full(n, 0.01),
            tic_id=tic_id, label=label, sector=1,
        )


def test_sync_one_item_copies_from_kaggle(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    kaggle_file = tmp_path / "staging" / "planet" / "42.npz"
    _write_source_npz(kaggle_file, 42, "planet")

    rows = {"42": {"tic_id": 42, "label": "planet"}}
    config_path = _write_config(tmp_path, output_dir, rows, {"42": str(kaggle_file)})
    monkeypatch.setenv("FETCHR_SYNC_CONFIG", str(config_path))

    result = _worker.sync_one_item("42")
    assert result["source"] == "kaggle"
    dest = verify.expected_path(output_dir, "planet", 42)
    assert dest.exists()
    verify.load_and_validate(dest)  # doesn't raise


def test_sync_one_item_skips_existing_valid_file(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    dest = verify.expected_path(output_dir, "planet", 42)
    _write_source_npz(dest, 42, "planet")

    rows = {"42": {"tic_id": 42, "label": "planet"}}
    # no kaggle entry at all -- if the worker tried to use it, this would KeyError
    config_path = _write_config(tmp_path, output_dir, rows, {})
    monkeypatch.setenv("FETCHR_SYNC_CONFIG", str(config_path))

    result = _worker.sync_one_item("42")
    assert result["source"] == "existing"


def test_sync_one_item_raises_missing_source_when_rebuild_disabled(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    rows = {"42": {"tic_id": 42, "label": "planet"}}
    config_path = _write_config(tmp_path, output_dir, rows, {}, rebuild_missing=False)
    monkeypatch.setenv("FETCHR_SYNC_CONFIG", str(config_path))

    with pytest.raises(_worker.MissingSourceError):
        _worker.sync_one_item("42")
