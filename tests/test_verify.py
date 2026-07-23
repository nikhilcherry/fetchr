from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from fetchr.verify import ContractError

# fetchr/__init__.py does `from .verify import verify`, which rebinds the
# `fetchr.verify` attribute to that function and shadows the submodule --
# importlib.import_module reads sys.modules directly, bypassing that.
verify = importlib.import_module("fetchr.verify")


def test_load_and_validate_accepts_valid_file(tmp_path, make_npz):
    path = make_npz(tmp_path, "planet", 12345)
    sample = verify.load_and_validate(path)
    assert sample["label"] == "planet"
    assert sample["tic_id"] == 12345


def test_load_and_validate_rejects_missing_file(tmp_path):
    with pytest.raises(ContractError, match="does not exist"):
        verify.load_and_validate(tmp_path / "nope.npz")


def test_load_and_validate_rejects_missing_required_array(tmp_path):
    path = tmp_path / "planet" / "1.npz"
    path.parent.mkdir(parents=True)
    with open(path, "wb") as f:
        np.savez(f, time=np.arange(10), flux=np.ones(10), tic_id=1, label="planet", sector=1)
    with pytest.raises(ContractError, match="missing required array"):
        verify.load_and_validate(path)


def test_load_and_validate_rejects_nans(tmp_path, make_npz):
    path = make_npz(tmp_path, "planet", 1, with_nan=True)
    with pytest.raises(ContractError, match="NaNs"):
        verify.load_and_validate(path)


def test_load_and_validate_rejects_bad_median_flux(tmp_path, make_npz):
    path = make_npz(tmp_path, "planet", 1, median_flux=1.5)
    with pytest.raises(ContractError, match="median flux"):
        verify.load_and_validate(path)


def test_load_and_validate_rejects_unknown_label(tmp_path):
    path = tmp_path / "bogus" / "1.npz"
    path.parent.mkdir(parents=True)
    n = 20
    with open(path, "wb") as f:
        np.savez(
            f, time=np.linspace(0, 1, n), flux=np.ones(n), flux_err=np.full(n, 0.01),
            tic_id=1, label="bogus", sector=1,
        )
    with pytest.raises(ContractError, match="not in"):
        verify.load_and_validate(path)


def test_load_and_validate_accepts_optional_flux_raw(tmp_path, make_npz):
    path = make_npz(tmp_path, "starspot", 1, flux_raw=np.ones(1000))
    sample = verify.load_and_validate(path)
    assert "flux_raw" in sample


def test_load_and_validate_rejects_too_few_cadences(tmp_path, make_npz):
    path = make_npz(tmp_path, "planet", 1, n=999)
    with pytest.raises(ContractError, match="cadences"):
        verify.load_and_validate(path)


def test_load_and_validate_rejects_infs(tmp_path, make_npz):
    path = make_npz(tmp_path, "planet", 1)
    with np.load(path, allow_pickle=True) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["flux"][0] = np.inf
    with open(path, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ContractError, match="infinite"):
        verify.load_and_validate(path)


def test_expected_path_matches_arvyo_data_layout():
    assert verify.expected_path("data/processed", "planet", 12345) == \
        Path("data/processed/planet/12345.npz")


def test_verify_buckets_present_missing_invalid(tmp_path, manifest_csv, make_npz):
    manifest_path = manifest_csv([
        {"tic_id": 1, "label": "planet"},
        {"tic_id": 2, "label": "eb"},
        {"tic_id": 3, "label": "null"},
    ])
    data_dir = tmp_path / "processed"
    make_npz(data_dir, "planet", 1)                       # present, valid
    make_npz(data_dir, "eb", 2, median_flux=1.5)           # present, invalid
    # tic_id 3 has no file at all -> missing

    report = verify.verify(manifest_path, data_dir)
    assert report.present == [1]
    assert report.missing == [3]
    assert report.schema_invalid == [2]
    assert "median flux" in report.errors[2]


def test_verify_handles_literal_null_label(tmp_path, manifest_csv, make_npz):
    # keep_default_na=False must be respected -- pandas' default NA
    # sentinels include the literal string "null", one of our label values.
    manifest_path = manifest_csv([{"tic_id": 9, "label": "null"}])
    data_dir = tmp_path / "processed"
    make_npz(data_dir, "null", 9)

    report = verify.verify(manifest_path, data_dir)
    assert report.present == [9]
    assert report.missing == []
