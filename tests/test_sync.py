from __future__ import annotations

import importlib
import shutil

import numpy as np

import fetchr
from fetchr import kaggle_source

# see test_worker.py for why this needs importlib rather than `from fetchr
# import verify` (fetchr/__init__.py shadows the submodule name with a function).
verify = importlib.import_module("fetchr.verify")


def test_sync_counts_pre_existing_valid_files(tmp_path, manifest_csv, make_npz):
    """A row whose output .npz already exists and validates should be
    reported as 'already present', not silently missing from every bucket
    in the summary (from_kaggle + from_mast + from_existing must add up)."""
    output_dir = tmp_path / "out"
    make_npz(output_dir, "planet", 42)

    manifest = manifest_csv([{"tic_id": 42, "label": "planet"}])

    report = fetchr.sync(
        manifest,
        output_dir=output_dir,
        cache_dir=tmp_path / ".fetchr_cache",
        workers=1,
    )

    assert report.total == 1
    assert report.from_existing == 1
    assert report.from_kaggle == 0
    assert report.from_mast == 0
    assert report.failed == 0
    assert "1 already present" in report.summary()
    assert (
        report.from_kaggle + report.from_mast + report.from_existing + report.failed
        == report.total
    )


def test_sync_force_redoes_pre_existing_valid_files(tmp_path, manifest_csv, make_npz, monkeypatch):
    # sync()'s own docstring: "pass force=True to redo everything" -- a
    # file that already exists and validates must actually be re-fetched
    # under force=True, not silently short-circuited as 'existing' just
    # because sync_one_item's own idempotency check doesn't know about force.
    output_dir = tmp_path / "out"
    make_npz(output_dir, "planet", 42, n=20)

    kaggle_dir = tmp_path / "kaggle_dataset"
    make_npz(kaggle_dir, "planet", 42, n=30)
    monkeypatch.setattr(
        kaggle_source, "download_and_extract",
        lambda dataset, staging_dir, force=False: (
            shutil.copytree(kaggle_dir, staging_dir, dirs_exist_ok=True), staging_dir
        )[1],
    )

    manifest = manifest_csv([{"tic_id": 42, "label": "planet"}])

    report = fetchr.sync(
        manifest, kaggle_dataset="fake/dataset", output_dir=output_dir,
        cache_dir=tmp_path / ".fetchr_cache", workers=1, force=True,
    )

    assert report.from_kaggle == 1
    assert report.from_existing == 0
    dest = verify.expected_path(output_dir, "planet", 42)
    assert len(np.load(dest)["time"]) == 30
