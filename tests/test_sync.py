from __future__ import annotations

import fetchr
from fetchr import verify


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
