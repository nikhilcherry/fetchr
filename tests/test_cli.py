from __future__ import annotations

from fetchr.cli import main


def test_cli_verify_reports_counts(tmp_path, manifest_csv, make_npz, capsys):
    manifest_path = manifest_csv([{"tic_id": 1, "label": "planet"}, {"tic_id": 2, "label": "eb"}])
    data_dir = tmp_path / "processed"
    make_npz(data_dir, "planet", 1)

    exit_code = main(["verify", "--manifest", str(manifest_path), "--data-dir", str(data_dir)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "present:        1" in out
    assert "missing:        1" in out
    assert "schema_invalid: 0" in out
