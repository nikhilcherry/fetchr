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


def test_cli_verify_missing_manifest_prints_clean_error(tmp_path, capsys):
    exit_code = main([
        "verify", "--manifest", str(tmp_path / "nope.csv"), "--data-dir", str(tmp_path),
    ])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "Traceback" not in err
    assert "Error:" in err


def test_cli_sync_missing_manifest_prints_clean_error(tmp_path, capsys):
    exit_code = main([
        "sync", "--manifest", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path / "out"),
    ])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "Traceback" not in err
    assert "Error:" in err
