from __future__ import annotations

import numpy as np

from fetchr import kaggle_source


def _touch_npz(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, flux=np.ones(3))


def test_index_by_tic_id_nested_layout(tmp_path):
    _touch_npz(tmp_path / "planet" / "111.npz")
    _touch_npz(tmp_path / "eb" / "222.npz")

    index = kaggle_source.index_by_tic_id(tmp_path)
    assert set(index) == {"111", "222"}
    assert index["111"] == tmp_path / "planet" / "111.npz"


def test_index_by_tic_id_flat_layout(tmp_path):
    _touch_npz(tmp_path / "333.npz")
    index = kaggle_source.index_by_tic_id(tmp_path)
    assert set(index) == {"333"}


def test_index_by_tic_id_duplicate_keeps_first(tmp_path):
    _touch_npz(tmp_path / "a" / "444.npz")
    _touch_npz(tmp_path / "b" / "444.npz")
    index = kaggle_source.index_by_tic_id(tmp_path)
    assert index["444"] == tmp_path / "a" / "444.npz"


def test_download_and_extract_skips_if_already_staged(tmp_path, monkeypatch):
    _touch_npz(tmp_path / "planet" / "1.npz")

    def _boom():
        raise AssertionError("should not call the kaggle API when already staged")

    monkeypatch.setattr(kaggle_source, "_kaggle_api", _boom)
    result = kaggle_source.download_and_extract("someone/some-dataset", tmp_path)
    assert result == tmp_path
