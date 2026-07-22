from __future__ import annotations

import numpy as np
import pytest


def _write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


@pytest.fixture
def make_npz(tmp_path):
    """Factory: write a schema-1.0-valid .npz at data_dir/{label}/{tic_id}.npz."""

    def _make(data_dir, label, tic_id, n=50, median_flux=1.0, with_nan=False, **extra):
        time = np.linspace(0, 10, n)
        flux = np.full(n, median_flux)
        if with_nan:
            flux[0] = np.nan
        flux_err = np.full(n, 0.001)
        path = data_dir / label / f"{tic_id}.npz"
        arrays = dict(
            time=time, flux=flux, flux_err=flux_err,
            tic_id=tic_id, label=label, sector=1,
            **extra,
        )
        _write_npz(path, **arrays)
        return path

    return _make


@pytest.fixture
def manifest_csv(tmp_path):
    """A tiny fixture manifest, same columns as arvyo-data's manifest.csv."""

    def _make(rows):
        path = tmp_path / "manifest.csv"
        header = "tic_id,label,source_catalog,disposition,period_days,epoch_btjd,depth_ppm,duration_hours,tmag,notes"
        lines = [header]
        for r in rows:
            lines.append(
                f"{r['tic_id']},{r['label']},exofop_toi,CP,1.0,2459000.0,100.0,1.0,10.0,"
            )
        path.write_text("\n".join(lines) + "\n")
        return path

    return _make
