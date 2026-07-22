"""Contract + sanity validation for arvyo-data .npz files.

Structural checks mirror arvyo-pipeline/arvyo/contract.py's load_sample()
(validated against fetchr/schema.py's copied constants, not by importing
arvyo.contract -- see schema.py's docstring for why). Sanity checks (NaNs,
median flux ~= 1.0) mirror arvyo-data/scripts/verify_dataset.py's
spot_check(), same tolerance (atol=0.05), so the two repos and fetchr never
disagree about what counts as a valid file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema

FLUX_MEDIAN_ATOL = 0.05  # matches arvyo-data/scripts/verify_dataset.py


class ContractError(ValueError):
    """Raised when a .npz sample violates the schema-1.0 contract or a basic sanity check."""


def _scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def load_and_validate(path: str | Path) -> dict:
    """Load one .npz and validate it against schema-1.0 + basic sanity checks.

    Raises ContractError naming the file and the violated rule. Returns the
    validated dict of arrays/meta on success.
    """
    path = Path(path)
    if not path.exists():
        raise ContractError(f"{path}: file does not exist")
    if path.stat().st_size == 0:
        raise ContractError(f"{path}: file is zero bytes")

    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ContractError(f"{path}: could not load npz ({exc})") from exc

    keys = set(npz.files)

    missing_arrays = [k for k in schema.REQUIRED_ARRAYS if k not in keys]
    if missing_arrays:
        raise ContractError(f"{path}: missing required array(s) {missing_arrays}")

    missing_meta = [k for k in schema.REQUIRED_META if k not in keys]
    if missing_meta:
        raise ContractError(f"{path}: missing required meta field(s) {missing_meta}")

    time = np.asarray(npz["time"], dtype=np.float64)
    flux = np.asarray(npz["flux"], dtype=np.float64)
    flux_err = np.asarray(npz["flux_err"], dtype=np.float64)

    if time.ndim != 1:
        raise ContractError(f"{path}: 'time' must be 1D, got shape {time.shape}")
    if flux.shape != time.shape:
        raise ContractError(f"{path}: 'flux' shape {flux.shape} != 'time' shape {time.shape}")
    if flux_err.shape != time.shape:
        raise ContractError(f"{path}: 'flux_err' shape {flux_err.shape} != 'time' shape {time.shape}")

    sample: dict[str, Any] = {"time": time, "flux": flux, "flux_err": flux_err}

    if "flux_raw" in keys:
        flux_raw = np.asarray(npz["flux_raw"], dtype=np.float64)
        if flux_raw.shape != time.shape:
            raise ContractError(f"{path}: 'flux_raw' shape {flux_raw.shape} != 'time' shape {time.shape}")
        sample["flux_raw"] = flux_raw

    label = str(_scalar(npz["label"]))
    if label not in schema.LABELS:
        raise ContractError(f"{path}: label {label!r} not in {schema.LABELS}")
    sample["label"] = label

    sample["tic_id"] = _scalar(npz["tic_id"])
    sample["sector"] = _scalar(npz["sector"])

    for key in schema.OPTIONAL_META:
        if key in keys:
            sample[key] = _scalar(npz[key])

    # Sanity checks -- matches arvyo-data/scripts/verify_dataset.py's spot_check().
    if np.isnan(flux).any() or np.isnan(time).any() or np.isnan(flux_err).any():
        raise ContractError(f"{path}: NaNs present in time/flux/flux_err")
    med = np.nanmedian(flux)
    if not np.isclose(med, 1.0, atol=FLUX_MEDIAN_ATOL):
        raise ContractError(f"{path}: median flux {med:.4f} != 1.0 (atol={FLUX_MEDIAN_ATOL})")

    return sample


@dataclass
class VerifyReport:
    present: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    schema_invalid: list = field(default_factory=list)
    # tic_id -> error message, populated alongside schema_invalid so the CLI
    # can explain *why* without a second pass over the files.
    errors: dict = field(default_factory=dict)


def _read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    # keep_default_na=False / na_values=[""]: pandas' default NA sentinels
    # include the literal string "null", which is one of the label values
    # (the quiet-star class) -- same convention as arvyo-data's own scripts.
    return pd.read_csv(manifest_path, keep_default_na=False, na_values=[""])


def expected_path(data_dir: str | Path, label, tic_id) -> Path:
    """{data_dir}/{label}/{tic_id}.npz -- arvyo-data's own on-disk layout."""
    return Path(data_dir) / str(label) / f"{tic_id}.npz"


def verify(manifest_path: str | Path, data_dir: str | Path) -> VerifyReport:
    """Offline, contract-only check of data_dir against manifest_path.

    For every manifest row: present (file exists and passes the contract +
    sanity checks), missing (no file at the expected path), or
    schema_invalid (a file exists but fails validation). Needs no Kaggle
    credentials and no network -- this is the first thing to run on a fresh
    machine to see what's missing.
    """
    manifest = _read_manifest(manifest_path)
    report = VerifyReport()

    for row in manifest.itertuples(index=False):
        row_dict = row._asdict()
        tic_id = row_dict["tic_id"]
        label = row_dict["label"]
        path = expected_path(data_dir, label, tic_id)
        if not path.exists():
            report.missing.append(tic_id)
            continue
        try:
            load_and_validate(path)
            report.present.append(tic_id)
        except ContractError as exc:
            report.schema_invalid.append(tic_id)
            report.errors[tic_id] = str(exc)

    return report
