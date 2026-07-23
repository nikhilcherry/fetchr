"""fetchr: contract-aware dataset acquisition for arvyo-data's processed .npz corpus.

Reads arvyo-data's manifest.csv, pulls the bulk .npz corpus from Kaggle as
the fast path, and falls back to a per-target MAST/lightkurve rebuild (via
mast_source.py, reusing arvyo-data's own build scripts) when a row isn't in
the Kaggle dataset or Kaggle is unavailable -- verifying every file against
schema-1.0 (schema.py, verify.py) as it goes, so "the sync finished" and
"the data is actually valid" are the same claim.

Public API: sync, verify, rebuild_one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from batchr import run_batch
from batchr.cache import load as _load_cached_value

from . import _worker, kaggle_source, schema
from .mast_source import rebuild_one
from .verify import ContractError, VerifyReport, verify

__all__ = ["sync", "verify", "rebuild_one", "SyncReport", "VerifyReport", "ContractError"]


def _json_safe(value):
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return value.item()
    return value


@dataclass
class SyncReport:
    total: int
    from_kaggle: int
    from_mast: int
    from_existing: int
    failed: int
    wall_time_s: float
    results: list = field(default_factory=list)

    def failed_items(self) -> list:
        return [r["tic_id"] for r in self.results if r["status"] == "failed"]

    def summary(self) -> str:
        return (
            f"{self.total} item(s): {self.from_kaggle} from kaggle, "
            f"{self.from_mast} from mast, {self.from_existing} already present, "
            f"{self.failed} failed, in {self.wall_time_s:.2f}s."
        )


def sync(
    manifest_path: str | Path,
    kaggle_dataset: str | None = None,
    output_dir: str | Path = "data/processed/",
    workers: int = 0,
    rebuild_missing: bool = False,
    cache_dir: str | Path = ".fetchr_cache",
    limit: int | None = None,
    arvyo_data_path: str | Path | None = None,
    force: bool = False,
) -> SyncReport:
    """Reconstruct ``{output_dir}/{label}/{tic_id}.npz`` for every manifest row.

    Kaggle is the fast path (requires ``fetchr[kaggle]`` + a
    ``~/.kaggle/kaggle.json`` token). Rows not found in the Kaggle dataset
    are left missing unless ``rebuild_missing=True``, in which case they're
    re-derived one at a time via ``mast_source.rebuild_one`` (requires
    ``fetchr[rebuild]``). Every written file is verified against schema-1.0
    before being counted as done -- a schema-invalid result is a failure,
    not a silent partial success.

    Resumability, dedup, and parallelism across rows come from batchr: the
    per-row worker (``fetchr._worker.sync_one_item``) is handed to
    ``batchr.run_batch(..., cache_dir=cache_dir, workers=workers)`` -- see
    batchr's README for the crash-safety guarantees that come with that for
    free. ``kaggle_dataset``/``output_dir``/``rebuild_missing`` are part of
    the config dict batchr hashes into its cache key, so changing any of
    them automatically invalidates just the cache entries they affect;
    changes to the *contents* of an already-downloaded Kaggle dataset are
    not auto-detected (pass ``force=True`` to redo everything).
    """
    manifest = pd.read_csv(manifest_path, keep_default_na=False, na_values=[""])
    if limit:
        manifest = manifest.head(limit)

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for row in manifest.itertuples(index=False):
        d = {k: _json_safe(v) for k, v in row._asdict().items()}
        rows[str(d["tic_id"])] = d

    kaggle_index: dict[str, str] = {}
    if kaggle_dataset:
        staging_dir = cache_dir / "kaggle_staging"
        kaggle_source.download_and_extract(kaggle_dataset, staging_dir, force=force)
        kaggle_index = {k: str(v) for k, v in kaggle_source.index_by_tic_id(staging_dir).items()}

    worker_config = {
        "output_dir": str(output_dir),
        "rebuild_missing": bool(rebuild_missing),
        "manifest_path": str(manifest_path),
        "arvyo_data_path": str(Path(arvyo_data_path).resolve()) if arvyo_data_path else None,
        "kaggle_index": kaggle_index,
        "rows": rows,
    }
    config_path = cache_dir / "sync_worker_config.json"
    with open(config_path, "w") as f:
        json.dump(worker_config, f)
    # Workers read this via FETCHR_SYNC_CONFIG -- see _worker.py's
    # docstring for why (batchr never forwards extra args to fn, only the
    # item string).
    os.environ["FETCHR_SYNC_CONFIG"] = str(config_path)
    _worker.reset_config_cache()

    items = list(rows.keys())
    batch_report = run_batch(
        _worker.sync_one_item,
        items,
        cache_dir=str(cache_dir),
        config={
            "output_dir": str(output_dir),
            "rebuild_missing": bool(rebuild_missing),
            "kaggle_dataset": kaggle_dataset,
            "schema_version": schema.SCHEMA_VERSION,
        },
        workers=workers,
        force=force,
    )

    from_kaggle = from_mast = from_existing = 0
    results = []
    for r in batch_report.results:
        if r.status == "failed":
            results.append({"tic_id": r.item, "status": "failed", "source": None, "error": r.error})
            continue
        value = _load_cached_value(r.output_path, "pickle")
        source = value.get("source")
        if source == "kaggle":
            from_kaggle += 1
        elif source == "mast":
            from_mast += 1
        elif source == "existing":
            from_existing += 1
        results.append({"tic_id": r.item, "status": r.status, "source": source, "error": None})

    return SyncReport(
        total=batch_report.total,
        from_kaggle=from_kaggle,
        from_mast=from_mast,
        from_existing=from_existing,
        failed=batch_report.failed,
        wall_time_s=batch_report.wall_time_s,
        results=results,
    )
