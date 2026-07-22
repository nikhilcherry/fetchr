"""The batchr worker function for fetchr.sync(), kept in its own module.

batchr requires ``fn`` to be a plain module-level function (picklable,
importable by worker processes) -- see batchr's README, "Non-picklable
functions". It also only ever calls ``fn(item)`` with the single item
string; there is no channel for run_batch to forward extra arguments to
fn (the ``config`` dict batchr accepts is hashed into the cache key, never
delivered to fn itself). So this worker reads everything else it needs --
the per-row manifest metadata, the Kaggle-dataset tic_id index, output_dir,
etc -- from a JSON file whose path is handed over via the
``FETCHR_SYNC_CONFIG`` environment variable. ``fetchr.sync()`` sets that
env var in the parent process just before calling ``run_batch()``; since
``ProcessPoolExecutor`` workers (both the "fork" and "spawn" start
methods) inherit the parent's environment at process-creation time, this
works regardless of platform.

The primary resumability mechanism is still batchr's own cache (item +
fn source + config -> cache key): a re-run with an unchanged manifest/
config skips calling this function at all for already-completed rows. The
on-disk existence check below is a secondary safety net for the case where
the batchr cache dir was purged or moved but the actual output files
weren't -- it should rarely be the thing that fires.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import mast_source, verify

_config_cache: dict | None = None


class MissingSourceError(RuntimeError):
    """Raised when a target isn't in the Kaggle dataset and rebuild_missing=False."""


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        import json

        config_path = os.environ["FETCHR_SYNC_CONFIG"]
        with open(config_path) as f:
            _config_cache = json.load(f)
    return _config_cache


def reset_config_cache() -> None:
    """Force the next call in this process to re-read FETCHR_SYNC_CONFIG.

    Matters when calling fetchr.sync() more than once within one Python
    session; each run_batch() call spawns fresh worker processes, but the
    parent process itself would otherwise reuse a stale cached config if it
    ever calls sync_one_item() directly.
    """
    global _config_cache
    _config_cache = None


def sync_one_item(tic_id: str) -> dict:
    """Download-or-rebuild one manifest row's .npz, then verify it. Returns a small dict."""
    config = _load_config()
    row = config["rows"].get(tic_id)
    if row is None:
        raise KeyError(f"tic_id {tic_id!r} not found in fetchr's sync config (internal error)")

    label = row["label"]
    output_dir = config["output_dir"]
    target_path = verify.expected_path(output_dir, label, tic_id)

    if target_path.exists():
        try:
            verify.load_and_validate(target_path)
            return {"tic_id": tic_id, "source": "existing", "path": str(target_path)}
        except verify.ContractError:
            pass  # present but invalid/stale -- fall through and redo it

    kaggle_path = config["kaggle_index"].get(tic_id)
    if kaggle_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(target_path.name + ".tmp")
        shutil.copyfile(kaggle_path, tmp_path)
        os.replace(tmp_path, target_path)
        source = "kaggle"
    elif config["rebuild_missing"]:
        mast_source.rebuild_one(
            tic_id,
            output_dir,
            label=label,
            mission=(row.get("mission") or "tess"),
            meta=row,
            manifest_path=config["manifest_path"],
            arvyo_data_path=config["arvyo_data_path"],
        )
        source = "mast"
    else:
        raise MissingSourceError(
            f"tic_id {tic_id} not present in the kaggle dataset and rebuild_missing=False "
            "(pass --rebuild-missing, or run `fetchr rebuild` on it directly)"
        )

    verify.load_and_validate(Path(target_path))
    return {"tic_id": tic_id, "source": source, "path": str(target_path)}
