"""Per-target MAST/lightkurve rebuild fallback -- the --rebuild-missing path.

Rather than reimplement TESS/Kepler download + detrending, this module
reuses arvyo-data's own build scripts directly:

  - TESS:   scripts/02_download_lightcurves.py (``_search_and_download``)
            scripts/03_preprocess.py           (``preprocess_one``)
  - Kepler: scripts/05_download_kepler.py       (``_search_and_download``)
            scripts/06_preprocess_kepler.py     (``preprocess_one``)

Why dynamic import instead of a normal ``import``: these filenames start
with a digit (``02_download_lightcurves.py``), which is not a valid
dotted module name -- ``import scripts.02_download_lightcurves`` is a
SyntaxError. arvyo-data is also not a pip-installable package (no
pyproject.toml/setup.py, not published anywhere), so it can't be added as
a normal dependency either. This module locates a local arvyo-data
checkout and loads the two scripts via
``importlib.util.spec_from_file_location``, which loads by file path and
sidesteps both problems. Critically, this means fetchr's rebuilt .npz
files are produced by the *exact same code* arvyo-data's own pipeline
uses (same search/download/retry logic, same sigma-clipping and wotan
detrending settings), so a MAST-rebuilt file can never silently diverge
from what a full arvyo-data rebuild would have produced. A lightweight
reimplementation was considered and rejected for exactly that reason --
the whole point of "prefer calling into arvyo-data's existing scripts" is
that fetchr and arvyo-data never quietly disagree on preprocessing.

Locating arvyo-data: fetchr's own CLI/API usage always takes a
``manifest_path`` that points *at* a manifest.csv living inside an
arvyo-data checkout (e.g. ``arvyo-data/manifest.csv``), so the default is
``Path(manifest_path).resolve().parent``. Override with the
``ARVYO_DATA_PATH`` env var or ``arvyo_data_path=`` if the manifest was
copied out of its checkout onto a machine that doesn't have the
``scripts/`` directory alongside it -- fetchr raises a clear
``ArvyoDataNotFoundError`` in that case rather than silently falling back
to a divergent reimplementation.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np

from .schema import SCHEMA_VERSION

logger = logging.getLogger("fetchr.mast_source")

_MODULE_CACHE: dict[str, ModuleType] = {}

# mission -> (download_script, preprocess_script), both under arvyo-data/scripts/
_SCRIPTS_BY_MISSION = {
    "tess": ("02_download_lightcurves.py", "03_preprocess.py"),
    "kepler": ("05_download_kepler.py", "06_preprocess_kepler.py"),
}


class ArvyoDataNotFoundError(RuntimeError):
    """Raised when a local arvyo-data checkout (with scripts/) can't be located."""


def resolve_arvyo_data_root(manifest_path=None, arvyo_data_path=None) -> Path:
    candidate = (
        arvyo_data_path
        or os.environ.get("ARVYO_DATA_PATH")
        or (Path(manifest_path).resolve().parent if manifest_path else None)
    )
    if candidate is None:
        raise ArvyoDataNotFoundError(
            "Could not locate an arvyo-data checkout: pass manifest_path, "
            "set the ARVYO_DATA_PATH env var, or pass arvyo_data_path explicitly."
        )
    root = Path(candidate).resolve()
    if not (root / "scripts").is_dir():
        raise ArvyoDataNotFoundError(
            f"{root} does not look like an arvyo-data checkout (no scripts/ "
            "directory found). The MAST rebuild path reuses arvyo-data's own "
            "download/preprocess scripts in-process, so a full local checkout "
            "(not just a copy of manifest.csv) is required. Set ARVYO_DATA_PATH "
            "to the correct location."
        )
    return root


def _load_script(root: Path, filename: str) -> ModuleType:
    path = root / "scripts" / filename
    key = str(path)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    if not path.exists():
        raise ArvyoDataNotFoundError(f"{path} not found under arvyo-data checkout {root}")
    # Numeric-prefixed filenames aren't valid dotted module names for a
    # normal `import` statement, so load by file path instead.
    module_name = f"_fetchr_arvyo_data_{filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def _lookup_manifest_row(tic_id, mission, manifest_path, root) -> dict:
    import pandas as pd

    mpath = manifest_path or (root / ("kepler_manifest.csv" if mission == "kepler" else "manifest.csv"))
    manifest = pd.read_csv(mpath, keep_default_na=False, na_values=[""])
    rows = manifest[manifest["tic_id"].astype(str) == str(tic_id)]
    if rows.empty:
        raise ValueError(f"tic_id {tic_id} not found in manifest {mpath}")
    return rows.iloc[0].to_dict()


def rebuild_one(
    tic_id,
    output_dir: str | Path,
    label: str | None = None,
    mission: str | None = None,
    meta: dict | None = None,
    manifest_path: str | Path | None = None,
    arvyo_data_path: str | Path | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    """Re-download + re-preprocess one target via MAST, matching arvyo-data's own recipe.

    Writes ``{output_dir}/{label}/{tic_id}.npz`` and returns its path.
    ``label``/``mission``/``meta`` (period_days, epoch_btjd, etc -- passed
    through as metadata by the preprocess step) are normally already known
    to the caller (``fetchr.sync()``'s worker reads them off the manifest
    row it's already holding); if omitted, this function looks the row up
    itself from ``manifest_path`` (or arvyo-data's default manifest,
    guessed from ``mission`` if given, else ``manifest.csv``) -- this is
    what the ``fetchr rebuild TIC_ID`` CLI path uses. An explicit
    ``mission`` always wins; otherwise it's read from the looked-up row's
    ``mission`` column if present (e.g. kepler_manifest.csv), else "tess".
    """
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"fetchr only supports schema_version={SCHEMA_VERSION!r}, got {schema_version!r}")

    root = resolve_arvyo_data_root(manifest_path=manifest_path, arvyo_data_path=arvyo_data_path)

    if label is None or meta is None:
        row = _lookup_manifest_row(tic_id, mission or "tess", manifest_path, root)
        label = label or row["label"]
        meta = meta or row
        if mission is None:
            mission = row.get("mission") or "tess"

    mission = (mission or "tess").strip().lower() or "tess"
    if mission not in _SCRIPTS_BY_MISSION:
        raise ValueError(f"Unknown mission {mission!r}; expected one of {list(_SCRIPTS_BY_MISSION)}")

    download_script, preprocess_script = _SCRIPTS_BY_MISSION[mission]
    dl_mod = _load_script(root, download_script)
    pp_mod = _load_script(root, preprocess_script)

    with tempfile.TemporaryDirectory() as tmpdir:
        fits_path = Path(tmpdir) / f"{tic_id}.fits"
        # Call _search_and_download directly rather than each script's own
        # download_target(): that wrapper hardcodes RAW_DIR under
        # arvyo-data's own data/ tree, not a path we control, and doesn't
        # accept a dest_path. _search_and_download does.
        if mission == "tess":
            ok, reason = dl_mod._search_and_download(tic_id, "SPOC", "short", fits_path)
            if not ok:
                ok, reason = dl_mod._search_and_download(tic_id, "TESS-SPOC", None, fits_path)
        else:
            ok, reason = dl_mod._search_and_download(tic_id, fits_path)
        if not ok:
            raise RuntimeError(f"MAST download failed for {mission} TIC/KIC {tic_id}: {reason}")

        result = pp_mod.preprocess_one(tic_id, label, fits_path, meta)

    out_path = Path(output_dir) / str(label) / f"{tic_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique-per-call tmp name: two independently invoked `fetchr sync
    # --rebuild-missing` (or `fetchr rebuild`) processes racing on the same
    # tic_id would otherwise both write to the exact same deterministic
    # tmp path, and one's os.replace() could find the other's tmp file
    # already gone.
    fd, tmp_name = tempfile.mkstemp(dir=out_path.parent, prefix=f".{out_path.name}-", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(tmp_path, "wb") as f:
        np.savez(f, **result)
    os.replace(tmp_path, out_path)
    return out_path
