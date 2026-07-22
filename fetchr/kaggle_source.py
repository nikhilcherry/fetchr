"""Kaggle Datasets download + extract for the bulk arvyo-data corpus.

The named Kaggle dataset does not exist publicly yet as of this writing --
arvyo-data's own README lists its "Data hosting" section as
``<placeholder -- Kaggle link TBD>``. That means the dataset's real file
layout (flat? nested by label? by mission?) cannot be confirmed against an
actual download right now, and this module was written without doing so --
per the task's "don't guess" instruction, it does not hardcode a single
assumed structure. Instead it downloads+extracts once, then *discovers*
the layout by scanning for ``*.npz`` files and indexing them by filename
stem (``<tic_id>.npz``), which works regardless of whether the uploader
nests them by label (arvyo-data's own ``{label}/{tic_id}.npz`` convention,
the most likely shape) or leaves them flat. Once the real dataset exists,
`fetchr sync` against it (verification step 3 in the task handout) is the
first real confirmation of its layout; if a future upload turns out to use
something index_by_tic_id can't discover (e.g. an ID embedded only in a
sidecar CSV, not the filename), that will surface as an accurate "0
from_kaggle" in the sync report rather than a silent wrong-path failure,
and this module's indexing would need a follow-up patch.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("fetchr.kaggle_source")


def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "The 'kaggle' package is required for fetchr's Kaggle download "
            "path. Install with `pip install fetchr[kaggle]`, and place "
            "your API token at ~/.kaggle/kaggle.json (see "
            "https://www.kaggle.com/docs/api)."
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return api


def download_and_extract(dataset: str, staging_dir: str | Path, force: bool = False) -> Path:
    """Download+unzip ``dataset`` (``owner/dataset-slug``) into staging_dir once.

    Idempotent: if staging_dir already has ``.npz`` files and force=False,
    skips the download+extract entirely, so repeated `fetchr sync` calls
    don't re-pull the whole bulk dataset every time.
    """
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(staging_dir.rglob("*.npz")):
        logger.info(
            "kaggle staging dir %s already has .npz files, skipping download "
            "(pass force=True to redo)", staging_dir,
        )
        return staging_dir

    api = _kaggle_api()
    logger.info("downloading kaggle dataset %s into %s", dataset, staging_dir)
    api.dataset_download_files(dataset, path=str(staging_dir), unzip=True, quiet=False)
    return staging_dir


def index_by_tic_id(staging_dir: str | Path) -> dict[str, Path]:
    """Map tic_id (str) -> extracted .npz path, tolerant of unknown layout.

    Indexes every ``*.npz`` under staging_dir by filename stem, regardless
    of nesting -- see module docstring for why this is deliberately
    layout-agnostic rather than assuming a fixed directory structure.
    """
    staging_dir = Path(staging_dir)
    index: dict[str, Path] = {}
    # sorted() so "keep the first" is deterministic -- rglob's order is
    # filesystem-dependent, not creation order.
    for npz_path in sorted(staging_dir.rglob("*.npz")):
        tic_id = npz_path.stem
        if tic_id in index:
            logger.warning(
                "duplicate tic_id %s found in kaggle dataset (%s and %s); keeping the first",
                tic_id, index[tic_id], npz_path,
            )
            continue
        index[tic_id] = npz_path
    return index
