"""The `fetchr` command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import mast_source, sync, verify
from .schema import SCHEMA_VERSION


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify(args.manifest, args.data_dir)
    print(f"present:        {len(report.present)}")
    print(f"missing:        {len(report.missing)}")
    print(f"schema_invalid: {len(report.schema_invalid)}")
    if report.schema_invalid:
        print("\nschema_invalid details:")
        for tic_id in report.schema_invalid:
            print(f"  {tic_id}: {report.errors[tic_id]}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    report = sync(
        manifest_path=args.manifest,
        kaggle_dataset=args.kaggle_dataset,
        output_dir=args.output_dir,
        workers=args.workers,
        rebuild_missing=args.rebuild_missing,
        cache_dir=args.cache_dir,
        limit=args.limit,
        arvyo_data_path=args.arvyo_data_path,
        force=args.force,
    )
    print(report.summary())
    if report.failed:
        print("\nfailed tic_ids:")
        for tic_id in report.failed_items():
            print(f"  {tic_id}")
    return 0 if report.failed == 0 else 1


def _cmd_rebuild(args: argparse.Namespace) -> int:
    try:
        out_path = mast_source.rebuild_one(
            args.tic_id,
            args.output_dir,
            label=args.label,
            mission=args.mission,
            manifest_path=args.manifest,
            arvyo_data_path=args.arvyo_data_path,
            schema_version=args.schema_version,
        )
    except Exception as e:
        print(f"rebuild failed for {args.tic_id}: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fetchr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser(
        "verify", help="Offline, contract-only check of data-dir against manifest (no Kaggle credentials needed)"
    )
    p_verify.add_argument("--manifest", required=True)
    p_verify.add_argument("--data-dir", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_sync = sub.add_parser("sync", help="Pull the bulk .npz corpus from Kaggle, falling back to MAST per-row")
    p_sync.add_argument("--manifest", required=True)
    p_sync.add_argument("--kaggle-dataset", default=None, help="owner/dataset-slug on Kaggle Datasets")
    p_sync.add_argument("--output-dir", required=True)
    p_sync.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    p_sync.add_argument("--rebuild-missing", action="store_true", dest="rebuild_missing")
    p_sync.add_argument("--limit", type=int, default=None, help="only sync the first N manifest rows (for testing)")
    p_sync.add_argument("--cache-dir", default=".fetchr_cache")
    p_sync.add_argument("--arvyo-data-path", default=None, help="local arvyo-data checkout, for --rebuild-missing")
    p_sync.add_argument("--force", action="store_true")
    p_sync.set_defaults(func=_cmd_sync)

    p_rebuild = sub.add_parser("rebuild", help="Single-target MAST fallback rebuild")
    p_rebuild.add_argument("tic_id")
    p_rebuild.add_argument("--output-dir", required=True)
    p_rebuild.add_argument("--schema-version", default=SCHEMA_VERSION, dest="schema_version")
    p_rebuild.add_argument(
        "--manifest", default="manifest.csv",
        help="manifest.csv to look up label/mission/period/epoch from (default: ./manifest.csv)",
    )
    p_rebuild.add_argument("--label", default=None, help="override the manifest's label for this target")
    p_rebuild.add_argument("--mission", default=None, help="override the manifest's mission (tess|kepler)")
    p_rebuild.add_argument("--arvyo-data-path", default=None, dest="arvyo_data_path")
    p_rebuild.set_defaults(func=_cmd_rebuild)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
