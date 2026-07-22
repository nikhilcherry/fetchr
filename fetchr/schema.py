"""fetchr's own copy of the arvyo-data <-> arvyo-pipeline schema-1.0 contract.

These constants are COPIED, not imported, from
``arvyo-pipeline/arvyo/contract.py`` (as of commit 0fa44bf, 2026-07-02,
"Add data contract"). fetchr intentionally does not depend on the ``arvyo``
package: fetchr is meant to run standalone, on a machine that may have
arvyo-data's manifest but not a full arvyo-pipeline checkout, and pulling in
arvyo-pipeline's own heavy dependency stack (torch, sbi, transitleastsquares,
...) just to read six constants would be a poor trade.

This is the same manual-sync convention arvyo-pipeline's own README
documents: "Any change to this schema requires bumping SCHEMA_VERSION and
updating BOTH repos' READMEs." fetchr is a third place these constants now
live, so a schema bump means updating arvyo-pipeline/arvyo/contract.py,
arvyo-data's README, arvyo-pipeline's README, AND this file — in practice,
grep both repos and this one for the old SCHEMA_VERSION when bumping it.
"""

SCHEMA_VERSION = "1.0"

REQUIRED_ARRAYS = ["time", "flux", "flux_err"]
OPTIONAL_ARRAYS = ["flux_raw"]  # present for starspot/null classes
REQUIRED_META = ["tic_id", "label", "sector"]
OPTIONAL_META = [
    "period_days", "epoch_btjd", "crowdsap", "mission",
    "augmented", "injection_params",
]
LABELS = ["planet", "eb", "blend", "starspot", "null", "unknown"]
