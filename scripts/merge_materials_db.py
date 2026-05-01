"""
scripts/merge_materials_db.py
==============================
Merge data/materials_database_extension.json into data/materials_database.json.

Usage:
    python scripts/merge_materials_db.py                  # dry run preview
    python scripts/merge_materials_db.py --apply          # write changes

The extension file holds new or updated materials. After merging, rerun:
    pytest tests/
    python scripts/run_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "materials_database.json"
EXT  = ROOT / "data" / "materials_database_extension.json"
OUT  = ROOT / "data" / "materials_database_full.json"


def load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def merge_bucket(main_bucket: dict, ext_bucket: dict, bucket_name: str) -> list[str]:
    changes = []
    for mat_name, entry in ext_bucket.items():
        if mat_name in main_bucket:
            changes.append(f"UPDATE {bucket_name}/{mat_name}")
        else:
            changes.append(f"ADD    {bucket_name}/{mat_name}")
        main_bucket[mat_name] = entry
    return changes


def merge_refs(main_refs: dict, ext_refs: dict) -> list[str]:
    changes = []
    for ref_id, entry in ext_refs.items():
        if ref_id in main_refs:
            if main_refs[ref_id] != entry:
                changes.append(f"UPDATE reference {ref_id}")
        else:
            changes.append(f"ADD    reference {ref_id}")
        main_refs[ref_id] = entry
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                     help="Actually write the merged file (default: dry run)")
    args = ap.parse_args()

    main_db = load(MAIN)
    ext_db  = load(EXT)

    changes: list[str] = []
    for bucket in ("htls", "absorbers", "etls"):
        if bucket in ext_db:
            changes += merge_bucket(main_db[bucket], ext_db[bucket], bucket)
    if "_references" in ext_db:
        changes += merge_refs(main_db.setdefault("_references", {}), ext_db["_references"])

    main_db["_meta"] = main_db.get("_meta", {})
    main_db["_meta"]["last_merged"] = datetime.utcnow().isoformat() + "Z"
    main_db["_meta"]["merge_count"] = main_db["_meta"].get("merge_count", 0) + 1

    print(f"Merge summary ({len(changes)} changes):")
    for c in changes:
        print(f"  {c}")

    if not args.apply:
        print("\nDry run only. Rerun with --apply to write.")
        return 0

    # Backup existing main file
    backup = MAIN.with_suffix(".json.bak")
    shutil.copy2(MAIN, backup)
    print(f"\nBacked up {MAIN.name} to {backup.name}")

    with open(OUT, "w") as f:
        json.dump(main_db, f, indent=2, sort_keys=False)
    print(f"Wrote merged database to {OUT}")
    print("\nTo activate: mv data/materials_database_full.json data/materials_database.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
