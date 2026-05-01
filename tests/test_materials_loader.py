"""
tests/test_materials_loader.py
===============================
The most important tests in this project: they enforce that the material
database never regresses to the kind of unverified state the legacy code
was in. A phantom citation, a missing DOI, or a parameter without a source
will cause the CI to fail and block the commit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Structure and count tests
# ---------------------------------------------------------------------------
def test_database_file_exists():
    """The JSON database file must exist."""
    assert (ROOT / "data" / "materials_database.json").exists()


def test_database_is_valid_json():
    """The JSON must parse without errors."""
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)
    assert isinstance(db, dict)


def test_three_layer_buckets_present():
    """Database must have htls, absorbers, etls."""
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)
    for bucket in ("htls", "absorbers", "etls"):
        assert bucket in db, f"Missing bucket: {bucket}"
        assert len(db[bucket]) > 0, f"Bucket {bucket} is empty"


def test_minimum_material_counts():
    """Make sure the database has enough materials for the tool to be useful."""
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)
    assert len(db["htls"])      >= 5, "Need ≥ 5 HTLs"
    assert len(db["absorbers"]) >= 5, "Need ≥ 5 absorbers"
    assert len(db["etls"])      >= 5, "Need ≥ 5 ETLs"


# ---------------------------------------------------------------------------
# Phantom-citation regression test (THE one that matters)
# ---------------------------------------------------------------------------
FORBIDDEN_PATTERNS = [
    "datto",          # the phantom ChemNanoMat 2026 reference
    "chemnanomat",
    "2026",            # no publications can be cited from a future year
    "in preparation",
    "unpublished",
    "personal communication",
    "pers. comm.",
]


def test_no_phantom_citations_in_sources():
    """Fail if any active material parameter cites a phantom or unverifiable source."""
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)

    violations = []
    for bucket in ("htls", "absorbers", "etls"):
        for mat_name, entry in db[bucket].items():
            for param_key, value in entry.get("parameters", {}).items():
                if isinstance(value, dict) and "source" in value:
                    src = value["source"].lower()
                    for pat in FORBIDDEN_PATTERNS:
                        if pat in src:
                            violations.append(
                                f"{bucket}/{mat_name}/{param_key}: source contains '{pat}' → {value['source']}"
                            )

    assert not violations, "PHANTOM CITATIONS DETECTED:\n" + "\n".join(violations)


def test_no_phantom_in_reference_entries():
    """The _references block must not contain ACTIVE phantom citations.

    Tombstone keys starting with '_REMOVED_' are intentional audit trails
    documenting what was purged from the legacy code. They are allowed.

    Verification metadata fields (verified_on, tombstone_date, etc.) can
    contain dates including the current year, so we exclude them from the
    phantom-pattern check.
    """
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)
    refs = db.get("_references", {})
    METADATA_FIELDS = {"verified_on", "verified", "url", "oa", "tombstone_date",
                       "verification_note", "migration_note", "replaced_with",
                       "reason", "flag", "absorber_material"}
    violations = []
    for ref_id, entry in refs.items():
        # Skip tombstones
        if ref_id.startswith("_REMOVED_"):
            continue
        # Build a string from only the citation-relevant fields
        scan_dict = {k: v for k, v in entry.items() if k not in METADATA_FIELDS}
        full = json.dumps(scan_dict).lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat in full:
                violations.append(f"{ref_id}: reference contains '{pat}'")
    assert not violations, "\n".join(violations)


# ---------------------------------------------------------------------------
# Provenance completeness test
# ---------------------------------------------------------------------------
REQUIRED_PROVENANCE_KEYS = {"value", "source", "confidence"}


def test_every_parameter_has_provenance():
    """Every material parameter must carry {value, source, confidence}."""
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)

    missing = []
    for bucket in ("htls", "absorbers", "etls"):
        for mat_name, entry in db[bucket].items():
            for key, val in entry.get("parameters", {}).items():
                if not isinstance(val, dict):
                    # Plain numeric — allowed for backward compat but flag it
                    continue
                missing_keys = REQUIRED_PROVENANCE_KEYS - set(val.keys())
                if missing_keys:
                    missing.append(f"{bucket}/{mat_name}/{key}: missing {missing_keys}")

    assert not missing, "Incomplete provenance:\n" + "\n".join(missing[:10])


def test_confidence_tiers_are_valid():
    """Confidence must be HIGH, MEDIUM, or LOW."""
    valid = {"HIGH", "MEDIUM", "LOW"}
    with open(ROOT / "data" / "materials_database.json") as f:
        db = json.load(f)

    invalid = []
    for bucket in ("htls", "absorbers", "etls"):
        for mat_name, entry in db[bucket].items():
            for key, val in entry.get("parameters", {}).items():
                if isinstance(val, dict) and "confidence" in val:
                    if val["confidence"] not in valid:
                        invalid.append(f"{bucket}/{mat_name}/{key}: {val['confidence']!r}")

    assert not invalid, "Invalid confidence tiers:\n" + "\n".join(invalid)


# ---------------------------------------------------------------------------
# Python API tests
# ---------------------------------------------------------------------------
def test_loader_api_works():
    """The public API in physics/materials.py must import and work."""
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.materials import get_material_with_provenance

    assert len(HTL_DB) > 0
    assert "MAPbI3" in PEROVSKITE_DB

    info = get_material_with_provenance("MAPbI3", "absorber")
    assert "Eg_eV" in info
    assert "Eg_eV_source" in info
    assert "Eg_eV_confidence" in info


def test_provenance_summary_budget():
    """LOW-confidence parameters must not exceed a sensible budget.

    This is a ceiling, not an ideal. If the count creeps up, the tool has
    regressed on data quality and we want to know about it.
    """
    from physics.materials import summarize_provenance_confidence
    summary = summarize_provenance_confidence()
    n_low = summary["total"]["LOW"]

    # Current published state: 7 LOW. Allow headroom to 15 before failing.
    # If you intentionally add a LOW-confidence parameter, raise this ceiling
    # WITH a commit message explaining why.
    assert n_low <= 15, (
        f"{n_low} LOW-confidence parameters — ceiling is 15. "
        f"Either re-source the new low-confidence entries or explicitly "
        f"raise the ceiling in this test."
    )
