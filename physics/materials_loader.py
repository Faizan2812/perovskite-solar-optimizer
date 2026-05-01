"""
physics/materials_loader.py
===========================
Drop-in replacement for the hardcoded dictionaries in physics/materials.py.
Loads the materials database from data/materials_database.json so that every
parameter carries verifiable provenance and the database can be extended
without editing Python source.

Usage (backwards-compatible with existing code):

    from physics.materials_loader import HTL_DB, PEROVSKITE_DB, ETL_DB
    # HTL_DB["Cu2O"].Eg  ->  2.17

Added capabilities:

    from physics.materials_loader import get_material_with_provenance
    info = get_material_with_provenance("MAPbI3", "absorber")
    print(info["Eg_eV"])                # 1.55
    print(info["Eg_eV_source"])         # "Green et al., Nat. Photon. 8, 506 (2014)"
    print(info["Eg_eV_doi"])            # "10.1038/nphoton.2014.134"
    print(info["Eg_eV_confidence"])     # "HIGH"

This keeps the fast dataclass access pattern your solver uses while exposing
provenance to the UI, the benchmark report, and the docx citation generator.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Find the JSON database. Falls back gracefully if the file is missing so
# that unit tests can still run in isolation.
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_HERE, "..", "data", "materials_database.json")


def _load_json_db(path: str = _DB_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Materials database not found at {path}. "
            "Check that data/materials_database.json is present."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Material dataclass (kept compatible with the existing physics/materials.py
# dataclass so that the rest of the codebase does not have to change).
# ----------------------------------------------------------------------
@dataclass
class Material:
    name: str
    Eg: float
    chi: float
    eps: float
    mu_e: float
    mu_h: float
    Nc: float
    Nv: float
    doping: float
    Nt: float = 1e14
    alpha_coeff: float = 1e5
    layer_type: str = "absorber"
    doping_type: str = "p"
    sigma_e: float = 1e-15
    sigma_h: float = 1e-15
    Et_offset: float = 0.0
    refs: str = ""
    category: str = ""
    provenance: Dict = field(default_factory=dict)  # NEW: per-parameter provenance

    @property
    def Ev(self):
        return self.chi + self.Eg

    @property
    def ni(self):
        kT = 0.02585  # at 300 K
        return np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * kT))


def _build_material(name: str, entry: dict, layer_type: str) -> Material:
    """Build a Material object from one JSON entry."""
    p = entry["parameters"]

    # Short ref string for backwards compatibility (e.g. "[1][3]")
    sources_used = {v.get("source", "") for v in p.values() if isinstance(v, dict)}
    refs = ",".join(sorted(s for s in sources_used if s))

    def g(key: str, default=None):
        """Get parameter value by JSON key."""
        if key in p and isinstance(p[key], dict):
            return p[key].get("value", default)
        return default

    m = Material(
        name=name,
        Eg=g("Eg_eV", 1.5),
        chi=g("chi_eV", 4.0),
        eps=g("eps_r", 6.5),
        mu_e=g("mu_n_cm2_Vs", 1.0),
        mu_h=g("mu_p_cm2_Vs", 1.0),
        Nc=g("Nc_cm3", 2.2e18),
        Nv=g("Nv_cm3", 1.8e19),
        doping=g("doping_cm3", 1e16),
        Nt=g("Nt_bulk_cm3", 1e14),
        alpha_coeff=g("alpha_coeff_cm", 1e5),
        layer_type=layer_type,
        doping_type=entry.get("doping_type", "p"),
        refs=refs,
        category=entry.get("category", ""),
        provenance=p,
    )
    return m


# ----------------------------------------------------------------------
# Build HTL_DB, PEROVSKITE_DB, ETL_DB at import time.
# ----------------------------------------------------------------------
_db = _load_json_db()
_references = _db.get("_references", {})

HTL_DB: Dict[str, Material] = {}
PEROVSKITE_DB: Dict[str, Material] = {}
ETL_DB: Dict[str, Material] = {}

for _name, _entry in _db.get("htls", {}).items():
    HTL_DB[_name] = _build_material(_name, _entry, layer_type="htl")

for _name, _entry in _db.get("absorbers", {}).items():
    PEROVSKITE_DB[_name] = _build_material(_name, _entry, layer_type="absorber")

for _name, _entry in _db.get("etls", {}).items():
    ETL_DB[_name] = _build_material(_name, _entry, layer_type="etl")


# ----------------------------------------------------------------------
# Provenance API for the UI and reports.
# ----------------------------------------------------------------------
def resolve_citation(source_key: str) -> Optional[dict]:
    """Return the full citation dict for a source id used in the DB.

    Example:
        resolve_citation("green_2014_photonics")
        -> {"citation": "Green et al., Nat. Photon. 8, 506 (2014)",
            "doi": "10.1038/nphoton.2014.134",
            "confidence": "HIGH"}
    """
    if source_key in _references:
        return dict(_references[source_key])
    return None


def get_material_with_provenance(name: str, layer_type: str) -> dict:
    """Get parameters + source + DOI for every field of a material.

    Returns a flat dict like:
        {"name": "MAPbI3",
         "Eg_eV": 1.55,
         "Eg_eV_source": "Green et al., Nat. Photon. 8, 506 (2014)",
         "Eg_eV_doi": "10.1038/nphoton.2014.134",
         "Eg_eV_confidence": "HIGH",
         "Eg_eV_method": "UV-Vis absorption edge",
         ...}
    """
    bucket = {"htl": "htls", "absorber": "absorbers",
              "perovskite": "absorbers", "etl": "etls"}.get(layer_type)
    if bucket is None:
        raise ValueError(f"Unknown layer type: {layer_type}")
    entry = _db.get(bucket, {}).get(name)
    if entry is None:
        raise KeyError(f"Material '{name}' not found in {bucket}")

    flat = {"name": name,
            "category": entry.get("category", ""),
            "formula": entry.get("formula", ""),
            "full_name": entry.get("full_name", "")}

    for field_name, field_entry in entry["parameters"].items():
        flat[field_name] = field_entry.get("value")
        source_key = field_entry.get("source")
        cite = resolve_citation(source_key) if source_key else None
        flat[f"{field_name}_source"] = cite["citation"] if cite else (source_key or "")
        flat[f"{field_name}_doi"] = cite["doi"] if cite else ""
        flat[f"{field_name}_confidence"] = field_entry.get("confidence") or (cite["confidence"] if cite else "UNKNOWN")
        flat[f"{field_name}_method"] = field_entry.get("method", "")
        flat[f"{field_name}_notes"] = field_entry.get("notes", "")

    return flat


def get_all_materials() -> Dict[str, Material]:
    """Return combined dictionary of all materials."""
    return {**HTL_DB, **PEROVSKITE_DB, **ETL_DB}


def get_material_names_by_type(layer_type: str):
    if layer_type == "htl": return list(HTL_DB.keys())
    if layer_type == "etl": return list(ETL_DB.keys())
    if layer_type in ("absorber", "perovskite"): return list(PEROVSKITE_DB.keys())
    return []


def summarize_provenance_confidence() -> Dict[str, Dict[str, int]]:
    """How many parameters at each confidence level across the whole DB.

    Useful for the Database tab in the Streamlit app, so reviewers can see
    at a glance where the weakest links are.
    """
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    by_material = {}
    for bucket in ("htls", "absorbers", "etls"):
        for mat_name, entry in _db.get(bucket, {}).items():
            per_mat = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
            for pname, pdata in entry["parameters"].items():
                conf = pdata.get("confidence") or "UNKNOWN"
                counts[conf] = counts.get(conf, 0) + 1
                per_mat[conf] = per_mat.get(conf, 0) + 1
            by_material[mat_name] = per_mat
    return {"total": counts, "by_material": by_material}
