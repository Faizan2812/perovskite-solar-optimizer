"""
physics/materials.py
=====================
Single source of truth for material parameters in this tool.

Every material parameter comes from data/materials_database.json which
carries per-value provenance (source paper, DOI, measurement method,
confidence tier).

This module is the stable import surface for the rest of the codebase:

    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.materials import get_material_with_provenance
    from physics.materials import CONTACTS, InterfaceDefects, DEFAULT_IDL

No hardcoded material parameters live here. Changing a value means editing
the JSON file, which forces a git diff reviewers can audit.
"""
from __future__ import annotations

from dataclasses import dataclass

# Route every material-data import through the JSON-backed loader.
from physics.materials_loader import (
    Material,
    HTL_DB,
    PEROVSKITE_DB,
    ETL_DB,
    get_all_materials,
    get_material_names_by_type,
    get_material_with_provenance,
    resolve_citation,
    summarize_provenance_confidence,
)


# ---------------------------------------------------------------------------
# Interface defect layer (IDL) parameter container.
# These are device-level settings, not material-level, so they stay here
# rather than in the JSON database.
# ---------------------------------------------------------------------------
@dataclass
class InterfaceDefects:
    """Parameters for a thin defective layer at a heterojunction.

    Used by physics/dd_solver.py to add a local recombination term
    R_if = (np - ni^2) / (n/S_p + p/S_n)
    at detected interface nodes.
    """
    Nt_if: float = 1e12     # interface trap density  [/cm^2]
    sigma_e: float = 1e-15  # electron capture cross-section [cm^2]
    sigma_h: float = 1e-15  # hole capture cross-section [cm^2]
    Et_offset: float = 0.0  # trap energy offset from midgap [eV]
    S_n: float = 1e4        # surface recombination velocity, electrons [cm/s]
    S_p: float = 1e4        # surface recombination velocity, holes [cm/s]


DEFAULT_IDL = InterfaceDefects()
HIGH_QUALITY_IDL = InterfaceDefects(Nt_if=1e10, S_n=1e2, S_p=1e2)
LOW_QUALITY_IDL = InterfaceDefects(Nt_if=1e14, S_n=1e6, S_p=1e6)


# ---------------------------------------------------------------------------
# Graded composition profile for absorber layers.
# ---------------------------------------------------------------------------
@dataclass
class GradedProfile:
    """Linear / V-shape / exponential bandgap grading across the absorber."""
    Eg_front: float = 1.55
    Eg_back: float = 1.55
    chi_front: float = 3.93
    chi_back: float = 3.93
    grading_type: str = "linear"  # 'linear' | 'exponential' | 'v-shaped'

    def get_Eg(self, x_frac):
        import numpy as np
        if self.grading_type == "linear":
            return self.Eg_front + (self.Eg_back - self.Eg_front) * x_frac
        if self.grading_type == "exponential":
            return self.Eg_front * np.exp(np.log(self.Eg_back / self.Eg_front) * x_frac)
        if self.grading_type == "v-shaped":
            if x_frac < 0.5:
                return self.Eg_front + (self.Eg_back - self.Eg_front) * 2 * x_frac
            return self.Eg_back + (self.Eg_front - self.Eg_back) * 2 * (x_frac - 0.5)
        return self.Eg_front

    def get_chi(self, x_frac):
        return self.chi_front + (self.chi_back - self.chi_front) * x_frac


# ---------------------------------------------------------------------------
# Tandem cell configuration container.
# ---------------------------------------------------------------------------
@dataclass
class TandemConfig:
    """Two-junction stack configuration for tandem simulations."""
    top_absorber: str = "FAPbI3"
    bottom_absorber: str = "CsSnI3"
    top_thickness_nm: float = 400
    bottom_thickness_nm: float = 500
    terminal_config: str = "2T"         # '2T' (series) or '4T' (independent)
    tunnel_junction_R: float = 1.0      # tunnel junction resistance  [Ohm cm^2]


# ---------------------------------------------------------------------------
# Contact work functions and surface recombination velocities.
# These come from standard references; kept here as a small lookup table.
# ---------------------------------------------------------------------------
CONTACTS = {
    "FTO":  {"work_function": 4.40, "Sn": 1e7, "Sp": 1e7},
    "ITO":  {"work_function": 4.60, "Sn": 1e7, "Sp": 1e7},
    "Au":   {"work_function": 5.10, "Sn": 1e5, "Sp": 1e7},
    "Ag":   {"work_function": 4.26, "Sn": 1e5, "Sp": 1e7},
    "C":    {"work_function": 5.00, "Sn": 1e5, "Sp": 1e7},
    "Al":   {"work_function": 4.06, "Sn": 1e7, "Sp": 1e5},
    "Ni":   {"work_function": 5.04, "Sn": 1e5, "Sp": 1e7},
    "Cu":   {"work_function": 4.65, "Sn": 1e5, "Sp": 1e7},
}


# ---------------------------------------------------------------------------
# Exported symbol list for clarity.
# ---------------------------------------------------------------------------
__all__ = [
    "Material",
    "HTL_DB", "PEROVSKITE_DB", "ETL_DB",
    "get_all_materials",
    "get_material_names_by_type",
    "get_material_with_provenance",
    "resolve_citation",
    "summarize_provenance_confidence",
    "InterfaceDefects", "DEFAULT_IDL", "HIGH_QUALITY_IDL", "LOW_QUALITY_IDL",
    "GradedProfile",
    "TandemConfig",
    "CONTACTS",
]
