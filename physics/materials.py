"""
Validated Material Database for Perovskite Solar Cell Simulation
================================================================
All parameters cross-validated against peer-reviewed SCAPS-1D studies.
Units: Eg(eV), chi(eV), eps(relative), mu_e/mu_h(cm²/Vs), 
       Nc/Nv(/cm³), doping(/cm³), Nt(/cm³), alpha(cm⁻¹)

References:
[1] Chabri et al., J. Electron. Mater. 52, 2722 (2023)
[2] Amjad et al., RSC Adv. 13, 23211 (2023)
[3] Hossain et al., ACS Omega 7, 43210 (2022)
[4] Uddin et al., Next Materials 9, 100980 (2025)
[5] Oyelade et al., Sci. Rep. 14 (2024)
[6] Datto, ChemNanoMat (2026)
[7] Chen et al., Energy & Fuels (2023)
[8] MDPI Photonics (2025)
[9] Araújo et al., RSC Sustainability 3, 4314 (2025)
[10] Saidarsan et al., Sol. Energy Mater. Sol. Cells 279, 113230 (2025)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Material:
    name: str
    Eg: float              # Bandgap [eV]
    chi: float             # Electron affinity [eV]
    eps: float             # Relative permittivity
    mu_e: float            # Electron mobility [cm²/Vs]
    mu_h: float            # Hole mobility [cm²/Vs]
    Nc: float              # Conduction band effective DOS [/cm³]
    Nv: float              # Valence band effective DOS [/cm³]
    doping: float          # Na (p-type) or Nd (n-type) [/cm³]
    Nt: float = 1e14       # Bulk trap density [/cm³]
    alpha_coeff: float = 1e5  # Absorption coefficient [/cm]
    layer_type: str = "absorber"
    doping_type: str = "p"  # "p" or "n"
    sigma_e: float = 1e-15  # Electron capture cross-section [cm²]
    sigma_h: float = 1e-15  # Hole capture cross-section [cm²]
    Et_offset: float = 0.0  # Trap energy offset from midgap [eV]
    refs: str = ""
    category: str = ""      # "htl", "etl", "perovskite", "oxide", etc.
    
    @property
    def Ev(self): return self.chi + self.Eg
    
    @property
    def ni(self):
        """Intrinsic carrier concentration [/cm³]."""
        kT = 0.02585  # at 300K
        return np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * kT))


# ═══════════════════════════════════════════════════════════════════════════════
# HTL MATERIALS (p-type)
# ═══════════════════════════════════════════════════════════════════════════════
HTL_DB = {
    "Cu2O": Material("Cu2O", 2.17, 3.20, 7.11, 200, 80, 2.02e17, 1.1e19,
                     1e18, 1e15, 5e4, "htl", "p", refs="[1][5][6]", category="Inorganic oxide"),
    "Spiro-OMeTAD": Material("Spiro-OMeTAD", 3.00, 2.05, 3.0, 2e-4, 2e-4,
                             2.2e18, 1.8e19, 1e18, 1e15, 0, "htl", "p", refs="[2][3]", category="Organic"),
    "NiO": Material("NiO", 3.80, 1.46, 11.75, 0.012, 0.028, 2.8e19, 1.0e19,
                    1e18, 1e15, 0, "htl", "p", refs="[1][3]", category="Inorganic oxide"),
    "CuSCN": Material("CuSCN", 3.60, 1.70, 10.0, 1e-4, 0.01, 2.2e19, 1.8e18,
                      1e18, 1e15, 0, "htl", "p", refs="[3]", category="Inorganic"),
    "CuI": Material("CuI", 3.10, 2.10, 6.5, 1.69e-2, 4.34e-1, 2.2e19, 1.8e19,
                    1e18, 1e15, 0, "htl", "p", refs="[8]", category="Inorganic halide"),
    "PEDOT:PSS": Material("PEDOT:PSS", 1.60, 3.40, 3.0, 0.001, 0.01, 2.2e18,
                          1.8e19, 1e18, 1e15, 0, "htl", "p", refs="[3]", category="Organic polymer"),
    "P3HT": Material("P3HT", 1.70, 3.20, 3.0, 1e-4, 1e-4, 2.2e18, 1.8e19,
                     1e18, 1e15, 0, "htl", "p", refs="[9]", category="Organic polymer"),
    "CuO": Material("CuO", 1.35, 4.07, 18.1, 0.01, 0.01, 2.2e19, 1.8e19,
                    1e18, 1e15, 0, "htl", "p", refs="[3]", category="Inorganic oxide"),
    "V2O5": Material("V2O5", 2.20, 4.70, 3.0, 0.01, 0.01, 2.2e18, 1.8e19,
                     1e18, 1e15, 0, "htl", "p", refs="[3]", category="Inorganic oxide"),
    "MoO3": Material("MoO3", 3.00, 2.30, 12.5, 100, 100, 2.2e18, 1.8e19,
                     1e18, 1e15, 0, "htl", "p", refs="[2]", category="Inorganic oxide"),
    "CBTS": Material("CBTS", 1.90, 3.60, 5.4, 30, 10, 2.2e18, 1.8e19,
                     1e18, 1e15, 0, "htl", "p", refs="[3]", category="Chalcogenide"),
    "SrCu2O2": Material("SrCu2O2", 3.30, 2.20, 9.8, 0.46, 0.46, 2.2e18, 1.8e19,
                        1e18, 1e15, 0, "htl", "p", refs="[4]", category="Inorganic oxide"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# PEROVSKITE ABSORBER MATERIALS
# ═══════════════════════════════════════════════════════════════════════════════
PEROVSKITE_DB = {
    # --- Lead-free tin halide ---
    "Cs2SnI6": Material("Cs2SnI6", 1.48, 4.01, 10.0, 53, 0.03, 2.2e18, 1.8e19,
                        1.2e16, 1e14, 1.2e5, "absorber", "p", refs="[1][4][5][7]",
                        category="Vacancy-ordered double perovskite"),
    "MASnI3": Material("MASnI3", 1.30, 4.17, 8.2, 1.6, 1.6, 1.0e18, 1.0e18,
                       3.2e15, 1e16, 5e4, "absorber", "p", refs="[5]",
                       category="Tin halide perovskite"),
    "CsSnI3": Material("CsSnI3", 1.30, 3.60, 9.93, 585, 585, 1e18, 1e18,
                       1e17, 1e16, 5e4, "absorber", "p", refs="[9]",
                       category="Tin halide perovskite"),
    "FASnI3": Material("FASnI3", 1.41, 3.52, 8.2, 22, 22, 1e18, 1e18,
                       1e17, 1e16, 5e4, "absorber", "p", refs="[9]",
                       category="Tin halide perovskite"),
    # --- Lead halide (reference) ---
    "MAPbI3": Material("MAPbI3", 1.55, 3.93, 6.5, 2.0, 2.0, 2.2e18, 1.8e19,
                       1e13, 1e14, 1.5e5, "absorber", "p", refs="[3][6]",
                       category="Lead halide perovskite"),
    "FAPbI3": Material("FAPbI3", 1.51, 4.00, 6.6, 2.7, 2.7, 2.2e18, 1.8e19,
                       1e13, 1e14, 1.4e5, "absorber", "p", refs="[3]",
                       category="Lead halide perovskite"),
    "CsPbI3": Material("CsPbI3", 1.73, 3.95, 6.3, 25, 25, 2.2e18, 1.8e19,
                       1e16, 1e14, 1.2e5, "absorber", "p", refs="[3]",
                       category="Lead halide perovskite"),
    "MAPbBr3": Material("MAPbBr3", 2.30, 3.38, 4.0, 1.0, 1.0, 2.2e18, 1.8e19,
                        1e13, 1e14, 5e4, "absorber", "p", refs="[3]",
                        category="Lead halide perovskite"),
    "CsPbI2Br": Material("CsPbI2Br", 1.88, 3.73, 6.5, 2.0, 2.0, 2.2e18, 1.8e19,
                         1e16, 1e14, 1e5, "absorber", "p", refs="[8]",
                         category="Mixed halide perovskite"),
    # --- Double perovskites (indirect bandgap → low α) ---
    "Cs2AgBiBr6": Material("Cs2AgBiBr6", 2.05, 3.80, 5.0, 11.81, 0.37, 1e19, 1e19,
                           1e16, 1e15, 5e3, "absorber", "p", refs="[3]",
                           category="Lead-free double perovskite"),
    "Cs2TiBr6": Material("Cs2TiBr6", 1.80, 3.90, 10.0, 2.36, 2.36, 3.8e20, 3.9e20,
                         1e16, 1e15, 2e4, "absorber", "p", refs="[9]",
                         category="Lead-free double perovskite"),
    # --- Chalcogenide perovskites ---
    "BaZrS3": Material("BaZrS3", 1.73, 3.59, 22.6, 4.4, 3.3, 3.2e18, 3.8e18,
                       1e16, 1e15, 8e4, "absorber", "p", refs="[9]",
                       category="Chalcogenide perovskite"),
    # --- Germanium-based ---
    "CsGeI3": Material("CsGeI3", 1.63, 3.90, 16.3, 2.54, 2.54, 1e18, 1e18,
                       1e16, 1e14, 1e5, "absorber", "p", refs="[9]",
                       category="Germanium halide perovskite"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# ETL MATERIALS (n-type)
# ═══════════════════════════════════════════════════════════════════════════════
ETL_DB = {
    "SnO2": Material("SnO2", 3.60, 4.00, 9.0, 100, 25, 2.2e18, 1.8e19,
                     1e17, 1e15, 0, "etl", "n", refs="[2][3][5]", category="Metal oxide"),
    "TiO2": Material("TiO2", 3.20, 4.00, 9.0, 20, 10, 2.2e18, 1.8e19,
                     1e17, 1e15, 0, "etl", "n", refs="[3][6]", category="Metal oxide"),
    "ZnO": Material("ZnO", 3.30, 4.40, 9.0, 100, 25, 2.2e18, 1.8e19,
                    1e18, 1e15, 0, "etl", "n", refs="[3][5]", category="Metal oxide"),
    "C60": Material("C60", 1.70, 3.90, 4.2, 0.08, 0.035, 8e19, 8e19,
                    2.6e17, 1e15, 0, "etl", "n", refs="[3]", category="Organic fullerene"),
    "PCBM": Material("PCBM", 2.00, 3.90, 3.9, 0.20, 0.20, 2.5e21, 2.5e21,
                     2.93e17, 1e15, 0, "etl", "n", refs="[3][6]", category="Organic fullerene"),
    "WS2": Material("WS2", 1.80, 3.95, 13.6, 100, 100, 2.2e18, 1.8e19,
                    1e18, 1e15, 0, "etl", "n", refs="[4]", category="TMD"),
    "CeO2": Material("CeO2", 3.50, 4.60, 9.0, 0.04, 0.04, 2.2e18, 1.8e19,
                     1e17, 1e15, 0, "etl", "n", refs="[3]", category="Rare-earth oxide"),
    "IGZO": Material("IGZO", 3.05, 4.16, 10.0, 15, 0.1, 5e18, 5e18,
                     1e18, 1e15, 0, "etl", "n", refs="[3]", category="Amorphous oxide"),
    "ZnSe": Material("ZnSe", 2.81, 4.09, 8.6, 400, 110, 1.5e18, 9.8e18,
                     1e18, 1e15, 0, "etl", "n", refs="[9]", category="II-VI semiconductor"),
    "WO3": Material("WO3", 2.60, 4.56, 5.76, 10, 10, 2.2e18, 1.8e19,
                    1e17, 1e15, 0, "etl", "n", refs="[9]", category="Metal oxide"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONTACT MATERIALS
# ═══════════════════════════════════════════════════════════════════════════════
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

def get_all_materials():
    """Return combined dictionary of all materials."""
    return {**HTL_DB, **PEROVSKITE_DB, **ETL_DB}

def get_material_names_by_type(layer_type):
    """Get material names for a given layer type."""
    if layer_type == "htl": return list(HTL_DB.keys())
    if layer_type == "etl": return list(ETL_DB.keys())
    if layer_type in ("absorber", "perovskite"): return list(PEROVSKITE_DB.keys())
    return []

# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL MATERIALS (expanding to 50+)
# ═══════════════════════════════════════════════════════════════════════════════
PEROVSKITE_DB.update({
    "MAPbCl3": Material("MAPbCl3", 2.97, 3.10, 4.0, 0.5, 0.5, 2.2e18, 1.8e19,
                        1e13, 1e15, 2e4, "absorber", "p", refs="[3]",
                        category="Lead halide perovskite"),
    "FASnBr3": Material("FASnBr3", 2.40, 3.30, 6.0, 5, 5, 1e18, 1e18,
                        1e16, 1e16, 2e4, "absorber", "p", refs="[9]",
                        category="Tin halide perovskite"),
    "CsPbBr3": Material("CsPbBr3", 2.30, 3.40, 5.0, 2.5, 2.5, 2.2e18, 1.8e19,
                        1e16, 1e15, 5e4, "absorber", "p", refs="[3]",
                        category="Lead halide perovskite"),
    "KSnI3": Material("KSnI3", 1.30, 3.60, 10.0, 35, 35, 1e18, 1e18,
                      1e16, 1e16, 5e4, "absorber", "p", refs="[9]",
                      category="Tin halide perovskite"),
    "RbSnI3": Material("RbSnI3", 1.32, 3.55, 9.5, 20, 20, 1e18, 1e18,
                       1e16, 1e16, 5e4, "absorber", "p", refs="[9]",
                       category="Tin halide perovskite"),
})

HTL_DB.update({
    "Me4PACz": Material("Me4PACz", 3.30, 2.10, 3.0, 1e-4, 1e-3, 2.2e18, 1.8e19,
                        1e18, 1e15, 0, "htl", "p", refs="[9]",
                        category="Self-assembled monolayer"),
    "2PACz": Material("2PACz", 3.50, 2.00, 3.0, 1e-4, 1e-3, 2.2e18, 1.8e19,
                      1e18, 1e15, 0, "htl", "p", refs="[9]",
                      category="Self-assembled monolayer"),
    "CuCrO2": Material("CuCrO2", 3.10, 2.30, 11.0, 0.1, 0.1, 2.2e18, 1.8e19,
                       1e18, 1e15, 0, "htl", "p", refs="[9]",
                       category="Delafossite oxide"),
})

ETL_DB.update({
    "BaSnO3": Material("BaSnO3", 3.10, 4.40, 20.0, 320, 320, 2.2e18, 1.8e19,
                       5e17, 1e15, 0, "etl", "n", refs="[9]", category="Perovskite oxide"),
    "SrTiO3": Material("SrTiO3", 3.20, 4.10, 300, 5, 5, 2.2e18, 1.8e19,
                       1e17, 1e15, 0, "etl", "n", refs="[9]", category="Perovskite oxide"),
    "Nb2O5": Material("Nb2O5", 3.40, 4.30, 25, 0.01, 0.01, 2.2e18, 1.8e19,
                      1e17, 1e15, 0, "etl", "n", refs="[9]", category="Metal oxide"),
})


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE DEFECT LAYER (IDL) PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class InterfaceDefects:
    """Interface defect layer parameters."""
    Nt_if: float = 1e12     # Interface trap density [/cm²]
    sigma_e: float = 1e-15  # Electron capture cross-section [cm²]
    sigma_h: float = 1e-15  # Hole capture cross-section [cm²]
    Et_offset: float = 0.0  # Trap energy offset from midgap [eV]
    S_n: float = 1e4        # Surface recombination velocity electrons [cm/s]
    S_p: float = 1e4        # Surface recombination velocity holes [cm/s]

DEFAULT_IDL = InterfaceDefects()
HIGH_QUALITY_IDL = InterfaceDefects(Nt_if=1e10, S_n=1e2, S_p=1e2)
LOW_QUALITY_IDL = InterfaceDefects(Nt_if=1e14, S_n=1e6, S_p=1e6)


# ═══════════════════════════════════════════════════════════════════════════════
# GRADED LAYER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GradedProfile:
    """Graded composition profile for absorber layers."""
    Eg_front: float = 1.55   # Bandgap at front (ETL side) [eV]
    Eg_back: float = 1.55    # Bandgap at back (HTL side) [eV]
    chi_front: float = 3.93  # Electron affinity at front [eV]
    chi_back: float = 3.93   # Electron affinity at back [eV]
    grading_type: str = "linear"  # "linear", "exponential", "v-shaped"
    
    def get_Eg(self, x_frac):
        """Get bandgap at fractional position x_frac (0=front, 1=back)."""
        if self.grading_type == "linear":
            return self.Eg_front + (self.Eg_back - self.Eg_front) * x_frac
        elif self.grading_type == "exponential":
            return self.Eg_front * np.exp(np.log(self.Eg_back / self.Eg_front) * x_frac)
        elif self.grading_type == "v-shaped":
            if x_frac < 0.5:
                return self.Eg_front + (self.Eg_back - self.Eg_front) * 2 * x_frac
            else:
                return self.Eg_back + (self.Eg_front - self.Eg_back) * 2 * (x_frac - 0.5)
        return self.Eg_front
    
    def get_chi(self, x_frac):
        """Get electron affinity at fractional position."""
        return self.chi_front + (self.chi_back - self.chi_front) * x_frac


# ═══════════════════════════════════════════════════════════════════════════════
# TANDEM CELL SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TandemConfig:
    """Configuration for tandem/multi-junction cells."""
    top_absorber: str = "FAPbI3"       # Wide-gap top cell
    bottom_absorber: str = "CsSnI3"    # Narrow-gap bottom cell
    top_thickness_nm: float = 400
    bottom_thickness_nm: float = 500
    terminal_config: str = "2T"        # "2T" (series) or "4T" (independent)
    tunnel_junction_R: float = 1.0     # Tunnel junction resistance [Ω·cm²]
