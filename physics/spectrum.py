"""
Reference Solar Spectrum — ASTM G173-03 AM1.5G
================================================
Canonical AM1.5G global-tilt spectrum. Tabulated data is the standard
ASTM G173-03 reference (extracted from pvlib's reference spectra),
resampled onto a 5 nm grid covering 280-1700 nm.

Integration:
    280-1700 nm : 947.4 W/m^2  (94.7% of total)
    full (ASTM) : 1000.37 W/m^2

The below-1700-nm truncation is inconsequential for any practical
photovoltaic material, whose bandgap forces lambda_cut < 1700 nm.

API
---
    photon_flux(lam)        : photons / (cm^2 s nm)
    irradiance(lam)         : W / (m^2 nm)
    integrated_power()      : mW/cm^2 in [280, lam_max]
    sq_jsc(Eg, EQE=1)       : ideal Jsc [mA/cm^2] for bandgap Eg [eV]
    photon_flux_in_band()   : photons / (cm^2 s) in [lam_min, lam_max]
"""
import numpy as np
from physics._astm_g173_data import WAVELENGTHS_NM, IRRADIANCE_W_M2_NM

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

H_PLANCK = 6.62607015e-34
C_LIGHT  = 2.99792458e8
Q_ELEC   = 1.602176634e-19
HC_EV_NM = 1239.841984

AM15G_WAVELENGTHS = WAVELENGTHS_NM.copy()
AM15G_IRRADIANCE  = IRRADIANCE_W_M2_NM.copy()

# Precompute photon flux in photons/(cm^2 s nm)
_lam_m = AM15G_WAVELENGTHS * 1e-9
AM15G_PHOTON_FLUX_CM2 = AM15G_IRRADIANCE * _lam_m / (H_PLANCK * C_LIGHT) * 1e-4


def irradiance(lam_nm):
    return np.interp(lam_nm, AM15G_WAVELENGTHS, AM15G_IRRADIANCE,
                     left=0.0, right=0.0)


def photon_flux(lam_nm):
    return np.interp(lam_nm, AM15G_WAVELENGTHS, AM15G_PHOTON_FLUX_CM2,
                     left=0.0, right=0.0)


def integrated_power(lam_max_nm=1700.0):
    mask = AM15G_WAVELENGTHS <= lam_max_nm
    P_w_m2 = _trapz(AM15G_IRRADIANCE[mask], AM15G_WAVELENGTHS[mask])
    return float(P_w_m2 * 0.1)


def sq_jsc(Eg_eV, EQE=1.0):
    if Eg_eV <= 0:
        return 0.0
    lam_edge_nm = HC_EV_NM / Eg_eV
    mask = AM15G_WAVELENGTHS <= lam_edge_nm
    if not np.any(mask):
        return 0.0
    phi = AM15G_PHOTON_FLUX_CM2[mask]
    lam = AM15G_WAVELENGTHS[mask]
    return float(Q_ELEC * _trapz(phi, lam) * EQE * 1000.0)


def photon_flux_in_band(lam_min_nm, lam_max_nm):
    mask = (AM15G_WAVELENGTHS >= lam_min_nm) & (AM15G_WAVELENGTHS <= lam_max_nm)
    if not np.any(mask):
        return 0.0
    return float(_trapz(AM15G_PHOTON_FLUX_CM2[mask], AM15G_WAVELENGTHS[mask]))
