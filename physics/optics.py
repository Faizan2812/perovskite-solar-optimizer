"""
Transfer Matrix Method (TMM) Optical Simulator
================================================
Computes reflection, transmission, absorption, and generation profiles
in multilayer thin-film solar cells accounting for coherent interference.

Uses complex refractive indices (n + ik) with Cauchy/Tauc-Lorentz models
when experimental data is not available.
"""
import numpy as np
from typing import List, Tuple, Dict


def cauchy_nk(lam_nm, n0, n1, n2, Eg, alpha_0=0, E_u=0.05):
    """
    Compute n and k from Cauchy model for n and Tauc-Lorentz for k.
    n(λ) = n0 + n1/λ² + n2/λ⁴  (λ in μm)
    k from absorption coefficient via Tauc model.
    """
    lam_um = lam_nm / 1000.0
    n = n0 + n1 / lam_um**2 + n2 / lam_um**4
    
    E_ph = 1240.0 / lam_nm  # eV
    if E_ph > Eg:
        alpha = alpha_0 * np.sqrt((E_ph - Eg) / max(Eg, 0.5))
    elif E_ph > Eg - E_u:
        alpha = alpha_0 * np.exp((E_ph - Eg) / E_u) * 0.1
    else:
        alpha = 0.0
    
    k = alpha * lam_nm * 1e-7 / (4 * np.pi)  # α = 4πk/λ
    return max(n, 1.0), max(k, 0.0)


# ── Material optical constants database ──────────────────────────────────
# Format: {name: (n0, n1, n2, Eg, alpha_0)}
# n0,n1,n2 are Cauchy coefficients; alpha_0 is absorption coefficient at Eg
OPTICAL_DB = {
    # Transport layers (wide bandgap → transparent)
    "FTO":          (1.90, 0.01, 0, 3.50, 0),
    "ITO":          (1.85, 0.01, 0, 3.60, 0),
    "Glass":        (1.52, 0.004, 0, 5.0, 0),
    # ETL
    "SnO2":         (2.00, 0.02, 0, 3.60, 0),
    "TiO2":         (2.50, 0.03, 0, 3.20, 1e3),
    "ZnO":          (2.00, 0.02, 0, 3.30, 0),
    "C60":          (2.00, 0.01, 0, 1.70, 5e4),
    "PCBM":         (2.00, 0.01, 0, 2.00, 3e4),
    "WS2":          (2.80, 0.05, 0, 1.80, 5e4),
    "CeO2":         (2.20, 0.02, 0, 3.50, 0),
    "IGZO":         (2.00, 0.01, 0, 3.05, 0),
    "ZnSe":         (2.60, 0.03, 0, 2.81, 1e4),
    "WO3":          (2.10, 0.02, 0, 2.60, 1e3),
    # Absorbers
    "Cs2SnI6":      (2.40, 0.05, 0, 1.48, 1.2e5),
    "MAPbI3":       (2.50, 0.06, 0, 1.55, 1.5e5),
    "FAPbI3":       (2.50, 0.06, 0, 1.51, 1.4e5),
    "CsPbI3":       (2.40, 0.05, 0, 1.73, 1.2e5),
    "MASnI3":       (2.40, 0.05, 0, 1.30, 5e4),
    "MAPbBr3":      (2.30, 0.04, 0, 2.30, 5e4),
    "CsSnI3":       (2.40, 0.05, 0, 1.30, 5e4),
    "FASnI3":       (2.40, 0.05, 0, 1.41, 5e4),
    "CsPbI2Br":     (2.40, 0.05, 0, 1.88, 1e5),
    "Cs2AgBiBr6":   (2.20, 0.03, 0, 2.05, 5e3),
    "Cs2TiBr6":     (2.30, 0.04, 0, 1.80, 2e4),
    "BaZrS3":       (2.60, 0.06, 0, 1.73, 8e4),
    "CsGeI3":       (2.40, 0.05, 0, 1.63, 1e5),
    # HTL
    "Cu2O":         (2.60, 0.04, 0, 2.17, 5e4),
    "Spiro-OMeTAD": (1.70, 0.01, 0, 3.00, 0),
    "NiO":          (2.30, 0.02, 0, 3.80, 0),
    "CuSCN":        (1.80, 0.01, 0, 3.60, 0),
    "CuI":          (2.10, 0.02, 0, 3.10, 0),
    "PEDOT:PSS":    (1.50, 0.01, 0, 1.60, 1e4),
    "P3HT":         (1.80, 0.02, 0, 1.70, 5e3),
    "CuO":          (2.50, 0.03, 0, 1.35, 5e4),
    "V2O5":         (2.20, 0.02, 0, 2.20, 1e3),
    "MoO3":         (2.10, 0.02, 0, 3.00, 0),
    "CBTS":         (2.40, 0.03, 0, 1.90, 3e4),
    "SrCu2O2":      (2.00, 0.02, 0, 3.30, 0),
    # Metals
    "Au":           (0.16, 0, 0, 0, 0),  # Special handling for metals
    "Ag":           (0.13, 0, 0, 0, 0),
}

# Metal k values (wavelength-independent approximation)
METAL_K = {"Au": 3.6, "Ag": 3.3, "Al": 6.0, "Cu": 2.9}


def get_nk(material_name, lam_nm):
    """Get complex refractive index for a material at wavelength."""
    if material_name in METAL_K:
        return complex(OPTICAL_DB.get(material_name, (0.2,0,0,0,0))[0], METAL_K[material_name])
    
    if material_name in OPTICAL_DB:
        n0, n1, n2, Eg, alpha_0 = OPTICAL_DB[material_name]
        n, k = cauchy_nk(lam_nm, n0, n1, n2, Eg, alpha_0)
        return complex(n, k)
    
    return complex(2.0, 0.0)  # Default


def transfer_matrix(n_layers, d_layers, lam_nm):
    """
    Compute transfer matrix for a multilayer stack.
    
    Args:
        n_layers: list of complex refractive indices [n + ik]
        d_layers: list of layer thicknesses [nm] (first and last are semi-infinite)
        lam_nm: wavelength [nm]
    
    Returns:
        M: 2x2 transfer matrix
        r: reflection coefficient
        t: transmission coefficient
    """
    M = np.eye(2, dtype=complex)
    
    for j in range(1, len(n_layers)):
        # Interface matrix (Fresnel coefficients for normal incidence)
        n_i = n_layers[j - 1]
        n_j = n_layers[j]
        r_ij = (n_i - n_j) / (n_i + n_j)
        t_ij = 2 * n_i / (n_i + n_j)
        
        D_ij = np.array([[1, r_ij], [r_ij, 1]]) / t_ij
        M = M @ D_ij
        
        # Propagation matrix (skip for last layer = semi-infinite)
        if j < len(n_layers) - 1 and d_layers[j] > 0:
            delta = 2 * np.pi * n_j * d_layers[j] / lam_nm
            P_j = np.array([[np.exp(-1j * delta), 0],
                           [0, np.exp(1j * delta)]])
            M = M @ P_j
    
    r = M[1, 0] / M[0, 0]
    t = 1.0 / M[0, 0]
    
    R = float(np.abs(r)**2)
    T = float(np.real(n_layers[-1] / n_layers[0]) * np.abs(t)**2)
    
    return M, R, T


def tmm_absorption_profile(layer_names, thicknesses_nm, lam_array, absorber_idx):
    """
    Compute absorption in each layer using TMM.
    
    Args:
        layer_names: list of material names (including ambient/substrate)
        thicknesses_nm: list of thicknesses (0 for semi-infinite ambient/substrate)
        lam_array: wavelength array [nm]
        absorber_idx: index of the absorber layer
    
    Returns:
        R_spectrum: reflectance vs wavelength
        A_absorber: absorption fraction in absorber vs wavelength
        A_total: total absorption vs wavelength
    """
    R_spectrum = np.zeros(len(lam_array))
    A_absorber = np.zeros(len(lam_array))
    A_total = np.zeros(len(lam_array))
    
    for i, lam in enumerate(lam_array):
        n_list = [get_nk(name, lam) for name in layer_names]
        _, R, T = transfer_matrix(n_list, thicknesses_nm, lam)
        
        R_spectrum[i] = R
        A_total[i] = 1 - R - T
        
        # Estimate absorber absorption from Beer-Lambert within TMM framework
        n_abs = n_list[absorber_idx]
        d_abs = thicknesses_nm[absorber_idx]
        alpha_abs = 4 * np.pi * n_abs.imag / (lam * 1e-7)  # /cm
        A_abs = (1 - R) * (1 - np.exp(-float(np.real(alpha_abs)) * d_abs * 1e-7))
        A_absorber[i] = min(float(np.real(A_abs)), A_total[i])
    
    return R_spectrum, A_absorber, A_total


def compute_tmm_generation(layer_names, thicknesses_nm, absorber_name, 
                           absorber_idx, d_abs_nm, Eg, N_spatial=100):
    """
    Compute generation profile G(x) in the absorber using TMM.
    Returns spatially-resolved generation rate [/cm³/s].
    """
    from physics.device import am15g_photon_flux
    
    lam_array = np.arange(300, min(1240/Eg*1.05, 1200), 5)
    R_spec, A_abs, _ = tmm_absorption_profile(layer_names, thicknesses_nm, 
                                               lam_array, absorber_idx)
    
    x = np.linspace(0, d_abs_nm * 1e-7, N_spatial)  # cm
    G = np.zeros(N_spatial)
    
    for i, lam in enumerate(lam_array):
        if lam < 300: continue
        n_complex = get_nk(absorber_name, lam)
        alpha = 4 * np.pi * n_complex.imag / (lam * 1e-7)  # /cm
        alpha = max(float(np.real(alpha)), 0)
        
        phi = am15g_photon_flux(lam)  # photons/cm²/s/nm
        incident = phi * (1 - R_spec[i])
        
        G += incident * alpha * np.exp(-alpha * x) * 5  # dlam=5nm
    
    return x, G


def compute_tmm_qe(layer_names, thicknesses_nm, absorber_name, absorber_idx,
                    Eg, eta_collection=0.95):
    """Compute EQE spectrum using TMM."""
    lam_array = np.arange(300, 1200, 5)
    R_spec, A_abs, _ = tmm_absorption_profile(layer_names, thicknesses_nm,
                                               lam_array, absorber_idx)
    
    qe = np.zeros(len(lam_array))
    for i, lam in enumerate(lam_array):
        E_ph = 1240.0 / lam
        if E_ph < Eg * 0.97:
            continue
        qe[i] = A_abs[i] * eta_collection * 100  # percentage
    
    return lam_array, qe, R_spec
