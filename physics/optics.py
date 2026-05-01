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


def transfer_matrix(n_layers, d_layers, lam_nm, return_layer_amplitudes=False):
    """
    Compute transfer matrix for a multilayer stack at normal incidence.

    Args:
        n_layers: list of complex refractive indices [n + ik]; length L+2,
                  where L is number of finite layers; first = ambient,
                  last = substrate (both semi-infinite)
        d_layers: list of layer thicknesses [nm]; first and last ignored
        lam_nm:   wavelength [nm]
        return_layer_amplitudes: if True, also return per-layer forward/backward
                  wave amplitudes (A_l, B_l) at the LEFT interface of each layer
                  in the normalization where incident amplitude A_0 = 1.

    Returns:
        M: 2x2 system transfer matrix
        R, T: power reflectance / transmittance
        (optional) A_all, B_all: arrays of length len(n_layers), the
                  forward and backward wave amplitudes at the left boundary
                  of each layer (for A_0 = 1 incoming wave).
    """
    L = len(n_layers)
    # Interface matrices D_{j-1, j}, propagation matrices P_j.
    # The full system matrix is M = D_01 P_1 D_12 P_2 ... P_{L-2} D_{L-2, L-1}.
    # But to extract amplitudes in each layer we build partial products.
    # [A_0, B_0]^T = M [A_{L-1}, B_{L-1}]^T, with B_{L-1} = 0 (no incoming from right).
    def D(ni, nj):
        r = (ni - nj) / (ni + nj)
        t = 2 * ni / (ni + nj)
        return np.array([[1, r], [r, 1]], dtype=complex) / t

    def P(n, d_nm):
        delta = 2 * np.pi * n * d_nm / lam_nm
        return np.array([[np.exp(-1j * delta), 0],
                         [0,                    np.exp(1j * delta)]], dtype=complex)

    # Build cumulative matrices from left: M_to_l transforms [A_l, B_l]
    # at the LEFT side of layer l into [A_0, B_0]: [A_0,B_0] = M_to_l [A_l,B_l]
    M_to_l = [np.eye(2, dtype=complex)]   # at layer 0 (ambient), identity
    for j in range(1, L):
        # Step from left edge of layer j-1 to left edge of layer j
        # For j==1 (first finite layer): no propagation through ambient
        # For 1<j<L-1: propagate through layer j-1, then interface j-1 -> j
        # For j==L-1 (substrate): propagate through layer L-2, then interface to substrate
        M_step = np.eye(2, dtype=complex)
        if 1 <= j - 1 <= L - 2 and j - 1 != 0:
            # Propagate through finite layer j-1
            M_step = P(n_layers[j - 1], d_layers[j - 1]) @ M_step
        # Interface j-1 -> j
        M_step = D(n_layers[j - 1], n_layers[j]) @ M_step
        M_to_l.append(M_to_l[-1] @ M_step)

    M = M_to_l[-1]
    # A_0 = 1 incident, B_0 = r. Then A_{L-1} = t, B_{L-1} = 0.
    # From [A_0, B_0]^T = M [A_{L-1}, B_{L-1}]^T with B_{L-1}=0:
    #   1     = M[0,0] A_{L-1}
    #   r     = M[1,0] A_{L-1}
    # => t = A_{L-1} = 1/M[0,0], r = M[1,0]/M[0,0]
    A_sub = 1.0 / M[0, 0]
    r = M[1, 0] / M[0, 0]
    t = A_sub
    R = float(np.abs(r) ** 2)
    T = float(np.real(n_layers[-1] / n_layers[0]) * np.abs(t) ** 2)

    if not return_layer_amplitudes:
        return M, R, T

    # Compute amplitudes at LEFT boundary of each layer:
    #   [A_0, B_0]^T = M_to_l[j] [A_j, B_j]^T
    # We know A_0 = 1, B_0 = r. Invert each M_to_l[j]:
    A_all = np.zeros(L, dtype=complex)
    B_all = np.zeros(L, dtype=complex)
    lhs = np.array([1.0 + 0j, r])
    for j in range(L):
        Minv = np.linalg.inv(M_to_l[j])
        vec = Minv @ lhs
        A_all[j] = vec[0]
        B_all[j] = vec[1]
    return M, R, T, A_all, B_all


def tmm_absorption_profile(layer_names, thicknesses_nm, lam_array, absorber_idx):
    """
    TMM absorption spectrum with proper coherent field calculation.

    For each wavelength: solves the full stack, extracts forward (A_j) and
    backward (B_j) wave amplitudes in the absorber, then integrates the
    absorbed power density
        a(x) = alpha_j * |E(x)|^2 / (n_j * |E_0|^2)
    across the absorber thickness, where
        E(x) = A_j e^{i k_j x} + B_j e^{-i k_j x},   k_j = 2 pi n_j / lam.
    The ratio of power absorbed in the absorber to incident power is
    returned as A_absorber; R and total (1-R-T) are also returned.
    """
    R_spectrum = np.zeros(len(lam_array))
    A_absorber = np.zeros(len(lam_array))
    A_total = np.zeros(len(lam_array))

    for i, lam in enumerate(lam_array):
        n_list = [get_nk(name, lam) for name in layer_names]
        M, R, T, A_all, B_all = transfer_matrix(n_list, thicknesses_nm, lam,
                                                return_layer_amplitudes=True)
        R_spectrum[i] = R
        A_total[i] = max(0.0, 1 - R - T)

        j = absorber_idx
        n_j = n_list[j]
        d_j = thicknesses_nm[j]           # nm
        if d_j <= 0:
            A_absorber[i] = 0.0
            continue
        # Field inside absorber at local coord x (0 at left interface of absorber):
        # E(x) = A_j e^{+i k_j x} + B_j e^{-i k_j x}, with k_j = 2 pi n_j / lam (nm^-1)
        k_j = 2 * np.pi * n_j / lam       # 1/nm
        # Absorbed power per unit volume: (omega eps_0 / 2) * Im(eps) * |E|^2
        # Integrated over absorber, normalized by incident intensity.
        # For Re-n_0 = 1 vacuum incidence, the fraction of incident power
        # absorbed in the absorber is:
        #   A_j_frac = (2 pi Im(n_j) / lam) * integral_0^d (|E(x)|^2 / |E_0|^2) dx / n_0
        # Since we take A_0 = 1 (ambient incident amp), |E_0|^2 = 1 and n_0 is real.
        # Compute |E(x)|^2 analytically:
        #   |E(x)|^2 = |A|^2 exp(-2 Im(k) x) + |B|^2 exp(+2 Im(k) x)
        #             + 2 Re( A B* exp(+2 i Re(k) x) )
        Ia = np.imag(k_j)
        Ra = np.real(k_j)
        Aj, Bj = A_all[j], B_all[j]
        # Analytic integral from 0 to d_j (nm):
        def safe_expm1(z):
            return np.expm1(z)
        # Term 1: |A|^2 * int_0^d exp(-2 Im(k) x) dx
        if abs(Ia) > 1e-12:
            I1 = np.abs(Aj) ** 2 * (1 - np.exp(-2 * Ia * d_j)) / (2 * Ia)
            I2 = np.abs(Bj) ** 2 * (np.exp(+2 * Ia * d_j) - 1) / (2 * Ia)
        else:
            I1 = np.abs(Aj) ** 2 * d_j
            I2 = np.abs(Bj) ** 2 * d_j
        # Cross term: 2 Re( A B* int_0^d exp(+2 i Re(k) x) dx )
        if abs(Ra) > 1e-12:
            I3 = 2 * np.real(Aj * np.conj(Bj) * (np.exp(+2j * Ra * d_j) - 1) / (2j * Ra))
        else:
            I3 = 2 * np.real(Aj * np.conj(Bj)) * d_j
        E2_integral = float(np.real(I1 + I2 + I3))    # nm (integral of |E|^2 dx)
        # Absorption coefficient (in 1/nm)
        alpha_nm = 4 * np.pi * n_j.imag / lam         # 1/nm
        # Absorbed fraction of incident power (accounting for n_0 = ambient index):
        # A_frac = alpha * Re(n_j) * integral(|E|^2/|E_0|^2) dx / (Re(n_0))
        # Standard Byrnes formulation — use Re(n_j) to convert field to power.
        n_0 = np.real(n_list[0]) if np.real(n_list[0]) > 1e-6 else 1.0
        A_frac = alpha_nm * np.real(n_j) * E2_integral / n_0
        A_frac = float(np.clip(A_frac, 0.0, A_total[i]))
        A_absorber[i] = A_frac

    return R_spectrum, A_absorber, A_total


def compute_tmm_generation(layer_names, thicknesses_nm, absorber_name,
                           absorber_idx, d_abs_nm, Eg, N_spatial=100):
    """
    Compute generation profile G(x) in the absorber using TMM + coherent fields.

    Returns spatially-resolved generation rate [/cm³/s] on a grid of N_spatial
    points from x=0 (front of absorber) to x=d_abs_nm. The computation uses
    the true coherent |E(x)|² inside the absorber rather than Beer-Lambert,
    so interference fringes are preserved.
    """
    from physics.spectrum import photon_flux as _pf, AM15G_WAVELENGTHS, HC_EV_NM

    lam_edge = HC_EV_NM / Eg
    m = (AM15G_WAVELENGTHS <= lam_edge) & (AM15G_WAVELENGTHS >= 280)
    lams = AM15G_WAVELENGTHS[m]
    x_nm = np.linspace(0, d_abs_nm, N_spatial)
    x_cm = x_nm * 1e-7
    G = np.zeros(N_spatial)

    for lam in lams:
        n_list = [get_nk(name, lam) for name in layer_names]
        M, R, T, A_all, B_all = transfer_matrix(n_list, thicknesses_nm, lam,
                                                return_layer_amplitudes=True)
        n_j = n_list[absorber_idx]
        k_j = 2 * np.pi * n_j / lam        # 1/nm
        Aj, Bj = A_all[absorber_idx], B_all[absorber_idx]
        # |E(x)|² for x in absorber local coord (nm)
        # Compute with complex exponentials; take |.|²
        E_x = Aj * np.exp(+1j * k_j * x_nm) + Bj * np.exp(-1j * k_j * x_nm)
        E2 = np.abs(E_x) ** 2                       # dimensionless (|E|² / |E_0|²)
        # Photon absorption rate per unit volume at depth x:
        #   g(x) [photons/cm³/s/nm] = phi_inc * alpha * Re(n_j) * |E|² / n_0
        phi = float(_pf(lam))                       # photons/cm²/s/nm
        alpha_cm = 4 * np.pi * n_j.imag / (lam * 1e-7)     # 1/cm
        n_0 = np.real(n_list[0]) if np.real(n_list[0]) > 1e-6 else 1.0
        g = phi * float(np.real(alpha_cm)) * float(np.real(n_j)) * E2 / n_0  # per cm³ per s per nm
        G += g * 5.0        # dlam = 5 nm

    return x_cm, G


def compute_tmm_qe(layer_names, thicknesses_nm, absorber_name, absorber_idx,
                   Eg, eta_collection=0.95):
    """Compute EQE(λ) from TMM absorption: EQE(λ) = A_abs(λ) · η_collection."""
    from physics.spectrum import AM15G_WAVELENGTHS, HC_EV_NM
    lam_array = np.arange(300, 1200, 5)
    R_spec, A_abs, _ = tmm_absorption_profile(layer_names, thicknesses_nm,
                                               lam_array, absorber_idx)
    qe = np.zeros(len(lam_array))
    for i, lam in enumerate(lam_array):
        E_ph = HC_EV_NM / lam
        if E_ph < Eg * 0.97:
            continue
        qe[i] = A_abs[i] * eta_collection * 100  # percentage
    return lam_array, qe, R_spec
