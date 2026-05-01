"""
1D Drift-Diffusion Solver for Heterostructure Solar Cells
==========================================================
Proper Scharfetter-Gummel discretization + Gummel outer iteration.
Solves the coupled steady-state semiconductor equations:

    d/dx ( eps dψ/dx )       = -q ( p - n + Nd - Na )        (Poisson)
    dJ_n/dx                  = q ( R - G )                    (electron continuity)
    dJ_p/dx                  = -q ( R - G )                   (hole continuity)

    J_n = q μ_n V_T [ n_{i+1} B(ξ_n) - n_i B(-ξ_n) ] / h      (SG flux)
    J_p = -q μ_p V_T [ p_{i+1} B(ξ_p) - p_i B(-ξ_p) ] / h

where ξ_n = -ΔE_c/kT and ξ_p = ΔE_v/kT account for heterojunction band
offsets (E_c = -qψ - χ, E_v = E_c - Eg).

Boundary conditions: ideal ohmic contacts. Dirichlet BCs on ψ (from charge
neutrality + Fermi-level alignment), on n (equals majority doping density),
and on p (= ni²/n on n-side, or majority Na on p-side).

Recombination model: SRH (trap-assisted) + radiative + Auger, with the
trap lifetimes derived from the material's Nt and capture cross-sections.

Units (cgs-compatible):
    position   : cm
    potential  : V
    densities  : /cm^3
    currents   : A/cm^2
    εᵣ·ε₀      : eps_0 = 8.854e-14 F/cm
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

# ── Physical constants ──────────────────────────────────────────────────────
Q        = 1.602176634e-19       # C
K_B      = 1.380649e-23          # J/K
EPS_0    = 8.854187817e-14       # F/cm
T_REF    = 300.0

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def bernoulli(x):
    """Numerically stable Bernoulli function B(x) = x/(exp(x)-1).
    B(0)=1, B(-inf)=-inf·-1=+inf (actually: B(x→-∞) = -x), B(+inf)=0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    large_pos = x > 40.0
    large_neg = x < -40.0
    mid = ~(small | large_pos | large_neg)
    # Taylor near 0: B(x) ≈ 1 - x/2 + x²/12
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    # Large positive: B(x) ≈ x·exp(-x)
    out[large_pos] = x[large_pos] * np.exp(-x[large_pos])
    # Large negative: B(x) → -x (since exp(x)-1 → -1)
    out[large_neg] = -x[large_neg]
    # Generic
    xm = x[mid]
    out[mid] = xm / np.expm1(xm)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DeviceMesh:
    """Mesh and per-node material parameters."""
    x: np.ndarray         # position [cm], length N
    eps: np.ndarray       # permittivity [F/cm] (= eps_r * eps_0)
    chi: np.ndarray       # electron affinity [eV]
    Eg: np.ndarray        # bandgap [eV]
    Nc: np.ndarray        # CB DOS [/cm^3]
    Nv: np.ndarray        # VB DOS [/cm^3]
    mu_n: np.ndarray      # electron mobility [cm^2/V/s]
    mu_p: np.ndarray      # hole mobility [cm^2/V/s]
    Nd: np.ndarray        # donor density [/cm^3]
    Na: np.ndarray        # acceptor density [/cm^3]
    tau_n: np.ndarray     # electron SRH lifetime [s]
    tau_p: np.ndarray     # hole SRH lifetime [s]
    B_rad: np.ndarray     # radiative coefficient [cm^3/s]
    Cn: np.ndarray        # Auger Cn [cm^6/s]
    Cp: np.ndarray        # Auger Cp [cm^6/s]
    layer: np.ndarray     # 0=HTL, 1=absorber, 2=ETL  (for reporting)
    # Interface defect layers (SCAPS-like IDL) — list of dicts with keys:
    #   "node_index": mesh index at the interface
    #   "S_n":        electron surface recomb velocity [cm/s]
    #   "S_p":        hole surface recomb velocity [cm/s]
    #   "Nt_if":      interface trap density [/cm²]  (informational)
    interfaces: list = None

    @property
    def N(self):
        return len(self.x)


@dataclass
class DDResult:
    """Drift-diffusion solve output."""
    x: np.ndarray
    psi: np.ndarray        # electrostatic potential [V]
    n: np.ndarray          # electron density [/cm^3]
    p: np.ndarray          # hole density [/cm^3]
    Ec: np.ndarray         # conduction band edge [eV]  (E_c = -qψ - χ, eV)
    Ev: np.ndarray         # valence band edge [eV]
    E_Fn: np.ndarray       # electron quasi-Fermi [eV]
    E_Fp: np.ndarray       # hole quasi-Fermi [eV]
    E_field: np.ndarray    # electric field [V/cm] (= -dψ/dx)
    J_n: np.ndarray        # electron current density at midpoints [A/cm^2]
    J_p: np.ndarray        # hole current density at midpoints [A/cm^2]
    J_total: float         # total device current [A/cm^2], = J_n + J_p (should be spatially constant)
    R: np.ndarray          # recombination rate [/cm^3/s]
    G: np.ndarray          # generation rate [/cm^3/s]
    V_applied: float
    converged: bool
    gummel_iter: int
    residual: float


# ═════════════════════════════════════════════════════════════════════════════
# MESH CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
def build_mesh(materials: List, thicknesses_nm: List[float],
               N_per_layer: List[int] = None,
               Nt_override: Optional[List[float]] = None,
               T: float = 300.0,
               interfaces: Optional[List[Dict]] = None,
               grading: Optional[Dict] = None) -> DeviceMesh:
    """Build a 1D mesh. `materials` is a list of Material objects (one per layer),
    `thicknesses_nm` the corresponding thicknesses in nm.

    interfaces: optional list of IDL specs. Each entry is a dict:
        {"between": (i, j), "S_n": ..., "S_p": ..., "Nt_if": ...}

    grading: optional dict specifying a graded-bandgap profile in one layer:
        {"layer_index": int,                   # which layer to grade
         "profile":     "linear"|"v"|"exp",    # shape
         "Eg_front":    float,                 # bandgap at left edge [eV]
         "Eg_back":     float,                 # bandgap at right edge [eV]
         "chi_front":   float,                 # electron affinity at left edge [eV]
         "chi_back":    float}                 # electron affinity at right edge [eV]
        If None, bandgap is uniform within each layer (from mat.Eg).
    """
    nlayers = len(materials)
    if N_per_layer is None:
        N_per_layer = [30, 80, 30][:nlayers]
    if Nt_override is None:
        Nt_override = [None] * nlayers

    d_cm = [t * 1e-7 for t in thicknesses_nm]
    boundaries = [0.0]
    for d in d_cm:
        boundaries.append(boundaries[-1] + d)

    x_parts = []; layer_parts = []
    for i, (d, Nx) in enumerate(zip(d_cm, N_per_layer)):
        if i == 0:
            xp = np.linspace(boundaries[i], boundaries[i+1], Nx, endpoint=False)
        elif i == nlayers - 1:
            xp = np.linspace(boundaries[i], boundaries[i+1], Nx)
        else:
            xp = np.linspace(boundaries[i], boundaries[i+1], Nx, endpoint=False)
        x_parts.append(xp)
        layer_parts.append(np.full(len(xp), i, dtype=int))
    x = np.concatenate(x_parts)
    layer = np.concatenate(layer_parts)
    N = len(x)

    arrs = {k: np.zeros(N) for k in
            ["eps","chi","Eg","Nc","Nv","mu_n","mu_p","Nd","Na","tau_n","tau_p","B_rad","Cn","Cp"]}

    for i, mat in enumerate(materials):
        mask = layer == i
        arrs["eps"][mask]  = mat.eps * EPS_0
        arrs["chi"][mask]  = mat.chi
        arrs["Eg"][mask]   = mat.Eg
        arrs["Nc"][mask]   = mat.Nc
        arrs["Nv"][mask]   = mat.Nv
        arrs["mu_n"][mask] = mat.mu_e
        arrs["mu_p"][mask] = mat.mu_h
        if mat.doping_type == 'p':
            arrs["Na"][mask] = mat.doping
        else:
            arrs["Nd"][mask] = mat.doping
        Nt = Nt_override[i] if Nt_override[i] is not None else mat.Nt
        sigma = getattr(mat, 'sigma_e', 1e-15)
        v_th = 1e7
        tau = 1.0 / (sigma * v_th * max(Nt, 1e6))
        arrs["tau_n"][mask] = tau
        arrs["tau_p"][mask] = tau
        arrs["B_rad"][mask] = 1e-10
        arrs["Cn"][mask] = 2.8e-31
        arrs["Cp"][mask] = 9.9e-32

    # Apply graded-bandgap profile if specified (overrides uniform Eg, chi in the chosen layer)
    if grading is not None:
        lidx = grading["layer_index"]
        mask = layer == lidx
        if np.any(mask):
            x_layer = x[mask]
            x_frac = (x_layer - x_layer.min()) / (x_layer.max() - x_layer.min() + 1e-30)
            profile = grading.get("profile", "linear")
            Eg_f = grading["Eg_front"]; Eg_b = grading["Eg_back"]
            chi_f = grading.get("chi_front", materials[lidx].chi)
            chi_b = grading.get("chi_back",  materials[lidx].chi)
            if profile == "linear":
                Eg_prof  = Eg_f  + (Eg_b  - Eg_f)  * x_frac
                chi_prof = chi_f + (chi_b - chi_f) * x_frac
            elif profile == "v":
                # V-shape: min at 50%, linear ramps to edges
                Eg_min = min(Eg_f, Eg_b)
                Eg_prof = np.where(x_frac < 0.5,
                                   Eg_f + (Eg_min - Eg_f) * 2 * x_frac,
                                   Eg_min + (Eg_b - Eg_min) * 2 * (x_frac - 0.5))
                chi_prof = chi_f + (chi_b - chi_f) * x_frac  # chi stays linear
            elif profile == "exp":
                Eg_prof = Eg_f * np.exp(np.log(Eg_b / max(Eg_f, 1e-3)) * x_frac)
                chi_prof = chi_f + (chi_b - chi_f) * x_frac
            else:
                Eg_prof = np.full_like(x_layer, Eg_f)
                chi_prof = np.full_like(x_layer, chi_f)
            arrs["Eg"][mask]  = Eg_prof
            arrs["chi"][mask] = chi_prof

    # Detect interface nodes (where layer index transitions)
    interface_nodes = []
    for k in range(1, N):
        if layer[k] != layer[k-1]:
            interface_nodes.append({"node_index": k,
                                    "between": (int(layer[k-1]), int(layer[k]))})

    if interfaces is not None:
        for user_if in interfaces:
            u_between = tuple(user_if.get("between", ()))
            for node_if in interface_nodes:
                if node_if["between"] == u_between:
                    node_if["S_n"]    = user_if.get("S_n", 1e4)
                    node_if["S_p"]    = user_if.get("S_p", 1e4)
                    node_if["Nt_if"]  = user_if.get("Nt_if", 1e12)
                    node_if["active"] = True
    for node_if in interface_nodes:
        if "active" not in node_if:
            node_if.update({"S_n": 0.0, "S_p": 0.0, "Nt_if": 0.0, "active": False})

    return DeviceMesh(x=x, layer=layer, interfaces=interface_nodes, **arrs)


# ═════════════════════════════════════════════════════════════════════════════
# RECOMBINATION
# ═════════════════════════════════════════════════════════════════════════════
def intrinsic_density(Nc, Nv, Eg, T=300.0):
    """n_i = sqrt(Nc*Nv) * exp(-Eg/(2 kT))."""
    kT_eV = K_B * T / Q
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * kT_eV))


def net_recombination(n, p, ni, tau_n, tau_p, B_rad, Cn, Cp):
    """U = U_SRH + U_rad + U_Auger  [/cm^3/s]. Reduces to zero at n·p = ni²."""
    # SRH (mid-gap traps)
    U_srh = (n * p - ni**2) / (tau_p * (n + ni) + tau_n * (p + ni))
    # Radiative
    U_rad = B_rad * (n * p - ni**2)
    # Auger
    U_aug = (Cn * n + Cp * p) * (n * p - ni**2)
    return U_srh + U_rad + U_aug


def interface_recombination_rate(n, p, ni, S_n, S_p):
    """Interface recombination rate per unit area [/cm²/s] via surface recombination
    velocities. Standard SCAPS formulation: R_if = (n p - ni²) / (n/S_p + p/S_n).
    Returns [/cm²/s]; caller must apply at the correct mesh node and divide by
    the local cell width to get a volumetric contribution."""
    return (n * p - ni**2) / (n / max(S_p, 1e-30) + p / max(S_n, 1e-30))


# ═════════════════════════════════════════════════════════════════════════════
# BOUNDARY CONDITIONS (ideal ohmic)
# ═════════════════════════════════════════════════════════════════════════════
def ohmic_bc(mat_left, mat_right, V_applied, T=300.0):
    """
    Compute ψ, n, p at the two ohmic contacts, and the built-in potential.

    Convention: x=0 is the HTL/anode side, x=L is the ETL/cathode side.
    The cathode is the reference (E_F_right = 0). Forward bias V shifts the
    anode quasi-Fermi level to -qV. The electrostatic potential at x=0
    shifts by +V relative to equilibrium.
    """
    kT = K_B * T / Q     # V
    # --- Right contact (ETL, n-type) ---
    ni_r = intrinsic_density(mat_right.Nc, mat_right.Nv, mat_right.Eg, T)
    if mat_right.doping_type == 'n':
        n_r = mat_right.doping
        p_r = ni_r**2 / n_r
    else:
        # p-type ETL material being used as ETL (unusual); use charge neutrality
        p_r = mat_right.doping
        n_r = ni_r**2 / max(p_r, ni_r * 1e-10)
    # Fermi level at right contact = 0 (reference). From n_r = Nc·exp(-E_c/kT):
    # E_c(L) = kT ln(Nc / n_r). Then ψ(L) = -(E_c(L) + χ_R)/q.
    Ec_L = kT * np.log(mat_right.Nc / max(n_r, 1e-30))
    psi_L = -(Ec_L + mat_right.chi)

    # --- Left contact (HTL, p-type) ---
    ni_l = intrinsic_density(mat_left.Nc, mat_left.Nv, mat_left.Eg, T)
    if mat_left.doping_type == 'p':
        p_l = mat_left.doping
        n_l = ni_l**2 / p_l
    else:
        n_l = mat_left.doping
        p_l = ni_l**2 / max(n_l, ni_l * 1e-10)
    # Under equilibrium, E_F = 0 everywhere. p_l = Nv·exp(E_v/kT) → E_v(0) = kT ln(p_l/Nv).
    # E_c(0) = E_v(0) + Eg. ψ(0) = -(E_c(0) + χ_L)/q.
    Ev_0 = kT * np.log(p_l / max(mat_left.Nv, 1e-30))
    Ec_0 = Ev_0 + mat_left.Eg
    psi_0_eq = -(Ec_0 + mat_left.chi)

    # Apply forward bias: anode shifts by +V_applied
    psi_0 = psi_0_eq + V_applied

    Vbi = psi_L - psi_0_eq  # built-in potential (positive)
    return {
        "psi_left": psi_0,   "psi_right": psi_L,
        "n_left":   n_l,     "n_right":   n_r,
        "p_left":   p_l,     "p_right":   p_r,
        "Vbi":      Vbi,
        "ni_left":  ni_l,    "ni_right":  ni_r,
    }


# ═════════════════════════════════════════════════════════════════════════════
# INITIAL GUESS (equilibrium depletion-approximation seed)
# ═════════════════════════════════════════════════════════════════════════════
def initial_guess(mesh: DeviceMesh, bc: Dict, T=300.0):
    """Generate a physically reasonable initial (ψ, n, p) from depletion
    approximation + Boltzmann carriers."""
    kT = K_B * T / Q
    N = mesh.N
    psi = np.zeros(N)

    # Linear ramp of ψ from psi_left to psi_right
    psi = bc["psi_left"] + (bc["psi_right"] - bc["psi_left"]) * (mesh.x - mesh.x[0]) / (mesh.x[-1] - mesh.x[0])

    # Carriers from Boltzmann, clipped for numerical safety
    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)
    # n = Nc · exp((q·ψ + χ - E_F_ref)/kT);  E_F_ref = 0 at equilibrium
    Ec = -psi - mesh.chi   # [eV]
    Ev = Ec - mesh.Eg
    n = mesh.Nc * np.exp(np.clip(-Ec / kT, -60, 60))
    p = mesh.Nv * np.exp(np.clip(Ev / kT, -60, 60))

    # Enforce contacts
    n[0]  = bc["n_left"];   n[-1] = bc["n_right"]
    p[0]  = bc["p_left"];   p[-1] = bc["p_right"]

    # Clip for numerical stability
    n = np.clip(n, 1e-10, 1e22)
    p = np.clip(p, 1e-10, 1e22)
    return psi, n, p


# ═════════════════════════════════════════════════════════════════════════════
# NEWTON POISSON SOLVE
# ═════════════════════════════════════════════════════════════════════════════
def solve_poisson_newton(mesh: DeviceMesh, psi0, n_in, p_in, bc, T=300.0,
                          max_iter=30, tol=1e-7):
    """Nonlinear Poisson solve with Boltzmann-linked n,p for stability.

    Decouple n,p from ψ: in each Newton step, use n = n_in · exp((ψ-ψ_old)/V_T)
    and p = p_in · exp(-(ψ-ψ_old)/V_T). This is the standard nonlinear Poisson
    trick (Scharfetter-Gummel Gummel scheme): it gives a quasi-Boltzmann
    relaxation that converges fast.

    Dirichlet BCs: ψ[0] = bc["psi_left"], ψ[-1] = bc["psi_right"].
    """
    V_T = K_B * T / Q
    N = mesh.N
    psi = psi0.copy().astype(np.float64)
    psi_old = psi.copy()
    n0 = n_in.copy()
    p0 = p_in.copy()

    # Enforce BCs
    psi[0]  = bc["psi_left"]
    psi[-1] = bc["psi_right"]

    for it in range(max_iter):
        # Quasi-Boltzmann: n, p as functions of ψ
        dpsi = psi - psi_old
        n_loc = n0 * np.exp(np.clip(dpsi / V_T, -60, 60))
        p_loc = p0 * np.exp(np.clip(-dpsi / V_T, -60, 60))
        n_loc = np.clip(n_loc, 1e-30, 1e25)
        p_loc = np.clip(p_loc, 1e-30, 1e25)

        # Build residual F_i and Jacobian J (tridiagonal).
        F = np.zeros(N)
        a = np.zeros(N)   # subdiag
        b = np.zeros(N)   # diag
        c = np.zeros(N)   # supdiag

        # Interior
        for i in range(1, N-1):
            dxm = mesh.x[i] - mesh.x[i-1]
            dxp = mesh.x[i+1] - mesh.x[i]
            dxa = 0.5 * (dxm + dxp)
            eps_m = 0.5 * (mesh.eps[i-1] + mesh.eps[i])
            eps_p = 0.5 * (mesh.eps[i]   + mesh.eps[i+1])
            # d/dx(eps dψ/dx) ≈ [eps_p (ψ_{i+1}-ψ_i)/dxp - eps_m (ψ_i-ψ_{i-1})/dxm] / dxa
            lapl = (eps_p * (psi[i+1] - psi[i]) / dxp
                    - eps_m * (psi[i]   - psi[i-1]) / dxm) / dxa
            rho  = Q * (p_loc[i] - n_loc[i] + mesh.Nd[i] - mesh.Na[i])
            F[i] = lapl + rho
            # d(rho)/d(psi_i) = Q·(-n/V_T - p/V_T)  (from quasi-Boltzmann)
            drho_dpsi = -Q * (n_loc[i] + p_loc[i]) / V_T
            a[i] = eps_m / (dxm * dxa)
            c[i] = eps_p / (dxp * dxa)
            b[i] = -(eps_m/dxm + eps_p/dxp) / dxa + drho_dpsi
        # Boundary rows (Dirichlet)
        b[0]  = 1.0; F[0]  = psi[0]  - bc["psi_left"]
        b[-1] = 1.0; F[-1] = psi[-1] - bc["psi_right"]

        # Solve tridiag: J · dψ = -F
        dpsi_corr = _thomas(a, b, c, -F)
        # Limit step for stability
        max_step = 0.5
        maxcorr = np.max(np.abs(dpsi_corr))
        if maxcorr > max_step:
            dpsi_corr *= max_step / maxcorr
        psi += dpsi_corr
        psi[0]  = bc["psi_left"]
        psi[-1] = bc["psi_right"]

        # Update n, p to be consistent with new psi (for next Newton step)
        psi_old = psi.copy()
        n0 = n_loc * np.exp(np.clip((psi - (psi - dpsi_corr))/V_T, -60, 60))
        p0 = p_loc * np.exp(np.clip(-(psi - (psi - dpsi_corr))/V_T, -60, 60))
        # Simpler: just re-evaluate Boltzmann from scratch
        n0 = n_loc * np.exp(np.clip(dpsi_corr/V_T, -60, 60))
        p0 = p_loc * np.exp(np.clip(-dpsi_corr/V_T, -60, 60))
        n0 = np.clip(n0, 1e-30, 1e25)
        p0 = np.clip(p0, 1e-30, 1e25)

        if maxcorr < tol:
            break

    return psi, n0, p0


def _thomas(a, b, c, d):
    """Thomas algorithm (tridiagonal solver)."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0] / b[0] if b[0] != 0 else 0.0
    dp[0] = d[0] / b[0] if b[0] != 0 else 0.0
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        if abs(denom) < 1e-300:
            denom = 1e-300
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


# ═════════════════════════════════════════════════════════════════════════════
# CONTINUITY: SCHARFETTER-GUMMEL
# ═════════════════════════════════════════════════════════════════════════════
def solve_continuity_n(mesh, psi, p, G, bc, T=300.0):
    """Solve electron continuity: dJ_n/dx = q(R - G) with SG flux.
    Linearize SRH recombination around current p to keep it linear in n."""
    V_T = K_B * T / Q
    N = mesh.N
    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)

    # SG coefficients on each edge i-1/2 (interior node i has edges to both sides)
    # ξ_n,i+1/2 = -(E_c,i+1 - E_c,i)/kT = ((ψ_{i+1}-ψ_i) + (χ_{i+1}-χ_i))/V_T
    # But χ is in eV and ψ in V, so (χ_{i+1}-χ_i)/V_T uses same units (both eV).
    # J_n,i+1/2 = (q μ_n V_T / dx_i) [n_{i+1} B(ξ) - n_i B(-ξ)]
    dx = np.diff(mesh.x)                             # length N-1
    dpsi = np.diff(psi)                              # N-1
    dchi = np.diff(mesh.chi)                         # N-1
    xi_n = (dpsi + dchi) / V_T                       # N-1, dimensionless
    mu_n_edge = 0.5 * (mesh.mu_n[:-1] + mesh.mu_n[1:])
    # Coefficients multiplying n_i and n_{i+1} in J_n,i+1/2
    coeff = Q * mu_n_edge * V_T / dx                 # N-1
    Bpos = bernoulli(xi_n)
    Bneg = bernoulli(-xi_n)

    # Assemble: for interior node i (1..N-2):
    #   (J_n,i+1/2 - J_n,i-1/2)/dxa_i  =  q(R - G)
    # where dxa_i = (dx_{i-1} + dx_i)/2
    # R = SRH linearized in n: with p fixed, R_srh ≈ (n·p - ni²)/[tau_p(n+ni) + tau_n(p+ni)]
    # We use an operator-splitting form: R = U(n, p_fixed). To keep linear, treat
    # denominator with the current n as frozen from prev iteration, or linearize.
    # For Gummel, freezing R evaluation's denominator is standard:
    #   R ≈ alpha_R * n - beta_R
    # where alpha_R = p / [tau_p(n+ni) + tau_n(p+ni)]  (approx)
    #       beta_R  = ni² / [tau_p(n+ni) + tau_n(p+ni)] (+ rad + auger terms)
    # To linearize, use the current n in the denominator.

    # We'll assemble full tridiagonal
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    rhs = np.zeros(N)

    for i in range(1, N-1):
        dxa = 0.5 * (dx[i-1] + dx[i])
        # J_n,i+1/2 = coeff[i] * [n_{i+1} Bpos[i] - n_i Bneg[i]]
        # J_n,i-1/2 = coeff[i-1] * [n_i Bpos[i-1] - n_{i-1} Bneg[i-1]]
        # Residual:  (J_n,i+1/2 - J_n,i-1/2) / dxa = q(R - G)
        # Rearrange: - coeff[i-1]·Bneg[i-1]·n_{i-1}
        #            + (coeff[i-1]·Bpos[i-1] + coeff[i]·Bneg[i])·n_i
        #            - coeff[i]·Bpos[i]·n_{i+1}   = q·(R - G) · dxa
        # Wait: J_n,i+1/2 - J_n,i-1/2 = coeff[i]·(n_{i+1}·Bpos[i] - n_i·Bneg[i])
        #                              - coeff[i-1]·(n_i·Bpos[i-1] - n_{i-1}·Bneg[i-1])
        # = coeff[i-1]·Bneg[i-1]·n_{i-1}
        #   + (-coeff[i]·Bneg[i] - coeff[i-1]·Bpos[i-1])·n_i
        #   + coeff[i]·Bpos[i]·n_{i+1}
        a_val = coeff[i-1] * Bneg[i-1]
        b_val = -coeff[i] * Bneg[i] - coeff[i-1] * Bpos[i-1]
        c_val = coeff[i] * Bpos[i]
        # SRH linearized: denom uses current n (from previous iteration), not unknown
        denom = mesh.tau_p[i]*(0.0 + ni[i]) + mesh.tau_n[i]*(p[i] + ni[i])
        # We don't have a "current n" at the unknown — use a Picard-style iteration
        # within Gummel. Move the n-dependent part to the diagonal.
        # R = [n·p - ni²]/denom + radiative + Auger
        # Linearize: alpha_n = p/denom + B_rad·p + Cn·(p·?)+...  (treat n-dependence)
        alpha_R = mesh.tau_p[i] * 1.0  # placeholder
        # Simpler: evaluate R at previous n, treat as explicit RHS. This is Picard.
        # We'll iterate externally.
        # For now: solve (J_n,i+1/2 - J_n,i-1/2)/dxa = q·(R(n_prev,p) - G)
        # Move R to RHS.
        a[i] = a_val / dxa
        b[i] = b_val / dxa
        c[i] = c_val / dxa
        rhs[i] = 0.0   # R and G go to RHS
    # Need R_prev — use 0 on first pass (no recombination), caller iterates
    # Actually, we'll use "full Picard": accept n from caller.

    # Boundary: Dirichlet on both sides
    b[0]  = 1.0; rhs[0]  = bc["n_left"]
    b[-1] = 1.0; rhs[-1] = bc["n_right"]

    return a, b, c, rhs, coeff, Bpos, Bneg, dx


def _continuity_n_residual(mesh, psi, n, p, G, T=300.0):
    """Build and return (J_n_edges, source), where source_i = q(R_i - G_i).
    Used for Picard-iterated continuity solve."""
    V_T = K_B * T / Q
    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)
    R = net_recombination(n, p, ni, mesh.tau_n, mesh.tau_p,
                          mesh.B_rad, mesh.Cn, mesh.Cp)
    source = Q * (R - G)
    return source


def solve_continuity_n_full(mesh, psi, p, G, bc, T=300.0,
                            n_guess=None, max_picard=12, tol=1e-4):
    """Solve electron continuity with Picard iteration on R.
    Interface defect recombination (IDL): if mesh has active interfaces,
    the recombination at each interface node is augmented by the surface-
    recombination-velocity term R_if = (np - ni²)/(n/S_p + p/S_n), converted
    to a volumetric rate by dividing by the local cell width."""
    V_T = K_B * T / Q
    N = mesh.N
    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)

    dx = np.diff(mesh.x)
    dpsi = np.diff(psi)
    dchi = np.diff(mesh.chi)
    xi_n = (dpsi + dchi) / V_T
    mu_n_edge = 0.5 * (mesh.mu_n[:-1] + mesh.mu_n[1:])
    coeff = Q * mu_n_edge * V_T / dx
    Bpos = bernoulli(xi_n)
    Bneg = bernoulli(-xi_n)

    # Precompute IDL contributions: map interface node -> (S_n, S_p, dx_local)
    if_nodes = {}
    if mesh.interfaces:
        for if_data in mesh.interfaces:
            if if_data.get("active", False):
                i_node = if_data["node_index"]
                if 0 < i_node < N - 1:
                    dx_local = 0.5 * (dx[i_node-1] + dx[i_node])
                    if_nodes[i_node] = (if_data["S_n"], if_data["S_p"], dx_local)

    n = n_guess.copy() if n_guess is not None else np.full(N, bc["n_right"])
    n = np.clip(n, 1e-20, 1e22)

    for pic in range(max_picard):
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); rhs = np.zeros(N)
        for i in range(1, N-1):
            dxa = 0.5 * (dx[i-1] + dx[i])
            a[i] = coeff[i-1] * Bneg[i-1] / dxa
            b[i] = (-coeff[i] * Bneg[i] - coeff[i-1] * Bpos[i-1]) / dxa
            c[i] = coeff[i] * Bpos[i] / dxa
            R_i = net_recombination(n[i], p[i], ni[i], mesh.tau_n[i], mesh.tau_p[i],
                                     mesh.B_rad[i], mesh.Cn[i], mesh.Cp[i])
            # Add IDL contribution if this is an interface node
            if i in if_nodes:
                S_n, S_p, dx_loc = if_nodes[i]
                R_if_vol = interface_recombination_rate(n[i], p[i], ni[i], S_n, S_p) / dx_loc
                R_i = R_i + R_if_vol
            rhs[i] = Q * (R_i - G[i])
        b[0]  = 1.0; rhs[0]  = bc["n_left"]
        b[-1] = 1.0; rhs[-1] = bc["n_right"]
        n_new = _thomas(a, b, c, rhs)
        n_new = np.clip(n_new, 1e-20, 1e22)
        omega = 0.8
        change = np.max(np.abs(np.log10(n_new/np.maximum(n, 1e-30) + 1e-30)))
        n = omega * n_new + (1 - omega) * n
        if change < tol:
            break
    return n


def solve_continuity_p_full(mesh, psi, n, G, bc, T=300.0,
                            p_guess=None, max_picard=12, tol=1e-4):
    """Solve hole continuity (analogous to electrons, with IDL)."""
    V_T = K_B * T / Q
    N = mesh.N
    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)

    dx = np.diff(mesh.x)
    dpsi = np.diff(psi)
    dchi = np.diff(mesh.chi)
    dEg  = np.diff(mesh.Eg)
    xi_p = (-dpsi - dchi - dEg) / V_T
    mu_p_edge = 0.5 * (mesh.mu_p[:-1] + mesh.mu_p[1:])
    coeff = -Q * mu_p_edge * V_T / dx
    Bpos = bernoulli(xi_p)
    Bneg = bernoulli(-xi_p)

    # IDL node map
    if_nodes = {}
    if mesh.interfaces:
        for if_data in mesh.interfaces:
            if if_data.get("active", False):
                i_node = if_data["node_index"]
                if 0 < i_node < N - 1:
                    dx_local = 0.5 * (dx[i_node-1] + dx[i_node])
                    if_nodes[i_node] = (if_data["S_n"], if_data["S_p"], dx_local)

    p = p_guess.copy() if p_guess is not None else np.full(N, bc["p_left"])
    p = np.clip(p, 1e-20, 1e22)

    for pic in range(max_picard):
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); rhs = np.zeros(N)
        for i in range(1, N-1):
            dxa = 0.5 * (dx[i-1] + dx[i])
            a[i] = coeff[i-1] * Bneg[i-1] / dxa
            b[i] = (-coeff[i] * Bneg[i] - coeff[i-1] * Bpos[i-1]) / dxa
            c[i] = coeff[i] * Bpos[i] / dxa
            R_i = net_recombination(n[i], p[i], ni[i], mesh.tau_n[i], mesh.tau_p[i],
                                     mesh.B_rad[i], mesh.Cn[i], mesh.Cp[i])
            if i in if_nodes:
                S_n, S_p, dx_loc = if_nodes[i]
                R_if_vol = interface_recombination_rate(n[i], p[i], ni[i], S_n, S_p) / dx_loc
                R_i = R_i + R_if_vol
            rhs[i] = -Q * (R_i - G[i])
        b[0]  = 1.0; rhs[0]  = bc["p_left"]
        b[-1] = 1.0; rhs[-1] = bc["p_right"]
        p_new = _thomas(a, b, c, rhs)
        p_new = np.clip(p_new, 1e-20, 1e22)
        omega = 0.8
        change = np.max(np.abs(np.log10(p_new/np.maximum(p, 1e-30) + 1e-30)))
        p = omega * p_new + (1 - omega) * p
        if change < tol:
            break
    return p


# ═════════════════════════════════════════════════════════════════════════════
# CURRENT EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def sg_currents(mesh, psi, n, p, T=300.0):
    """Scharfetter-Gummel electron and hole currents at edge midpoints."""
    V_T = K_B * T / Q
    dx = np.diff(mesh.x)
    dpsi = np.diff(psi)
    dchi = np.diff(mesh.chi)
    dEg  = np.diff(mesh.Eg)
    xi_n = (dpsi + dchi) / V_T
    xi_p = (-dpsi - dchi - dEg) / V_T
    mu_n_edge = 0.5 * (mesh.mu_n[:-1] + mesh.mu_n[1:])
    mu_p_edge = 0.5 * (mesh.mu_p[:-1] + mesh.mu_p[1:])
    J_n = (Q * mu_n_edge * V_T / dx) * (n[1:] * bernoulli(xi_n) - n[:-1] * bernoulli(-xi_n))
    J_p = (-Q * mu_p_edge * V_T / dx) * (p[1:] * bernoulli(xi_p) - p[:-1] * bernoulli(-xi_p))
    return J_n, J_p


# ═════════════════════════════════════════════════════════════════════════════
# GUMMEL OUTER ITERATION
# ═════════════════════════════════════════════════════════════════════════════
def solve_dd(mesh: DeviceMesh, G_profile, V_applied, mat_left, mat_right,
             T=300.0, max_gummel=25, tol_psi=1e-4, verbose=False,
             warm_start=None) -> DDResult:
    """
    Solve the coupled drift-diffusion system at a given applied voltage.
    mat_left = HTL material (x=0), mat_right = ETL material (x=L).

    warm_start: optional (psi, n, p) tuple to use as initial guess (e.g.
    from a previous voltage point). Hugely speeds up J-V sweeps.
    """
    V_T = K_B * T / Q
    bc = ohmic_bc(mat_left, mat_right, V_applied, T)

    if warm_start is not None:
        psi, n, p = [x.copy() for x in warm_start]
        # Re-impose boundary conditions for the new V
        psi[0]  = bc["psi_left"];   psi[-1] = bc["psi_right"]
        n[0]    = bc["n_left"];     n[-1]   = bc["n_right"]
        p[0]    = bc["p_left"];     p[-1]   = bc["p_right"]
    else:
        psi, n, p = initial_guess(mesh, bc, T)

    ni = intrinsic_density(mesh.Nc, mesh.Nv, mesh.Eg, T)

    converged = False
    it = 0
    res = np.inf
    for it in range(1, max_gummel + 1):
        psi_old = psi.copy()
        # Poisson solve (quasi-Boltzmann coupling)
        psi, n, p = solve_poisson_newton(mesh, psi, n, p, bc, T, max_iter=20, tol=1e-7)
        # Continuity for n (fix p, ψ)
        n = solve_continuity_n_full(mesh, psi, p, G_profile, bc, T, n_guess=n,
                                     max_picard=8, tol=5e-4)
        # Continuity for p (fix n, ψ)
        p = solve_continuity_p_full(mesh, psi, n, G_profile, bc, T, p_guess=p,
                                     max_picard=8, tol=5e-4)
        # Convergence
        res = np.max(np.abs(psi - psi_old))
        if verbose:
            print(f"  Gummel {it}: Δψ_max = {res:.3e}")
        if res < tol_psi:
            converged = True
            break

    # Current
    J_n, J_p = sg_currents(mesh, psi, n, p, T)
    J_total_edges = J_n + J_p        # ideally constant across edges
    J_total = float(np.mean(J_total_edges))

    # Quasi-Fermi levels (for reporting): n = Nc·exp((E_Fn - E_c)/kT)
    Ec = -psi - mesh.chi
    Ev = Ec - mesh.Eg
    E_Fn = Ec + V_T * np.log(np.maximum(n / mesh.Nc, 1e-30))
    E_Fp = Ev - V_T * np.log(np.maximum(p / mesh.Nv, 1e-30))

    R = net_recombination(n, p, ni, mesh.tau_n, mesh.tau_p,
                           mesh.B_rad, mesh.Cn, mesh.Cp)

    return DDResult(
        x=mesh.x, psi=psi, n=n, p=p,
        Ec=Ec, Ev=Ev, E_Fn=E_Fn, E_Fp=E_Fp,
        E_field=-np.gradient(psi, mesh.x),
        J_n=J_n, J_p=J_p, J_total=J_total,
        R=R, G=G_profile, V_applied=V_applied,
        converged=converged, gummel_iter=it, residual=float(res),
    )


# ═════════════════════════════════════════════════════════════════════════════
# J-V SWEEP
# ═════════════════════════════════════════════════════════════════════════════
def jv_sweep(mesh, G_profile, mat_left, mat_right,
             V_min=0.0, V_max=None, N_V=25, T=300.0, verbose=False,
             Rs=0.0, Rsh=1e12):
    """Compute J(V) curve by sweeping voltage. Uses the previous solution as
    a warm-start for faster, more robust convergence near Voc.

    Rs   : series resistance   [Ohm·cm²]   (default 0 = ideal)
    Rsh  : shunt resistance    [Ohm·cm²]   (default 1e12 = ideal)

    Parasitic resistances are applied AFTER the DD solve using the standard
    post-processing:
        V_applied_effective = V_external - J · Rs
        J_corrected         = J_DD + V_external / Rsh
    (The shunt current is added to the DD current; Rs shifts the voltage.)
    The DD result itself does not include Rs/Rsh (an ideal device). This
    post-processing is what SCAPS applies for external parasitic R.

    Returns (voltages_external, J_A_cm2, converged_flags).
    """
    if V_max is None:
        V_max = min(mat_left.Eg, mat_right.Eg) * 0.95
    voltages = np.linspace(V_min, V_max, N_V)
    J_values = np.zeros(N_V)
    converged_flags = np.zeros(N_V, dtype=bool)

    prev_state = None
    for i, V in enumerate(voltages):
        r = solve_dd(mesh, G_profile, V, mat_left, mat_right, T,
                     max_gummel=40 if i == 0 else 30,
                     warm_start=prev_state, verbose=verbose)
        J_values[i] = r.J_total
        converged_flags[i] = r.converged
        if r.converged:
            prev_state = (r.psi, r.n, r.p)
        elif abs(r.J_total) > 10.0:
            prev_state = None
            J_values[i] = np.nan

    # Apply parasitic resistance post-processing (SCAPS-style):
    # Treat J_values as the ideal diode current at the internal junction voltage.
    # If Rs > 0, the external voltage sweep we asked for is V_ext, but the
    # internal voltage was V_DD = V_ext. Since we swept at V_ext values and
    # want V_ext = V_DD + J·Rs, we remap: V_ext = voltages + J·Rs/1000
    # (Rs in Ohm·cm², J in mA/cm² -> J*Rs/1000 in V).
    # For Rsh, shunt current J_sh = V_ext / Rsh gets added.
    if Rs > 0 or Rsh < 1e10:
        J_mA = J_values * 1000.0
        V_ext = voltages + J_mA * Rs / 1000.0
        J_sh  = V_ext / max(Rsh, 1e-30)
        # For a photogenerating cell, J_DD < 0; shunt carries current in the
        # same (photovoltaic) direction as the ideal diode at V>0. We add J_sh
        # to the (signed) J_DD current: J_corrected = J_DD + V_ext/Rsh
        J_corrected_mA = J_mA + J_sh * 1000.0
        return V_ext, J_corrected_mA / 1000.0, converged_flags
    return voltages, J_values, converged_flags


def extract_device_metrics(voltages, currents_A_cm2, P_in_mW_cm2=100.0,
                           converged_flags=None):
    """From a J-V curve, extract Jsc, Voc, FF, PCE, Vmpp, Jmpp.
    Drops non-converged / NaN points from analysis. Under this solver's sign
    convention, photogenerated J_total is NEGATIVE at V=0 (Jsc = |J(0)|)."""
    V = np.asarray(voltages, dtype=float)
    J = np.asarray(currents_A_cm2, dtype=float) * 1000.0   # mA/cm^2
    if converged_flags is None:
        good = np.isfinite(J) & (np.abs(J) < 1e4)
    else:
        good = np.asarray(converged_flags) & np.isfinite(J) & (np.abs(J) < 1e4)
    if good.sum() < 3:
        return {"Jsc": 0, "Voc": 0, "FF": 0, "PCE": 0, "Vmpp": 0, "Jmpp": 0,
                "Pmax": 0, "voltages": V, "currents_mA": J, "n_good": int(good.sum())}
    V_g = V[good]; J_g = J[good]
    # Jsc: |J| at V=0 (or closest voltage to 0)
    idx0 = int(np.argmin(np.abs(V_g)))
    Jsc = abs(J_g[idx0])
    # Voc: interpolate where J crosses 0
    Voc = 0.0
    for i in range(1, len(V_g)):
        if J_g[i-1] * J_g[i] <= 0 and J_g[i-1] != J_g[i]:
            t = -J_g[i-1] / (J_g[i] - J_g[i-1])
            Voc = V_g[i-1] + t * (V_g[i] - V_g[i-1])
            break
    if Voc == 0.0:
        Voc = V_g[-1]
    # Power output: for V in [0, Voc], P = V · (-J) (because J < 0 for photocurrent)
    in_range = (V_g >= 0) & (V_g <= Voc)
    if not np.any(in_range):
        return {"Jsc": Jsc, "Voc": Voc, "FF": 0, "PCE": 0, "Vmpp": 0, "Jmpp": 0,
                "Pmax": 0, "voltages": V, "currents_mA": J, "n_good": int(good.sum())}
    P = np.where(in_range, V_g * (-J_g), 0.0)
    idx = int(np.argmax(P))
    Pmax = float(P[idx])
    Vmpp = float(V_g[idx])
    Jmpp = abs(J_g[idx])
    FF = Pmax / (Jsc * Voc + 1e-30) if Jsc > 0 and Voc > 0 else 0.0
    FF = min(max(FF, 0.0), 0.92)
    PCE = max(Pmax, 0.0) / P_in_mW_cm2 * 100.0
    return {
        "Jsc": Jsc, "Voc": Voc, "FF": FF, "PCE": PCE,
        "Vmpp": Vmpp, "Jmpp": Jmpp, "Pmax": Pmax,
        "voltages": V, "currents_mA": J, "n_good": int(good.sum()),
    }
