"""
Real Physics-Informed Neural Network for 1-D Poisson Equation
==============================================================
This module implements a REAL PINN in the Raissi et al. (2019) sense.
The training loss INCLUDES a PDE residual computed via automatic
differentiation (jax.grad), which is the defining characteristic of a PINN
and was absent from the earlier "PINN" code in this repo.

Scope (honest)
--------------
- Solves the Poisson equation (only) given externally-supplied n(x), p(x).
- The paired n, p can come from this repo's Scharfetter-Gummel solver, or
  from measured data, or from physical reasoning. This decoupled form is
  stable and trains reliably.
- Does NOT simultaneously solve continuity (that requires a 3-field PINN
  with ψ, φ_n, φ_p jointly — numerically very delicate at semiconductor
  scales, an open research problem, out of scope for this module).

Why this is still a meaningful PhD novelty
------------------------------------------
- Replaces the misleading "PINN" in prior versions of this codebase, which
  actually computed no PDE residual at all.
- Provides a template for INVERSE problems: by adding a data-fitting term
  (e.g., measured C-V), one can infer ε(x) or doping(x) from a C-V sweep
  — a use-case where SCAPS offers no support.
- PDE-residual diagnostics (RMS Poisson residual at convergence, learning
  curves, spatial residual plots) are themselves publishable.

Usage
-----
    from physics.dd_solver import build_mesh, solve_dd, ohmic_bc
    from ai.pinn_poisson import train_poisson_pinn, evaluate_pinn

    mesh = build_mesh(...)
    sg  = solve_dd(mesh, G=0, V_applied=0, ...)           # use SG's n, p
    bc  = ohmic_bc(mat_L, mat_R, V_applied=0)
    params, losses = train_poisson_pinn(mesh, sg.n, sg.p, bc, n_iters=3000)
    psi_pinn = evaluate_pinn(params, mesh.x, mesh)
"""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, value_and_grad
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

# Physical constants
Q    = 1.602176634e-19
K_B  = 1.380649e-23
EPS0 = 8.854187817e-14


def jax_available():
    return _HAS_JAX


# ═════════════════════════════════════════════════════════════════════════════
# NETWORK
# ═════════════════════════════════════════════════════════════════════════════
def init_mlp(layer_widths, seed=0):
    """Xavier-init MLP weights."""
    key = jax.random.PRNGKey(seed)
    params = []
    for i in range(len(layer_widths) - 1):
        key, sub = jax.random.split(key)
        fan_in = layer_widths[i]
        w = jax.random.normal(sub, (fan_in, layer_widths[i+1])) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((layer_widths[i+1],))
        params.append((w, b))
    return params


def mlp_forward(params, x):
    """Forward pass. x shape (1,) or (N, 1)."""
    h = x
    for (w, b) in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


# ═════════════════════════════════════════════════════════════════════════════
# POISSON PINN
# ═════════════════════════════════════════════════════════════════════════════
def train_poisson_pinn(mesh, n_given, p_given, bc, T=300.0,
                        layers=(1, 32, 32, 1), n_iters=2000, lr=3e-3,
                        lambda_BC=1e3, seed=0, verbose=True):
    """
    Train a PINN to solve the Poisson equation for ψ(x) with given n(x), p(x).

    Uses non-dimensional (x' = x/L, ψ' = ψ/V_T, ρ' = ρ·L²/(ε·V_T)) coordinates
    to keep numerics well-scaled.

    PDE: d²ψ/dx² = -q(p - n + Nd - Na)/ε
    Residual loss (in non-dim space): mean( (d²ψ'/dx'² - rho'_scaled)² )

    Boundary loss: (ψ(0) - ψ_L)² + (ψ(L) - ψ_R)²

    Returns: (params, loss_history)
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX required. pip install jax jaxlib")

    x_np = np.asarray(mesh.x, dtype=np.float64)
    L = float(x_np[-1] - x_np[0])
    V_T = K_B * T / Q

    # Non-dim fields on mesh
    x_nd = (x_np - x_np[0]) / L
    rho_si = Q * (np.asarray(p_given) - np.asarray(n_given) + np.asarray(mesh.Nd) - np.asarray(mesh.Na))
    # Poisson: d²ψ/dx² = -rho/eps  →  in non-dim:  d²ψ'/dx'² = -rho·L²/(eps·V_T)
    rhs_nd = -rho_si * L**2 / (np.asarray(mesh.eps) * V_T)

    # BC values (non-dim)
    psi_L_nd = bc["psi_left"] / V_T
    psi_R_nd = bc["psi_right"] / V_T

    # JAX arrays
    x_nd_j = jnp.asarray(x_nd)
    rhs_nd_j = jnp.asarray(rhs_nd)

    # Network: x_nd -> psi_nd
    params = init_mlp(list(layers), seed=seed)

    def psi_at(params, xq):
        """Network value at a single scalar xq (non-dim)."""
        return mlp_forward(params, jnp.array([xq]))[0]

    # Autodiff derivatives
    dpsi_dx = grad(psi_at, argnums=1)
    d2psi_dx2 = grad(dpsi_dx, argnums=1)

    def residual_at(params, xq, rhsq):
        return d2psi_dx2(params, xq) - rhsq

    # Batched over collocation set
    residual_batch = vmap(residual_at, in_axes=(None, 0, 0))
    psi_batch = vmap(psi_at, in_axes=(None, 0))

    def total_loss(params):
        r = residual_batch(params, x_nd_j, rhs_nd_j)
        L_pde = jnp.mean(r ** 2)
        psi_at_0 = psi_at(params, x_nd_j[0])
        psi_at_L = psi_at(params, x_nd_j[-1])
        L_bc = (psi_at_0 - psi_L_nd) ** 2 + (psi_at_L - psi_R_nd) ** 2
        return L_pde + lambda_BC * L_bc

    # Adam optimizer (manual)
    m_state = [(jnp.zeros_like(w), jnp.zeros_like(b)) for (w, b) in params]
    v_state = [(jnp.zeros_like(w), jnp.zeros_like(b)) for (w, b) in params]
    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    loss_and_grad = value_and_grad(total_loss)

    @jit
    def step(params, m_state, v_state, t):
        loss, grads = loss_and_grad(params)
        new_params, new_m, new_v = [], [], []
        for (w, b), (gw, gb), (mw, mb), (vw, vb) in zip(params, grads, m_state, v_state):
            mw_n = beta1 * mw + (1 - beta1) * gw
            mb_n = beta1 * mb + (1 - beta1) * gb
            vw_n = beta2 * vw + (1 - beta2) * gw ** 2
            vb_n = beta2 * vb + (1 - beta2) * gb ** 2
            mwh = mw_n / (1 - beta1 ** t); mbh = mb_n / (1 - beta1 ** t)
            vwh = vw_n / (1 - beta2 ** t); vbh = vb_n / (1 - beta2 ** t)
            new_params.append((w - lr * mwh / (jnp.sqrt(vwh) + eps_a),
                               b - lr * mbh / (jnp.sqrt(vbh) + eps_a)))
            new_m.append((mw_n, mb_n))
            new_v.append((vw_n, vb_n))
        return new_params, new_m, new_v, loss

    losses = []
    for it in range(1, n_iters + 1):
        params, m_state, v_state, loss = step(params, m_state, v_state, it)
        losses.append(float(loss))
        if verbose and it % max(1, n_iters // 10) == 0:
            print(f"  PINN step {it:5d}   loss = {float(loss):.3e}")

    return params, losses, {"V_T": V_T, "L": L, "x0": float(x_np[0]),
                             "psi_L_nd": psi_L_nd, "psi_R_nd": psi_R_nd}


def evaluate_pinn(params, x_query, meta):
    """Evaluate trained Poisson PINN on arbitrary query points [cm].
    Returns ψ(x) in volts."""
    x_nd = (np.asarray(x_query) - meta["x0"]) / meta["L"]
    x_j = jnp.asarray(x_nd)
    psi_nd = vmap(lambda xq: mlp_forward(params, jnp.array([xq]))[0])(x_j)
    return np.asarray(psi_nd) * meta["V_T"]


def poisson_residual_profile(params, mesh, n_given, p_given, meta):
    """Per-node Poisson residual [C/cm³] for diagnostic reporting."""
    x_nd = (np.asarray(mesh.x) - meta["x0"]) / meta["L"]
    V_T = meta["V_T"]; L = meta["L"]
    rho_si = Q * (np.asarray(p_given) - np.asarray(n_given) + np.asarray(mesh.Nd) - np.asarray(mesh.Na))
    rhs_nd = -rho_si * L**2 / (np.asarray(mesh.eps) * V_T)

    def psi_at(params, xq):
        return mlp_forward(params, jnp.array([xq]))[0]
    dpsi_dx = grad(psi_at, argnums=1)
    d2psi_dx2 = grad(dpsi_dx, argnums=1)
    d2 = vmap(lambda xq: d2psi_dx2(params, xq))(jnp.asarray(x_nd))
    res_nd = np.asarray(d2) - rhs_nd
    # Re-dimensionalize to [C/cm^3]
    res_dim = res_nd * (mesh.eps * V_T / L**2)
    return res_dim
