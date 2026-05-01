"""
tests/test_physics.py
======================
Tests that enforce fundamental physics the DD solver must obey regardless
of material choice.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_thermal_voltage_at_300K():
    from ai.pinn_real import VT_300K
    assert abs(VT_300K - 0.02585) < 0.001


def test_intrinsic_density_formula():
    from ai.pinn_real import DeviceSpec
    dev = DeviceSpec()
    expected = np.sqrt(dev.Nc * dev.Nv) * np.exp(-dev.Eg / (2 * dev.Vt))
    assert abs(dev.ni - expected) / expected < 1e-6


def test_permittivity_units():
    from ai.pinn_real import DeviceSpec, EPS0
    dev = DeviceSpec(eps_r=6.5)
    assert abs(dev.eps - 6.5 * EPS0) < 1e-20


def test_pinn_forward_pass():
    import torch
    from ai.pinn_real import PerovskitePINN
    torch.manual_seed(0)
    model = PerovskitePINN()
    x = torch.rand(50, 1)
    psi, log_n, log_p = model(x)
    assert psi.shape    == (50, 1)
    assert log_n.shape  == (50, 1)
    assert log_p.shape  == (50, 1)
    assert torch.isfinite(psi).all()
    assert torch.isfinite(log_n).all()
    assert torch.isfinite(log_p).all()


def test_fourier_features_shape():
    import torch
    from ai.pinn_real import FourierFeatures
    ff = FourierFeatures(in_dim=1, mapping_size=32, scale=8.0)
    x = torch.rand(10, 1)
    z = ff(x)
    assert z.shape == (10, 64)


def test_poisson_residual_finite():
    import torch
    from ai.pinn_real import PerovskitePINN, DeviceSpec, poisson_residual
    torch.manual_seed(0)
    model = PerovskitePINN()
    dev = DeviceSpec()
    x = torch.rand(16, 1)
    r = poisson_residual(model, x, dev, dev.L_total)
    assert torch.isfinite(r).all()
    assert r.shape == (16, 1)


def test_fast_sim_returns_reasonable_pce():
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    r = fast_simulate(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
        d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e14, T=300,
    )
    assert 5 < r["PCE"] < 35
    assert 0.5 < r["Voc"] < 1.5
    assert 5 < r["Jsc"] < 30
    assert 0.3 < r["FF"] < 0.95


def test_fast_sim_produces_valid_curve():
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    r = fast_simulate(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
        d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e14, T=300,
    )
    V = np.array(r["voltages"])
    J = np.array(r["currents"])
    assert len(V) == len(J) > 0
    assert r["Jsc"] > 0
