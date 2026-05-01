"""
tests/test_app_imports.py
==========================
Smoke test that every module can be imported.

This catches missing deps, broken imports, and typos that would crash
`streamlit run app.py` at startup.
"""


def test_physics_modules_import():
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device    import simulate_iv_curve, fast_simulate, solve_drift_diffusion
    from physics.dd_solver import DDResult, DeviceMesh, build_mesh, jv_sweep
    from physics.optics    import compute_tmm_generation, transfer_matrix
    from physics.spectrum  import photon_flux, sq_jsc
    from physics._astm_g173_data import WAVELENGTHS_NM

    assert len(HTL_DB) > 0
    assert len(PEROVSKITE_DB) > 0


def test_ai_modules_import():
    from ai.ml_models   import RandomForestRegressor, compare_all_models, ANNRegressor
    from ai.optimizer   import bayesian_optimization
    from ai.pinn_real   import PerovskitePINN, DeviceSpec, train_pinn, poisson_residual
    from ai.pinn_pde    import JVSurrogate
    from ai.pinn_poisson import evaluate_pinn


def test_utils_modules_import():
    from utils.benchmark              import run_full_benchmark, BenchmarkResult
    from utils.experimental_benchmark import run_experimental_benchmark, ExperimentalResult
    from utils.helpers                import export_iv_csv, generate_html_report


def test_pinn_module_docstring_has_key_concepts():
    """Basic sanity check: the PINN module documents the engineering choices
    that make it a real PINN and not an MLP."""
    import ai.pinn_real as p
    doc = (p.__doc__ or "").lower()
    for concept in ("poisson", "continuity", "autograd", "log(n)", "fourier"):
        assert concept in doc, f"PINN docstring missing: {concept}"


def test_materials_json_loads():
    """The database JSON must parse and have the expected buckets."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    with open(root / "data" / "materials_database.json") as f:
        db = json.load(f)
    for bucket in ("htls", "absorbers", "etls", "_references"):
        assert bucket in db
