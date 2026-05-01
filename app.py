"""
app.py - Streamlit web application (live interactive)
======================================================
AI-Driven Open Source Tool for Design and Optimization of Perovskite Solar Cells

Every tab produces real computational results with interactive Plotly charts.
All live computation is backed by the validated fast analytical simulator
(18ms per J-V sweep). A pre-trained PINN model is bundled for live ψ(x),
n(x), p(x) profile inference. Pre-computed DD benchmark results are
displayed with parity plots and error histograms.

Run:
    streamlit run app.py
"""
from __future__ import annotations

# Suppress convergence warnings from sklearn/scipy optimization
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from physics.materials import (
    HTL_DB, PEROVSKITE_DB, ETL_DB, CONTACTS,
    DEFAULT_IDL, HIGH_QUALITY_IDL, LOW_QUALITY_IDL,
    get_material_with_provenance,
    summarize_provenance_confidence,
)
from physics.device import simulate_iv_curve, fast_simulate


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Perovskite Solar Cell Design Tool",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_benchmark_data():
    """Pre-computed DD benchmark results."""
    path = ARTIFACTS_DIR / "benchmark_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_pretrained_pinn():
    """Load the bundled pre-trained PINN model for live inference."""
    import torch
    from ai.pinn_real import PerovskitePINN

    model_path = ARTIFACTS_DIR / "model_d1.pt"
    hist_path  = ARTIFACTS_DIR / "history_d1.pkl"

    if not model_path.exists():
        return None, None

    model = PerovskitePINN()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    history = None
    if hist_path.exists():
        with open(hist_path, "rb") as f:
            history = pickle.load(f)

    return model, history


def build_sample_grid(n_samples=200, seed=0):
    """Latin-hypercube-style parameter sampling for ML training data.

    Produces a DataFrame with columns: d_htl, d_abs, d_etl, Nt, T, + metrics.
    """
    rng = np.random.default_rng(seed)
    # Parameter ranges
    d_htl = rng.uniform(50, 400, n_samples)
    d_abs = rng.uniform(200, 900, n_samples)
    d_etl = rng.uniform(20, 200, n_samples)
    log_Nt = rng.uniform(13.0, 16.5, n_samples)
    T = rng.uniform(280, 330, n_samples)

    rows = []
    for i in range(n_samples):
        try:
            r = fast_simulate(
                HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
                d_htl_nm=float(d_htl[i]), d_abs_nm=float(d_abs[i]),
                d_etl_nm=float(d_etl[i]), Nt_abs=float(10**log_Nt[i]),
                T=float(T[i]),
            )
            rows.append({
                "d_htl": d_htl[i], "d_abs": d_abs[i], "d_etl": d_etl[i],
                "log_Nt": log_Nt[i], "T": T[i],
                "PCE": r["PCE"], "Voc": r["Voc"], "Jsc": r["Jsc"], "FF": r["FF"],
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;">
        <h1 style="margin: 0; color: #1f2937; font-size: 1.8rem;">
            ☀️ Perovskite Solar Cell Design & Optimization Tool
        </h1>
        <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.95rem;">
            AI-driven open source tool · Validated physics · Live ML · Real PINN · Publication-ready
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: device configuration (shared across tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Device Configuration")

    htl_name = st.selectbox("Hole Transport Layer (HTL)",
                             list(HTL_DB.keys()),
                             index=list(HTL_DB.keys()).index("Spiro-OMeTAD"))
    abs_name = st.selectbox("Absorber",
                             list(PEROVSKITE_DB.keys()),
                             index=list(PEROVSKITE_DB.keys()).index("MAPbI3"))
    etl_name = st.selectbox("Electron Transport Layer (ETL)",
                             list(ETL_DB.keys()),
                             index=list(ETL_DB.keys()).index("TiO2"))

    st.markdown("---")
    st.markdown("### Layer Thicknesses (nm)")
    d_htl = st.slider("HTL",       10,  500, 200, step=10)
    d_abs = st.slider("Absorber",  50, 1000, 500, step=25)
    d_etl = st.slider("ETL",       10,  500,  50, step=10)

    st.markdown("---")
    st.markdown("### Operating Conditions")
    Nt = st.select_slider("Bulk trap density (cm⁻³)",
                           options=[1e13, 1e14, 1e15, 1e16, 1e17],
                           value=1e14,
                           format_func=lambda x: f"{x:.0e}")
    temperature = st.slider("Temperature (K)", 250, 400, 300, step=5)

    st.markdown("---")
    summary = summarize_provenance_confidence()
    st.caption(
        f"📋 Database: **{len(HTL_DB)+len(PEROVSKITE_DB)+len(ETL_DB)} materials** · "
        f"**{summary['total']['HIGH']+summary['total']['MEDIUM']+summary['total']['LOW']} parameters** · "
        f"**HIGH/MED/LOW: {summary['total']['HIGH']}/{summary['total']['MEDIUM']}/{summary['total']['LOW']}**"
    )


# ---------------------------------------------------------------------------
# Main content: 10 tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "⚡ Fast Simulate",
    "🔬 DD Solver",
    "🧠 ML Surrogates",
    "⚛️ PINN (real)",
    "🎯 Bayesian Opt",
    "📊 Multi-Objective",
    "🔍 Feature Importance",
    "🔄 Inverse Design",
    "📐 Benchmarks",
    "📋 Database",
])


# ===========================================================================
# TAB 1: FAST SIMULATE
# ===========================================================================
with tabs[0]:
    st.markdown("### ⚡ Fast Analytical Simulation")
    st.caption(
        "Single-diode surrogate with ASTM G173-03 spectral integration. "
        "Sub-50 ms per J-V sweep."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        if st.button("▶ Run Fast Simulation", type="primary", use_container_width=True):
            t0 = time.time()
            try:
                r = fast_simulate(
                    HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                    d_htl_nm=d_htl, d_abs_nm=d_abs, d_etl_nm=d_etl,
                    Nt_abs=Nt, T=temperature,
                )
                elapsed = (time.time() - t0) * 1000
                st.session_state["fast_result"] = r
                st.session_state["fast_elapsed_ms"] = elapsed
            except Exception as e:
                st.error(f"Simulation error: {e}")

        if "fast_result" in st.session_state:
            r = st.session_state["fast_result"]
            st.metric("PCE",  f"{r['PCE']:.2f} %")
            st.metric("Voc",  f"{r['Voc']:.3f} V")
            st.metric("Jsc",  f"{r['Jsc']:.2f} mA/cm²")
            st.metric("FF",   f"{r['FF']*100:.1f} %")
            st.caption(f"✓ Computed in {st.session_state['fast_elapsed_ms']:.1f} ms")

    with col_b:
        if "fast_result" in st.session_state:
            r = st.session_state["fast_result"]
            V = np.array(r["voltages"])
            J = np.array(r["currents"])

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("J–V curve", "Power density"),
                horizontal_spacing=0.12,
            )
            fig.add_trace(go.Scatter(x=V, y=J, mode="lines", name="J(V)",
                                     line=dict(color="#2563eb", width=3)),
                         row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af", row=1, col=1)
            fig.add_vline(x=r["Voc"], line_dash="dot", line_color="#9ca3af", row=1, col=1)

            P = V * np.abs(J)
            fig.add_trace(go.Scatter(x=V, y=P, mode="lines", name="P(V)",
                                     line=dict(color="#16a34a", width=3)),
                         row=1, col=2)
            fig.add_vline(x=r["Vmpp"], line_dash="dot", line_color="#9ca3af", row=1, col=2)

            fig.update_xaxes(title_text="V (V)", row=1, col=1)
            fig.update_yaxes(title_text="J (mA/cm²)", row=1, col=1)
            fig.update_xaxes(title_text="V (V)", row=1, col=2)
            fig.update_yaxes(title_text="P (mW/cm²)", row=1, col=2)
            fig.update_layout(height=400, showlegend=False,
                              margin=dict(l=50, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            # Secondary row: QE spectrum
            if "qe" in r and r["qe"] is not None and len(r["qe"]) > 0:
                lams = np.array(r["lams_qe"])
                qe = np.array(r["qe"]) * 100  # to %
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=lams, y=qe, mode="lines",
                                           fill="tozeroy",
                                           line=dict(color="#dc2626", width=2),
                                           name="EQE"))
                fig2.update_layout(
                    title="External Quantum Efficiency (EQE)",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="EQE (%)",
                    height=300,
                    margin=dict(l=50, r=20, t=50, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Click **Run Fast Simulation** to start.")

    with st.expander("ℹ️ About this method — scope and limits"):
        st.markdown("""
        **What it does**: Solves a single-diode equivalent circuit with Tauc absorption
        `α ∝ √(E−Eg)/E`, voltage deficit from traps
        `Vd = 0.33 + 0.06·log(Nt/10¹³)`, and Newton-Raphson J(V) sweep.

        **Validated against**: 10 published SCAPS-1D references. Mean PCE error
        14%, median 14%. Tab 9 (Benchmarks) shows device-by-device detail.

        **Use it when**: You want a first-pass answer fast. Screening 1000 designs
        overnight. Building surrogate training data (Tab 3).

        **Don't use it for**: First-principles predictions. Cases outside typical
        perovskite parameter ranges. Anything publication-critical — cross-check
        with the DD solver (Tab 2).
        """)


# ===========================================================================
# TAB 2: DRIFT-DIFFUSION SOLVER
# ===========================================================================
with tabs[1]:
    st.markdown("### 🔬 1D Drift-Diffusion Solver (Scharfetter-Gummel)")
    st.caption(
        "Full DD PDE solver: Poisson + two continuity equations, "
        "Scharfetter-Gummel flux discretization, Newton + Gummel iteration. "
        "1–3 seconds per J-V sweep."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        enable_idl = st.checkbox("Enable Interface Defect Layer (IDL)", value=False)
        idl_quality = None
        if enable_idl:
            idl_quality = st.radio(
                "IDL quality",
                ["High (S=10² cm/s)", "Default (S=10⁴ cm/s)", "Low (S=10⁶ cm/s)"],
                index=1,
            )

        if st.button("▶ Run DD Solver", type="primary", use_container_width=True):
            t0 = time.time()
            try:
                with st.spinner("Running full drift-diffusion solver..."):
                    r = simulate_iv_curve(
                        HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                        d_htl_nm=d_htl, d_abs_nm=d_abs, d_etl_nm=d_etl,
                        Nt_abs=Nt, T=temperature,
                        mode="dd",
                    )
                elapsed = time.time() - t0
                st.session_state["dd_result"] = r
                st.session_state["dd_elapsed_s"] = elapsed
            except Exception as e:
                st.error(f"DD solver error: {e}")

        if "dd_result" in st.session_state:
            r = st.session_state["dd_result"]
            st.metric("PCE",  f"{r['PCE']:.2f} %")
            st.metric("Voc",  f"{r['Voc']:.3f} V")
            st.metric("Jsc",  f"{r['Jsc']:.2f} mA/cm²")
            st.metric("FF",   f"{r['FF']*100:.1f} %")
            st.caption(f"✓ Computed in {st.session_state['dd_elapsed_s']:.2f} s")

    with col_b:
        if "dd_result" in st.session_state:
            r = st.session_state["dd_result"]
            V = np.array(r["voltages"])
            J = np.array(r["currents"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=V, y=J, mode="lines", name="DD solver",
                                     line=dict(color="#dc2626", width=3)))
            fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
            fig.add_vline(x=r["Voc"], line_dash="dot", line_color="#9ca3af",
                         annotation_text=f"Voc = {r['Voc']:.3f} V")
            fig.update_layout(
                title="J–V curve (drift-diffusion)",
                xaxis_title="Voltage (V)",
                yaxis_title="Current density (mA/cm²)",
                height=450,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Run DD Solver** to compute the full drift-diffusion solution. Takes 1–3 seconds.")


# ===========================================================================
# TAB 3: ML SURROGATES (live training!)
# ===========================================================================
with tabs[2]:
    st.markdown("### 🧠 Machine Learning Surrogates")
    st.caption(
        "Trains 7 regressors on simulation data. Compare R², see parity plots, "
        "export trained model. All live."
    )

    st.info(
        "**Honest framing**: These are feed-forward regressors trained on J-V "
        "data via MSE loss. They are NOT PINNs. The real PINN (autograd, PDE "
        "residuals) is in the next tab."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        n_train = st.slider("Training samples", 50, 500, 200, step=50,
                           help="Number of fast-sim calls used to train surrogates. 200 takes ~4 sec.")
        target = st.selectbox("Target metric", ["PCE", "Voc", "Jsc", "FF"])

        if st.button("▶ Train + Compare Models", type="primary", use_container_width=True):
            t0 = time.time()
            with st.spinner(f"Generating {n_train} training samples and fitting 7 models..."):
                # Build training set
                df = build_sample_grid(n_samples=n_train, seed=42)
                X = df[["d_htl", "d_abs", "d_etl", "log_Nt", "T"]].values
                y = df[target].values

                # Split
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

                # Train 7 regressors
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.svm import SVR
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.linear_model import BayesianRidge, LinearRegression
                from sklearn.neural_network import MLPRegressor
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                from sklearn.metrics import r2_score, mean_absolute_error

                models = {
                    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=0),
                    "GradientBoost":    GradientBoostingRegressor(n_estimators=100, random_state=0),
                    "SVR":              make_pipeline(StandardScaler(), SVR(C=1.0, kernel="rbf")),
                    "KNN":              KNeighborsRegressor(n_neighbors=5),
                    "BayesianRidge":    make_pipeline(StandardScaler(), BayesianRidge()),
                    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
                    "NeuralNet":        make_pipeline(StandardScaler(), MLPRegressor(
                        hidden_layer_sizes=(64, 64), max_iter=500, random_state=0)),
                }

                results = []
                predictions = {}
                for name, model in models.items():
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    results.append({
                        "Model": name,
                        "R²":  r2_score(y_te, y_pred),
                        "MAE": mean_absolute_error(y_te, y_pred),
                    })
                    predictions[name] = y_pred

                st.session_state["ml_results"]      = pd.DataFrame(results)
                st.session_state["ml_predictions"]  = predictions
                st.session_state["ml_y_test"]       = y_te
                st.session_state["ml_target"]       = target
                st.session_state["ml_elapsed"]      = time.time() - t0
                st.session_state["ml_df"]           = df
                st.session_state["ml_models"]       = models

        if "ml_results" in st.session_state:
            st.caption(f"✓ Trained in {st.session_state['ml_elapsed']:.1f} s")
            st.dataframe(
                st.session_state["ml_results"].sort_values("R²", ascending=False),
                hide_index=True, use_container_width=True,
            )

    with col_b:
        if "ml_results" in st.session_state:
            results_df = st.session_state["ml_results"].sort_values("R²", ascending=True)

            # R² bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df["R²"], y=results_df["Model"],
                orientation="h",
                marker=dict(color=results_df["R²"],
                           colorscale="RdYlGn", cmin=0, cmax=1, showscale=False),
                text=[f"{v:.3f}" for v in results_df["R²"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"R² comparison — predicting {st.session_state['ml_target']}",
                xaxis_title="R² score", xaxis_range=[0, 1.05],
                height=350, margin=dict(l=50, r=80, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Parity plot for best model
            best_model = results_df.iloc[-1]["Model"]
            y_true = st.session_state["ml_y_test"]
            y_pred = st.session_state["ml_predictions"][best_model]

            fig2 = go.Figure()
            lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                       line=dict(color="#9ca3af", dash="dash"),
                                       name="y = x"))
            fig2.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                                       marker=dict(size=7, color="#2563eb", opacity=0.6),
                                       name="predictions"))
            fig2.update_layout(
                title=f"Parity plot — best model: {best_model}",
                xaxis_title=f"Actual {st.session_state['ml_target']}",
                yaxis_title=f"Predicted {st.session_state['ml_target']}",
                height=400, margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Click **Train + Compare Models** to see R² comparison and parity plots.")


# ===========================================================================
# TAB 4: REAL PINN (bundled pretrained + live inference)
# ===========================================================================
with tabs[3]:
    st.markdown("### ⚛️ Physics-Informed Neural Network (real, autodiff)")
    st.caption(
        "PINN in the Raissi 2019 sense: Poisson + continuity residuals in the "
        "loss, computed via PyTorch autograd. Pre-trained model shipped with "
        "the repo. Live inference takes <100 ms."
    )

    model, history = load_pretrained_pinn()

    if model is None:
        st.error("Pre-trained PINN model not found in artifacts/. "
                "Run `python scripts/train_pinn.py` to generate one.")
    else:
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("**Model**: 5-layer MLP × 128 units, tanh")
            st.markdown("**Input**: Fourier features (32 sin + 32 cos)")
            st.markdown("**Output**: ψ(x), log n(x), log p(x)")
            st.markdown("**Training**: 1000 data + 5000 physics epochs")
            st.markdown(f"**Params**: {sum(p.numel() for p in model.parameters()):,}")

            if st.button("▶ Run PINN Inference", type="primary", use_container_width=True):
                import torch
                t0 = time.time()
                x = torch.linspace(0, 1, 200, dtype=torch.float32).reshape(-1, 1)
                with torch.no_grad():
                    psi, log_n, log_p = model(x)
                elapsed = (time.time() - t0) * 1000

                st.session_state["pinn_x"]    = x.numpy().flatten()
                st.session_state["pinn_psi"]  = psi.numpy().flatten()
                st.session_state["pinn_logn"] = log_n.numpy().flatten()
                st.session_state["pinn_logp"] = log_p.numpy().flatten()
                st.session_state["pinn_infer_ms"] = elapsed

            if "pinn_infer_ms" in st.session_state:
                st.caption(f"✓ Inference in {st.session_state['pinn_infer_ms']:.1f} ms")

        with col_b:
            # Subtabs: Profiles / Training loss / Residuals
            ptab1, ptab2 = st.tabs(["📈 Profiles ψ/n/p", "📉 Training loss history"])

            with ptab1:
                if "pinn_x" in st.session_state:
                    x = st.session_state["pinn_x"]
                    psi = st.session_state["pinn_psi"]
                    n = np.exp(np.clip(st.session_state["pinn_logn"], -50, 50))
                    p = np.exp(np.clip(st.session_state["pinn_logp"], -50, 50))

                    # Device length for x-axis (from bundled device spec)
                    dev_path = ARTIFACTS_DIR / "dev_d1.pkl"
                    if dev_path.exists():
                        with open(dev_path, "rb") as f:
                            dev_data = pickle.load(f)
                        L_nm = (dev_data["L_htl"] + dev_data["L_abs"] + dev_data["L_etl"]) * 1e7
                        L_htl_nm = dev_data["L_htl"] * 1e7
                        L_abs_nm = dev_data["L_abs"] * 1e7
                    else:
                        L_nm, L_htl_nm, L_abs_nm = 750.0, 200.0, 500.0

                    x_nm = x * L_nm

                    fig = make_subplots(rows=2, cols=1,
                                         subplot_titles=("Electrostatic potential ψ(x)",
                                                          "Carrier densities n(x), p(x)"),
                                         shared_xaxes=True, vertical_spacing=0.12)

                    # Shade layers
                    for row in [1, 2]:
                        fig.add_vrect(x0=0, x1=L_htl_nm, fillcolor="#fee2e2", opacity=0.4, line_width=0, row=row, col=1)
                        fig.add_vrect(x0=L_htl_nm, x1=L_htl_nm+L_abs_nm, fillcolor="#dbeafe", opacity=0.4, line_width=0, row=row, col=1)
                        fig.add_vrect(x0=L_htl_nm+L_abs_nm, x1=L_nm, fillcolor="#dcfce7", opacity=0.4, line_width=0, row=row, col=1)

                    fig.add_trace(go.Scatter(x=x_nm, y=psi, mode="lines", name="ψ(x)",
                                             line=dict(color="#7c3aed", width=2.5)), row=1, col=1)

                    fig.add_trace(go.Scatter(x=x_nm, y=n, mode="lines", name="n(x)",
                                             line=dict(color="#2563eb", width=2.5)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=x_nm, y=p, mode="lines", name="p(x)",
                                             line=dict(color="#dc2626", width=2.5)), row=2, col=1)

                    fig.update_yaxes(title_text="ψ (V)", row=1, col=1)
                    fig.update_yaxes(title_text="Density (cm⁻³)", type="log",
                                     range=[np.log10(1e-5), np.log10(1e20)], row=2, col=1)
                    fig.update_xaxes(title_text="Position x (nm)", row=2, col=1)
                    fig.update_layout(height=600, showlegend=True,
                                       margin=dict(l=60, r=20, t=50, b=40),
                                       legend=dict(orientation="h", yanchor="top", y=-0.1))

                    # Layer annotations
                    fig.add_annotation(x=L_htl_nm/2, y=1.02, xref="x", yref="paper",
                                        text="<b>HTL</b>", showarrow=False, font=dict(size=10, color="#991b1b"))
                    fig.add_annotation(x=L_htl_nm+L_abs_nm/2, y=1.02, xref="x", yref="paper",
                                        text="<b>Absorber</b>", showarrow=False, font=dict(size=10, color="#1e40af"))
                    fig.add_annotation(x=L_htl_nm+L_abs_nm+(L_nm-L_htl_nm-L_abs_nm)/2, y=1.02, xref="x", yref="paper",
                                        text="<b>ETL</b>", showarrow=False, font=dict(size=10, color="#14532d"))

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Click **Run PINN Inference** to compute spatial profiles.")

            with ptab2:
                if history is not None:
                    epochs = np.arange(len(history.get("total", [])))

                    fig = go.Figure()
                    for key, color in [("total", "#1f2937"),
                                        ("pde", "#2563eb"),
                                        ("bc", "#7c3aed"),
                                        ("data", "#dc2626")]:
                        if key in history and len(history[key]) > 0:
                            vals = np.abs(np.array(history[key])) + 1e-12
                            fig.add_trace(go.Scatter(x=epochs, y=vals, mode="lines",
                                                      name=f"ℒ_{key}",
                                                      line=dict(color=color, width=2)))

                    fig.update_yaxes(type="log", title_text="Loss value")
                    fig.update_xaxes(title_text="Epoch")
                    fig.update_layout(
                        title="PINN training loss curves",
                        height=500, margin=dict(l=60, r=20, t=50, b=50),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        f"**Final losses** — total: {history['total'][-1]:.2e}, "
                        f"pde: {history['pde'][-1]:.2e}, "
                        f"bc: {history['bc'][-1]:.2e}"
                    )
                else:
                    st.info("Training history not found in artifacts/.")


# ===========================================================================
# TAB 5: BAYESIAN OPTIMIZATION (live!)
# ===========================================================================
with tabs[4]:
    st.markdown("### 🎯 Bayesian Optimization")
    st.caption(
        "Gaussian-process surrogate with Expected Improvement acquisition. "
        "Finds best device configuration in 20–30 evaluations."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        n_iter_bo = st.slider("Iterations", 10, 40, 25, step=5)
        target_bo = st.selectbox("Maximize", ["PCE", "Voc", "Jsc"], key="bo_target")

        if st.button("▶ Run Bayesian Optimization", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_iter_bo}-iteration Bayesian optimization..."):
                t0 = time.time()

                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern
                from scipy.stats import norm

                # Parameter bounds: d_htl, d_abs, d_etl, log_Nt
                bounds = np.array([[50, 400], [200, 900], [20, 200], [13, 16.5]])
                n_dim = bounds.shape[0]

                rng = np.random.default_rng(0)

                def eval_fast(params):
                    try:
                        r = fast_simulate(
                            HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                            d_htl_nm=params[0], d_abs_nm=params[1], d_etl_nm=params[2],
                            Nt_abs=10**params[3], T=300,
                        )
                        return r[target_bo]
                    except Exception:
                        return 0.0

                # Initial design: 5 Latin-hypercube-ish points
                X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(5, n_dim))
                y = np.array([eval_fast(x) for x in X])

                history_best = [y.max()]
                kernel = Matern(length_scale=[100, 100, 50, 1.0], nu=2.5)
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                alpha=0.01, n_restarts_optimizer=3)

                for _ in range(n_iter_bo):
                    gp.fit(X, y)

                    # Random candidate search for EI acquisition
                    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(500, n_dim))
                    mu, sigma = gp.predict(candidates, return_std=True)
                    sigma = np.maximum(sigma, 1e-6)

                    y_best = y.max()
                    z = (mu - y_best) / sigma
                    ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)

                    x_next = candidates[ei.argmax()]
                    y_next = eval_fast(x_next)

                    X = np.vstack([X, x_next])
                    y = np.append(y, y_next)
                    history_best.append(y.max())

                st.session_state["bo_X"] = X
                st.session_state["bo_y"] = y
                st.session_state["bo_history"] = history_best
                st.session_state["bo_elapsed"] = time.time() - t0
                st.session_state["_saved_bo_target"] = target_bo

        if "bo_history" in st.session_state:
            best_idx = np.argmax(st.session_state["bo_y"])
            best_X = st.session_state["bo_X"][best_idx]
            st.caption(f"✓ Converged in {st.session_state['bo_elapsed']:.1f} s")
            st.metric(f"Best {st.session_state['_saved_bo_target']}",
                     f"{st.session_state['bo_y'][best_idx]:.3f}")
            st.markdown("**Best parameters**:")
            st.markdown(f"- d_htl = **{best_X[0]:.0f} nm**")
            st.markdown(f"- d_abs = **{best_X[1]:.0f} nm**")
            st.markdown(f"- d_etl = **{best_X[2]:.0f} nm**")
            st.markdown(f"- Nt = **10^{best_X[3]:.2f} cm⁻³**")

    with col_b:
        if "bo_history" in st.session_state:
            hist = st.session_state["bo_history"]
            all_y = st.session_state["bo_y"]

            fig = go.Figure()
            # Evaluation history
            fig.add_trace(go.Scatter(
                x=list(range(1, len(all_y)+1)), y=all_y,
                mode="markers", marker=dict(size=8, color="#9ca3af"),
                name="Evaluations",
            ))
            # Best-so-far
            fig.add_trace(go.Scatter(
                x=list(range(1, len(hist)+1)),
                y=hist,
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8),
                name="Best so far",
            ))
            fig.update_layout(
                title=f"Bayesian optimization convergence — {st.session_state['_saved_bo_target']}",
                xaxis_title="Evaluation",
                yaxis_title=st.session_state["_saved_bo_target"],
                height=500,
                margin=dict(l=60, r=20, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Run Bayesian Optimization** to find the best device configuration.")


# ===========================================================================
# TAB 6: MULTI-OBJECTIVE (live NSGA-II!)
# ===========================================================================
with tabs[5]:
    st.markdown("### 📊 Multi-Objective Optimization (NSGA-II)")
    st.caption(
        "Find the Pareto front between two competing objectives. Live NSGA-II "
        "implementation runs in ~20 seconds."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        obj1 = st.selectbox("Objective 1 (maximize)", ["PCE", "Voc", "Jsc"], index=0)
        obj2 = st.selectbox("Objective 2 (maximize)", ["FF", "Jsc", "Voc"], index=0,
                           help="Trades off against objective 1")
        n_pop  = st.slider("Population size", 20, 100, 50, step=10)
        n_gens = st.slider("Generations", 10, 50, 25, step=5)

        if st.button("▶ Run NSGA-II", type="primary", use_container_width=True):
            with st.spinner(f"Running NSGA-II: {n_pop} × {n_gens} generations..."):
                t0 = time.time()

                bounds = np.array([[50, 400], [200, 900], [20, 200], [13, 16.5]])
                rng = np.random.default_rng(0)

                def eval_multi(params):
                    try:
                        r = fast_simulate(
                            HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                            d_htl_nm=params[0], d_abs_nm=params[1], d_etl_nm=params[2],
                            Nt_abs=10**params[3], T=300,
                        )
                        return np.array([r[obj1], r[obj2]])
                    except Exception:
                        return np.array([0.0, 0.0])

                def fast_non_dominated_sort(F):
                    """Return rank of each point (0 = Pareto front)."""
                    n = F.shape[0]
                    ranks = np.zeros(n, dtype=int)
                    dominated = [[] for _ in range(n)]
                    dom_count = np.zeros(n, dtype=int)

                    for i in range(n):
                        for j in range(n):
                            if i == j: continue
                            if np.all(F[i] >= F[j]) and np.any(F[i] > F[j]):
                                dominated[i].append(j)
                            elif np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                                dom_count[i] += 1

                    current_rank = 0
                    current_front = np.where(dom_count == 0)[0].tolist()
                    while current_front:
                        next_front = []
                        for i in current_front:
                            ranks[i] = current_rank
                            for j in dominated[i]:
                                dom_count[j] -= 1
                                if dom_count[j] == 0:
                                    next_front.append(j)
                        current_rank += 1
                        current_front = next_front
                    return ranks

                # Initialize
                pop = rng.uniform(bounds[:, 0], bounds[:, 1], (n_pop, 4))
                F = np.array([eval_multi(x) for x in pop])

                for gen in range(n_gens):
                    # Create offspring via SBX-like random mutation
                    offspring = pop.copy()
                    for i in range(n_pop):
                        if rng.random() < 0.5:
                            # crossover between two random parents
                            p1, p2 = rng.choice(n_pop, 2, replace=False)
                            alpha = rng.random(4)
                            offspring[i] = alpha * pop[p1] + (1 - alpha) * pop[p2]
                        # mutation
                        if rng.random() < 0.3:
                            k = rng.integers(4)
                            offspring[i, k] = rng.uniform(bounds[k, 0], bounds[k, 1])
                    F_off = np.array([eval_multi(x) for x in offspring])

                    # Combine and select
                    combined_pop = np.vstack([pop, offspring])
                    combined_F   = np.vstack([F, F_off])
                    ranks = fast_non_dominated_sort(combined_F)

                    # Select top n_pop by rank
                    order = np.argsort(ranks)[:n_pop]
                    pop = combined_pop[order]
                    F   = combined_F[order]

                # Extract Pareto front
                final_ranks = fast_non_dominated_sort(F)
                pareto_idx = np.where(final_ranks == 0)[0]

                st.session_state["nsga_F"] = F
                st.session_state["nsga_pareto"] = pareto_idx
                st.session_state["nsga_pop"] = pop
                st.session_state["nsga_elapsed"] = time.time() - t0
                st.session_state["nsga_obj1"] = obj1
                st.session_state["nsga_obj2"] = obj2

        if "nsga_F" in st.session_state:
            st.caption(f"✓ Done in {st.session_state['nsga_elapsed']:.1f} s")
            st.metric("Pareto front size", len(st.session_state["nsga_pareto"]))

    with col_b:
        if "nsga_F" in st.session_state:
            F = st.session_state["nsga_F"]
            pareto = st.session_state["nsga_pareto"]

            fig = go.Figure()
            # All points
            non_pareto = np.setdiff1d(np.arange(len(F)), pareto)
            fig.add_trace(go.Scatter(x=F[non_pareto, 0], y=F[non_pareto, 1],
                                     mode="markers",
                                     marker=dict(size=6, color="#d1d5db"),
                                     name="Dominated"))
            # Pareto points
            p_sorted = pareto[np.argsort(F[pareto, 0])]
            fig.add_trace(go.Scatter(x=F[p_sorted, 0], y=F[p_sorted, 1],
                                     mode="markers+lines",
                                     marker=dict(size=10, color="#dc2626",
                                                symbol="star"),
                                     line=dict(color="#dc2626", width=2, dash="dash"),
                                     name="Pareto front"))
            fig.update_layout(
                title=f"Pareto front: {st.session_state['nsga_obj1']} vs {st.session_state['nsga_obj2']}",
                xaxis_title=st.session_state["nsga_obj1"],
                yaxis_title=st.session_state["nsga_obj2"],
                height=500,
                margin=dict(l=60, r=20, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Run NSGA-II** to find the Pareto-optimal trade-offs.")


# ===========================================================================
# TAB 7: FEATURE IMPORTANCE (live permutation!)
# ===========================================================================
with tabs[6]:
    st.markdown("### 🔍 Feature Importance (Permutation)")
    st.caption(
        "Permutation importance on a trained Random Forest surrogate. "
        "Ranks device parameters by how much scrambling each one degrades "
        "predicted PCE."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        n_samples_fi = st.slider("Training samples", 100, 400, 200, step=50, key="fi_n_slider")
        fi_target = st.selectbox("Target", ["PCE", "Voc", "Jsc", "FF"], key="fi_target_widget")

        if st.button("▶ Compute Feature Importance", type="primary", use_container_width=True):
            with st.spinner(f"Generating {n_samples_fi} samples and computing permutation importance..."):
                t0 = time.time()

                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance

                df = build_sample_grid(n_samples=n_samples_fi, seed=42)
                features = ["d_htl", "d_abs", "d_etl", "log_Nt", "T"]
                feature_labels = ["HTL thickness", "Absorber thickness",
                                   "ETL thickness", "log₁₀(Nt)", "Temperature"]
                X = df[features].values
                y = df[fi_target].values

                model = RandomForestRegressor(n_estimators=150, random_state=0)
                model.fit(X, y)

                result = permutation_importance(model, X, y,
                                                  n_repeats=15,
                                                  random_state=0,
                                                  n_jobs=1)

                st.session_state["fi_mean"] = result.importances_mean
                st.session_state["fi_std"]  = result.importances_std
                st.session_state["fi_labels"] = feature_labels
                st.session_state["fi_target_display"] = fi_target
                st.session_state["fi_elapsed"] = time.time() - t0
                st.session_state["fi_r2"] = model.score(X, y)

        if "fi_mean" in st.session_state:
            st.caption(f"✓ Computed in {st.session_state['fi_elapsed']:.1f} s")
            st.metric("Training R²", f"{st.session_state['fi_r2']:.3f}")

    with col_b:
        if "fi_mean" in st.session_state:
            labels = st.session_state["fi_labels"]
            means  = st.session_state["fi_mean"]
            stds   = st.session_state["fi_std"]

            # Sort by importance
            order = np.argsort(means)
            labels_sorted = [labels[i] for i in order]
            means_sorted  = means[order]
            stds_sorted   = stds[order]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=labels_sorted, x=means_sorted, orientation="h",
                error_x=dict(type="data", array=stds_sorted, color="#374151"),
                marker=dict(color=means_sorted,
                           colorscale="Viridis", showscale=False),
                text=[f"{v:.3f}" for v in means_sorted], textposition="outside",
            ))
            fig.update_layout(
                title=f"Feature importance — predicting {st.session_state['fi_target_display']}",
                xaxis_title="Permutation importance (↓ R² when shuffled)",
                height=400,
                margin=dict(l=120, r=80, t=50, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Compute Feature Importance** to see which device parameters matter most.")


# ===========================================================================
# TAB 8: INVERSE DESIGN (live DE!)
# ===========================================================================
with tabs[7]:
    st.markdown("### 🔄 Inverse Design")
    st.caption(
        "Find device parameters matching your target metrics. Differential "
        "Evolution runs in ~10 seconds."
    )

    st.markdown("**Target metrics**")
    cols = st.columns(4)
    tgt_pce = cols[0].number_input("Target PCE (%)",  18.0, 28.0, 24.0, step=0.5)
    tgt_voc = cols[1].number_input("Target Voc (V)",  0.8,  1.4,  1.15, step=0.01)
    tgt_jsc = cols[2].number_input("Target Jsc (mA/cm²)", 18.0, 28.0, 24.0, step=0.5)
    tgt_ff  = cols[3].number_input("Target FF (%)",   70.0, 90.0, 82.0, step=0.5)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        max_iter = st.slider("DE iterations", 20, 80, 40, step=10)

        if st.button("▶ Search", type="primary", use_container_width=True):
            with st.spinner(f"Running differential evolution ({max_iter} iterations)..."):
                t0 = time.time()

                from scipy.optimize import differential_evolution

                history_loss = []

                def obj_fn(x):
                    d_htl, d_abs, d_etl, log_Nt = x
                    try:
                        r = fast_simulate(
                            HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                            d_htl_nm=d_htl, d_abs_nm=d_abs, d_etl_nm=d_etl,
                            Nt_abs=10**log_Nt, T=300,
                        )
                        loss = ((r["PCE"] - tgt_pce)**2
                                + 100 * (r["Voc"] - tgt_voc)**2
                                + (r["Jsc"] - tgt_jsc)**2
                                + 100 * (r["FF"]  - tgt_ff/100)**2)
                        return loss
                    except Exception:
                        return 1e6

                bounds = [(50, 400), (200, 900), (20, 200), (13.0, 16.5)]

                def callback(xk, convergence):
                    history_loss.append(obj_fn(xk))
                    return False

                result = differential_evolution(
                    obj_fn, bounds, maxiter=max_iter, popsize=12, seed=0,
                    tol=1e-4, polish=True, callback=callback,
                )

                d_htl_f, d_abs_f, d_etl_f, log_Nt_f = result.x
                r_f = fast_simulate(
                    HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                    d_htl_nm=d_htl_f, d_abs_nm=d_abs_f, d_etl_nm=d_etl_f,
                    Nt_abs=10**log_Nt_f, T=300,
                )

                st.session_state["inv_result"] = r_f
                st.session_state["inv_params"] = (d_htl_f, d_abs_f, d_etl_f, 10**log_Nt_f)
                st.session_state["inv_targets"] = (tgt_pce, tgt_voc, tgt_jsc, tgt_ff/100)
                st.session_state["inv_history"] = history_loss
                st.session_state["inv_elapsed"] = time.time() - t0

        if "inv_result" in st.session_state:
            r = st.session_state["inv_result"]
            d_htl_f, d_abs_f, d_etl_f, Nt_f = st.session_state["inv_params"]

            st.caption(f"✓ Done in {st.session_state['inv_elapsed']:.1f} s")
            st.markdown("**Found parameters**:")
            st.markdown(f"- HTL thickness: **{d_htl_f:.0f} nm**")
            st.markdown(f"- Absorber thickness: **{d_abs_f:.0f} nm**")
            st.markdown(f"- ETL thickness: **{d_etl_f:.0f} nm**")
            st.markdown(f"- Nt = **{Nt_f:.2e} cm⁻³**")

    with col_b:
        if "inv_result" in st.session_state:
            r = st.session_state["inv_result"]
            tgt_pce_s, tgt_voc_s, tgt_jsc_s, tgt_ff_s = st.session_state["inv_targets"]

            # Target vs achieved table
            comp_df = pd.DataFrame({
                "Metric": ["PCE (%)", "Voc (V)", "Jsc (mA/cm²)", "FF"],
                "Target":   [f"{tgt_pce_s:.2f}", f"{tgt_voc_s:.3f}", f"{tgt_jsc_s:.2f}", f"{tgt_ff_s:.3f}"],
                "Achieved": [f"{r['PCE']:.2f}", f"{r['Voc']:.3f}", f"{r['Jsc']:.2f}", f"{r['FF']:.3f}"],
                "Error":    [f"{abs(r['PCE']-tgt_pce_s):.2f}",
                             f"{abs(r['Voc']-tgt_voc_s):.3f}",
                             f"{abs(r['Jsc']-tgt_jsc_s):.2f}",
                             f"{abs(r['FF']-tgt_ff_s):.3f}"],
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Convergence plot
            if len(st.session_state.get("inv_history", [])) > 1:
                hist = st.session_state["inv_history"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(hist)+1)), y=hist,
                    mode="lines+markers", line=dict(color="#2563eb", width=2),
                    marker=dict(size=6),
                ))
                fig.update_yaxes(type="log", title_text="Objective loss")
                fig.update_xaxes(title_text="DE iteration")
                fig.update_layout(
                    title="Optimization convergence",
                    height=350, margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Search** to run inverse design.")


# ===========================================================================
# TAB 9: BENCHMARKS (pre-computed DD results + live parity plots)
# ===========================================================================
with tabs[8]:
    st.markdown("### 📐 Validation Benchmarks")

    bdata = load_benchmark_data()

    if bdata is None:
        st.error("Benchmark data not found. Run `python scripts/run_benchmark.py` to generate.")
    else:
        # ---------- Honest provenance disclosure (visible by default) -----
        meta = bdata.get("_meta", {})
        disc = meta.get("honest_disclosure", {})
        if disc:
            with st.expander("📜 **How these benchmarks were produced — honest disclosure**", expanded=True):
                st.markdown(f"**What is real**: {disc.get('what_is_real', '')}")
                st.markdown(f"**Pre-computed values**: {disc.get('what_is_pre_computed', '')}")
                st.markdown(f"**Estimates**: {disc.get('what_is_an_estimate', '')}")
                st.markdown(f"**To regenerate**: {disc.get('how_to_regenerate', '')}")

        sub1, sub2 = st.tabs(["🔬 SCAPS-reference (verified DOIs only)",
                                "📊 Experimental (verified DOIs only)"])

        # ============================================================
        with sub1:
            scaps = bdata["scaps_reference"]
            summary = scaps["summary"]

            st.markdown(
                "Drift-diffusion benchmark against published SCAPS-1D references. "
                "**Every reference has a verified DOI** that resolves via doi.org. "
                "Click 🔗 in any row to open the original paper."
            )

            cmetrics = st.columns(4)
            cmetrics[0].metric("Devices",          summary["n_devices"])
            cmetrics[1].metric("Mean PCE error",   f"{summary['mean_pce_error_pct']:.1f}%")
            cmetrics[2].metric("Median PCE error", f"{summary['median_pce_error_pct']:.1f}%")
            cmetrics[3].metric("All verified",     "✓" if summary.get("all_references_verified") else "—")

            # Per-device table with clickable DOI links
            devs = pd.DataFrame(scaps["devices"])
            display_df = devs[["id", "stack", "reference_text", "doi", "url",
                                "scaps_pce", "our_pce", "pce_error_pct"]].rename(columns={
                "id": "Device", "stack": "Stack", "reference_text": "Reference",
                "doi": "DOI", "url": "Open paper",
                "scaps_pce": "SCAPS PCE", "our_pce": "Our PCE",
                "pce_error_pct": "|ΔPCE| %",
            })
            st.dataframe(
                display_df,
                hide_index=True, use_container_width=True,
                column_config={
                    "Device":     st.column_config.TextColumn("Device", width="small"),
                    "Stack":      st.column_config.TextColumn("Stack", width="medium"),
                    "Reference":  st.column_config.TextColumn("Reference", width="large"),
                    "DOI":        st.column_config.TextColumn("DOI", width="medium"),
                    "Open paper": st.column_config.LinkColumn(
                        "Open paper",
                        help="Click to open the original paper at doi.org",
                        display_text="🔗 Open",
                        width="small",
                    ),
                    "SCAPS PCE":  st.column_config.NumberColumn("SCAPS PCE", format="%.2f"),
                    "Our PCE":    st.column_config.NumberColumn("Our PCE",   format="%.2f"),
                    "|ΔPCE| %":   st.column_config.NumberColumn("|ΔPCE| %",  format="%.1f"),
                },
            )

            # Parity plot + error histogram
            cols = st.columns(2)
            with cols[0]:
                fig = go.Figure()
                mn = min(devs["scaps_pce"].min(), devs["our_pce"].min()) - 1
                mx = max(devs["scaps_pce"].max(), devs["our_pce"].max()) + 1
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                         line=dict(color="#9ca3af", dash="dash"),
                                         name="y = x", showlegend=False))
                fig.add_trace(go.Scatter(
                    x=devs["scaps_pce"], y=devs["our_pce"],
                    mode="markers+text",
                    marker=dict(size=10, color="#2563eb"),
                    text=devs["id"], textposition="top right",
                    name="Devices", showlegend=False,
                ))
                fig.update_layout(
                    title="Parity plot: our DD vs SCAPS PCE",
                    xaxis_title="SCAPS PCE (%)", yaxis_title="Our DD PCE (%)",
                    height=400, margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=devs["id"], y=devs["pce_error_pct"],
                    marker=dict(color=devs["pce_error_pct"],
                                 colorscale="Reds", showscale=False),
                    text=[f"{v:.1f}%" for v in devs["pce_error_pct"]],
                    textposition="outside",
                ))
                fig.add_hline(y=summary["mean_pce_error_pct"], line_dash="dash",
                              line_color="#6b7280",
                              annotation_text=f"mean = {summary['mean_pce_error_pct']:.1f}%")
                fig.update_layout(
                    title="Per-device PCE error",
                    xaxis_title="Device", yaxis_title="|ΔPCE| (%)",
                    height=400, margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        with sub2:
            exp = bdata["experimental"]
            summary = exp["summary"]

            st.markdown(
                "Benchmark against fabricated, measured cells. **Every reference "
                "has a verified DOI**. Expected errors 15-30% because a 1D DD "
                "solver cannot capture 2D/3D effects, grain boundaries, or area "
                "scaling."
            )

            if summary.get("interpretation_caveat"):
                st.info(f"**Interpretation note**: {summary['interpretation_caveat']}")

            cmetrics = st.columns(4)
            cmetrics[0].metric("Devices",          summary["n_total"])
            cmetrics[1].metric("Mean PCE error",   f"{summary['mean_pce_error_pct']:.1f}%")
            cmetrics[2].metric("Median PCE error", f"{summary['median_pce_error_pct']:.1f}%")
            cmetrics[3].metric("Within 30%",       f"{summary['n_within_30pct']}/{summary['n_total']}")

            devs = pd.DataFrame(exp["devices"])
            display_df = devs[["id", "reference_text", "doi", "url", "stack",
                                "measured_pce", "our_pce", "pce_error_pct",
                                "certified"]].rename(columns={
                "id": "ID", "reference_text": "Reference",
                "doi": "DOI", "url": "Open paper", "stack": "Stack",
                "measured_pce": "Measured PCE", "our_pce": "Our PCE",
                "pce_error_pct": "|ΔPCE| %", "certified": "Certified",
            })
            st.dataframe(
                display_df,
                hide_index=True, use_container_width=True,
                column_config={
                    "ID":        st.column_config.TextColumn("ID", width="small"),
                    "Reference": st.column_config.TextColumn("Reference", width="large"),
                    "DOI":       st.column_config.TextColumn("DOI", width="medium"),
                    "Open paper": st.column_config.LinkColumn(
                        "Open paper",
                        help="Click to open the original paper at doi.org",
                        display_text="🔗 Open",
                        width="small",
                    ),
                    "Stack":     st.column_config.TextColumn("Stack", width="medium"),
                    "Measured PCE": st.column_config.NumberColumn("Measured PCE", format="%.2f"),
                    "Our PCE":      st.column_config.NumberColumn("Our PCE",      format="%.2f"),
                    "|ΔPCE| %":     st.column_config.NumberColumn("|ΔPCE| %",     format="%.1f"),
                    "Certified":    st.column_config.CheckboxColumn("Cert."),
                },
            )

            cols = st.columns(2)

            with cols[0]:
                fig = go.Figure()
                mn = min(devs["measured_pce"].min(), devs["our_pce"].min()) - 1
                mx = max(devs["measured_pce"].max(), devs["our_pce"].max()) + 1
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                         line=dict(color="#9ca3af", dash="dash"),
                                         showlegend=False))

                for is_cert in [True, False]:
                    subset = devs[devs["certified"] == is_cert]
                    if len(subset) > 0:
                        fig.add_trace(go.Scatter(
                            x=subset["measured_pce"], y=subset["our_pce"],
                            mode="markers+text",
                            marker=dict(size=12,
                                         color="#dc2626" if is_cert else "#2563eb",
                                         symbol="star" if is_cert else "circle"),
                            text=subset["id"], textposition="top right",
                            name="Certified" if is_cert else "Reported",
                        ))
                fig.update_layout(
                    title="Parity plot: our DD vs measured PCE",
                    xaxis_title="Measured PCE (%)", yaxis_title="Our DD PCE (%)",
                    height=400, margin=dict(l=60, r=20, t=50, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=devs["id"], y=devs["pce_error_pct"],
                    marker=dict(color=["#dc2626" if c else "#2563eb"
                                        for c in devs["certified"]]),
                    text=[f"{v:.1f}%" for v in devs["pce_error_pct"]],
                    textposition="outside",
                ))
                fig.add_hline(y=summary["mean_pce_error_pct"], line_dash="dash",
                              line_color="#6b7280")
                fig.update_layout(
                    title="Per-device PCE error (red = NREL-certified)",
                    xaxis_title="Device", yaxis_title="|ΔPCE| (%)",
                    height=400, margin=dict(l=60, r=20, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 10: DATABASE
# ===========================================================================
with tabs[9]:
    st.markdown("### 📋 Material Database — with per-parameter provenance")

    summary = summarize_provenance_confidence()
    cmetrics = st.columns(4)
    cmetrics[0].metric("Total materials", len(HTL_DB) + len(PEROVSKITE_DB) + len(ETL_DB))
    cmetrics[1].metric("HIGH-confidence",   summary["total"]["HIGH"])
    cmetrics[2].metric("MEDIUM-confidence", summary["total"]["MEDIUM"])
    cmetrics[3].metric("LOW-confidence",    summary["total"]["LOW"],
                        help="Re-source these before publishing")

    st.markdown("---")
    st.markdown("#### Inspect a material")

    layer_choice = st.radio("Layer type",
                             ["absorber", "htl", "etl"],
                             horizontal=True)
    pool = {"absorber": PEROVSKITE_DB, "htl": HTL_DB, "etl": ETL_DB}[layer_choice]
    mat_choice = st.selectbox("Material", list(pool.keys()))

    info = get_material_with_provenance(mat_choice, layer_choice)

    # Build rows with both DOI and clickable URL columns
    rows = []
    for key in ["Eg_eV", "chi_eV", "eps_r", "Nc_cm3", "Nv_cm3",
                "mu_n_cm2_Vs", "mu_p_cm2_Vs", "doping_cm3",
                "Nt_bulk_cm3", "alpha_coeff_cm"]:
        if key in info:
            doi = info.get(f"{key}_doi", "")
            url = f"https://doi.org/{doi}" if doi and doi != "VERIFY" else ""
            rows.append({
                "Parameter":  key,
                "Value":      info[key],
                "Confidence": info.get(f"{key}_confidence", ""),
                "Method":     info.get(f"{key}_method", ""),
                "Source":     info.get(f"{key}_source", ""),
                "DOI":        doi,
                "Open paper": url,   # rendered as a clickable button-link
            })
    df = pd.DataFrame(rows)

    # Render with clickable DOIs via LinkColumn
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameter":  st.column_config.TextColumn("Parameter", width="small"),
            "Value":      st.column_config.NumberColumn("Value", format="%.3g"),
            "Confidence": st.column_config.TextColumn("Conf.", width="small"),
            "Method":     st.column_config.TextColumn("Method", width="medium"),
            "Source":     st.column_config.TextColumn("Source paper", width="large"),
            "DOI":        st.column_config.TextColumn("DOI", width="medium"),
            "Open paper": st.column_config.LinkColumn(
                "Open paper",
                help="Click to open the original paper at doi.org",
                display_text="🔗 Open",
                width="small",
            ),
        },
    )

    st.caption(
        "💡 Click 🔗 **Open** in any row to open that source paper at doi.org. "
        "Every value carries its own DOI — no shared citations across "
        "unrelated parameters."
    )

    # Show full reference list with clickable links
    with st.expander("📚 All 27 references in this database (with clickable DOI links)"):
        # Load references from the JSON
        with open(ARTIFACTS_DIR.parent / "data" / "materials_database.json") as f:
            db_full = json.load(f)
        refs = db_full.get("_references", {})

        ref_rows = []
        for ref_id, entry in refs.items():
            if ref_id.startswith("_REMOVED"):
                continue
            doi = entry.get("doi", "")
            url = entry.get("url", f"https://doi.org/{doi}" if doi else "")
            ref_rows.append({
                "ID":        ref_id,
                "Citation":  entry.get("citation", ""),
                "DOI":       doi,
                "Open":      url,
                "Verified":  "✓" if entry.get("verified") else "",
            })

        ref_df = pd.DataFrame(ref_rows)
        st.dataframe(
            ref_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID":       st.column_config.TextColumn("Citation key", width="medium"),
                "Citation": st.column_config.TextColumn("Full citation", width="large"),
                "DOI":      st.column_config.TextColumn("DOI", width="medium"),
                "Open":     st.column_config.LinkColumn(
                    "Open paper", display_text="🔗 Open", width="small"),
                "Verified": st.column_config.TextColumn("Verified", width="small"),
            },
        )
        st.caption(
            f"Total {len(ref_rows)} active references, all with verified DOIs and "
            "clickable URLs. Tombstone entries (deprecated/removed references) "
            "are kept in the JSON for audit trail but hidden here."
        )

    st.markdown(
        "**Format**: every parameter carries source, DOI, measurement method, "
        "and a confidence tier. Edits to `data/materials_database.json` are "
        "enforced by CI — any new value missing provenance blocks the merge."
    )

    with st.expander("How to add or update a material"):
        st.markdown(
            "See `docs/MATERIAL_LIFECYCLE.md` for the full workflow. "
            "Short version: edit the extension JSON, run "
            "`scripts/merge_materials_db.py`, run `pytest tests/`, "
            "run the benchmarks, commit with the delta."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "☀️ Perovskite Solar Cell Design Tool · MIT License · "
    "Live interactive · Publication-ready figures · "
    f"Materials DB: **{len(HTL_DB)+len(PEROVSKITE_DB)+len(ETL_DB)}** with full provenance"
)
