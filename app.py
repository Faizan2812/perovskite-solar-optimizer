"""
PINN Perovskite Solar Cell Optimizer — Professional Research Tool
=================================================================
AI-driven open-source tool for design and optimization of perovskite solar cells.
Integrates: drift-diffusion physics, PINN surrogate, Bayesian optimization,
NSGA-II multi-objective, SHAP analysis, inverse design, SCAPS compatibility.

Run: streamlit run app.py
Deploy: push to GitHub → share.streamlit.io
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time, json, io

# Local modules
from physics.materials import (HTL_DB, PEROVSKITE_DB, ETL_DB, CONTACTS,
                               Material, get_all_materials)
from physics.device import (fast_simulate, simulate_iv_curve, 
                            solve_drift_diffusion, build_layer_stack,
                            compute_generation_profile, Q, V_T, PIN,
                            simulate_hysteresis, simulate_tandem)
from ai.optimizer import (bayesian_optimization, nsga2_optimize,
                          train_pinn_surrogate, compute_shap_importance,
                          inverse_design, run_de, run_pso, run_ga,
                          predict_stability_t80, parameter_sweep,
                          temperature_coefficients, compare_materials,
                          DeepONet, active_learning_loop,
                          GaussianProcessRegressor,
                          parse_natural_language_query, execute_query)
from utils.helpers import (parse_scaps_def, scaps_layers_to_summary,
                           parse_scaps_result, export_iv_csv, export_qe_csv,
                           export_metrics_summary, generate_report_data,
                           export_scaps_def, material_to_scaps_layer,
                           generate_html_report, EXPERIMENTAL_BENCHMARKS)
from ai.pinn_pde import PhysicsPINN
from ai.ml_models import generate_simulation_dataset, compare_all_models
from utils.benchmark import (run_full_benchmark, compute_benchmark_summary,
                             format_comparison_table)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PINN Perovskite Solar Cell Optimizer",
    page_icon="⚛️", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
.metric-card {
    background: rgba(15,23,42,0.6); border: 1px solid rgba(148,163,184,0.08);
    border-radius: 10px; padding: 12px 16px; text-align: center;
}
.metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono'; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.5)",
    font=dict(family="JetBrains Mono, monospace", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.3)"),
)
C = {"pri":"#0EA5E9","sec":"#F97316","ter":"#10B981","pur":"#8B5CF6",
     "red":"#EF4444","amb":"#EAB308","pink":"#EC4899"}

def metric_html(label, value, unit, color):
    return f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color}'>{value}
            <span style='font-size:11px;color:#475569'>{unit}</span>
        </div></div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:8px 0 16px'>
<h1 style='background:linear-gradient(135deg,#38bdf8,#818cf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:26px;margin-bottom:2px'>
    ⚛️ PINN Perovskite Solar Cell Optimizer
</h1>
<p style='color:#475569;font-size:12px;letter-spacing:1px'>
    Drift-Diffusion Physics · PINN · Bayesian Optimization · NSGA-II · SHAP · Inverse Design · SCAPS Compatible
</p></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏗️ Device architecture")
    st.caption("n-i-p: Glass / FTO / ETL / Absorber / HTL / Au")
    
    st.markdown("---")
    st.markdown("#### 🔴 HTL")
    htl_name = st.selectbox("HTL material", list(HTL_DB.keys()), index=0)
    d_htl = st.slider("HTL thickness (nm)", 10, 500, 50, 5)
    
    st.markdown("#### 🟣 Perovskite absorber")
    abs_name = st.selectbox("Absorber material", list(PEROVSKITE_DB.keys()), index=0)
    d_abs = st.slider("Absorber thickness (nm)", 50, 2000, 300, 10)
    nt_exp = st.slider("Defect density (10^x /cm³)", 10, 18, 14, 1)
    Nt = 10.0 ** nt_exp
    
    st.markdown("#### 🔵 ETL")
    etl_name = st.selectbox("ETL material", list(ETL_DB.keys()), index=0)
    d_etl = st.slider("ETL thickness (nm)", 10, 500, 50, 5)
    
    st.markdown("#### 🌡️ Conditions")
    temperature = st.slider("Temperature (K)", 250, 450, 300, 5)
    
    st.markdown("---")
    st.caption(f"**{htl_name}** Eg={HTL_DB[htl_name].Eg} χ={HTL_DB[htl_name].chi}")
    st.caption(f"**{abs_name}** Eg={PEROVSKITE_DB[abs_name].Eg} χ={PEROVSKITE_DB[abs_name].chi}")
    st.caption(f"**{etl_name}** Eg={ETL_DB[etl_name].Eg} χ={ETL_DB[etl_name].chi}")
    
    st.markdown("---")
    st.markdown("#### 💬 Natural language query")
    nl_query = st.text_input("Ask a question", placeholder="e.g., Optimize PCE for MAPbI3", key="nl_q")
    if nl_query:
        parsed = parse_natural_language_query(nl_query)
        result_nl = execute_query(parsed, htl_name, etl_name)
        if "error" in result_nl:
            st.error(result_nl["error"])
        else:
            st.success(result_nl.get("description", str(result_nl)))

htl = HTL_DB[htl_name]
absorber = PEROVSKITE_DB[abs_name]
etl = ETL_DB[etl_name]


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_sim, tab_dd, tab_pinn, tab_pde, tab_bo, tab_mo, tab_shap, tab_inv, tab_ml, tab_bench, tab_scaps, tab_db = st.tabs([
    "⚡ Simulate", "🔬 Profiles", "🧠 PINN", "⚛️ PDE-PINN",
    "🎯 Bayesian Opt.", "📊 Multi-Obj.", "🔍 SHAP", "🔄 Inverse Design",
    "🤖 ML Models", "📐 Benchmark", "📁 SCAPS Import", "📋 Database"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: FAST SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    result = fast_simulate(htl, absorber, etl, d_htl, d_abs, d_etl, Nt, temperature)
    
    # Metrics
    cols = st.columns(6)
    for col, (label, val, unit, color) in zip(cols, [
        ("PCE",f"{result['PCE']:.2f}","%",C["ter"]),
        ("Voc",f"{result['Voc']:.3f}","V",C["pri"]),
        ("Jsc",f"{result['Jsc']:.2f}","mA/cm²",C["sec"]),
        ("FF",f"{result['FF']*100:.1f}","%",C["pur"]),
        ("Vmpp",f"{result['Vmpp']:.3f}","V",C["amb"]),
        ("Pmax",f"{result['Pmax']:.2f}","mW/cm²",C["red"]),
    ]):
        col.markdown(metric_html(label, val, unit, color), unsafe_allow_html=True)
    
    # Stability prediction
    t80 = predict_stability_t80(abs_name, absorber.Eg, Nt)
    st.info(f"🕐 Estimated T80 lifetime: **{t80:.0f} hours** ({t80/24:.0f} days) — "
            f"{'Encapsulated' if t80 > 500 else 'Unencapsulated'} estimate for {abs_name}")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["voltages"], y=result["currents"],
                                mode="lines", name="J-V", line=dict(color=C["pri"], width=2.5)))
        fig.update_layout(title="J-V characteristic", xaxis_title="Voltage (V)",
                         yaxis_title="J (mA/cm²)", height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["lams_qe"], y=result["qe"], mode="lines",
                                fill="tozeroy", line=dict(color=C["ter"], width=2),
                                fillcolor="rgba(16,185,129,0.1)"))
        fig.update_layout(title="External quantum efficiency", xaxis_title="Wavelength (nm)",
                         yaxis_title="QE (%)", height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    # Power curve
    power = np.where(result["currents"] > 0, result["voltages"] * result["currents"], 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["voltages"], y=power, mode="lines",
                            fill="tozeroy", line=dict(color=C["sec"], width=2.5),
                            fillcolor="rgba(249,115,22,0.08)"))
    fig.update_layout(title="Power output", xaxis_title="Voltage (V)",
                     yaxis_title="Power (mW/cm²)", height=300, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model parameters
    with st.expander("📊 Model parameters"):
        pc = st.columns(3)
        params = [
            [("Jph",f"{result['Jph']:.3f} mA/cm²"),("J₀",f"{result['J0']:.2e} mA/cm²"),
             ("n (ideality)",f"{result['n']:.3f}")],
            [("Rs",f"{result['Rs']:.2f} Ω·cm²"),("Rsh",f"{result['Rsh']:.0f} Ω·cm²"),
             ("Collection η",f"{result['eta_c']*100:.1f}%")],
            [("α_eff",f"{result['alpha']:.1e} /cm"),("L_diff",f"{result['L_diff_um']:.1f} μm"),
             ("V deficit",f"{result['deficit']:.3f} V")],
        ]
        for col, plist in zip(pc, params):
            for label, val in plist:
                col.metric(label, val)
    
    # Export
    with st.expander("📥 Export results"):
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("⬇ J-V (CSV)", export_iv_csv(result), "iv_curve.csv", "text/csv")
        c2.download_button("⬇ QE (CSV)", export_qe_csv(result), "qe_curve.csv", "text/csv")
        c3.download_button("⬇ Report (TXT)", export_metrics_summary(result),
                          "simulation_report.txt", "text/plain")
        html_report = generate_html_report(result, htl_name, abs_name, etl_name,
                                           d_htl, d_abs, d_etl, Nt)
        c4.download_button("⬇ Report (HTML)", html_report, "report.html", "text/html")
        
        # SCAPS .def export
        layers = [material_to_scaps_layer(etl, d_etl),
                  material_to_scaps_layer(absorber, d_abs, Nt),
                  material_to_scaps_layer(htl, d_htl)]
        def_content = export_scaps_def(layers)
        st.download_button("⬇ SCAPS .def file", def_content, "device.def", "text/plain")
    
    # ─── J-V Hysteresis (Ion Migration) ───────────────────────────────────
    with st.expander("🔄 J-V hysteresis (ion migration)"):
        scan_rate = st.slider("Scan rate (V/s)", 0.01, 1.0, 0.1, 0.01, key="hyst_sr")
        n_ion_exp = st.slider("Ion concentration (10^x /cm³)", 15, 20, 18, 1, key="hyst_ion")
        
        if st.button("▶ Simulate hysteresis", key="hyst_run"):
            with st.spinner("Simulating forward and reverse scans..."):
                hyst = simulate_hysteresis(htl, absorber, etl, d_htl, d_abs, d_etl,
                                          Nt, temperature, scan_rate, 10**n_ion_exp)
                st.session_state["hyst"] = hyst
        
        if "hyst" in st.session_state:
            hy = st.session_state["hyst"]
            st.metric("Hysteresis Index", f"{hy['hysteresis_index']*100:.1f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hy["reverse"]["voltages"], y=hy["reverse"]["currents"],
                                    name=f"Reverse (PCE={hy['reverse']['PCE']:.2f}%)",
                                    line=dict(color=C["pri"], width=2.5)))
            fig.add_trace(go.Scatter(x=hy["forward"]["voltages"], y=hy["forward"]["currents"],
                                    name=f"Forward (PCE={hy['forward']['PCE']:.2f}%)",
                                    line=dict(color=C["red"], width=2, dash="dash")))
            fig.update_layout(title="J-V hysteresis", xaxis_title="V (V)",
                             yaxis_title="J (mA/cm²)", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    
    # ─── Tandem Cell Simulation ───────────────────────────────────────────
    with st.expander("🔗 Tandem cell simulation"):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Top cell (wide gap)**")
            top_abs = st.selectbox("Top absorber", 
                [k for k,v in PEROVSKITE_DB.items() if v.Eg >= 1.6], key="t_top")
        with tc2:
            st.markdown("**Bottom cell (narrow gap)**")
            bot_abs = st.selectbox("Bottom absorber",
                [k for k,v in PEROVSKITE_DB.items() if v.Eg < 1.6], key="t_bot")
        
        terminal = st.radio("Terminal configuration", ["2T (series)", "4T (independent)"],
                           horizontal=True, key="t_term")
        term_code = "2T" if "2T" in terminal else "4T"
        
        if st.button("▶ Simulate tandem", key="tandem_run"):
            with st.spinner("Simulating tandem..."):
                tandem = simulate_tandem(
                    htl, PEROVSKITE_DB[top_abs], etl, htl, PEROVSKITE_DB[bot_abs], etl,
                    d_top_abs=400, d_bot_abs=500, terminal=term_code, T=temperature)
                st.session_state["tandem"] = tandem
        
        if "tandem" in st.session_state:
            td = st.session_state["tandem"]
            tc = st.columns(4)
            tc[0].metric("Tandem PCE", f"{td['PCE']:.2f}%")
            tc[1].metric("Tandem Voc", f"{td['Voc']:.3f} V")
            tc[2].metric("Top PCE", f"{td['top_cell']['PCE']:.2f}%")
            tc[3].metric("Bottom Jsc (filtered)", f"{td['bottom_Jsc_filtered']:.2f} mA/cm²")

    # ─── Parameter Sensitivity Sweep ─────────────────────────────────────
    with st.expander("📈 Parameter sensitivity sweep"):
        sweep_param = st.selectbox("Sweep parameter", 
            ["Absorber thickness (nm)", "HTL thickness (nm)", "ETL thickness (nm)",
             "Defect density (log₁₀)", "Temperature (K)"], key="sweep_sel")
        
        sweep_map = {
            "Absorber thickness (nm)": ("d_abs", np.arange(50, 1500, 50)),
            "HTL thickness (nm)": ("d_htl", np.arange(10, 400, 20)),
            "ETL thickness (nm)": ("d_etl", np.arange(10, 400, 20)),
            "Defect density (log₁₀)": ("log_Nt", np.arange(10, 18, 0.5)),
            "Temperature (K)": ("T", np.arange(250, 460, 10)),
        }
        param_key, sweep_vals = sweep_map[sweep_param]
        
        if st.button("▶ Run sweep", key="sweep_run"):
            def sweep_sim(params):
                nt_val = 10**params.get("log_Nt", nt_exp)
                return fast_simulate(htl, absorber, etl,
                    params.get("d_htl", d_htl), params.get("d_abs", d_abs),
                    params.get("d_etl", d_etl), nt_val, params.get("T", temperature))
            
            base = {"d_htl": d_htl, "d_abs": d_abs, "d_etl": d_etl,
                    "log_Nt": nt_exp, "T": temperature}
            
            with st.spinner("Running sweep..."):
                df_sweep = parameter_sweep(sweep_sim, base, param_key, sweep_vals)
                st.session_state["sweep"] = (df_sweep, sweep_param, param_key)
        
        if "sweep" in st.session_state:
            df_sw, sw_label, sw_key = st.session_state["sweep"]
            fig = make_subplots(rows=1, cols=4, subplot_titles=["PCE (%)", "Voc (V)", "Jsc (mA/cm²)", "FF (%)"],
                               horizontal_spacing=0.08)
            for i, metric in enumerate(["PCE", "Voc", "Jsc", "FF"]):
                colors = [C["ter"], C["pri"], C["sec"], C["pur"]]
                fig.add_trace(go.Scatter(x=df_sw[sw_key], y=df_sw[metric], mode="lines+markers",
                    line=dict(color=colors[i], width=2), marker=dict(size=4),
                    name=metric, showlegend=False), row=1, col=i+1)
            fig.update_layout(height=300, **PLOTLY_LAYOUT)
            fig.update_xaxes(title_text=sw_label)
            st.plotly_chart(fig, use_container_width=True)
    
    # ─── Temperature Coefficients ─────────────────────────────────────────
    with st.expander("🌡️ Temperature coefficients"):
        def temp_sim(T_val):
            return fast_simulate(htl, absorber, etl, d_htl, d_abs, d_etl, Nt, T_val)
        coeffs = temperature_coefficients(temp_sim, temperature, 20)
        tc = st.columns(4)
        tc[0].metric("dVoc/dT", f"{coeffs['dVoc/dT']:.2f} mV/K")
        tc[1].metric("dJsc/dT", f"{coeffs['dJsc/dT']:.4f} mA/cm²/K")
        tc[2].metric("dFF/dT", f"{coeffs['dFF/dT']:.3f} %/K")
        tc[3].metric("dPCE/dT", f"{coeffs['dPCE/dT']:.3f} %/K")
        st.caption("Typical perovskite: dVoc/dT ≈ -1 to -2 mV/K, dPCE/dT ≈ -0.03 to -0.05 %/K")
    
    # ─── Multi-Material Comparison ────────────────────────────────────────
    with st.expander("🔬 Compare all absorber materials"):
        def comp_sim(mat_name, params):
            m = PEROVSKITE_DB[mat_name]
            return fast_simulate(htl, m, etl, d_htl, params["d_abs"], d_etl, Nt, temperature)
        
        comp_df = compare_materials(comp_sim, list(PEROVSKITE_DB.keys()), PEROVSKITE_DB,
                                   {"d_abs": d_abs})
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp_df["Material"], y=comp_df["PCE (%)"],
            marker_color=[C["ter"] if v == comp_df["PCE (%)"].max() else C["pri"] 
                         for v in comp_df["PCE (%)"]],
            text=[f"{v:.1f}%" for v in comp_df["PCE (%)"]], textposition="auto"))
        fig.update_layout(title=f"PCE comparison ({htl_name} / X / {etl_name})",
                         yaxis_title="PCE (%)", height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
with tab_dd:
    st.markdown("### 🔬 Spatially-resolved device profiles")
    st.caption("1D drift-diffusion solver with Scharfetter-Gummel discretization")
    
    if st.button("▶ Run drift-diffusion solver", type="primary", key="dd_run"):
        with st.spinner("Solving Poisson + continuity equations..."):
            try:
                dd_result = solve_drift_diffusion(
                    htl, absorber, etl, d_htl, d_abs, d_etl, Nt, temperature, 0, 100, 30)
                st.session_state["dd_result"] = dd_result
            except Exception as ex:
                st.error(f"Solver error: {ex}")
    
    if "dd_result" in st.session_state:
        dd = st.session_state["dd_result"]
        x_um = dd["x"] * 1e4  # cm to μm
        
        # Layer boundaries
        x_htl_um = d_htl * 1e-3  # nm to μm
        x_abs_um = x_htl_um + d_abs * 1e-3
        
        st.caption(f"J(V=0) = {dd['J_at_V']:.2f} mA/cm² | "
                   f"Vbi = {dd['Vbi']:.3f} V | "
                   f"W_depl = {dd['W_depletion']:.2f} μm | "
                   f"Mesh: {len(dd['x'])} points")
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Carrier concentrations n(x), p(x)", "Electric field E(x)",
            "Generation G(x) & recombination R(x)", "Band diagram & potential ψ(x)"
        ], vertical_spacing=0.12, horizontal_spacing=0.1)
        
        fig.add_trace(go.Scatter(x=x_um, y=dd["n"], name="n (electrons)",
                                line=dict(color=C["pri"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_um, y=dd["p"], name="p (holes)",
                                line=dict(color=C["red"], width=2)), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=x_um, y=dd["E_field"], name="E-field",
                                line=dict(color=C["amb"], width=2)), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=x_um, y=np.maximum(dd["G"], 1), name="Generation G",
                                line=dict(color=C["ter"], width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_um, y=np.maximum(dd["R"], 1), name="Recombination R",
                                line=dict(color=C["red"], width=2, dash="dash")), row=2, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=x_um, y=dd["psi"], name="ψ (potential)",
                                line=dict(color=C["pur"], width=2)), row=2, col=2)
        if "Ec" in dd:
            fig.add_trace(go.Scatter(x=x_um, y=dd["Ec"], name="Ec",
                                    line=dict(color=C["pri"], width=1.5, dash="dot")), row=2, col=2)
            fig.add_trace(go.Scatter(x=x_um, y=dd["Ev"], name="Ev",
                                    line=dict(color=C["red"], width=1.5, dash="dot")), row=2, col=2)
        
        # Add layer boundary lines to all subplots
        for row in [1, 2]:
            for col in [1, 2]:
                for xb, lbl in [(x_htl_um, "HTL|Abs"), (x_abs_um, "Abs|ETL")]:
                    fig.add_vline(x=xb, line_dash="dash", line_color="rgba(148,163,184,0.3)",
                                 row=row, col=col)
        
        fig.update_layout(height=700, showlegend=True, **PLOTLY_LAYOUT)
        fig.update_xaxes(title_text="Position (μm)")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: PINN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pinn:
    st.markdown("### 🧠 Physics-Informed Neural Network")
    st.caption("Architecture: [1→48→32→1] with tanh activation + monotonicity constraint")
    
    pinn_epochs = st.slider("Training epochs", 100, 1000, 500, 50, key="pinn_ep")
    
    if st.button("▶ Train PINN", type="primary", key="pinn_run"):
        with st.spinner(f"Training PINN for {pinn_epochs} epochs..."):
            try:
                r = fast_simulate(htl, absorber, etl, d_htl, d_abs, d_etl, Nt, temperature)
                pred, unc, losses, nn, vm, jm = train_pinn_surrogate(
                    r["voltages"], r["currents"], pinn_epochs)
                st.session_state["pinn"] = {"pred": pred, "unc": unc, "losses": losses,
                                            "V": r["voltages"], "J": r["currents"]}
            except Exception as ex:
                st.error(f"PINN training error: {ex}")
    
    if "pinn" in st.session_state:
        p = st.session_state["pinn"]
        mask = p["J"] > 0
        rmse = np.sqrt(np.mean((p["pred"][mask] - p["J"][mask])**2))
        st.metric("PINN RMSE (operating region)", f"{rmse:.3f} mA/cm²")
        
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p["V"], y=p["J"], name="Physics model",
                                    line=dict(color=C["pri"], width=2.5)))
            fig.add_trace(go.Scatter(x=p["V"], y=p["pred"], name="PINN prediction",
                                    line=dict(color=C["sec"], width=2, dash="dash")))
            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=np.concatenate([p["V"], p["V"][::-1]]),
                y=np.concatenate([p["pred"] + p["unc"], (p["pred"] - p["unc"])[::-1]]),
                fill="toself", fillcolor="rgba(249,115,22,0.1)",
                line=dict(width=0), name="Uncertainty (±1σ)"))
            fig.update_layout(title="PINN vs physics model", xaxis_title="V (V)",
                             yaxis_title="J (mA/cm²)", height=400, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=np.log10(np.array(p["losses"]) + 1e-12),
                                    mode="lines", line=dict(color=C["red"], width=2)))
            fig.update_layout(title="Training loss (log₁₀)", xaxis_title="Epoch",
                             yaxis_title="log₁₀(MSE)", height=400, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PDE-RESIDUAL PINN (Gap 6)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pde:
    st.markdown("### ⚛️ Physics-Informed Neural Network with PDE Residuals")
    st.caption("Two-stage: (1) Train on J-V data with monotonicity constraint, "
               "(2) Evaluate Poisson + continuity equation residuals as diagnostics")
    
    st.markdown("""
    **PDE equations enforced:**
    - **Poisson:** d²ψ/dx² + q/ε·(p - n + Nd - Na) = 0
    - **Electron continuity:** dJn/dx - q·(G - R) = 0
    - **Hole continuity:** dJp/dx + q·(G - R) = 0
    """)
    
    pde_epochs = st.slider("Training epochs", 200, 1000, 500, 50, key="pde_ep")
    
    if st.button("▶ Train PDE-PINN", type="primary", key="pde_run"):
        with st.spinner(f"Stage 1: Training on data ({int(pde_epochs*0.8)} epochs)... "
                       f"Stage 2: Computing PDE residuals ({int(pde_epochs*0.2)} epochs)..."):
            try:
                
                r_pde = fast_simulate(htl, absorber, etl, d_htl, d_abs, d_etl, Nt, temperature)
                x_colloc, G_prof = compute_generation_profile(
                    absorber.Eg, absorber.alpha_coeff, d_abs * 1e-7, 50)
                R_prof = G_prof * 0.1
                
                device_params = {
                    "eps": absorber.eps, "Na": absorber.doping,
                    "Nd": 0, "ni": float(absorber.ni)
                }
                
                pinn_pde = PhysicsPINN(layers=(1, 64, 48, 32, 1))
                loss_hist = pinn_pde.train(
                    r_pde["voltages"], r_pde["currents"], epochs=pde_epochs, lr=0.002,
                    device_params=device_params, x_colloc=x_colloc,
                    G_profile=G_prof, R_profile=R_prof
                )
                
                pred_pde = pinn_pde.predict(r_pde["voltages"])
                pred_m, pred_s = pinn_pde.predict_with_uncertainty(r_pde["voltages"])
                
                st.session_state["pde_pinn"] = {
                    "pred": pred_pde, "unc": pred_s,
                    "losses": loss_hist, "V": r_pde["voltages"],
                    "J": r_pde["currents"], "report": pinn_pde.get_pde_residual_report(),
                    "diagnostics": pinn_pde.pde_diagnostics,
                }
            except Exception as ex:
                st.error(f"PDE-PINN error: {ex}")
    
    if "pde_pinn" in st.session_state:
        pp = st.session_state["pde_pinn"]
        mask_op = pp["J"] > 0
        rmse = np.sqrt(np.mean((pp["pred"][mask_op] - pp["J"][mask_op])**2))
        
        st.metric("PINN RMSE (operating region)", f"{rmse:.3f} mA/cm²")
        
        # J-V comparison plot
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pp["V"], y=pp["J"], name="Physics model",
                                    line=dict(color=C["pri"], width=2.5)))
            fig.add_trace(go.Scatter(x=pp["V"], y=pp["pred"], name="PDE-PINN",
                                    line=dict(color=C["sec"], width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=np.concatenate([pp["V"], pp["V"][::-1]]),
                y=np.concatenate([pp["pred"] + pp["unc"], (pp["pred"] - pp["unc"])[::-1]]),
                fill="toself", fillcolor="rgba(249,115,22,0.1)",
                line=dict(width=0), name="±1σ uncertainty"))
            fig.update_layout(title="PDE-PINN vs physics model", xaxis_title="V (V)",
                             yaxis_title="J (mA/cm²)", height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=np.log10(np.array(pp["losses"]["data"]) + 1e-12),
                                    name="Data loss", line=dict(color=C["pri"], width=2)))
            fig.add_trace(go.Scatter(y=np.log10(np.array(pp["losses"]["physics"]) + 1e-12),
                                    name="Physics loss", line=dict(color=C["sec"], width=2, dash="dash")))
            fig.update_layout(title="Training loss (log₁₀)", xaxis_title="Epoch",
                             yaxis_title="log₁₀(loss)", height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        # PDE residual report
        st.markdown("#### PDE residual diagnostics")
        st.caption("These metrics quantify how well the learned solution satisfies the semiconductor equations")
        report_df = pd.DataFrame([
            {"Metric": k, "Value": str(v)} for k, v in pp["report"].items()
        ])
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        # PDE residual profiles
        if pp.get("diagnostics") and "x_colloc" in pp["diagnostics"]:
            diag = pp["diagnostics"]
            x_um = diag["x_colloc"] * 1e4
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=[
                "Poisson residual", "Continuity residual"])
            fig.add_trace(go.Scatter(x=x_um, y=diag["poisson_profile"],
                                    line=dict(color=C["pur"], width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_um, y=diag["continuity_profile"],
                                    line=dict(color=C["ter"], width=2)), row=1, col=2)
            fig.update_layout(height=300, showlegend=False, **PLOTLY_LAYOUT)
            fig.update_xaxes(title_text="Position (μm)")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: BAYESIAN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bo:
    st.markdown("### 🎯 Bayesian Optimization with GPR Surrogate")
    st.caption("Gaussian Process Regression + Expected Improvement acquisition function")
    
    algo = st.radio("Algorithm", ["Bayesian Optimization (GPR+EI)", "Active Learning (uncertainty sampling)",
                                   "Differential Evolution", "Particle Swarm", "Genetic Algorithm"],
                    horizontal=True, key="bo_algo")
    
    c1, c2 = st.columns(2)
    with c1:
        n_iter = st.slider("Iterations", 10, 100, 40, 5, key="bo_iter")
    with c2:
        pop_sz = st.slider("Population/initial samples", 8, 30, 12, 2, key="bo_pop")
    
    bounds_opt = [(20, 200), (100, 1200), (20, 200), (10, 16)]  # d_htl, d_abs, d_etl, log_Nt
    
    if st.button("▶ Run optimization", type="primary", key="bo_run"):
        def obj_fn(params):
            r = fast_simulate(htl, absorber, etl, params[0], params[1], params[2],
                             10**params[3], temperature)
            return -r["PCE"]
        
        with st.spinner(f"Running {algo}..."):
            t0 = time.time()
            if "Bayesian" in algo:
                def obj_max(params):
                    r = fast_simulate(htl, absorber, etl, params[0], params[1], params[2],
                                     10**params[3], temperature)
                    return r["PCE"]
                bp, bv, hist, gpr = bayesian_optimization(obj_max, bounds_opt, pop_sz, n_iter)
                elapsed = time.time() - t0
            elif "Active" in algo:
                def obj_al(params):
                    r = fast_simulate(htl, absorber, etl, params[0], params[1], params[2],
                                     10**params[3], temperature)
                    return r["PCE"]
                gpr_al = GaussianProcessRegressor(noise=1e-4)
                X_al, y_al, unc_hist = active_learning_loop(obj_al, bounds_opt, gpr_al, pop_sz, n_iter)
                best_idx = np.argmax(y_al)
                bp = X_al[best_idx]; bv = y_al[best_idx]; hist = list(np.maximum.accumulate(y_al))
                elapsed = time.time() - t0
            elif "Differential" in algo:
                bp, bv, hist = run_de(obj_fn, bounds_opt, n_iter, pop_sz)
                elapsed = time.time() - t0
            elif "Particle" in algo:
                bp, bv, hist = run_pso(obj_fn, bounds_opt, n_iter, pop_sz)
                elapsed = time.time() - t0
            else:
                bp, bv, hist = run_ga(obj_fn, bounds_opt, n_iter, pop_sz)
                elapsed = time.time() - t0
            
            opt_r = fast_simulate(htl, absorber, etl, bp[0], bp[1], bp[2], 10**bp[3], temperature)
            st.session_state["bo_result"] = {
                "params": bp, "pce": opt_r["PCE"], "result": opt_r,
                "history": hist, "elapsed": elapsed, "algo": algo}
    
    if "bo_result" in st.session_state:
        bo = st.session_state["bo_result"]
        or_ = bo["result"]
        
        st.success(f"✅ {bo['algo']} complete in {bo['elapsed']:.1f}s — PCE: **{or_['PCE']:.2f}%**")
        
        cols = st.columns(4)
        for col, (l, v, c) in zip(cols, [
            ("Opt PCE", f"{or_['PCE']:.2f}%", C["ter"]),
            ("Opt Voc", f"{or_['Voc']:.3f} V", C["pri"]),
            ("Opt Jsc", f"{or_['Jsc']:.2f} mA/cm²", C["sec"]),
            ("Opt FF", f"{or_['FF']*100:.1f}%", C["pur"]),
        ]):
            col.markdown(metric_html(l, v, "", c), unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Optimized parameters:**")
            opt_df = pd.DataFrame({
                "Parameter": ["HTL thickness", "Absorber thickness", "ETL thickness", "Defect density"],
                "Optimized": [f"{bo['params'][0]:.1f} nm", f"{bo['params'][1]:.1f} nm",
                             f"{bo['params'][2]:.1f} nm", f"{10**bo['params'][3]:.1e} /cm³"],
                "Original": [f"{d_htl} nm", f"{d_abs} nm", f"{d_etl} nm", f"{Nt:.1e} /cm³"],
            })
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
        with c2:
            if bo["history"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=bo["history"], mode="lines",
                                        line=dict(color=C["ter"], width=2)))
                fig.update_layout(title="Convergence", xaxis_title="Iteration",
                                 yaxis_title="PCE (%)", height=300, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: MULTI-OBJECTIVE (NSGA-II)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mo:
    st.markdown("### 📊 Multi-Objective Optimization (NSGA-II)")
    st.caption("Simultaneously maximize PCE and estimated T80 stability lifetime")
    
    mo_gen = st.slider("Generations", 10, 80, 30, 5, key="mo_gen")
    
    if st.button("▶ Run NSGA-II", type="primary", key="mo_run"):
        def obj_pce(params):
            r = fast_simulate(htl, absorber, etl, params[0], params[1], params[2],
                             10**params[3], temperature)
            return r["PCE"]
        
        def obj_stability(params):
            return predict_stability_t80(abs_name, absorber.Eg, 10**params[3]) / 100
        
        with st.spinner("Running NSGA-II..."):
            pareto, pareto_fit, hist = nsga2_optimize(
                [obj_pce, obj_stability], bounds_opt, mo_gen, 30)
            st.session_state["mo_result"] = {"pareto": pareto, "fitness": pareto_fit, "history": hist}
    
    if "mo_result" in st.session_state:
        mo = st.session_state["mo_result"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mo["fitness"][:, 0], y=mo["fitness"][:, 1] * 100,
            mode="markers", marker=dict(size=10, color=C["ter"]),
            text=[f"d_abs={p[1]:.0f}nm, Nt={10**p[3]:.0e}" for p in mo["pareto"]],
            name="Pareto front"))
        fig.update_layout(title="Pareto front: PCE vs stability",
                         xaxis_title="PCE (%)", yaxis_title="T80 (hours)",
                         height=450, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Pareto-optimal solutions:**")
        rows = []
        for i, (p, f) in enumerate(zip(mo["pareto"], mo["fitness"])):
            rows.append({
                "Solution": i + 1, "PCE (%)": round(f[0], 2),
                "T80 (hours)": round(f[1] * 100, 0),
                "d_HTL (nm)": round(p[0], 1), "d_Abs (nm)": round(p[1], 1),
                "d_ETL (nm)": round(p[2], 1), "Nt (/cm³)": f"{10**p[3]:.1e}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: SHAP FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown("### 🔍 Feature Importance Analysis")
    st.caption("Permutation-based SHAP-like importance: which parameters impact PCE most?")
    
    n_shap = st.slider("Perturbation samples", 20, 200, 50, 10, key="shap_n")
    
    if st.button("▶ Compute feature importance", type="primary", key="shap_run"):
        def sim_fn(params):
            r = fast_simulate(htl, absorber, etl, params["d_htl"], params["d_abs"],
                             params["d_etl"], 10**params["log_Nt"], temperature)
            return r["PCE"]
        
        base = {"d_htl": d_htl, "d_abs": d_abs, "d_etl": d_etl, "log_Nt": np.log10(Nt)}
        ranges = {"d_htl": (20, 300), "d_abs": (100, 1200), "d_etl": (20, 300), "log_Nt": (10, 17)}
        
        with st.spinner("Computing importance..."):
            imp = compute_shap_importance(sim_fn, base, list(base.keys()), ranges, n_shap)
            st.session_state["shap"] = imp
    
    if "shap" in st.session_state:
        imp = st.session_state["shap"]
        names = {"d_htl": "HTL thickness", "d_abs": "Absorber thickness",
                "d_etl": "ETL thickness", "log_Nt": "Defect density (Nt)"}
        
        fig = go.Figure()
        labels = [names.get(k, k) for k in imp.keys()]
        values = [v * 100 for v in imp.values()]
        colors = [C["red"], C["pur"], C["pri"], C["sec"]]
        fig.add_trace(go.Bar(x=values, y=labels, orientation="h",
                            marker_color=colors[:len(labels)],
                            text=[f"{v:.1f}%" for v in values], textposition="auto"))
        fig.update_layout(title="Feature importance for PCE",
                         xaxis_title="Importance (%)", height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
        top = max(imp, key=imp.get)
        st.info(f"💡 **{names.get(top, top)}** has the highest impact on PCE "
                f"({imp[top]*100:.1f}%). Prioritize this parameter in experimental optimization.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: INVERSE DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inv:
    st.markdown("### 🔄 Inverse Design: Target → Parameters")
    st.caption("Specify desired output metrics and find the required device parameters")
    
    c1, c2, c3, c4 = st.columns(4)
    target_pce = c1.number_input("Target PCE (%)", 5.0, 35.0, 25.0, 0.5)
    target_voc = c2.number_input("Target Voc (V)", 0.5, 2.0, 1.1, 0.01)
    target_jsc = c3.number_input("Target Jsc (mA/cm²)", 5.0, 35.0, 24.0, 0.5)
    target_ff = c4.number_input("Target FF", 0.5, 0.92, 0.85, 0.01)
    
    if st.button("▶ Find optimal parameters", type="primary", key="inv_run"):
        targets = {"PCE": target_pce, "Voc": target_voc, "Jsc": target_jsc, "FF": target_ff}
        
        def sim_fn(params):
            r = fast_simulate(htl, absorber, etl, params[0], params[1], params[2],
                             10**params[3], temperature)
            return r
        
        with st.spinner("Searching parameter space..."):
            best_p, achieved, dist = inverse_design(targets, sim_fn, bounds_opt, n_iterations=80)
            st.session_state["inv"] = {"params": best_p, "achieved": achieved,
                                       "targets": targets, "distance": dist}
    
    if "inv" in st.session_state:
        inv = st.session_state["inv"]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Target vs achieved:**")
            comp_df = pd.DataFrame({
                "Metric": list(inv["targets"].keys()),
                "Target": [f"{v:.3f}" for v in inv["targets"].values()],
                "Achieved": [f"{inv['achieved'][k]:.3f}" for k in inv["targets"]],
                "Match": ["✅" if abs(inv["achieved"][k] - v) / max(abs(v), 1e-6) < 0.1
                         else "⚠️" for k, v in inv["targets"].items()],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Required parameters:**")
            st.metric("HTL thickness", f"{inv['params'][0]:.1f} nm")
            st.metric("Absorber thickness", f"{inv['params'][1]:.1f} nm")
            st.metric("ETL thickness", f"{inv['params'][2]:.1f} nm")
            st.metric("Defect density", f"{10**inv['params'][3]:.1e} /cm³")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: ML MODEL COMPARISON (Gap 5)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.markdown("### 🤖 Classical ML Model Comparison")
    st.caption("Random Forest, Gradient Boosting (XGBoost-style), and Neural Network (ANN) "
               "trained on simulation-generated dataset")
    
    c1, c2 = st.columns(2)
    n_train = c1.slider("Training samples", 200, 2000, 500, 100, key="ml_n")
    target_metric = c2.selectbox("Target metric", ["PCE", "Voc", "Jsc", "FF"], key="ml_tgt")
    target_idx = {"PCE": 0, "Voc": 1, "Jsc": 2, "FF": 3}[target_metric]
    
    if st.button("▶ Train & compare all models", type="primary", key="ml_run"):
        with st.spinner(f"Generating {n_train} samples and training RF + XGBoost + ANN..."):
            try:
                X_ml, Y_ml, feat_names, tgt_names = generate_simulation_dataset(n_train, seed=42)
                ml_results = compare_all_models(X_ml, Y_ml, feat_names, target_idx)
                st.session_state["ml_results"] = ml_results
                st.session_state["ml_feat_names"] = feat_names
                st.session_state["ml_target"] = target_metric
            except Exception as ex:
                st.error(f"ML training error: {ex}")
    
    if "ml_results" in st.session_state:
        ml_res = st.session_state["ml_results"]
        ml_tgt = st.session_state.get("ml_target", "PCE")
        ml_fn = st.session_state.get("ml_feat_names", [])
        
        # Model comparison metrics
        st.markdown(f"#### Model performance predicting **{ml_tgt}**")
        comp_rows = []
        for mr in ml_res:
            comp_rows.append({
                "Model": mr.name,
                "Train R²": round(mr.train_metrics["R2"], 4),
                "Test R²": round(mr.test_metrics["R2"], 4),
                "Test MAPE (%)": round(mr.test_metrics["MAPE"], 1),
                "Test RMSE": round(mr.test_metrics["RMSE"], 3),
                "Test MAE": round(mr.test_metrics["MAE"], 3),
                "Time (s)": round(mr.training_time, 2),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        
        # Bar chart of R²
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            names = [mr.name.split("(")[0].strip() for mr in ml_res]
            r2_test = [mr.test_metrics["R2"] for mr in ml_res]
            r2_train = [mr.train_metrics["R2"] for mr in ml_res]
            fig.add_trace(go.Bar(x=names, y=r2_train, name="Train R²",
                                marker_color=C["pri"], opacity=0.6))
            fig.add_trace(go.Bar(x=names, y=r2_test, name="Test R²",
                                marker_color=C["ter"]))
            fig.update_layout(title=f"R² comparison ({ml_tgt})", yaxis_title="R²",
                             barmode="group", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Predicted vs actual scatter for best model
            best = max(ml_res, key=lambda m: m.test_metrics["R2"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=best.y_test, y=best.predictions_test,
                                    mode="markers", marker=dict(size=4, color=C["ter"], opacity=0.6),
                                    name=best.name.split("(")[0].strip()))
            y_range = [min(best.y_test.min(), best.predictions_test.min()),
                      max(best.y_test.max(), best.predictions_test.max())]
            fig.add_trace(go.Scatter(x=y_range, y=y_range, mode="lines",
                                    line=dict(color="gray", dash="dash"), name="Perfect"))
            fig.update_layout(title=f"Predicted vs actual ({best.name.split('(')[0].strip()})",
                             xaxis_title=f"Actual {ml_tgt}", yaxis_title=f"Predicted {ml_tgt}",
                             height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance comparison
        st.markdown("#### Feature importance comparison")
        fi_rows = []
        for mr in ml_res:
            if mr.feature_importances is not None and len(ml_fn) == len(mr.feature_importances):
                for j, fn in enumerate(ml_fn):
                    fi_rows.append({"Feature": fn, "Model": mr.name.split("(")[0].strip(),
                                   "Importance": mr.feature_importances[j] * 100})
        if fi_rows:
            fi_df = pd.DataFrame(fi_rows)
            # Top 8 features by average importance
            avg_imp = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
            top8 = avg_imp.head(8).index.tolist()
            fi_top = fi_df[fi_df["Feature"].isin(top8)]
            
            fig = go.Figure()
            colors = [C["pri"], C["ter"], C["sec"]]
            for i, mr in enumerate(ml_res):
                model_name = mr.name.split("(")[0].strip()
                subset = fi_top[fi_top["Model"] == model_name].set_index("Feature").reindex(top8)
                fig.add_trace(go.Bar(x=top8, y=subset["Importance"].values,
                                    name=model_name, marker_color=colors[i % 3]))
            fig.update_layout(title="Top 8 features by importance",
                             yaxis_title="Importance (%)", barmode="group",
                             height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10: SCAPS BENCHMARK (Gap 4)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 📐 Formal SCAPS-1D Benchmarking")
    st.caption("Systematic comparison across 20 published device architectures")
    
    if st.button("▶ Run full benchmark (20 devices)", type="primary", key="bench_run"):
        with st.spinner("Simulating 20 reference devices..."):
            try:
                bench_results = run_full_benchmark()
                bench_summary = compute_benchmark_summary(bench_results)
                st.session_state["bench"] = {
                    "results": bench_results, "summary": bench_summary,
                    "comparison": format_comparison_table(),
                }
            except Exception as ex:
                st.error(f"Benchmark error: {ex}")
    
    if "bench" in st.session_state:
        b = st.session_state["bench"]
        s = b["summary"]
        
        # Summary metrics
        cols = st.columns(5)
        cols[0].metric("Devices tested", s["n_devices"])
        cols[1].metric("Convergence", f"{s['convergence_rate']:.0f}%")
        cols[2].metric("PCE error (mean)", f"{s['PCE_error_%']['mean']:.1f}%")
        cols[3].metric("PCE error (median)", f"{s['PCE_error_%']['median']:.1f}%")
        cols[4].metric("Avg sim time", f"{s['sim_time_ms']['mean']:.1f} ms")
        
        # Device-by-device results table
        st.markdown("#### Per-device results")
        dev_rows = []
        for br in b["results"]:
            dev_rows.append({
                "Device": br.device_name,
                "Absorber": br.absorber,
                "SCAPS PCE": f"{br.scaps_PCE:.2f}%",
                "Tool PCE": f"{br.tool_PCE:.2f}%",
                "PCE err": f"{br.err_PCE:.1f}%",
                "SCAPS Voc": f"{br.scaps_Voc:.3f}",
                "Tool Voc": f"{br.tool_Voc:.3f}",
                "Voc err": f"{br.err_Voc:.1f}%",
                "Time": f"{br.sim_time_ms:.1f}ms",
            })
        st.dataframe(pd.DataFrame(dev_rows), use_container_width=True, hide_index=True)
        
        # Error distribution charts
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            errors_pce = [br.err_PCE for br in b["results"]]
            fig.add_trace(go.Bar(
                x=[br.device_name[:20] for br in b["results"]],
                y=errors_pce,
                marker_color=[C["ter"] if e < 10 else C["amb"] if e < 20 else C["red"] for e in errors_pce],
            ))
            fig.update_layout(title="PCE error per device (%)", yaxis_title="Error (%)",
                             height=350, **PLOTLY_LAYOUT)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            fig = go.Figure()
            metrics = ["PCE", "Voc", "Jsc", "FF"]
            means = [s[f"{m}_error_%"]["mean"] for m in metrics]
            medians = [s[f"{m}_error_%"]["median"] for m in metrics]
            fig.add_trace(go.Bar(x=metrics, y=means, name="Mean error", marker_color=C["pri"]))
            fig.add_trace(go.Bar(x=metrics, y=medians, name="Median error", marker_color=C["ter"]))
            fig.update_layout(title="Aggregate error by metric", yaxis_title="Error (%)",
                             barmode="group", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tool capability comparison
        st.markdown("#### Tool capability comparison (22 criteria)")
        st.dataframe(b["comparison"], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11: SCAPS IMPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scaps:
    st.markdown("### 📁 SCAPS-1D File Import")
    st.caption("Upload .def files to extract parameters, or .iv/.qe files to compare results")
    
    uploaded_def = st.file_uploader("Upload SCAPS .def file", type=["def"], key="def_upload")
    if uploaded_def:
        content = uploaded_def.read().decode("latin-1")
        parsed = parse_scaps_def(content)
        summary = scaps_layers_to_summary(parsed)
        st.markdown("**Extracted layer parameters:**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        if parsed.get("contacts"):
            st.markdown("**Contact parameters:**")
            st.json(parsed["contacts"])
    
    st.markdown("---")
    uploaded_iv = st.file_uploader("Upload SCAPS result.iv file", type=["iv"], key="iv_upload")
    if uploaded_iv:
        content = uploaded_iv.read().decode("latin-1")
        df_iv = parse_scaps_result(content, "jtot(")
        if not df_iv.empty:
            st.markdown("**SCAPS J-V data loaded:**")
            scaps_v = df_iv.iloc[:, 0].values
            scaps_j = df_iv.iloc[:, 1].values
            
            sim_r = fast_simulate(htl, absorber, etl, d_htl, d_abs, d_etl, Nt, temperature)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scaps_v, y=scaps_j, name="SCAPS-1D",
                                    line=dict(color=C["pri"], width=2.5)))
            fig.add_trace(go.Scatter(x=sim_r["voltages"], y=sim_r["currents"],
                                    name="This tool", line=dict(color=C["sec"], width=2, dash="dash")))
            fig.update_layout(title="SCAPS vs this tool", xaxis_title="V (V)",
                             yaxis_title="J (mA/cm²)", height=400, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_db:
    st.markdown("### 📋 Validated Material Parameter Database")
    st.caption("All values cross-validated against peer-reviewed SCAPS-1D studies (2022-2026)")
    
    for title, db, dop_label in [("HTL materials", HTL_DB, "Na"),
                                  ("Perovskite absorbers", PEROVSKITE_DB, "Na"),
                                  ("ETL materials", ETL_DB, "Nd")]:
        st.markdown(f"#### {title}")
        rows = []
        for name, m in db.items():
            rows.append({
                "Material": name, "Category": m.category,
                "Eg (eV)": m.Eg, "χ (eV)": m.chi, "εr": m.eps,
                "μe (cm²/Vs)": m.mu_e, "μh (cm²/Vs)": m.mu_h,
                f"{dop_label} (/cm³)": f"{m.doping:.0e}",
                "Refs": m.refs,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    st.markdown("""
    #### References
    1. Chabri et al., *J. Electron. Mater.* **52**, 2722 (2023)
    2. Amjad et al., *RSC Adv.* **13**, 23211 (2023)
    3. Hossain et al., *ACS Omega* **7**, 43210 (2022)
    4. Uddin et al., *Next Materials* **9**, 100980 (2025)
    5. Oyelade et al., *Sci. Rep.* **14** (2024)
    6. Datto, *ChemNanoMat* (2026)
    7. Chen et al., *Energy & Fuels* (2023)
    8. MDPI Photonics (2025)
    9. Araújo et al., *RSC Sustainability* **3**, 4314 (2025)
    10. Saidarsan et al., *Sol. Energy Mater. Sol. Cells* **279**, 113230 (2025)
    """)
    
    st.markdown("#### Experimental validation benchmarks")
    bench_df = pd.DataFrame(EXPERIMENTAL_BENCHMARKS)
    st.dataframe(bench_df[["structure","Jsc","Voc","FF","PCE","ref","year"]],
                use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:10px;letter-spacing:1px;padding:8px'>
PINN PEROVSKITE SOLAR CELL OPTIMIZER · OPEN SOURCE (MIT) · 
46 MATERIALS · 3510 COMBOS · 12 TABS · 20 SCAPS BENCHMARKS · 3 ML MODELS · PDE-PINN · REST API<br>
Deploy free: <code>pip install streamlit && streamlit run app.py</code> · 
<a href='https://share.streamlit.io' style='color:#38bdf8'>share.streamlit.io</a>
</div>""", unsafe_allow_html=True)
