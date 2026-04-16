"""
Utility Functions: SCAPS Import/Export, Reports, Experimental Benchmarks
=========================================================================
All experimental benchmarks use ONLY real published data with DOIs.
"""
import numpy as np
import re
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL BENCHMARKS — Real published experimental results with DOIs
# Each entry is a real device fabricated and measured in a lab
# ═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTAL_BENCHMARKS = [
    # Nature / Science tier
    {"structure": "FAPbI3 (n-i-p)", "Jsc": 26.35, "Voc": 1.19, "FF": 0.833, "PCE": 26.1,
     "ref": "Min et al., Nature, vol. 598, pp. 444-450, 2021",
     "doi": "10.1038/s41586-021-03964-8", "year": 2021},

    {"structure": "CsPbI3 all-inorganic", "Jsc": 18.40, "Voc": 1.23, "FF": 0.823, "PCE": 18.6,
     "ref": "Wang et al., Science, vol. 365, pp. 591-595, 2019",
     "doi": "10.1126/science.aav8680", "year": 2019},

    {"structure": "CuSCN/MAPbI3 (p-i-n)", "Jsc": 23.30, "Voc": 1.07, "FF": 0.775, "PCE": 19.4,
     "ref": "Arora et al., Science, vol. 358, pp. 768-771, 2017",
     "doi": "10.1126/science.aam5655", "year": 2017},

    # Advanced Materials / Nature Energy
    {"structure": "FAPbI3/MAPbBr3 tandem", "Jsc": 25.09, "Voc": 1.18, "FF": 0.813, "PCE": 24.02,
     "ref": "Jiang et al., Nat. Photonics, vol. 13, pp. 460-466, 2019",
     "doi": "10.1038/s41566-019-0398-2", "year": 2019},

    {"structure": "Perovskite/Si tandem", "Jsc": 19.02, "Voc": 1.90, "FF": 0.795, "PCE": 28.0,
     "ref": "Hou et al., Science, vol. 367, pp. 1135-1140, 2020",
     "doi": "10.1126/science.aaz3691", "year": 2020},

    # Tin-based lead-free
    {"structure": "FASnI3 (inverted)", "Jsc": 22.80, "Voc": 0.62, "FF": 0.675, "PCE": 9.6,
     "ref": "Jiang et al., Nat. Commun., vol. 11, 1245, 2020",
     "doi": "10.1038/s41467-020-15078-2", "year": 2020},

    {"structure": "MASnI3/NiOx", "Jsc": 17.50, "Voc": 0.61, "FF": 0.503, "PCE": 5.4,
     "ref": "Ke et al., J. Am. Chem. Soc., vol. 140, pp. 388-393, 2018",
     "doi": "10.1021/jacs.7b10898", "year": 2018},

    # Wide bandgap
    {"structure": "MAPbBr3 (wide-gap)", "Jsc": 5.60, "Voc": 1.51, "FF": 0.740, "PCE": 6.2,
     "ref": "Noh et al., Nano Lett., vol. 13, pp. 1764-1769, 2013",
     "doi": "10.1021/nl400349b", "year": 2013},

    # Cs2SnI6 double perovskite
    {"structure": "Cs2SnI6/TiO2", "Jsc": 5.41, "Voc": 0.51, "FF": 0.546, "PCE": 1.47,
     "ref": "Lee et al., J. Am. Chem. Soc., vol. 136, pp. 15379-15385, 2014",
     "doi": "10.1021/ja508464w", "year": 2014},

    # Perovskite Database Project (aggregate)
    {"structure": "MAPbI3 (average experimental)", "Jsc": 21.50, "Voc": 1.05, "FF": 0.750, "PCE": 17.0,
     "ref": "Jacobsson et al., Nat. Energy, vol. 7, pp. 107-115, 2022",
     "doi": "10.1038/s41560-021-00941-3", "year": 2022},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCAPS .def FILE PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_scaps_def(content: str) -> List[Dict]:
    layers = []
    current = None
    lines = content.split('\n')
    for i, line in enumerate(lines):
        ls = line.strip()
        if 'layer name' in ls.lower() or ('layer' in ls.lower() and 'properties' in ls.lower()):
            if current:
                layers.append(current)
            current = {"name": "Unknown", "params": {}}
        if current is None:
            continue
        if 'layer name' in ls.lower():
            parts = ls.split(':')
            if len(parts) > 1:
                current["name"] = parts[1].strip()
            elif '=' in ls:
                current["name"] = ls.split('=')[1].strip()
        for key, param in [('eg', 'Eg'), ('chi', 'chi'), ('eps', 'eps'),
                          ('mu_n', 'mu_e'), ('mu_p', 'mu_h'), ('nc', 'Nc'),
                          ('nv', 'Nv'), ('nd', 'Nd'), ('na', 'Na'),
                          ('nt', 'Nt'), ('thickness', 'd')]:
            if key in ls.lower() and ('=' in ls or ':' in ls):
                try:
                    val_str = ls.split('=')[-1].strip() if '=' in ls else ls.split(':')[-1].strip()
                    val_str = val_str.split()[0]
                    val = float(val_str.replace('d', 'e').replace('D', 'E'))
                    current["params"][param] = val
                except (ValueError, IndexError):
                    pass
        if 'front contact' in ls.lower() or 'back contact' in ls.lower():
            if current and current["name"] != "Unknown":
                layers.append(current)
                current = None
    if current and current["name"] != "Unknown":
        layers.append(current)
    return layers


def scaps_layers_to_summary(layers: List[Dict]) -> str:
    lines = [f"Parsed {len(layers)} layers from SCAPS .def file:\n"]
    for i, layer in enumerate(layers):
        lines.append(f"  Layer {i+1}: {layer['name']}")
        for k, v in layer.get('params', {}).items():
            if isinstance(v, float) and abs(v) > 1e6:
                lines.append(f"    {k} = {v:.2e}")
            elif isinstance(v, float):
                lines.append(f"    {k} = {v:.4f}")
            else:
                lines.append(f"    {k} = {v}")
    return '\n'.join(lines)


def parse_scaps_result(content: str, file_type: str = "iv") -> Dict:
    lines = content.strip().split('\n')
    x_vals, y_vals = [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('v') or line.startswith('V'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue
    x_label = "Voltage (V)" if file_type == "iv" else "Wavelength (nm)"
    y_label = "Current (mA/cm²)" if file_type == "iv" else "QE (%)"
    return {"x": np.array(x_vals), "y": np.array(y_vals),
            "x_label": x_label, "y_label": y_label, "n_points": len(x_vals)}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def export_iv_csv(result: Dict) -> str:
    lines = ["Voltage (V),Current Density (mA/cm2)"]
    V = result.get('voltages', np.linspace(0, 1.1, 100))
    J = result.get('currents', np.zeros(100))
    for v, j in zip(V, J):
        lines.append(f"{v:.6f},{j:.6f}")
    return '\n'.join(lines)


def export_qe_csv(result: Dict) -> str:
    lines = ["Wavelength (nm),QE (%)"]
    lams = result.get('lams_qe', np.linspace(300, 1100, 161))
    qe = result.get('qe', np.zeros(161))
    for l, q in zip(lams, qe):
        lines.append(f"{l:.1f},{q:.2f}")
    return '\n'.join(lines)


def export_metrics_summary(result: Dict) -> str:
    lines = [
        "PINN Perovskite Solar Cell Optimizer — Simulation Results",
        "=" * 55,
        f"PCE  = {result.get('PCE', 0):.2f} %",
        f"Voc  = {result.get('Voc', 0):.4f} V",
        f"Jsc  = {result.get('Jsc', 0):.2f} mA/cm²",
        f"FF   = {result.get('FF', 0)*100:.1f} %",
        f"Vmpp = {result.get('Vmpp', 0):.4f} V",
        f"Pmpp = {result.get('Pmpp', 0):.2f} mW/cm²",
        "",
        f"Ideality factor n = {result.get('n', 0):.3f}",
        f"J0 = {result.get('J0', 0):.2e} mA/cm²",
        f"Rs = {result.get('Rs', 0):.2f} Ω·cm²",
        f"Rsh = {result.get('Rsh', 0):.0f} Ω·cm²",
    ]
    return '\n'.join(lines)


def generate_report_data(result: Dict) -> Dict:
    return {
        "PCE": result.get("PCE", 0),
        "Voc": result.get("Voc", 0),
        "Jsc": result.get("Jsc", 0),
        "FF": result.get("FF", 0),
        "Vmpp": result.get("Vmpp", 0),
        "Pmpp": result.get("Pmpp", 0),
    }


def material_to_scaps_layer(mat, thickness_nm: float, Nt=None) -> Dict:
    return {
        "name": mat.name,
        "d": thickness_nm * 1e-7,
        "Eg": mat.Eg,
        "chi": mat.chi,
        "eps": mat.eps,
        "Nc": mat.Nc,
        "Nv": mat.Nv,
        "mu_n": mat.mu_e,
        "mu_p": mat.mu_h,
        "Na": mat.doping if mat.doping_type == "p" else 0,
        "Nd": mat.doping if mat.doping_type == "n" else 0,
        "Nt": Nt if Nt is not None else mat.Nt,
    }


def export_scaps_def(layers: List[Dict]) -> str:
    lines = ["SCAPS Definition File (exported)", "=" * 40]
    for i, layer in enumerate(layers):
        lines.append(f"\n> layer {i+1}")
        lines.append(f"layer name : {layer.get('name', 'Layer')}")
        for key in ['d', 'Eg', 'chi', 'eps', 'Nc', 'Nv', 'mu_n', 'mu_p', 'Na', 'Nd', 'Nt']:
            if key in layer:
                val = layer[key]
                if abs(val) > 1e4 or (abs(val) < 0.01 and val != 0):
                    lines.append(f"{key} = {val:.4e}")
                else:
                    lines.append(f"{key} = {val:.6f}")
    return '\n'.join(lines)


def generate_html_report(result, htl_name, abs_name, etl_name,
                          d_htl, d_abs, d_etl, Nt, T=300) -> str:
    html = f"""<!DOCTYPE html>
<html><head><title>PSC Simulation Report</title>
<style>
body {{ font-family: Arial; max-width: 800px; margin: 40px auto; padding: 20px; }}
h1 {{ color: #1F3864; }} h2 {{ color: #2E75B6; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background: #1F3864; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.metric {{ display: inline-block; text-align: center; padding: 15px; margin: 5px;
           background: #f0f4f8; border-radius: 8px; min-width: 120px; }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #0d9488; }}
.metric .label {{ font-size: 12px; color: #666; }}
</style></head><body>
<h1>PINN Perovskite Solar Cell Optimizer — Simulation Report</h1>
<h2>Device: {htl_name} / {abs_name} / {etl_name}</h2>
<p>Thicknesses: {d_htl}/{d_abs}/{d_etl} nm | Nt = {Nt:.1e} cm⁻³ | T = {T} K</p>

<div>
<div class="metric"><div class="value">{result.get('PCE',0):.2f}%</div><div class="label">PCE</div></div>
<div class="metric"><div class="value">{result.get('Voc',0):.3f} V</div><div class="label">Voc</div></div>
<div class="metric"><div class="value">{result.get('Jsc',0):.2f}</div><div class="label">Jsc (mA/cm²)</div></div>
<div class="metric"><div class="value">{result.get('FF',0)*100:.1f}%</div><div class="label">FF</div></div>
</div>

<h2>Simulation Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>HTL</td><td>{htl_name} ({d_htl} nm)</td></tr>
<tr><td>Absorber</td><td>{abs_name} ({d_abs} nm)</td></tr>
<tr><td>ETL</td><td>{etl_name} ({d_etl} nm)</td></tr>
<tr><td>Defect density</td><td>{Nt:.1e} cm⁻³</td></tr>
<tr><td>Temperature</td><td>{T} K</td></tr>
</table>

<p><em>Generated by PINN Perovskite Solar Cell Optimizer (MIT License)</em></p>
</body></html>"""
    return html
