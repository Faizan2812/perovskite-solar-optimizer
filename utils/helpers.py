"""
Utilities: CSV export, report generation, experimental benchmarks.

Note on SCAPS .def interoperability
-----------------------------------
Earlier versions of this module included `parse_scaps_def`, `parse_scaps_result`,
`scaps_layers_to_summary`, `export_scaps_def`, and `material_to_scaps_layer`.
Those functions targeted the SCAPS-1D `.def` text format but were never
round-trip validated against real SCAPS inputs/outputs — the format has
required sections (defect layers, contact parameters, numerical settings,
illumination specifications) that the stubs did not emit, and the parser
covered only a minimal subset of SCAPS's actual keyword set.

Rather than ship half-working parsers that would silently produce wrong
results when a user round-tripped through SCAPS, the functions were removed
in this release and the UI tab that exposed them ("SCAPS Import") was
removed as well. The thesis positions the tool as validated *against*
published SCAPS outputs (see utils/benchmark.py), not interoperable *with*
SCAPS files. A future release may add a properly validated parser; for now,
users who need SCAPS interop should author their .def files directly in SCAPS.
"""
import numpy as np
import pandas as pd
import io
from typing import Dict, Optional



# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def export_iv_csv(result: Dict) -> str:
    """Export J-V curve as CSV string."""
    power = np.where(result["currents"] > 0, 
                     result["voltages"] * result["currents"], 0)
    df = pd.DataFrame({
        "Voltage_V": result["voltages"],
        "Current_Density_mA_cm2": result["currents"],
        "Power_mW_cm2": power,
    })
    return df.to_csv(index=False)


def export_qe_csv(result: Dict) -> str:
    """Export QE curve as CSV string."""
    df = pd.DataFrame({
        "Wavelength_nm": result["lams_qe"],
        "QE_percent": result["qe"],
    })
    return df.to_csv(index=False)


def export_metrics_summary(result: Dict) -> str:
    """Export simulation metrics as formatted text."""
    lines = [
        "=" * 50,
        "PEROVSKITE SOLAR CELL SIMULATION RESULTS",
        "=" * 50,
        "",
        "OUTPUT METRICS:",
        f"  PCE  = {result['PCE']:.4f} %",
        f"  Voc  = {result['Voc']:.4f} V",
        f"  Jsc  = {result['Jsc']:.4f} mA/cm²",
        f"  FF   = {result['FF']*100:.2f} %",
        f"  Vmpp = {result['Vmpp']:.4f} V",
        f"  Jmpp = {result['Jmpp']:.4f} mA/cm²",
        f"  Pmax = {result['Pmax']:.4f} mW/cm²",
        "",
        "MODEL PARAMETERS:",
        f"  Jph      = {result['Jph']:.4f} mA/cm²",
        f"  J0       = {result['J0']:.4e} mA/cm²",
        f"  n        = {result['n']:.4f}",
        f"  Rs       = {result['Rs']:.4f} Ω·cm²",
        f"  Rsh      = {result['Rsh']:.1f} Ω·cm²",
        f"  η_c      = {result['eta_c']*100:.2f} %",
        f"  α_eff    = {result['alpha']:.2e} /cm",
        f"  L_diff   = {result['L_diff_um']:.2f} μm",
        f"  τ        = {result['tau_ns']:.2f} ns",
        "",
        "INTERFACE ANALYSIS:",
        f"  CBO (ETL/Abs) = {result['CBO']:.4f} eV",
        f"  VBO (HTL/Abs) = {result['VBO']:.4f} eV",
        f"  V deficit     = {result['deficit']:.4f} V",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)


def generate_report_data(result: Dict, htl_name: str, abs_name: str, 
                         etl_name: str, d_htl: float, d_abs: float, 
                         d_etl: float) -> Dict:
    """Generate structured report data for export."""
    return {
        "Device Architecture": {
            "Structure": f"FTO/{etl_name}/{abs_name}/{htl_name}/Au",
            "HTL": f"{htl_name} ({d_htl:.0f} nm)",
            "Absorber": f"{abs_name} ({d_abs:.0f} nm)",
            "ETL": f"{etl_name} ({d_etl:.0f} nm)",
        },
        "Output Metrics": {
            "PCE (%)": round(result["PCE"], 4),
            "Voc (V)": round(result["Voc"], 4),
            "Jsc (mA/cm²)": round(result["Jsc"], 4),
            "FF": round(result["FF"], 4),
            "Vmpp (V)": round(result["Vmpp"], 4),
            "Pmax (mW/cm²)": round(result["Pmax"], 4),
        },
        "Model Parameters": {
            "Jph (mA/cm²)": round(result["Jph"], 4),
            "J0 (mA/cm²)": f"{result['J0']:.2e}",
            "Ideality n": round(result["n"], 4),
            "Rs (Ω·cm²)": round(result["Rs"], 4),
            "Rsh (Ω·cm²)": round(result["Rsh"], 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PDF/HTML REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_html_report(result, htl_name, abs_name, etl_name, 
                         d_htl, d_abs, d_etl, Nt):
    """Generate a publication-quality HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Perovskite Solar Cell Simulation Report</title>
<style>
body {{ font-family: 'Times New Roman', serif; max-width: 800px; margin: 40px auto; 
       padding: 20px; color: #333; line-height: 1.6; }}
h1 {{ font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 5px; }}
h2 {{ font-size: 14pt; color: #555; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #999; padding: 6px 10px; text-align: left; font-size: 10pt; }}
th {{ background: #f0f0f0; font-weight: bold; }}
.metric {{ font-size: 14pt; font-weight: bold; color: #006400; }}
.caption {{ font-size: 9pt; color: #666; margin-top: 5px; }}
</style>
</head>
<body>
<h1>Perovskite Solar Cell Simulation Report</h1>
<p><em>Generated by PINN Perovskite Solar Cell Optimizer</em></p>

<h2>1. Device Architecture</h2>
<table>
<tr><th>Layer</th><th>Material</th><th>Thickness</th><th>Eg (eV)</th></tr>
<tr><td>Front contact</td><td>FTO</td><td>—</td><td>3.50</td></tr>
<tr><td>ETL</td><td>{etl_name}</td><td>{d_etl} nm</td><td>{result.get('Voc',0):.0f}</td></tr>
<tr><td>Absorber</td><td>{abs_name}</td><td>{d_abs} nm</td><td>—</td></tr>
<tr><td>HTL</td><td>{htl_name}</td><td>{d_htl} nm</td><td>—</td></tr>
<tr><td>Back contact</td><td>Au</td><td>—</td><td>—</td></tr>
</table>

<h2>2. Photovoltaic Performance</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>PCE</td><td class="metric">{result['PCE']:.4f} %</td></tr>
<tr><td>Voc</td><td>{result['Voc']:.4f} V</td></tr>
<tr><td>Jsc</td><td>{result['Jsc']:.4f} mA/cm²</td></tr>
<tr><td>FF</td><td>{result['FF']*100:.2f} %</td></tr>
<tr><td>Vmpp</td><td>{result['Vmpp']:.4f} V</td></tr>
<tr><td>Pmax</td><td>{result['Pmax']:.4f} mW/cm²</td></tr>
</table>

<h2>3. Model Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Photocurrent (Jph)</td><td>{result['Jph']:.4f} mA/cm²</td></tr>
<tr><td>Saturation current (J0)</td><td>{result['J0']:.2e} mA/cm²</td></tr>
<tr><td>Ideality factor (n)</td><td>{result['n']:.4f}</td></tr>
<tr><td>Series resistance (Rs)</td><td>{result['Rs']:.4f} Ω·cm²</td></tr>
<tr><td>Shunt resistance (Rsh)</td><td>{result['Rsh']:.1f} Ω·cm²</td></tr>
<tr><td>Collection efficiency</td><td>{result['eta_c']*100:.2f} %</td></tr>
<tr><td>Diffusion length</td><td>{result['L_diff_um']:.2f} μm</td></tr>
<tr><td>Defect density (Nt)</td><td>{Nt:.1e} /cm³</td></tr>
</table>

<h2>4. Interface Analysis</h2>
<table>
<tr><td>CBO (ETL/Absorber)</td><td>{result['CBO']:.4f} eV</td></tr>
<tr><td>VBO (HTL/Absorber)</td><td>{result['VBO']:.4f} eV</td></tr>
<tr><td>Voltage deficit</td><td>{result['deficit']:.4f} V</td></tr>
</table>

<p class="caption">
Report generated by PINN Perovskite Solar Cell Optimizer. 
Material parameters validated against peer-reviewed SCAPS-1D studies (2022-2026).
</p>
</body></html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL VALIDATION DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTAL_BENCHMARKS = [
    {"structure": "Spiro-OMeTAD/MAPbI3/TiO2", "Jsc": 24.9, "Voc": 1.16, "FF": 0.81, "PCE": 23.3,
     "ref": "NREL Best Research-Cell Efficiencies (2024)", "year": 2024},
    {"structure": "Spiro-OMeTAD/FAPbI3/SnO2", "Jsc": 26.3, "Voc": 1.19, "FF": 0.83, "PCE": 26.1,
     "ref": "Min et al., Nature (2021)", "year": 2021},
    {"structure": "NiO/CsPbI3/SnO2", "Jsc": 18.5, "Voc": 1.26, "FF": 0.82, "PCE": 19.1,
     "ref": "Wang et al., Science (2019)", "year": 2019},
    {"structure": "PEDOT:PSS/MAPbI3/PCBM", "Jsc": 22.0, "Voc": 1.10, "FF": 0.79, "PCE": 19.1,
     "ref": "Meng et al., Acc. Chem. Res. (2016)", "year": 2016},
    {"structure": "Cu2O/Cs2SnI6/SnO2", "Jsc": 24.31, "Voc": 1.055, "FF": 0.856, "PCE": 21.94,
     "ref": "This work (SCAPS validation)", "year": 2025},
    {"structure": "Spiro-OMeTAD/MAPbBr3/TiO2", "Jsc": 5.6, "Voc": 1.51, "FF": 0.73, "PCE": 6.2,
     "ref": "Noh et al., Nano Lett. (2013)", "year": 2013},
    {"structure": "CuSCN/MAPbI3/TiO2", "Jsc": 23.1, "Voc": 1.09, "FF": 0.77, "PCE": 19.4,
     "ref": "Arora et al., Science (2017)", "year": 2017},
    {"structure": "Spiro-OMeTAD/CsPbI3/TiO2", "Jsc": 18.4, "Voc": 1.23, "FF": 0.82, "PCE": 18.6,
     "ref": "Wang et al., Joule (2019)", "year": 2019},
    {"structure": "NiO/Cs2AgBiBr6/TiO2", "Jsc": 3.9, "Voc": 1.01, "FF": 0.69, "PCE": 2.73,
     "ref": "Greul et al., J. Mater. Chem. A (2017)", "year": 2017},
    {"structure": "Spiro-OMeTAD/MASnI3/TiO2", "Jsc": 20.3, "Voc": 0.88, "FF": 0.59, "PCE": 10.5,
     "ref": "Ke et al., J. Am. Chem. Soc. (2018)", "year": 2018},
    {"structure": "PEDOT:PSS/FASnI3/C60", "Jsc": 22.5, "Voc": 0.76, "FF": 0.72, "PCE": 12.4,
     "ref": "Jiang et al., Nat. Commun. (2022)", "year": 2022},
    {"structure": "Me4PACz/FAPbI3/C60", "Jsc": 25.8, "Voc": 1.18, "FF": 0.84, "PCE": 25.6,
     "ref": "Jiang et al., Nature (2024)", "year": 2024},
]
