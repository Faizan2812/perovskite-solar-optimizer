"""
scripts/run_inverse.py
=======================
Inverse design: given target PCE / Voc / Jsc / FF, find device parameters
(thicknesses, doping, defect density) that produce them.

Uses scipy.optimize.differential_evolution on the fast-simulator objective.

Usage:
    python scripts/run_inverse.py --target-pce 24 --target-voc 1.15
    python scripts/run_inverse.py --target-pce 26 --absorber FAPbI3 --etl SnO2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-pce", type=float, default=24.0, help="Target PCE (%)")
    ap.add_argument("--target-voc", type=float, default=None,  help="Target Voc (V)")
    ap.add_argument("--target-jsc", type=float, default=None,  help="Target Jsc (mA/cm²)")
    ap.add_argument("--target-ff",  type=float, default=None,  help="Target FF (fraction 0-1)")
    ap.add_argument("--htl",      type=str, default="Spiro-OMeTAD")
    ap.add_argument("--absorber", type=str, default="MAPbI3")
    ap.add_argument("--etl",      type=str, default="TiO2")
    ap.add_argument("--maxiter",  type=int, default=40)
    ap.add_argument("--out",      type=str, default="inverse_result.json")
    args = ap.parse_args()

    from scipy.optimize import differential_evolution
    from physics.device    import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    htl_mat = HTL_DB[args.htl]
    abs_mat = PEROVSKITE_DB[args.absorber]
    etl_mat = ETL_DB[args.etl]

    print(f"Inverse design — stack: {args.htl} / {args.absorber} / {args.etl}")
    print(f"  Target PCE: {args.target_pce}%")
    if args.target_voc: print(f"  Target Voc: {args.target_voc} V")
    if args.target_jsc: print(f"  Target Jsc: {args.target_jsc} mA/cm²")
    if args.target_ff:  print(f"  Target FF:  {args.target_ff}")

    # Search-space bounds:  d_htl, d_abs, d_etl, log10(Nt)
    bounds = [
        (50,  500),    # d_htl_nm
        (100, 900),    # d_abs_nm
        (20,  500),    # d_etl_nm
        (13.0, 17.0),  # log10(Nt_abs)
    ]

    def objective(x):
        d_htl, d_abs, d_etl, log_Nt = x
        try:
            r = fast_simulate(htl_mat, abs_mat, etl_mat,
                              d_htl_nm=d_htl, d_abs_nm=d_abs, d_etl_nm=d_etl,
                              Nt_abs=10**log_Nt, T=300)
            loss = (r["PCE"] - args.target_pce)**2
            if args.target_voc: loss += 100 * (r["Voc"] - args.target_voc)**2
            if args.target_jsc: loss += (r["Jsc"] - args.target_jsc)**2
            if args.target_ff:  loss += 100 * (r["FF"]  - args.target_ff)**2
            return loss
        except Exception:
            return 1e6

    result = differential_evolution(
        objective, bounds,
        maxiter=args.maxiter,
        popsize=15, tol=1e-4, seed=0,
        polish=True,
    )

    d_htl, d_abs, d_etl, log_Nt = result.x
    Nt = 10**log_Nt

    # Evaluate final
    r = fast_simulate(htl_mat, abs_mat, etl_mat,
                       d_htl_nm=d_htl, d_abs_nm=d_abs, d_etl_nm=d_etl,
                       Nt_abs=Nt, T=300)

    out = {
        "target": {
            "PCE": args.target_pce,
            "Voc": args.target_voc, "Jsc": args.target_jsc, "FF": args.target_ff,
        },
        "stack": {"htl": args.htl, "absorber": args.absorber, "etl": args.etl},
        "found_parameters": {
            "d_htl_nm": float(d_htl),
            "d_abs_nm": float(d_abs),
            "d_etl_nm": float(d_etl),
            "Nt_abs":   float(Nt),
        },
        "predicted_metrics": {
            "PCE": r["PCE"], "Voc": r["Voc"],
            "Jsc": r["Jsc"], "FF": r["FF"],
        },
        "optimizer_final_loss": float(result.fun),
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 50)
    print(f"  Result:  PCE = {r['PCE']:.2f}%   Voc = {r['Voc']:.3f} V")
    print(f"           Jsc = {r['Jsc']:.2f} mA/cm²   FF = {r['FF']*100:.1f}%")
    print(f"  Params:  {d_htl:.0f}/{d_abs:.0f}/{d_etl:.0f} nm,  Nt = {Nt:.2e}")
    print("=" * 50)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
