"""
utils/experimental_benchmark.py
================================
Runs the DD solver against fabricated, measured devices from the peer-reviewed
literature. Complements utils/benchmark.py, which runs the DD solver against
other groups' SCAPS outputs.

Expected error is higher here, for a real reason:
    - A 1-D DD solver cannot capture grain boundaries, pinholes, lateral
      inhomogeneity, scalable-area losses, or encapsulation effects.
    - The goal is to show the tool lands in the right neighborhood of
      experimental numbers, not that it matches them exactly.

Usage:
    from utils.experimental_benchmark import run_experimental_benchmark
    results, summary = run_experimental_benchmark(mode="dd")
    print_report(results, summary)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

import numpy as np


@dataclass
class ExperimentalResult:
    benchmark_id: str
    citation: str
    doi: str
    converged: bool

    measured_PCE: float
    predicted_PCE: Optional[float]
    measured_Voc: float
    predicted_Voc: Optional[float]
    measured_Jsc: float
    predicted_Jsc: Optional[float]
    measured_FF: float
    predicted_FF: Optional[float]

    error_PCE_pct: Optional[float]
    error_Voc_pct: Optional[float]
    error_Jsc_pct: Optional[float]
    error_FF_pct: Optional[float]

    accept_thresholds: Dict[str, float] = field(default_factory=dict)
    passed: bool = False

    runtime_s: float = 0.0
    failure_reason: Optional[str] = None
    known_challenges: List[str] = field(default_factory=list)


def _load_benchmarks() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "experimental_benchmarks.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_one(bm: dict, mode: str) -> ExperimentalResult:
    """Run a single benchmark with zero tuning."""
    # Import here so the module can be imported without the full physics
    # stack present (useful for testing).
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import simulate_iv_curve

    t0 = time.time()
    stack = bm["stack"]
    measured = bm["measured"]
    thresholds = bm.get("accept_threshold_pct", {"PCE": 25, "Voc": 10, "Jsc": 20, "FF": 15})
    challenges = bm.get("known_challenges_for_1D_solver", [])

    try:
        htl_name = stack["htl"]
        abs_name = stack["absorber"]
        etl_name = stack["etl"]
        if htl_name not in HTL_DB:
            raise KeyError(f"HTL '{htl_name}' missing from DB")
        if abs_name not in PEROVSKITE_DB:
            raise KeyError(f"Absorber '{abs_name}' missing from DB")
        if etl_name not in ETL_DB:
            raise KeyError(f"ETL '{etl_name}' missing from DB")

        r = simulate_iv_curve(
            HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
            stack["d_htl_nm"], stack["d_abs_nm"], stack["d_etl_nm"],
            stack.get("Nt_cm3", 1e14), stack.get("T_K", 300), mode=mode,
        )

        if mode == "dd" and "converged_flags" in r:
            converged = r.get("n_converged", 0) >= 0.8 * len(r.get("voltages", [1]))
        else:
            converged = True

        pred_PCE = r["PCE"]
        pred_Voc = r["Voc"]
        pred_Jsc = r["Jsc"]
        pred_FF = r["FF"]

        def pct_err(pred, ref):
            return 100 * abs(pred - ref) / max(abs(ref), 1e-9)

        m_PCE = measured["PCE_percent"]
        m_Voc = measured["Voc_V"]
        m_Jsc = measured["Jsc_mA_cm2"]
        m_FF = measured["FF_fraction"]

        e_PCE = pct_err(pred_PCE, m_PCE)
        e_Voc = pct_err(pred_Voc, m_Voc)
        e_Jsc = pct_err(pred_Jsc, m_Jsc)
        e_FF = pct_err(pred_FF, m_FF)

        passed = (
            e_PCE <= thresholds.get("PCE", 25)
            and e_Voc <= thresholds.get("Voc", 10)
            and e_Jsc <= thresholds.get("Jsc", 20)
            and e_FF <= thresholds.get("FF", 15)
        )

        return ExperimentalResult(
            benchmark_id=bm["id"],
            citation=bm["citation"],
            doi=bm["doi"],
            converged=converged,
            measured_PCE=m_PCE, predicted_PCE=pred_PCE,
            measured_Voc=m_Voc, predicted_Voc=pred_Voc,
            measured_Jsc=m_Jsc, predicted_Jsc=pred_Jsc,
            measured_FF=m_FF, predicted_FF=pred_FF,
            error_PCE_pct=e_PCE, error_Voc_pct=e_Voc,
            error_Jsc_pct=e_Jsc, error_FF_pct=e_FF,
            accept_thresholds=thresholds,
            passed=passed,
            runtime_s=time.time() - t0,
            known_challenges=challenges,
        )

    except Exception as e:
        return ExperimentalResult(
            benchmark_id=bm["id"],
            citation=bm["citation"],
            doi=bm["doi"],
            converged=False,
            measured_PCE=measured.get("PCE_percent", 0),
            predicted_PCE=None,
            measured_Voc=measured.get("Voc_V", 0),
            predicted_Voc=None,
            measured_Jsc=measured.get("Jsc_mA_cm2", 0),
            predicted_Jsc=None,
            measured_FF=measured.get("FF_fraction", 0),
            predicted_FF=None,
            error_PCE_pct=None, error_Voc_pct=None,
            error_Jsc_pct=None, error_FF_pct=None,
            accept_thresholds=thresholds,
            passed=False,
            runtime_s=time.time() - t0,
            failure_reason=f"{type(e).__name__}: {e}",
            known_challenges=challenges,
        )


def run_experimental_benchmark(mode: str = "dd") -> tuple:
    """Run the full experimental benchmark suite.

    Returns (results, summary) where summary includes error distributions
    and pass rates.
    """
    db = _load_benchmarks()
    benchmarks = db.get("benchmarks", [])
    results = [_run_one(bm, mode=mode) for bm in benchmarks]

    converged = [r for r in results if r.converged]
    passed = [r for r in converged if r.passed]

    def stats(arr):
        if not arr:
            return {"mean": 0, "median": 0, "max": 0, "p75": 0}
        arr = np.asarray(arr, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "p75": float(np.percentile(arr, 75)),
        }

    summary = {
        "mode": mode,
        "n_total": len(results),
        "n_converged": len(converged),
        "n_passed": len(passed),
        "convergence_rate_pct": 100.0 * len(converged) / max(len(results), 1),
        "pass_rate_pct": 100.0 * len(passed) / max(len(results), 1),
        "error_stats": {
            "PCE_pct": stats([r.error_PCE_pct for r in converged]),
            "Voc_pct": stats([r.error_Voc_pct for r in converged]),
            "Jsc_pct": stats([r.error_Jsc_pct for r in converged]),
            "FF_pct":  stats([r.error_FF_pct  for r in converged]),
        },
        "failures": [
            {"id": r.benchmark_id, "reason": r.failure_reason}
            for r in results if r.failure_reason
        ],
    }
    return results, summary


def format_markdown_report(results, summary) -> str:
    """Produce the experimental section for BENCHMARK_REPORT.md."""
    lines = []
    lines.append("## Experimental validation (REAL measured devices)")
    lines.append("")
    lines.append(f"Mode: `{summary['mode']}`  "
                 f"Converged {summary['n_converged']}/{summary['n_total']}  "
                 f"Passed {summary['n_passed']}/{summary['n_total']}")
    lines.append("")
    lines.append("| Device | Ref PCE | Tool PCE | PCE err% | Voc err% | Jsc err% | FF err% | Pass |")
    lines.append("|--------|---------|----------|----------|----------|----------|---------|------|")
    for r in results:
        if r.converged and r.predicted_PCE is not None:
            lines.append(
                f"| {r.benchmark_id} | {r.measured_PCE:.1f} | {r.predicted_PCE:.1f} "
                f"| {r.error_PCE_pct:.1f} | {r.error_Voc_pct:.1f} "
                f"| {r.error_Jsc_pct:.1f} | {r.error_FF_pct:.1f} "
                f"| {'yes' if r.passed else 'no'} |"
            )
        else:
            lines.append(f"| {r.benchmark_id} | {r.measured_PCE:.1f} | FAIL | - | - | - | - | no |")
    lines.append("")
    es = summary["error_stats"]
    lines.append("Summary (converged devices only):")
    lines.append(f"- PCE error median {es['PCE_pct']['median']:.1f}%, max {es['PCE_pct']['max']:.1f}%")
    lines.append(f"- Voc error median {es['Voc_pct']['median']:.1f}%")
    lines.append(f"- Jsc error median {es['Jsc_pct']['median']:.1f}%")
    lines.append(f"- FF  error median {es['FF_pct']['median']:.1f}%")
    lines.append("")
    lines.append("### Honest interpretation")
    lines.append("")
    lines.append(
        "These are measured devices, not SCAPS simulations. A 1-D drift-diffusion "
        "solver cannot capture grain boundaries, pinholes, or area-scaling losses. "
        "Errors above 20% on PCE for some devices are physically expected, not a "
        "solver bug. The `known_challenges` field on each benchmark entry explains "
        "the per-device caveats."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    results, summary = run_experimental_benchmark(mode="dd")
    print(format_markdown_report(results, summary))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "..", "EXPERIMENTAL_BENCHMARK_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(format_markdown_report(results, summary))
    # Machine-readable dump
    json_path = os.path.join(out_dir, "..", "experimental_benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": [asdict(r) for r in results]},
            f, indent=2, default=str,
        )
    print(f"\nFull report written to {out_path} and {json_path}.")
