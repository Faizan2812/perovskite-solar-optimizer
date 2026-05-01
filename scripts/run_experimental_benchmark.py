"""
scripts/run_experimental_benchmark.py
======================================
Run the 5-device experimental validation suite and write
EXPERIMENTAL_BENCHMARK_REPORT.md.

Validates against real fabricated, measured cells — not SCAPS results.

Expected errors are higher (15-30% on PCE) because a 1D DD solver cannot
capture grain boundaries, pinholes, or area-scaling losses. See
docs/VALIDATION.md for interpretation guidance.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.experimental_benchmark import run_experimental_benchmarks


def write_report(results: dict, outpath: Path) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = [
        "# Experimental Validation Report",
        "",
        f"*Generated: {now}*",
        "",
        "Validation of the drift-diffusion solver against **5 fabricated, "
        "measured devices** from peer-reviewed publications.",
        "",
        "## Summary",
        "",
        f"- **Mean PCE error**: {results.get('mean_pce_error_pct', 0):.2f}%",
        f"- **Median PCE error**: {results.get('median_pce_error_pct', 0):.2f}%",
        f"- **Within 30% of measured**: "
        f"{results.get('n_within_30pct', 0)}/{results.get('n_total', 0)}",
        "",
        "## Per-device results",
        "",
        "| Device | Reference | Certified | Measured PCE | Our PCE | |ΔPCE| |",
        "|---|---|---|---|---|---|",
    ]
    for d in results.get("devices", []):
        cert = "✓" if d.get("certified") else ""
        lines.append(
            f"| {d['id']} | {d['reference']} | {cert} | "
            f"{d['measured_pce']:.2f}% | {d['our_pce']:.2f}% | "
            f"{abs(d['error_pct']):.1f}% |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "A 1D DD solver is fundamentally limited when comparing to real devices:",
        "",
        "- **2D/3D effects**: grain boundaries, pinholes, lateral inhomogeneity",
        "- **Area scaling**: measured cells are ≥ 0.1 cm²; lab-scale ≤ 1 mm²",
        "- **Contact resistance**: series resistance often understated in sims",
        "- **Measurement conditions**: spectral mismatch, hysteresis direction",
        "",
        "Errors of 15-30% against measured cells are expected and not a bug. "
        "Against other groups' **SCAPS-1D simulations** this tool hits 8-10% "
        "median error (see BENCHMARK_REPORT.md).",
        "",
        "## References",
        "",
        "- **Saliba 2016** — Energy Environ. Sci. 9, 1989 (2016). DOI: 10.1039/C5EE03874J",
        "- **Jeon 2018** (certified) — Nat. Energy 3, 682 (2018). DOI: 10.1038/s41560-018-0200-6",
        "- **Liu 2013** — Nature 501, 395 (2013). DOI: 10.1038/nature12509",
        "- **Wang 2019** (lead-free) — ACS Energy Lett. 4, 222 (2019). DOI: 10.1021/acsenergylett.8b02058",
        "- **Kim 2019** (certified) — Joule 3, 2179 (2019). DOI: 10.1016/j.joule.2019.06.014",
    ]

    outpath.write_text("\n".join(lines))
    print(f"Wrote {outpath}")


def main() -> int:
    print("Running experimental benchmark suite...")
    t0 = time.time()
    results = run_experimental_benchmarks()
    print(f"Done in {time.time()-t0:.1f}s")

    outpath = ROOT / "EXPERIMENTAL_BENCHMARK_REPORT.md"
    write_report(results, outpath)

    json_path = ROOT / "validation_report" / "experimental_results.json"
    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
