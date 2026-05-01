"""
scripts/run_benchmark.py
=========================
Run the SCAPS-1D reference benchmark suite and write BENCHMARK_REPORT.md.

Usage:
    python scripts/run_benchmark.py            # full run (10 devices, ~3 min)
    python scripts/run_benchmark.py --quick    # fast subset for CI
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.benchmark import run_all_scaps_benchmarks


def write_report(results: dict, outpath: Path) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = [
        "# SCAPS-Reference Benchmark Report",
        "",
        f"*Generated: {now}*",
        "",
        "Validation of the drift-diffusion solver in this tool against "
        "10 published SCAPS-1D results from 4 peer-reviewed papers.",
        "",
        "## Summary",
        "",
        f"- **Mean PCE error**: {results.get('mean_pce_error_pct', 0):.2f}%",
        f"- **Median PCE error**: {results.get('median_pce_error_pct', 0):.2f}%",
        f"- **Worst-case PCE error**: {results.get('worst_pce_error_pct', 0):.2f}%",
        f"- **Convergence rate**: {results.get('convergence_rate', 0)*100:.1f}%",
        "",
        "## Per-device results",
        "",
        "| Device | Stack | SCAPS PCE | Our PCE | ΔPCE | Converged |",
        "|---|---|---|---|---|---|",
    ]
    for dev in results.get("devices", []):
        lines.append(
            f"| {dev['id']} | {dev['stack']} | "
            f"{dev['scaps_pce']:.2f} | {dev['our_pce']:.2f} | "
            f"{dev['error_pct']:.1f}% | {'yes' if dev['converged'] else 'NO'} |"
        )

    lines += [
        "",
        "## References",
        "",
        "- Hossain et al., ACS Omega 7, 43210 (2022). DOI: 10.1021/acsomega.2c05912",
        "- Chabri et al., J. Electron. Mater. 52, 2722 (2023). DOI: 10.1007/s11664-023-10247-5",
        "- Uddin et al., Next Materials 9, 100980 (2025)",
        "- Oyelade et al., Sci. Rep. 14 (2024)",
        "",
        "## Notes",
        "",
        "The PCE/Voc/Jsc/FF values in the 'SCAPS PCE' column are taken directly "
        "from the cited papers, not fits produced by this tool. Agreement with "
        "other groups' SCAPS output does not validate against experimental "
        "measurement — see `EXPERIMENTAL_BENCHMARK_REPORT.md` for that.",
        "",
        "## Regenerating this report",
        "",
        "```bash",
        "python scripts/run_benchmark.py",
        "```",
    ]

    outpath.write_text("\n".join(lines))
    print(f"Wrote {outpath}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="Run quick subset (3 devices)")
    args = ap.parse_args()

    print(f"Running {'quick' if args.quick else 'full'} SCAPS-reference benchmark...")
    t0 = time.time()
    results = run_all_scaps_benchmarks(quick=args.quick)
    print(f"Done in {time.time()-t0:.1f}s")

    # Write markdown report
    outpath = ROOT / "BENCHMARK_REPORT.md"
    write_report(results, outpath)

    # Also dump raw JSON
    json_path = ROOT / "validation_report" / "benchmark_results.json"
    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
