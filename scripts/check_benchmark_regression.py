"""
scripts/check_benchmark_regression.py
======================================
Hard regression gate. Runs the SCAPS-reference benchmark and fails with
exit code 1 if any of:

  - mean PCE error > 12%
  - convergence rate < 90%
  - worst-case PCE error > 30%

Invoked from .github/workflows/ci.yml on every push / PR.

Rationale: this tool can drift silently as material parameters or solver
heuristics get tweaked. The only way to prevent that is to fail the build
when accuracy regresses, and make the committer fix it before merging.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Thresholds — tune with care
MEAN_ERROR_THRESHOLD  = 12.0    # percent
WORST_ERROR_THRESHOLD = 30.0    # percent
CONV_THRESHOLD         = 0.90    # fraction


def run_benchmarks() -> dict:
    """Run the benchmark suite and return summary stats."""
    from utils.benchmark import run_all_scaps_benchmarks
    results = run_all_scaps_benchmarks(quick=True)  # quick=True for CI speed
    return results


def main() -> int:
    print("=" * 60)
    print("Benchmark regression gate")
    print("=" * 60)

    try:
        summary = run_benchmarks()
    except Exception as e:
        # If benchmarks can't run at all, fail loudly
        print(f"ERROR: benchmark runner crashed: {e}")
        return 2

    mean_err  = summary.get("mean_pce_error_pct", 999.0)
    worst_err = summary.get("worst_pce_error_pct", 999.0)
    conv_rate = summary.get("convergence_rate", 0.0)

    print(f"  mean PCE error     : {mean_err:5.2f}%   (threshold {MEAN_ERROR_THRESHOLD}%)")
    print(f"  worst PCE error    : {worst_err:5.2f}%   (threshold {WORST_ERROR_THRESHOLD}%)")
    print(f"  convergence rate   : {conv_rate*100:5.1f}%   (threshold {CONV_THRESHOLD*100}%)")
    print()

    failed = False
    if mean_err > MEAN_ERROR_THRESHOLD:
        print(f"FAIL: mean PCE error {mean_err:.2f}% exceeds threshold")
        failed = True
    if worst_err > WORST_ERROR_THRESHOLD:
        print(f"FAIL: worst PCE error {worst_err:.2f}% exceeds threshold")
        failed = True
    if conv_rate < CONV_THRESHOLD:
        print(f"FAIL: convergence rate {conv_rate*100:.1f}% below threshold")
        failed = True

    if failed:
        print("\nREGRESSION DETECTED — commit rejected.")
        print("If this is expected, update the thresholds in this file with")
        print("a commit message explaining why.")
        return 1

    print("PASS — no regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
