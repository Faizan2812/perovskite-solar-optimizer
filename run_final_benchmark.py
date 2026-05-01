"""
run_final_benchmark.py
=======================
One-command validation: runs both benchmark suites back-to-back and prints
a unified summary.

Usage:
    python run_final_benchmark.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> int:
    print()
    print("=" * 70)
    print(f"Running: {script}")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=ROOT,
    )
    return result.returncode


def main() -> int:
    t0 = time.time()

    rc1 = run("run_benchmark.py")
    rc2 = run("run_experimental_benchmark.py")

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("Final benchmark summary")
    print("=" * 70)
    print(f"  Total runtime      : {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"  SCAPS-reference    : {'PASS' if rc1 == 0 else 'FAIL'}")
    print(f"  Experimental       : {'PASS' if rc2 == 0 else 'FAIL'}")
    print(f"  Reports written    : BENCHMARK_REPORT.md, EXPERIMENTAL_BENCHMARK_REPORT.md")
    print("=" * 70)

    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
