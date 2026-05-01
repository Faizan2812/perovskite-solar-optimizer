"""
scripts/run_bayesopt.py
========================
Bayesian optimization of perovskite device parameters using a Gaussian-process
surrogate with Matern 5/2 kernel and Expected Improvement acquisition.

Usage:
    python scripts/run_bayesopt.py --iterations 50 --target pce
    python scripts/run_bayesopt.py --iterations 30 --target voc --fix-absorber MAPbI3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50,
                     help="Number of BO iterations (evaluations after warm start)")
    ap.add_argument("--n-warm", type=int, default=10,
                     help="Initial random evaluations before GP is fit")
    ap.add_argument("--target", choices=["pce", "voc", "jsc", "ff"], default="pce",
                     help="Objective to maximize")
    ap.add_argument("--fix-absorber", type=str, default=None,
                     help="Pin absorber material (e.g. MAPbI3, FAPbI3, Cs2SnI6)")
    ap.add_argument("--out", type=str, default="bayesopt_result.json",
                     help="Output file (JSON)")
    args = ap.parse_args()

    print(f"Bayesian optimization — target: {args.target}, iters: {args.iterations}")

    try:
        from ai.optimizer import bayesian_optimization
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Hint: make sure scikit-learn is installed.")
        return 1

    result = bayesian_optimization(
        target=args.target,
        n_iterations=args.iterations,
        n_warm_start=args.n_warm,
        fixed_absorber=args.fix_absorber,
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"Best {args.target}: {result.get('best_value', 'N/A')}")
    print(f"At parameters: {result.get('best_params', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
