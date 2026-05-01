"""
scripts/run_nsga.py
====================
NSGA-II multi-objective optimization. Typical use: Pareto front of
PCE vs T80 stability, or PCE vs fabrication cost.

Usage:
    python scripts/run_nsga.py --pop 100 --gens 50
    python scripts/run_nsga.py --pop 60 --gens 30 --objectives pce,t80
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
    ap.add_argument("--pop",  type=int, default=100,  help="Population size")
    ap.add_argument("--gens", type=int, default=50,   help="Number of generations")
    ap.add_argument("--objectives", type=str, default="pce,t80",
                     help="Comma-separated objective list (pce, t80, cost, voc)")
    ap.add_argument("--out", type=str, default="pareto_front.json")
    args = ap.parse_args()

    try:
        from ai.optimizer import NSGAIIOptimizer
    except ImportError:
        # Optimizer module might expose it differently; try the alternate name
        try:
            import ai.optimizer as opt
            NSGAIIOptimizer = getattr(opt, "NSGAIIOptimizer",
                                       getattr(opt, "nsga2", None))
            if NSGAIIOptimizer is None:
                print("NSGA-II entry point not found in ai/optimizer.py.")
                print("Inspect ai/optimizer.py for available names.")
                return 1
        except ImportError as e:
            print(f"ERROR: {e}")
            return 1

    print(f"NSGA-II — pop={args.pop}, gens={args.gens}, "
          f"objectives={args.objectives}")
    objectives = args.objectives.split(",")

    # Route to the optimizer
    result = NSGAIIOptimizer(
        population_size=args.pop,
        n_generations=args.gens,
        objectives=objectives,
    ).run()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
