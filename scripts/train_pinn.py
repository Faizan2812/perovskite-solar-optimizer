"""
scripts/train_pinn.py
======================
Train the physics-informed neural network (ai/pinn_real.py) on a specific
device from the benchmark suite.

Usage:
    python scripts/train_pinn.py --device D1 --epochs-data 1000 --epochs-pde 5000
    python scripts/train_pinn.py --device D3 --lr 5e-4 --lambda-bc 20

Produces:
    checkpoints/pinn_<device>.pt        — trained model weights
    checkpoints/pinn_<device>_hist.json — training history
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Reference device specs matching the 5 benchmark devices in the paper
DEVICE_PRESETS = {
    "D1": dict(   # Spiro / MAPbI3 / TiO2 (500 nm abs)
        L_htl=200e-7, L_abs=500e-7, L_etl=50e-7,
        Eg=1.55, chi=3.93, eps_r=6.5,
        Nc=2.2e18, Nv=1.8e19,
        mu_n=2.0, mu_p=2.0,
        Na=1e16, Nd=0.0, Nt=1e14,
        tau_n=1e-8, tau_p=1e-8,
        G0=2.5e21, T=300.0,
    ),
    "D2": dict(   # Spiro / MAPbI3 / SnO2
        L_htl=200e-7, L_abs=500e-7, L_etl=50e-7,
        Eg=1.55, chi=3.93, eps_r=6.5,
        Nc=2.2e18, Nv=1.8e19, mu_n=2.0, mu_p=2.0,
        Na=1e16, Nd=0.0, Nt=1e14,
        tau_n=1e-8, tau_p=1e-8, G0=2.5e21, T=300.0,
    ),
    "D3": dict(   # Cu2O / Cs2SnI6 / SnO2
        L_htl=150e-7, L_abs=500e-7, L_etl=50e-7,
        Eg=1.60, chi=3.90, eps_r=5.0,
        Nc=1e19, Nv=1e19, mu_n=53.0, mu_p=0.03,
        Na=1e15, Nd=0.0, Nt=1e14,
        tau_n=1e-8, tau_p=1e-8, G0=2.5e21, T=300.0,
    ),
    "D4": dict(   # NiO / FAPbI3 / SnO2
        L_htl=100e-7, L_abs=600e-7, L_etl=50e-7,
        Eg=1.48, chi=3.90, eps_r=6.5,
        Nc=2e18, Nv=2e19, mu_n=2.0, mu_p=2.0,
        Na=1e16, Nd=0.0, Nt=1e14,
        tau_n=1e-8, tau_p=1e-8, G0=2.5e21, T=300.0,
    ),
    "D5": dict(   # PEDOT:PSS / MAPbI3 / C60 (inverted p-i-n)
        L_htl=30e-7, L_abs=400e-7, L_etl=30e-7,
        Eg=1.55, chi=3.93, eps_r=6.5,
        Nc=2.2e18, Nv=1.8e19, mu_n=2.0, mu_p=2.0,
        Na=1e16, Nd=0.0, Nt=1e14,
        tau_n=1e-8, tau_p=1e-8, G0=2.5e21, T=300.0,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=list(DEVICE_PRESETS),
                     default="D1",
                     help="Reference device to train on")
    ap.add_argument("--epochs-data", type=int, default=1000,
                     help="Stage A: data pretraining epochs")
    ap.add_argument("--epochs-pde", type=int, default=5000,
                     help="Stage B: physics fine-tuning epochs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-data", type=float, default=1.0)
    ap.add_argument("--lambda-pde",  type=float, default=1.0)
    ap.add_argument("--lambda-bc",   type=float, default=10.0)
    ap.add_argument("--n-collocation", type=int, default=256)
    args = ap.parse_args()

    from ai.pinn_real import DeviceSpec, train_pinn
    import torch

    torch.manual_seed(0)

    preset = DEVICE_PRESETS[args.device]
    dev = DeviceSpec(**preset)

    print(f"\n=== Training PINN on device {args.device} ===")
    print(f"  L_total  = {dev.L_total*1e7:.0f} nm")
    print(f"  Eg       = {dev.Eg} eV")
    print(f"  ni       = {dev.ni:.3e} /cm^3")
    print(f"  epochs   = {args.epochs_data} data + {args.epochs_pde} physics")
    print()

    t0 = time.time()
    model, history = train_pinn(
        dev,
        n_collocation=args.n_collocation,
        n_epochs_data=args.epochs_data,
        n_epochs_pde=args.epochs_pde,
        lr=args.lr,
        lambda_data=args.lambda_data,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining took {elapsed:.1f}s ({elapsed/60:.2f} min)")

    # Save checkpoint
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / f"pinn_{args.device}.pt"
    hist_path  = ckpt_dir / f"pinn_{args.device}_hist.json"
    torch.save(model.state_dict(), model_path)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {model_path}")
    print(f"Saved {hist_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
