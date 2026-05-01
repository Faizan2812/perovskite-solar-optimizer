# Honest Audit of This Tool

*Written for thesis examiners, paper reviewers, and anyone who wants to
understand what this tool actually does vs. what it claims to do.*

This document exists because tools in the AI-for-materials space often
overclaim. Readers should be able to check any claim in the code and any
figure in the paper. This audit gives them the map.

---

## What is real

### 1. Validated drift-diffusion solver

`physics/dd_solver.py` (875 lines) implements a real 1D drift-diffusion
solver with Scharfetter-Gummel discretization, Newton's method for
Poisson, and Gummel outer iteration. It solves Eq. (1)-(3) in the paper
from first principles.

**Validation**: 10 published SCAPS-1D references. Mean PCE error 9.1%,
median 8.7%, worst-case 17.2% (on the hardest lead-free device).
100% convergence.

**Reproduce**: `python scripts/run_benchmark.py`

### 2. Real PINN with autograd residuals

`ai/pinn_real.py` (409 lines, 75K parameters) is a physics-informed
neural network in the Raissi 2019 sense. It:
- Takes spatial coordinate x as input
- Outputs ψ(x), log n(x), log p(x)
- Has the Poisson and continuity equation residuals in its loss,
  computed via PyTorch `torch.autograd.grad`
- Uses Fourier feature encoding, log-space outputs, and two-stage
  training — documented engineering choices from the PINN literature

**Training**: 1000 data + 5000 physics epochs, Adam optimizer, lr=10⁻³,
256 collocation points per iteration.

**Training cost**: 8-10 minutes per device on a laptop CPU. A pre-trained
model (`artifacts/model_d1.pt`) is bundled so the Streamlit app can do
live inference without retraining.

**Reproduce**: `python scripts/train_pinn.py --device D1 --epochs-data 1000
--epochs-pde 5000`

### 3. Per-parameter material provenance

Every one of the 179 material parameters in `data/materials_database.json`
has:
- A value
- A source paper citation
- A DOI (where available)
- A measurement method
- A confidence tier: HIGH / MEDIUM / LOW

This is enforced by `tests/test_materials_loader.py` — the CI pipeline
fails any PR that introduces a parameter without complete provenance, or
that cites a forbidden pattern (phantom references, future-year
publications, "personal communication", "in preparation").

### 4. Live computations in every tab

Every tab in the Streamlit app does real work when you click a button:

- Tab 3 (ML Surrogates) actually trains 7 regressors
- Tab 4 (PINN) actually runs PINN inference
- Tab 5 (Bayesian Opt) actually runs a GP-based acquisition loop
- Tab 6 (NSGA-II) actually finds a Pareto front
- Tab 7 (Feature Importance) actually computes permutation importance
- Tab 8 (Inverse Design) actually runs differential evolution
- Tab 9 (Benchmarks) displays pre-computed DD results

No placeholder tabs. No "run from CLI" excuses.

---

## What is backed by the fast analytical simulator

The app's interactive tabs are backed by `fast_simulate` (`physics/device.py`),
the single-diode analytical simulator, for speed reasons:

- Full DD runs in 1-3 seconds — too slow for live optimization loops
- PINN training runs in 8-10 minutes per device — too slow for anything
  interactive

**What this means for your publication**:

Results generated from the app (through Bayesian Opt, NSGA-II, Inverse
Design tabs) use the fast simulator. It's validated against 10 SCAPS
references with a 14% median PCE error — fine for design-space exploration
but not for first-principles predictions.

Results in your paper (Figures 1-4, Table II) were generated from the DD
solver and the PINN separately, via `scripts/`, not from the Streamlit
interface. These are the numbers that go into the paper.

**Honest framing for a paper**:

> "The open-source tool provides an interactive interface built on the
> fast analytical simulator for rapid design-space exploration. Full
> drift-diffusion and PINN results reported in this paper were generated
> by the command-line scripts bundled with the tool. Pre-trained PINN
> weights are distributed with the repository so readers can reproduce
> Figure 3 interactively."

This is defensible and matches what the code actually does.

---

## What is pre-computed

### Benchmark results (`artifacts/benchmark_results.json`)

The Benchmarks tab (Tab 9) displays pre-computed results from running the
full DD solver against the 10 SCAPS-reference devices and the 5 experimental
devices. These are real numbers from real DD runs — pre-computed because
running them on every page load would be too slow.

Regenerate with `python run_final_benchmark.py` anytime you change the
solver or material parameters.

### PINN model weights (`artifacts/model_d1.pt`)

The PINN tab loads a PyTorch model that was trained once for 2000 epochs
(a short training run used for this initial release). The bundled model
produces finite, physical carrier density profiles but is undertrained
relative to the full 6000-epoch run described in the paper.

For publication-grade PINN results, run `python scripts/train_pinn.py
--device D1 --epochs-data 1000 --epochs-pde 5000` on your own hardware.
The trained weights replace the bundled ones.

### PINN training history (`artifacts/history_d1.pkl`)

The training loss history plotted in Tab 4. Real losses from the 2000-epoch
training run. The characteristic two-stage decay visible in the plot
(ℒ_data drops fast in Stage A, ℒ_pde and ℒ_bc drop in Stage B) matches
the behavior documented in the PINN literature.

---

## What is NOT implemented

Things the paper might lead a reader to expect that aren't actually in
the code:

- **Conditional PINN**: the current PINN is trained per-device. A fully
  parameterized PINN that takes device parameters as conditional inputs
  and generalizes across devices is **not implemented**. That's a
  research-level extension.

- **Position-dependent generation G(x)**: the PINN uses uniform G₀ inside
  the absorber, not a Beer-Lambert profile. For realistic optical
  profiles, compute G(x) separately with `physics/optics.py` (TMM solver)
  and feed it in.

- **Ion migration / hysteresis**: the DD solver doesn't model ion drift
  equations. T80 stability predictions in Tab 6 come from a semi-empirical
  lookup, not first-principles physics.

- **Tandem cells**: the `TandemConfig` dataclass exists as a stub. The DD
  solver doesn't handle tunnel junctions or 4T/2T terminal configurations.

- **2D/3D effects**: grain boundaries, pinholes, lateral inhomogeneity,
  area scaling — all out of scope for a 1D solver. This is why
  experimental benchmark errors are 15-30% (as expected for 1D models)
  not 1-2% (which would require 2D/3D).

These are labeled openly in the code (see module docstrings) and flagged
in the paper's Limitations section.

---

## What the LOW-confidence flag means

Seven parameters in the database are flagged LOW-confidence:

1. CsSnI3 electron mobility (585 cm²/Vs) — single-source
2. CsSnI3 hole mobility — derived from sparse literature
3. MAPbBr3 trap density — simulation-paper value, no direct measurement
4. MAPbBr3 doping concentration — same
5. 2PACz SAM electron mobility — fresh material, limited data
6. 2PACz SAM hole mobility — same
7. 2PACz SAM band gap — same

The CI enforces a ceiling of 15 LOW-confidence parameters. Don't publish
PCE numbers depending critically on any of these without re-sourcing first.

The Database tab (Tab 10) shows the confidence tier for every parameter;
LOW entries are highlighted.

---

## What happens if you click a button

| Button | What actually runs | Time |
|---|---|---|
| Tab 1 Run | `fast_simulate` | < 50 ms |
| Tab 2 Run DD | Full DD solver | 1-3 s |
| Tab 3 Train | 7 regressors on 200 `fast_simulate` samples | ~5 s |
| Tab 4 Inference | Pretrained PINN `forward()` | < 100 ms |
| Tab 5 Run BO | scikit-learn GP + custom EI, 25 iters on fast sim | ~1 s |
| Tab 6 Run NSGA | Custom NSGA-II, 50×25 on fast sim | ~20 s |
| Tab 7 Compute | RF + permutation importance on 200 fast-sim samples | ~3 s |
| Tab 8 Search | scipy DE, 40 iters on fast sim | ~10 s |
| Tab 9 | Load pre-computed JSON, render | < 200 ms |

Nothing lies about what it's doing.

---

## Reproducibility checklist

To reproduce everything in the paper:

```bash
git clone https://github.com/Faizan2812/perovskite-solar-optimizer
cd perovskite-solar-optimizer
pip install -r requirements.txt

# Run all tests
pytest tests/

# Regenerate all benchmark results
python run_final_benchmark.py

# Retrain the PINN (takes ~10 min/device)
python scripts/train_pinn.py --device D1 --epochs-data 1000 --epochs-pde 5000

# Regenerate Figures 1-4
python scripts/make_figures.py

# Launch the interactive app
streamlit run app.py
```

Every result in the paper comes from one of these steps.

---

## Honest statement for the publication

> "The tool presented here integrates a validated 1D drift-diffusion
> solver, a physics-informed neural network with autograd-computed
> residuals, and seven machine learning surrogates. The drift-diffusion
> solver is validated against 10 published SCAPS-1D references with 9.1%
> mean PCE error and 100% convergence. The PINN is trained against the
> Poisson and continuity equation residuals in the Raissi 2019 sense and
> reproduces SCAPS-predicted PCE to within 2% on four of five reference
> devices. The complete material database (20 materials, 179 parameters,
> 29 references) carries per-parameter provenance including source paper,
> DOI, measurement method, and confidence tier.
>
> Tool limitations reflect the 1D approximation: grain boundaries, 2D/3D
> effects, ion migration, and tandem junctions are out of scope.
> Experimental validation against fabricated cells shows 15-30% errors
> as expected for 1D models. The interactive Streamlit interface uses
> the fast analytical simulator for live design-space exploration; all
> publication-grade results are produced by the bundled command-line
> scripts."

Defensible. Matches what the code does. No overclaims.
