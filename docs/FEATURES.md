# Features Guide

Tab-by-tab description of what the Streamlit app does live, what returns in
real time, and which code backs each feature.

All results are live-computed. No placeholder tabs. Every plot is an
interactive Plotly chart — hover for values, zoom, export as high-resolution
PNG for publications.

---

## Tab 1: ⚡ Fast Simulate

**Live**: click Run → J-V curve + P-V curve + EQE spectrum in under 50 ms.

**Backing code**: `physics/device.py::fast_simulate`

**What you see**:
- J-V curve with Voc marker
- P-V curve with Vmpp marker
- External Quantum Efficiency (EQE) vs wavelength

**Use for**: rapid screening, building training data for ML surrogates,
first-pass checks before DD.

---

## Tab 2: 🔬 DD Solver

**Live**: click Run → full drift-diffusion J-V curve in 1–3 seconds.

**Backing code**: `physics/dd_solver.py` (875 lines) via `simulate_iv_curve(mode="dd")`

**What you see**:
- J-V curve from Scharfetter-Gummel PDE solver
- PCE, Voc, Jsc, FF extracted metrics

**Supports**: Interface defect layer toggle (High / Default / Low quality).

---

## Tab 3: 🧠 ML Surrogates

**Live**: click Train → 200 samples generated + 7 models trained in ~5 sec.

**Backing code**: scikit-learn (RandomForest, GradientBoosting, SVR, KNN,
BayesianRidge, LinearRegression, MLPRegressor).

**What you see**:
- R² comparison bar chart across all 7 models
- Parity plot (actual vs predicted) for the best model
- Table of R² and MAE per model

**Typical accuracy** (200 training samples): RF and GradientBoost reach
R² ≈ 0.95-0.98 on PCE.

**Honest framing**: these are standard regressors trained on MSE loss. They
are NOT PINNs. The real PINN is in Tab 4.

**For publication**: export the parity plot and R² bar chart for your paper's
ML surrogate section.

---

## Tab 4: ⚛️ PINN (real, autodiff)

**Live**: click Inference → ψ(x), n(x), p(x) profiles across the device in
under 100 ms. Plus the full training loss history from the original 2000-epoch
training run.

**Backing code**: `ai/pinn_real.py` (409 lines), pretrained weights bundled
in `artifacts/model_d1.pt`.

**What you see**:
- Electrostatic potential ψ(x) across the device with layer shading
- Carrier densities n(x), p(x) on log scale, spanning ~20 orders of magnitude
- Training loss history: ℒ_total, ℒ_pde, ℒ_bc, ℒ_data on log scale

**Key engineering choices (documented in `ai/pinn_real.py`)**:
- Log-space outputs for carrier densities
- Fourier feature encoding for spatial coordinate
- Two-stage training: data pretraining → physics fine-tuning
- 74K parameters, 5-layer MLP × 128 units tanh

**For publication**: the ψ/n/p profile plots and training loss curves are
directly exportable for paper Figure 2 and Figure 3.

**To retrain from scratch**: `python scripts/train_pinn.py --device D1`

---

## Tab 5: 🎯 Bayesian Opt

**Live**: click Run → 25 BO iterations with GP surrogate in ~1 second.

**Backing code**: scikit-learn `GaussianProcessRegressor` with Matern 5/2
kernel + custom Expected Improvement acquisition.

**What you see**:
- Convergence plot showing all evaluations + best-so-far trajectory
- Best-found parameters (HTL/absorber/ETL thickness, trap density)
- Best achieved PCE/Voc/Jsc

**For publication**: export the convergence plot to show BO efficiency vs
random search.

---

## Tab 6: 📊 Multi-Objective (NSGA-II)

**Live**: click Run → Pareto front between two competing objectives in
~1-20 seconds depending on population/generations.

**Backing code**: custom NSGA-II implementation with fast non-dominated
sorting.

**What you see**:
- 2D scatter plot: dominated solutions (grey) + Pareto front (red stars)
- Pareto front size (count of non-dominated solutions)

**Typical results**: Pareto front of 20-40 points showing the PCE vs FF (or
other) trade-off.

**For publication**: the Pareto scatter is directly exportable for your
design-space analysis section.

---

## Tab 7: 🔍 Feature Importance

**Live**: click Compute → permutation importance on trained RF in ~3 seconds.

**Backing code**: scikit-learn `permutation_importance`.

**What you see**:
- Horizontal bar chart with error bars (15 permutation repeats)
- Ranked features: usually log(Nt) > absorber thickness > ETL thickness > HTL thickness > T

**Honest framing**: This is permutation importance, not exact Shapley
decomposition. Good enough for ranking, not for additive attribution.

**For publication**: export the bar chart with error bars for your sensitivity
analysis section.

---

## Tab 8: 🔄 Inverse Design

**Live**: click Search → differential evolution finds matching device
parameters in ~2-15 seconds.

**Backing code**: `scipy.optimize.differential_evolution` on the fast
simulator.

**What you see**:
- Table: target vs achieved for PCE, Voc, Jsc, FF
- Convergence plot: loss vs DE iteration (log scale)
- Found device parameters

**What SCAPS-1D doesn't have**: an inverse-design workflow. This is a
genuine capability unique to AI-driven design tools.

**Limitations**: the optimizer finds parameters consistent with the target.
It doesn't tell you whether those parameters are fabricable.

---

## Tab 9: 📐 Benchmarks

**Pre-computed DD results**, displayed with live interactive plots.

**Backing code**: `artifacts/benchmark_results.json`, regenerated by
`scripts/run_benchmark.py` and `scripts/run_experimental_benchmark.py`.

### SCAPS-reference subtab

- **10 devices** from 4 peer-reviewed SCAPS-1D papers
- **Mean PCE error**: 9.1%
- **Convergence**: 100%

**What you see**:
- Summary metrics (mean, median, worst-case error + convergence rate)
- Per-device table
- Parity plot: our DD vs SCAPS PCE (all 10 points labeled)
- Per-device error histogram

### Experimental subtab

- **5 fabricated cells** from Saliba 2016, Jeon 2018, Liu 2013, Wang 2019,
  Kim 2019 (2 NREL-certified)
- **Mean PCE error**: 19.3% — expected because 1D DD can't capture 2D/3D
  effects
- **All 5 within 30%** of measured values

**What you see**:
- Summary metrics
- Per-device table with certified flag
- Parity plot showing certified (red star) vs reported (blue)
- Per-device error bars

**For publication**: both parity plots are the clearest visualization of
the tool's validation and go directly into your paper's validation section.

---

## Tab 10: 📋 Database

**Live**: interactive material database inspection.

**Backing code**: `physics/materials_loader.py` with JSON-backed provenance.

**What you see**:
- Summary: 20 materials, HIGH/MEDIUM/LOW confidence counts
- Per-material table: every parameter shows value, confidence, source, DOI,
  measurement method

**The novelty vs SCAPS-1D**: SCAPS ships with material files listing parameter
values but not origins. This tool carries the DOI for every number. Reviewers
can audit where each value came from.

**Current stats**:
- 20 materials (6 HTLs, 8 absorbers, 6 ETLs)
- 179 parameters, 29 verified references
- 25 HIGH + 147 MEDIUM + 7 LOW confidence
- 7 LOW-confidence flagged for re-sourcing before publication

See `docs/MATERIAL_LIFECYCLE.md` for how to add/update materials safely.

---

## Timing summary

Every live action completes in seconds, not minutes:

| Tab | Action | Typical time |
|---|---|---|
| Fast Simulate | J-V + EQE | < 50 ms |
| DD Solver | Full DD J-V | 1-3 s |
| ML Surrogates | Train 7 models on 200 samples | ~5 s |
| PINN | Inference on 200 points | < 100 ms |
| Bayesian Opt | 25 iterations | ~1 s |
| NSGA-II | 50 pop × 25 gens | ~20 s |
| Feature Importance | 200 samples + perm importance | ~3 s |
| Inverse Design | DE, 40 iterations | ~10 s |
| Benchmarks | Load + render | < 200 ms |
| Database | Inspect material | < 50 ms |

Everything fits inside Streamlit Cloud's request timeout.

---

## Publication-ready figures

Every Plotly chart in the app has an export button (camera icon in top-right
when hovering over a chart). Exports as PNG at high DPI suitable for papers
and theses.

For even higher quality, matplotlib-based versions of the same figures are
in `scripts/make_figures.py` (generates PDF and 300-DPI PNG for figures 1-4).
