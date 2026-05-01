# Validation Guide

How to interpret the two benchmark suites this tool ships with, and how to
use the results responsibly in your own work.

---

## Two different benchmarks, two different questions

This tool ships with two independent validation suites that answer two
different questions:

| Suite | Question it answers | Expected errors |
|---|---|---|
| **SCAPS-reference** (`BENCHMARK_REPORT.md`) | "Does my DD solver agree with SCAPS-1D's DD solver?" | 5-12% median PCE |
| **Experimental** (`EXPERIMENTAL_BENCHMARK_REPORT.md`) | "Do either of our 1D DD solvers agree with what happens in a real lab?" | 15-30% median PCE |

Both matter. One is a numerical consistency check. The other is a reality
check.

---

## 1. SCAPS-reference benchmark

### What it tests

For 10 device configurations from 4 peer-reviewed SCAPS-1D papers, we:
1. Build the same stack with the same material parameters
2. Run our DD solver
3. Compare PCE, Voc, Jsc, FF to the numbers reported in the paper

### What "agreement" means here

Both SCAPS and our solver solve the same PDEs (Poisson + continuity +
SRH/radiative/Auger recombination). If both are implemented correctly,
they should produce PCE values within a few percent of each other for the
same inputs.

Residual errors come from:
- Different Scharfetter-Gummel discretization variants
- Different Newton-solver step-size heuristics
- Different spectral sampling for photon flux
- Small differences in thermal-voltage or ni computation

These are numerical artifacts, not physics differences. Our tool's median
error of 8-9% on this suite is in the normal range for independently
implemented 1D DD solvers.

### What does NOT a good SCAPS-reference error tell you

It does NOT tell you that either tool matches a real experimental
measurement. That's the second benchmark's job.

---

## 2. Experimental benchmark

### What it tests

For 5 fabricated, measured perovskite cells (including 2 NREL-certified
champion cells), we:
1. Build the device stack as reported in the paper
2. Run our DD solver with measured or reported material parameters
3. Compare to the measured J-V metrics

### Why errors are larger here

A 1D DD solver is a physics model, not a clone of a real lab. Expected
sources of systematic error:

- **2D/3D effects (not in the model)**: grain boundaries, pinholes,
  lateral field non-uniformity, shunt paths
- **Area scaling**: reported champion cells are 0.1 cm² or smaller; module
  values are different again
- **Interface quality**: real HTL/absorber interfaces have fabrication
  variation we can't capture with a single `Nt_if` number
- **Measurement conditions**: spectral mismatch factors, hysteresis
  direction, scan rate, pre-conditioning
- **Parasitic resistance**: series and shunt resistance often understated
  in the literature values we use

A 15-30% error against the certified PCE is NOT a failure of the physics.
Anything below ~10% against a certified cell using a 1D model would be
suspicious.

### How to use this suite

- As a sanity check before publication: your solver should produce PCE
  values within the same ballpark as real devices made from the same
  materials
- As an uncertainty quantifier: if the experimental benchmark shows 25%
  median error, you should not claim 1% precision on your own device
  simulations regardless of how well they match SCAPS
- As a way to debug: if your solver disagrees with SCAPS by 5% AND with
  experiment by 50%, the problem is in the physics model, not the
  numerics

---

## Interpreting per-device results

### Hossain 2022 (reference papers [10] in paper, D1/D2 in benchmark)

The original paper used SCAPS-1D to scan different HTL/ETL combinations
for MAPbI₃ cells. PCE values in the 23-25% range. Our solver should agree
to within 1-2% absolute PCE.

### Chabri 2023 (D3 in benchmark, lead-free Cs₂SnI₆)

Lead-free chalcogenide. Known hard case because the mobility is very
asymmetric (53 cm²/Vs electrons, 0.03 cm²/Vs holes) and this produces a
sharp J-V knee that 1D DD solvers often under-resolve. Expect 2-5% error.

### Saliba 2016 / Jeon 2018 / Kim 2019 (experimental suite)

NREL-certified 21.1% / 22.6% / 23.7% cells. Our model will get
within 15-25% relative error. That is acceptable.

### Wang 2019 (experimental, lead-free)

Lead-free Sn-based device. Substantially lower PCE (6.36%) because of
high hole doping from Sn²⁺ → Sn⁴⁺ oxidation. Hard to match precisely;
15-30% error is acceptable.

---

## When to re-run the benchmark

Run `python run_final_benchmark.py` whenever you:

- Modify `physics/dd_solver.py` or `physics/device.py`
- Edit any numerical value in `data/materials_database.json`
- Change the spectrum file in `physics/_astm_g173_data.py`
- Tune any solver tolerance or step-size heuristic

The CI runs both suites on every PR. If mean PCE error climbs above 12%
or worst-case above 30%, the merge is blocked.

---

## When NOT to trust a benchmark

- When the benchmark passes but the validation suite was quick-mode
  (`--quick`). Quick mode uses only 3 of 10 devices. Always run the full
  suite before claiming results.
- When you just added a new material with LOW-confidence parameters.
  Re-source the parameters first.
- When the specific device type you care about is NOT in the benchmark.
  For example, neither benchmark contains 2D/Ruddlesden-Popper perovskites
  or formamidinium-cesium mixes. If that's your target, add benchmark
  entries before trusting PCE numbers from this tool.

---

## Citing validation results

When using this tool in a publication, you should:

1. Report mean and worst-case PCE error from the SCAPS-reference suite
2. Report median PCE error from the experimental suite
3. Cite the original benchmark references (all 9 papers are in the
   References of BENCHMARK_REPORT.md and EXPERIMENTAL_BENCHMARK_REPORT.md)
4. If you made any custom modifications to the solver, re-run both suites
   and report the new numbers

Do not report only the best-case device. Readers will spot it.
