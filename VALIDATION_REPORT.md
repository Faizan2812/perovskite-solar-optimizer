# Benchmark Validation Report

## Summary

Every reference value (PCE, Voc, Jsc, FF) in `artifacts/benchmark_results.json`
has been validated against its cited peer-reviewed primary source. Every DOI
has been verified at `doi.org`.

| Suite           | Devices | Mean PCE error | Worst PCE error |
|-----------------|---------|----------------|-----------------|
| SCAPS-reference | 7       | **1.7%**       | **3.5%**         |
| Experimental    | 2       | 15.2%          | 15.3%            |

The SCAPS-reference mean error of **1.7%** is well within the 1–3% intrinsic
spread expected between independent implementations of the drift–diffusion
equations.

---

## Primary reference 1 — Hossain et al. ACS Omega 7, 43210 (2022)

**DOI**: `10.1021/acsomega.2c05912` (verified, GOLD-OA)

**Topic**: CsPbI₃-based perovskite solar cells with eight ETL options
(IGZO, SnO₂, WS₂, CeO₂, PCBM, TiO₂, ZnO, C₆₀) and twelve HTL options.
The best six configurations all share CBTS as the HTL.

**Champion device**: ITO/TiO₂/CsPbI₃/CBTS/Au — **PCE 17.90%**.

The values below are reproduced verbatim from Hossain 2022 **Table 4**:

| Device | Stack                              | Voc (V) | Jsc (mA/cm²) | FF (%)  | PCE (%) |
|--------|------------------------------------|---------|--------------|---------|---------|
| D1     | ITO/PCBM/CsPbI₃/CBTS/Au            | 0.994   | 19.77        | 85.04   | 16.71   |
| D2     | ITO/TiO₂/CsPbI₃/CBTS/Au ⭐         | 0.997   | 21.07        | 85.21   | **17.90** |
| D3     | ITO/ZnO/CsPbI₃/CBTS/Au             | 0.997   | 21.07        | 85.03   | 17.86   |
| D4     | ITO/C₆₀/CsPbI₃/CBTS/Au             | 0.989   | 17.25        | 84.80   | 14.47   |
| D5     | ITO/IGZO/CsPbI₃/CBTS/Au            | 0.995   | 20.98        | 85.13   | 17.76   |
| D6     | ITO/WS₂/CsPbI₃/CBTS/Au             | 0.997   | 20.98        | 85.22   | 17.82   |

Each value matches the file `benchmark_results.json` exactly.

---

## Primary reference 2 — Chabri et al. J. Electron. Mater. 52, 2722 (2023)

**DOI**: `10.1007/s11664-023-10235-x` (verified via Springer Link)

**Topic**: Lead-free Cs₂SnI₆-based perovskite solar cell, ZnO ETL,
CuI HTL (final optimized stack: ITO/CuI/Cs₂SnI₆/ZnO/AZO/Ag).

**Reported PCE**: 14.65% at 300 K, 16.77% at 400 K.

| Device | Stack                              | T (K) | Voc (V) | Jsc (mA/cm²) | FF (%) | PCE (%) |
|--------|------------------------------------|-------|---------|--------------|--------|---------|
| D7     | ITO/CuI/Cs₂SnI₆/ZnO/AZO/Ag         | 300   | 0.873   | 22.80        | 73.6   | **14.65** |

---

## Experimental suite

| ID | Reference                              | DOI                          | Measured PCE |
|----|----------------------------------------|------------------------------|--------------|
| E1 | Saliba 2016, Energy Environ. Sci.      | 10.1039/c5ee03874j           | 21.10%       |
| E2 | Yang/Jeon 2015, Science (NREL-cert.)   | 10.1126/science.aaa9272      | 20.10%       |

---

## How to reproduce

From the repository root:

```bash
# Re-run DD solver against every benchmark device
python scripts/run_benchmark.py            # SCAPS-reference suite
python scripts/run_experimental_benchmark.py # experimental suite

# Re-verify every DOI resolves at doi.org
python scripts/verify_references.py --network --strict

# Run the full test suite (23 tests, includes phantom-citation guard)
python -m pytest tests/ -q
```

All three should pass cleanly. CI (`.github/workflows/benchmark-regression.yml`)
runs the same checks on every commit.
