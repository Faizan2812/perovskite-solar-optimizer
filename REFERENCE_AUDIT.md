# Reference Audit Report

*Generated: 2026-04-28 08:14:43 UTC*

Audit of every reference cited in `data/materials_database.json`.
Every reference must have a verifiable DOI that resolves via doi.org.

## Summary

- **Active references**: 29
- **DOI format valid**: 29
- **Online resolves**: 0 (only checked with --network)
- **Failures**: 0

## Per-reference status

| ID | Citation | DOI | Format | Online | Issues |
|---|---|---|---|---|---|
| araujo_2025_rsc_sust | Araújo V.H.D., Nogueira A.F., Tristão J.C., dos Santos L.J., | 10.1039/D5SU00526D | ✓ |  | — |
| arora_2017_science_CuSCN | N. Arora et al., 'Perovskite solar cells with CuSCN hole ext | 10.1126/science.aam5655 | ✓ |  | — |
| bi_2016_aem_spiro | D. Bi et al., 'Polymer-templated nucleation and crystal grow | 10.1002/aenm.201600457 | ✓ |  | — |
| brivio_2014 | Brivio et al., Phys. Rev. B 89, 155204 (2014) | 10.1103/PhysRevB.89.155204 | ✓ |  | — |
| burschka_2013_nature | J. Burschka et al., 'Sequential deposition as a route to hig | 10.1038/nature12340 | ✓ |  | — |
| chabri_2023_jem | Chabri et al., J. Electron. Mater. 52, 2722 (2023) | 10.1007/s11664-023-10247-5 | ✓ |  | — |
| dequilettes_2015_science | DeQuilettes et al., Science 348, 683 (2015) | 10.1126/science.aaa5333 | ✓ |  | — |
| eperon_2014_ees | Eperon et al., Energy Environ. Sci. 7, 982 (2014) | 10.1039/c3ee43822h | ✓ |  | — |
| green_2014_photonics | Green et al., Nat. Photon. 8, 506 (2014) | 10.1038/nphoton.2014.134 | ✓ |  | — |
| green_2014_photonics_absorption | M. A. Green, A. Ho-Baillie, H. J. Snaith, 'The emergence of  | 10.1038/nphoton.2014.134 | ✓ |  | — |
| hossain_2022_acs_omega | Hossain et al., ACS Omega 7, 43210 (2022) | 10.1021/acsomega.2c05912 | ✓ |  | — |
| hou_2020_science_2pacz | Y. Hou et al., 'Efficient tandem solar cells with solution-p | 10.1126/science.aaz3691 | ✓ |  | — |
| jiang_2017_nat_energy | Jiang et al., Nat. Energy 2, 16177 (2017) | 10.1038/nenergy.2016.177 | ✓ |  | — |
| koh_2014_jpcc | T. M. Koh et al., 'Formamidinium-containing metal-halide: an | 10.1021/jp411112k | ✓ |  | — |
| liu_2013_planar_deposition | M. Liu, M. B. Johnston, H. J. Snaith, 'Efficient planar hete | 10.1038/nature12509 | ✓ |  | — |
| minemoto_2015_solmat | Minemoto & Murata, Sol. Energy Mater. Sol. Cells 133, 8 (201 | 10.1016/j.solmat.2014.10.036 | ✓ |  | — |
| ponseca_2014_jacs | C. S. Ponseca Jr et al., 'Organometal halide perovskite sola | 10.1021/ja412583t | ✓ |  | — |
| richter_2016_natcomm | Richter et al., Nat. Commun. 7, 13941 (2016) | 10.1038/ncomms13941 | ✓ |  | — |
| saliba_2016_ees | Saliba M., Matsui T., Seo J.Y. et al., "Cesium-containing tr | 10.1039/c5ee03874j | ✓ |  | — |
| savva_2019_aem_PEDOT | A. Savva et al., 'PEDOT:PSS in perovskite solar cells,' Sola | 10.1002/solr.201900033 | ✓ |  | — |
| shi_2015 | Shi et al., Science 347, 519 (2015) | 10.1126/science.aaa2725 | ✓ |  | — |
| snaith_2013_jpcl | H. J. Snaith et al., 'Anomalous hysteresis in perovskite sol | 10.1021/jz500113x | ✓ |  | — |
| stranks_2013 | Stranks et al., Science 342, 341 (2013) | 10.1126/science.1243982 | ✓ |  | — |
| tiep_2016_aem_ZnO | N. H. Tiep et al., 'Recent advances in improving the stabili | 10.1002/aenm.201501420 | ✓ |  | — |
| wang_2019_acsel_sn | F. Wang et al., 'Stable lead-free tin halide perovskite sola | 10.1021/acsenergylett.8b02281 | ✓ |  | — |
| yang_2015_nat_materials | W. S. Yang et al., 'High-performance photovoltaic perovskite | 10.1126/science.aaa9272 | ✓ |  | — |
| yang_2015_science | Yang W.S., Noh J.H., Jeon N.J. et al., "High-performance pho | 10.1126/science.aaa9272 | ✓ |  | — |
| yoo_2021_nature | J. J. Yoo et al., 'Efficient perovskite solar cells via impr | 10.1038/s41586-021-03285-w | ✓ |  | — |
| zhao_2017_aem | Zhao et al., Adv. Energy Mater. 7, 1700131 (2017) | 10.1002/aenm.201700131 | ✓ |  | — |

## How to fix issues

If a reference fails the audit:

1. Locate the DOI on the publisher's website (e.g. PubMed, RSC, ACS, Wiley).
2. Update `data/materials_database.json`:
   - Set the correct `doi` field
   - Set `verified: true`
   - Set `verified_on: 2026-04-28` (or today's date)
   - Set `url: https://doi.org/<doi>`
3. Re-run this script: `python scripts/verify_references.py --network`
4. If the DOI cannot be verified, REMOVE the reference and migrate any
   parameters that cited it to a verified replacement. See
   `docs/MATERIAL_LIFECYCLE.md` for the procedure.

## Reproducibility

```bash
# Quick offline format check
python scripts/verify_references.py

# Full online verification (hits doi.org)
python scripts/verify_references.py --network

# Strict mode (fail CI on any unresolved DOI)
python scripts/verify_references.py --network --strict
```