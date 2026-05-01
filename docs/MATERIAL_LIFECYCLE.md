# Material Lifecycle Guide

How to add a new material, update an existing one, or remove one from the
database — while keeping the validation cascade intact.

This is the workflow that makes the tool publication-worthy. Every numerical
change to a material parameter triggers tests and benchmarks; CI blocks
anything that regresses.

---

## The validation cascade

```
  Edit JSON        ┐
  ↓                │
  pytest tests/    │   ← tests block if phantom citations,
  ↓                │     missing provenance, or LOW-confidence budget breach
  run_benchmark    │
  ↓                │   ← regression gate blocks if PCE error worsens > 12%
  Commit + push    │
  ↓                │
  CI re-runs       │   ← on every PR, GitHub Actions runs all of the above
  ↓                ┘
  Merge
```

---

## Adding a new material

### 1. Edit the extension file

All new materials go in `data/materials_database_extension.json`, NOT the
main file. This keeps diffs clean.

Example: adding a new HTL material, let's say PTAA polymer.

```json
{
  "htls": {
    "PTAA": {
      "role": "HTL",
      "chemistry": "poly(triarylamine) polymer",
      "typical_thickness_nm": 40,
      "parameters": {
        "Eg_eV": {
          "value": 2.80,
          "source": "Heo et al., Nat. Photon. 7, 486 (2013)",
          "doi": "10.1038/nphoton.2013.80",
          "method": "optical absorption onset",
          "confidence": "HIGH"
        },
        "chi_eV": {
          "value": 2.30,
          "source": "Heo et al., Nat. Photon. 7, 486 (2013)",
          "doi": "10.1038/nphoton.2013.80",
          "method": "UPS",
          "confidence": "HIGH"
        },
        "mu_p_cm2_Vs": {
          "value": 1.0e-4,
          "source": "Heo et al., Nat. Photon. 7, 486 (2013)",
          "doi": "10.1038/nphoton.2013.80",
          "method": "SCLC",
          "confidence": "MEDIUM"
        },
        "..."
      }
    }
  }
}
```

### 2. Required fields per parameter

Every material parameter must have these four fields:

| Field | Required | Notes |
|---|---|---|
| `value` | yes | The numerical value |
| `source` | yes | Author, journal, volume, pages, year |
| `confidence` | yes | `HIGH`, `MEDIUM`, or `LOW` |
| `doi` | strongly preferred | Needed for reproducibility |
| `method` | preferred | How it was measured |
| `notes` | optional | Temperature, substrate, any caveat |

### 3. Confidence tiers

- **HIGH**: parameter value from an original experimental measurement,
  widely cited (50+ papers or a review). Examples: MAPbI₃ Eg = 1.55 eV,
  Spiro-OMeTAD HOMO = -5.22 eV.

- **MEDIUM**: literature consensus value, but not a single authoritative
  source. Most parameters fall here.

- **LOW**: value used in simulation papers without a clear experimental
  origin, or where sources disagree by more than ~30%. Flag for
  re-sourcing before publication.

### 4. Run the merge script

```bash
python scripts/merge_materials_db.py            # dry run
python scripts/merge_materials_db.py --apply    # actually merge
mv data/materials_database_full.json data/materials_database.json
```

### 5. Run tests

```bash
pytest tests/
```

All 10 `test_materials_loader.py` tests must pass. If any fail, don't merge.
Fix the JSON first.

### 6. Run benchmarks

```bash
python scripts/run_benchmark.py
```

Check `BENCHMARK_REPORT.md`. The regression gate allows mean PCE error up
to 12% and worst-case up to 30%. If a new material or parameter update
pushes things past those, investigate before committing.

### 7. Commit

Commit the JSON change + any test updates with a message showing the
benchmark delta:

```
Add PTAA as HTL option

Sources: Heo et al. Nat. Photon. 7, 486 (2013) - main reference
        Baier et al. JMCA 2017 for mobility cross-check

Benchmark impact: mean PCE err unchanged at 9.1%, no regressions.
```

---

## Updating an existing material

Same workflow, but edit the main database file directly instead of the
extension. The diff will show the old-vs-new values on review.

If you're updating a parameter because a newer authoritative source
contradicts the old one, include the new DOI in the commit message.

---

## Removing a material

Three-step process to keep audit trail intact:

1. In the `_references` block of `materials_database.json`, add a
   tombstone key starting with `_REMOVED_` documenting why the material
   was removed.
2. Delete the material from `htls`, `absorbers`, or `etls`.
3. Update any benchmark that referenced it.

The `_REMOVED_datto_chemnanomat_2026` entry in the current database is an
example — it tracks the phantom reference that was purged during the
initial audit.

---

## What NOT to do

- **Don't add a parameter without provenance**. Tests will fail and CI
  will block.
- **Don't use future-year publications** (2026, 2027, etc.) as sources.
  Tests catch this.
- **Don't cite "personal communication" or "in preparation"**. Tests catch
  this too.
- **Don't bump the LOW-confidence ceiling in tests** without a commit
  message explaining why. The ceiling is there to prevent drift.
- **Don't update a value without updating the source**. If you found a
  new paper with a different number, change both together.

---

## Current database stats

- 20 materials (6 HTLs, 8 absorbers, 6 ETLs)
- 179 parameters with full provenance
- 29 verified references with DOIs
- Confidence distribution: 25 HIGH, 147 MEDIUM, 7 LOW

See the Database tab in the Streamlit app for live inspection.
