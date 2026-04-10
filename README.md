# PINN Perovskite Solar Cell Optimizer

**Professional AI-driven open-source tool for design and optimization of perovskite solar cells.**

46 literature-validated materials, 10 analysis tabs, TMM optics, ion migration, tandem cells,
Bayesian optimization, PINN surrogate, NSGA-II, SHAP, inverse design, REST API, and free web deployment.
Replaces SCAPS-1D with zero convergence errors.

---

## 22 Implemented Features

### Phase 1: Physics Engine
- 1.1 Drift-diffusion spatial profiles: n(x), p(x), E(x), G(x), R(x), Ec(x), Ev(x)
- 1.2 SRH + radiative + Auger recombination with interface defect layers (IDL)
- 1.3 Graded layer composition: linear, exponential, v-shaped bandgap profiles
- 1.4 Transfer Matrix Method optics with 40 material n,k entries
- 1.5 Fowler-Nordheim tunneling + ion migration hysteresis model

### Phase 2: AI/ML Integration
- 2.1 Bayesian Optimization with GPR (Matern 5/2) + Expected Improvement
- 2.2 NSGA-II multi-objective: PCE vs T80 stability Pareto front
- 2.3 DeepONet surrogate: branch [6-32-16] + trunk [1-32-16] operator network
- 2.4 SHAP permutation-based feature importance
- 2.5 Inverse design: target PCE/Voc/Jsc/FF to required parameters
- 2.6 Active learning with uncertainty sampling
- 2.7 Natural language query interface

### Phase 3: Data and Validation
- 3.1 12 experimental benchmarks from Nature, Science, JACS
- 3.2 46 materials: 15 HTL + 18 absorber + 13 ETL = 3,510 combinations
- 3.3 SCAPS .def file import AND export
- 3.4 Validated against 12 published experimental devices
- 3.5 Uncertainty quantification via MC dropout and GPR posterior

### Phase 4: Publication-Grade Features
- 4.1 Tandem cell simulation: 2T (current-matched) and 4T (independent)
- 4.2 T80 stability prediction model
- 4.3 HTML publication report + TXT summary
- 4.4 5 export formats: J-V CSV, QE CSV, TXT, HTML, SCAPS .def
- 4.5 PyPI packaging (pyproject.toml)
- 4.6 FastAPI REST API: /simulate, /optimize, /query endpoints

---

## Validation

Cu2O / Cs2SnI6 / SnO2 (50/300/50 nm, Nt=1e14, 300K):

| Parameter | This Tool | SCAPS-1D | Error |
|-----------|-----------|----------|-------|
| Jsc       | 24.08     | 24.31    | 1.0%  |
| Voc       | 1.090     | 1.055    | 3.4%  |
| FF        | 87.1%     | 85.6%    | 1.8%  |
| PCE       | 22.85%    | 21.94%   | 4.2%  |

3,510 material combinations: 0 failures. All below SQ limit.

---

## Quick Start

```bash
bash setup_and_run.sh
# or: pip install -r requirements.txt && streamlit run app.py
```

## REST API

```bash
pip install fastapi uvicorn
uvicorn utils.api:app --port 8000
# Swagger docs at http://localhost:8000/docs
```

## Deploy Free

Push to GitHub then deploy at share.streamlit.io

---

## Project Structure

```
app.py                  812 lines   Streamlit GUI (10 tabs)
physics/device.py       720 lines   DD solver + fast analytical + tandem + hysteresis
physics/materials.py    288 lines   46 materials + IDL + graded profiles
physics/optics.py       224 lines   TMM optical simulator
ai/optimizer.py         945 lines   BO, NSGA-II, PINN, DeepONet, SHAP, inverse, NL
utils/helpers.py        449 lines   SCAPS parser/exporter, reports, benchmarks
utils/api.py            122 lines   FastAPI REST API
pyproject.toml           43 lines   PyPI packaging
TOTAL                 3,560 lines
```

## License

MIT License - free for academic and commercial use.
