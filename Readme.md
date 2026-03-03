# TMDSI: Drought Index Toolkit

**Tripathy–Mishra Drought Severity Index (TMDSI)** and **scPDSI** in a clean, Python package.

> Tripathy, K.P. and Mishra, A.K. (in prep.)  
> *Beyond Empirical Constants: A Physically Calibrated Drought Severity Index (PCDSI)*  
> Water Resources Research

---

## Repository Structure

```
tmdsi/
├── tmdsi/                    # main package
│   ├── pet/
│   │   ├── thornthwaite.py   # Thornthwaite (1948) PET [implemented]
│   │   └── thornthwaite.py   # Hargreaves & Penman-Monteith [stubs]
│   ├── indices/
│   │   ├── scpdsi.py         # Wells et al. (2004) scPDSI
│   │   └── tmdsi.py          # Tripathy-Mishra DSI [coming next]
│   ├── water_balance.py      # Palmer two-layer soil bucket model
│   └── utils.py              # helpers: reshape, classify, DataFrame export
│
├── notebooks/
│   ├── 01_scpdsi_example.py  # single-station walkthrough (VSCode Jupyter)
│   └── 02_tmdsi_example.py   # TMDSI walkthrough (coming next)
│
├── data/sample/
│   └── generate_sample.py    # generates synthetic station CSV
│
├── tests/
│   └── test_scpdsi.py        # pytest test suite
│
├── environment.yml           # conda environment
└── pyproject.toml            # package metadata
```

---

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate tmdsi

# 2. Install in editable mode (for development)
pip install -e ".[dev]"

# 3. VSCode: select 'tmdsi' as Python interpreter
#    Ctrl+Shift+P → Python: Select Interpreter → tmdsi
```

---

## Quick Start

```python
from tmdsi.indices.scpdsi import compute

result = compute(
    precip                 = precip_mm,     # 1-D array, monthly (mm)
    tmax                   = tmax_c,        # monthly Tmax (°C)
    tmin                   = tmin_c,        # monthly Tmin (°C)
    awc                    = 150.0,         # available water capacity (mm)
    latitude               = 35.0,          # decimal degrees
    data_start_year        = 1970,
    calibration_start_year = 1970,
    calibration_end_year   = 2019,
    pet_method             = "thornthwaite",  # 'hargreaves' or 'penman_monteith' soon
)

result.pdsi   # scPDSI time series
result.z      # Z-index
result.cafec  # fitted CAFEC coefficients
```

---

## PET Methods (current status)

| Method | Status | Extra inputs needed |
|---|---|---|
| `thornthwaite` | ✅ Implemented | latitude only |
| `hargreaves` | 🔜 Stub | latitude |
| `penman_monteith` | 🔜 Stub | wind, RH, solar radiation, elevation |

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Citation

```bibtex
@article{tripathy2025tmdsi,
  title   = {Beyond Empirical Constants: A Physically Calibrated Drought Severity Index},
  author  = {Tripathy, Kshitij P. and Mishra, Ashok K.},
  journal = {Water Resources Research},
  year    = {2025},
  note    = {in preparation}
}
```