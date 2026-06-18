# Salinity ML Lab

Everything runs from **this folder** (`Salinity 2`).

## Files

| File | Purpose |
|------|---------|
| `Data.xlsx` | Your dataset (preferred location) |
| `app.py` | Streamlit UI |
| `Salinity_2.py` | Command-line experiments |
| `run_ui.bat` | Double-click to open the UI |
| `run_cli.bat` | Double-click for a CLI run |
| `requirements.txt` | Python dependencies |

## Requirements

- **Python**: 3.9+ (tested with 3.9)
- **pip**: installed with Python

## Install (once)

From this folder:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

- **UI (recommended):** double-click `run_ui.bat`
  - or:

```bash
python -m streamlit run app.py
```

- **CLI:** double-click `run_cli.bat`
  - or:

```bash
python Salinity_2.py --runs 30 --algorithm random_forest
```

If `python` is not recognized in PowerShell, use the Windows launcher:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
py Salinity_2.py
```

## Data notes

- The app creates the label from **`Ece mean`** using your chosen threshold.
- The code automatically blocks ID/leakage columns from being used as features.
- If you replace `Data.xlsx` with a new dataset, keep the same column names (or update `config.py`).

## UI tabs

| Tab | What you control |
|-----|------------------|
| **Data** | Preview rows and class balance |
| **Features** | Drop columns *or* use only selected columns |
| **Hyperparameters** | Full sklearn settings per algorithm (save/load JSON) |
| **Run** | Results, feature importance chart, JSON export |

**Sidebar:** resampling, SMOTE neighbors, hold-out vs stratified CV, auto-tune decision threshold for recall/F1.

Open this folder in Visual Studio and set **Working Directory** to `.` if you debug from the IDE.
