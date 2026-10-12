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


## What this tool does (research workflow)
This app trains a **binary classifier** for salinity using your spreadsheet features and a label derived from:

- **Target column**: `Ece mean`
- **Label rule**: class **1 (saline)** if \(Ece\_mean \ge\) **Ece threshold**, else class **0**

Because the dataset can be **highly imbalanced** (very few saline samples), you should focus on:
- **Recall (saline)**: how many saline profiles you detect
- **Balanced accuracy**: accuracy averaged across classes
- **F1**: balance of precision/recall when you care about false alarms too

## How to use the UI
### 1) Data tab
- Confirm the file loads and columns look correct.
- Adjust the **Ece threshold** and watch how the **class balance** changes.

### 2) Features tab (choose which columns are used)
You have two modes:
- **Exclude mode**: use all allowed columns except the ones you drop.
- **Include mode**: use **only** the selected columns (best when you want full control).

Notes:
- ID columns and leakage columns (like `Ece mean`) are automatically blocked from training.
- “Spectral only” is a fast way to avoid geographic memorization (longitude/latitude/year) and categorical soil/LU columns.

### 3) Hyperparameters tab (full algorithm control)
- Modify every hyperparameter exposed for the chosen algorithm.
- **Save hyperparameters (JSON)** to create a reproducible “model preset”.
- Load a saved JSON later (or share it with another researcher).

### 4) Run tab (evaluation + export)
Evaluation options:
- **Repeated random hold-out**: repeatedly split train/validation and average metrics.
- **Stratified k-fold CV**: splits data into k folds with similar class ratios; better when positives are rare.

Decision threshold options:
- **Fixed threshold**: predicts saline when \(P(saline)\ge\) threshold (default 0.5).
- **Auto-tune threshold**: searches probability cutoffs on each validation fold to maximize **recall** or **F1**.

Result tables report mean and standard deviation across runs or folds. The detail view shows an aggregate summary table above the individual per-run/per-fold metrics. The individual table labels **Training F1** and **Validation F1** separately; both are calculated from their respective confusion matrices. Training F1 is measured on the original training samples used to fit the model, so it is an in-sample diagnostic and not an independent performance estimate. Validation F1 is measured on the held-out fold. When automatic threshold tuning is enabled, the current pipeline selects the threshold using that fold's validation labels and scores F1 on those same labels, so that validation F1 can be optimistic and should not be treated as an independent estimate.

Exports:
- Download a JSON containing the full config, summary metrics, selected features, and feature importance (when available).

## Training controls (what they mean)
### Resampling (SMOTE / ADASYN)
Used **only on the training split** to reduce class imbalance.

- **none**: no oversampling; rely on class weighting (if the algorithm supports it).
- **SMOTE**: creates synthetic minority samples by interpolating neighbors.
  - **neighbors (k)**: larger k makes smoother synthetic samples; too large can create unrealistic points when the minority class is tiny.
- **ADASYN**: like SMOTE but generates more synthetic samples where the minority class is harder to learn (near boundaries).
  - can help, but may also amplify noise when data is very small.

### Standardize features
Normalization (z-score scaling) is important for distance-based / margin-based models:
- **SVM** and **KNN** usually benefit strongly.
- Tree models (Random Forest / Extra Trees / Gradient Boosting) are less sensitive.

## Algorithms: how they work, when to use them, and hyperparameters
Below, “positive class” means **saline (1)**.

### 1) Random Forest (`random_forest`)
**How it works**: trains many decision trees on bootstrapped samples; averages votes. Handles non-linear feature interactions well.

**When to use**: strong default on tabular data, robust, gives feature importance.

**Hyperparameters**
- **n_estimators (trees)**: more trees → more stable, slower.
- **max_depth**: limits tree depth. Lower values reduce overfitting on small datasets.
- **min_samples_split**: minimum samples to split a node. Higher → smoother/less complex trees.
- **min_samples_leaf**: minimum samples in a leaf. Higher → less variance, often better generalization.
- **max_features**:
  - `sqrt` (common): decorrelates trees, usually good default.
  - `log2`: similar, slightly fewer features per split.
  - `all`: strongest splits, but trees become more similar → can overfit.
- **class_weight**:
  - `balanced`: increases penalty for misclassifying the rare class; often improves recall.
  - `none`: can increase accuracy but often collapses recall when positives are rare.
- **bootstrap**: if enabled, each tree sees a bootstrap sample → more diversity.

### 2) Extra Trees (`extra_trees`)
**How it works**: like Random Forest but splits are more randomized; often faster and sometimes generalizes better.

**When to use**: strong alternative to Random Forest; can be more stable with very small data.

**Hyperparameters**: same meaning as Random Forest.

### 3) Gradient Boosting (`gradient_boosting`)
**How it works**: builds trees sequentially; each new tree corrects errors of the previous ensemble.

**When to use**: can be very accurate, but is easier to overfit on tiny or noisy data.

**Hyperparameters**
- **n_estimators**: more stages → more capacity; combine with smaller learning rate.
- **learning_rate**: shrink contribution of each tree.
  - smaller → usually needs more estimators, but often generalizes better.
- **max_depth**: depth of each weak learner tree.
- **min_samples_split / min_samples_leaf**: regularization, same concept as RF.
- **subsample**: < 1.0 enables stochastic boosting (adds regularization).

### 4) Logistic Regression (`logistic_regression`)
**How it works**: linear decision boundary in feature space; outputs probabilities via sigmoid.

**When to use**: baseline, interpretable, good when signal is mostly linear; works well with scaling.

**Hyperparameters**
- **C**: inverse of regularization strength.
  - smaller C → stronger regularization → simpler model.
- **penalty**:
  - `l2`: smooth shrinkage (default).
  - `l1`: sparse coefficients (feature selection effect).
  - `elasticnet`: mix of l1/l2 (requires solver `saga`).
- **solver**: optimization method; some solvers support only some penalties.
- **max_iter**: increase if convergence warnings occur.
- **class_weight**: `balanced` usually improves minority recall.

### 5) Support Vector Machine (`svm_rbf`)
**How it works**: finds a separating boundary with maximum margin; RBF kernel creates a non-linear boundary.

**When to use**: small datasets where scaling is possible; can work well with few positives but needs careful tuning.

**Hyperparameters**
- **C**: trade-off between margin size and training errors.
  - higher C fits training data more tightly (risk overfitting).
  - lower C is smoother (may improve generalization).
- **kernel**:
  - `rbf`: non-linear (default).
  - `linear`: linear boundary (faster).
  - `poly`: polynomial boundary (rarely needed here).
- **gamma**:
  - `scale` (recommended): adapts to feature variance.
  - `auto`: uses 1 / n_features; sometimes too aggressive/too weak.
- **class_weight**: `balanced` often required to get any recall on rare positives.

### 6) K-Nearest Neighbors (`knn`)
**How it works**: predicts based on the labels of the closest points in feature space.

**When to use**: quick baseline; can work if classes cluster well; sensitive to scaling and noise.

**Hyperparameters**
- **n_neighbors (k)**:
  - small k → sensitive, can overfit.
  - larger k → smoother, may miss rare positives.
- **weights**:
  - `uniform`: each neighbor equally weighted.
  - `distance`: closer points matter more; often better.
- **p**:
  - 1 = Manhattan distance; 2 = Euclidean distance (default).

### 7) Gaussian Naive Bayes (`naive_bayes`)
**How it works**: assumes each feature is conditionally independent given the class and normally distributed.

**When to use**: very small datasets; quick baseline; can perform surprisingly well but assumptions may be violated.

**Hyperparameters**
- **var_smoothing**: adds small variance to stabilize numerical behavior.
  - larger smoothing can prevent overconfidence but can underfit.

## Recommended defaults (for your current class imbalance)
If saline samples are extremely rare:
1. Use **Stratified k-fold CV** (3–5 folds) and multiple runs.
2. Enable **Auto-tune threshold → recall**.
3. Start with **Random Forest** or **Extra Trees**, `class_weight=balanced`, and shallow `max_depth` (3–8).
4. Use SMOTE carefully: keep **neighbors (k)** small when minority class count is tiny.

## CLI usage (same pipeline, no UI)
```bash
python Salinity_2.py --runs 30 --algorithm random_forest
```

Tip: for research reproducibility, keep a fixed `--seed`, and save the UI hyperparameters as JSON.
        """
