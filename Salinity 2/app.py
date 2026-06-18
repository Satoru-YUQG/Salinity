"""
Salinity ML Lab — full control over features, training, and algorithm hyperparameters.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BEST_RESULT_TITLES,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RUNS,
    DEFAULT_SALINITY_THRESHOLD,
    DEFAULT_TEST_SIZE,
    LABEL_MODES,
    MULTI_TARGET_COMBINE,
    PROJECT_ROOT,
    RANK_METRICS,
    SAMPLING_CHOICES,
    TARGET_COLUMN,
)
from data import (
    build_labels,
    class_distribution,
    get_selectable_feature_columns,
    get_target_candidates,
    load_raw_dataframe,
    prepare_dataset,
    resolve_data_path,
)
from models import list_algorithms
from param_presets import rank_rows
from pipeline import RunConfig, config_to_dict, run_experiment, run_multi_preset_experiment
from ui_controls import render_param_grid

st.set_page_config(page_title="Salinity ML Lab", layout="wide", initial_sidebar_state="expanded")

if "algo_params_store" not in st.session_state:
    st.session_state.algo_params_store = {}

st.session_state.setdefault("target_columns", [TARGET_COLUMN])
st.session_state.setdefault("label_mode", "threshold")
st.session_state.setdefault("class_threshold", float(DEFAULT_SALINITY_THRESHOLD))
st.session_state.setdefault("multi_target_combine", "any")

grid_presets: list = []


def _metric_means(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k.endswith("_mean")}


def _best_heading(metric: str, score) -> str:
    title = BEST_RESULT_TITLES.get(metric, "Best result")
    if score is None:
        return title
    return f"{title}: {score:.3f}"


def _cache_single_result(algorithm: str, name: str, params: dict, result) -> None:
    st.session_state.last_results = {
        "mode": "single",
        "algorithm": algorithm,
        "rows": [
            {
                "combo_name": name,
                "params_summary": name,
                **_metric_means(result.summary),
            }
        ],
        "detail": {
            "name": name,
            "params": params,
            "summary": result.summary,
            "per_run": result.summary_dataframe().to_dict(orient="records"),
            "feature_importance": result.feature_importance,
            "class_balance": result.class_balance,
            "features": result.feature_names,
            "config": config_to_dict(result.config),
        },
    }


def _cache_grid_results(algorithm: str, multi) -> None:
    rows = []
    details = []
    for pname, pparams, pres in multi.preset_results:
        rows.append(
            {
                "combo_name": pname,
                "params_summary": pname,
                **_metric_means(pres.summary),
            }
        )
        details.append(
            {
                "name": pname,
                "params": pparams,
                "summary": pres.summary,
                "per_run": pres.summary_dataframe().to_dict(orient="records"),
                "feature_importance": pres.feature_importance,
            }
        )
    st.session_state.last_results = {
        "mode": "grid",
        "algorithm": algorithm,
        "rows": rows,
        "details": details,
    }


def _cache_algo_compare(rows: list) -> None:
    st.session_state.last_results = {
        "mode": "algo_compare",
        "algorithm": "all",
        "rows": [
            {
                "combo_name": r["algorithm"],
                "params_summary": r["algorithm"],
                "recall_mean": r.get("recall_mean"),
                "f1_mean": r.get("f1_mean"),
                "balanced_accuracy_mean": r.get("balanced_accuracy_mean"),
                "roc_auc_mean": r.get("roc_auc_mean"),
            }
            for r in rows
        ],
    }


def _display_cached_results() -> None:
    cached = st.session_state.get("last_results")
    if not cached:
        return

    st.divider()
    st.subheader("Results")
    st.caption("Change the sort metric below.")

    sort_metric = st.selectbox(
        "Sort / pick best by",
        options=list(RANK_METRICS.keys()),
        format_func=lambda k: RANK_METRICS[k],
        key="results_sort_metric",
    )

    rows = cached["rows"]
    sorted_rows, best_name = rank_rows(rows, metric=sort_metric)
    best_score = sorted_rows[0].get(sort_metric) if sorted_rows else None

    st.markdown(f"### {_best_heading(sort_metric, best_score)}")
    if best_name:
        st.markdown(f"**Settings:** `{best_name}`")

    table = pd.DataFrame(sorted_rows)
    if "combo_name" in table.columns:
        table = table.rename(columns={"combo_name": "configuration"})
    st.dataframe(table, use_container_width=True)

    mode = cached["mode"]
    algo = cached.get("algorithm", "")

    if mode == "grid" and best_name:
        best_detail = next(
            (d for d in cached["details"] if d["name"] == best_name),
            cached["details"][0] if cached.get("details") else None,
        )
        if best_detail:
            m1, m2, m3, m4, m5 = st.columns(5)
            s = best_detail["summary"]
            m1.metric("Recall", f"{s.get('recall_mean', 0):.3f}")
            m2.metric("Precision", f"{s.get('precision_mean', 0):.3f}")
            m3.metric("F1", f"{s.get('f1_mean', 0):.3f}")
            m4.metric("Bal. accuracy", f"{s.get('balanced_accuracy_mean', 0):.3f}")
            roc = s.get("roc_auc_mean")
            m5.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "—")

            with st.expander("All combinations — detailed results"):
                for d in cached["details"]:
                    st.markdown(f"#### {d['name']}")
                    st.json(d["params"])
                    st.dataframe(pd.DataFrame(d["per_run"]), use_container_width=True)

            st.download_button(
                "Download grid comparison (JSON)",
                data=json.dumps(
                    {
                        "algorithm": algo,
                        "sort_metric": sort_metric,
                        "best_configuration": best_name,
                        "comparison": sorted_rows,
                        "details": cached["details"],
                    },
                    indent=2,
                    default=str,
                ),
                file_name=f"{algo}_grid_comparison.json",
                mime="application/json",
                key="dl_grid_cached",
            )

    elif mode == "single" and cached.get("detail"):
        d = cached["detail"]
        s = d["summary"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Recall", f"{s.get('recall_mean', 0):.3f}")
        m2.metric("Precision", f"{s.get('precision_mean', 0):.3f}")
        m3.metric("F1", f"{s.get('f1_mean', 0):.3f}")
        m4.metric("Bal. accuracy", f"{s.get('balanced_accuracy_mean', 0):.3f}")
        roc = s.get("roc_auc_mean")
        m5.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "—")

        st.dataframe(pd.DataFrame([s]).T.rename(columns={0: "value"}), use_container_width=True)
        st.dataframe(pd.DataFrame(d["per_run"]), use_container_width=True)

        if d.get("feature_importance"):
            imp_df = pd.DataFrame(
                list(d["feature_importance"].items()),
                columns=["feature", "importance"],
            ).sort_values("importance", ascending=False)
            st.bar_chart(imp_df.set_index("feature"))

        st.download_button(
            "Download full results (JSON)",
            data=json.dumps(d, indent=2, default=str),
            file_name="salinity_run.json",
            mime="application/json",
            key="dl_single_cached",
        )


st.title("Salinity classification lab")
st.caption("Customize every algorithm, pick features, and tune training for small imbalanced datasets.")

# ── Sidebar: data & run settings ──────────────────────────────────────────
st.link_button(
    "📂 GitHub Repository",
    "https://github.com/Satoru-YUQG/Salinity"
)
with st.sidebar:
    st.header("Data")
    use_upload = st.checkbox("Upload Excel file", value=False)
    data_path = None

    if use_upload:
        uploaded = st.file_uploader("Data.xlsx", type=["xlsx", "xls"])
        if uploaded is not None:
            cache_dir = PROJECT_ROOT / "data" / "uploads"
            cache_dir.mkdir(parents=True, exist_ok=True)
            data_path = str(cache_dir / uploaded.name)
            Path(data_path).write_bytes(uploaded.getvalue())
    else:
        try:
            default_path = str(resolve_data_path())
        except FileNotFoundError:
            default_path = ""
        data_path = st.text_input("Data file path", value=default_path)

    st.header("Algorithm")
    algo_labels = list_algorithms()
    algorithm = st.selectbox(
        "Model",
        options=list(algo_labels.keys()),
        format_func=lambda k: algo_labels[k],
    )
    compare_all = st.checkbox("Compare all algorithms", value=False)

    st.header("Training")
    sampling = st.selectbox(
        "Resampling",
        options=list(SAMPLING_CHOICES.keys()),
        format_func=lambda k: SAMPLING_CHOICES[k],
    )
    smote_k = st.number_input(
        "SMOTE / ADASYN neighbors (0 = auto)",
        min_value=0,
        max_value=10,
        value=0,
        help="Auto uses min(5, minority_count - 1).",
    )
    normalize = st.checkbox("Standardize features", value=True)

    evaluation_mode = st.selectbox(
        "Evaluation",
        options=["holdout", "stratified_cv"],
        format_func=lambda x: {
            "holdout": "Repeated random hold-out",
            "stratified_cv": "Stratified k-fold (per run)",
        }[x],
    )
    cv_folds = st.slider("CV folds", 2, 10, 5)
    n_runs = st.number_input("Runs", 1, 200, DEFAULT_RUNS)
    test_size = st.slider("Hold-out validation %", 0.1, 0.4, float(DEFAULT_TEST_SIZE), 0.05)

    st.subheader("Decision threshold")
    auto_tune_threshold = st.checkbox(
        "Auto-tune on validation set",
        value=False,
        help="Scans probability thresholds each fold to maximize recall or F1.",
    )
    tune_metric = st.selectbox("Optimize metric", ["recall", "f1"], disabled=not auto_tune_threshold)
    decision_threshold = st.slider(
        "Fixed P(saline) threshold",
        0.05,
        0.95,
        float(DEFAULT_DECISION_THRESHOLD),
        0.05,
        disabled=auto_tune_threshold,
    )

    random_seed = st.number_input("Random seed (0 = random each run)", 0, 10_000, 0)

tab_data, tab_features, tab_params, tab_run, tab_help = st.tabs(
    ["Data", "Features", "Hyperparameters", "Run", "Guide"]
)

feature_mode = "exclude"
drop_columns: list[str] = []
use_columns: list[str] | None = None
target_columns: list[str] = list(st.session_state.target_columns)
label_mode: str = st.session_state.label_mode
class_threshold: float = float(st.session_state.class_threshold)
multi_target_combine: str = st.session_state.multi_target_combine

# ── Data tab ────────────────────────────────────────────────────────────────
with tab_data:
    if not data_path:
        st.warning("Set a data path in the sidebar or upload a file.")
    else:
        try:
            raw = load_raw_dataframe(data_path)
            st.dataframe(raw.head(25), use_container_width=True)
            st.write(f"**Rows:** {len(raw)} · **Columns:** {len(raw.columns)}")

            y_preview = build_labels(
                raw,
                target_columns,
                label_mode=label_mode,
                class_threshold=class_threshold,
                multi_target_combine=multi_target_combine,
            )
            balance = class_distribution(y_preview.dropna())
            cols = st.columns(max(len(balance), 1))
            for idx, (label, info) in enumerate(balance.items()):
                name = "Saline (1)" if label == 1 else "Non-saline (0)"
                cols[idx].metric(name, f"{info['count']} ({info['percent']}%)")

            if balance.get(1, {}).get("count", 0) < 5:
                st.info("Few saline samples — tune threshold, use recall, and many repeated runs.")
        except Exception as exc:
            st.error(str(exc))

# ── Features tab ──────────────────────────────────────────────────────────
with tab_features:
    if not data_path:
        st.warning("Load data first.")
    else:
        raw = load_raw_dataframe(data_path)
        target_options = get_target_candidates(raw)

        st.subheader("Target Y (label)")
        label_mode = st.radio(
            "Label mode",
            options=list(LABEL_MODES.keys()),
            format_func=lambda k: LABEL_MODES[k],
            horizontal=True,
            key="label_mode_radio",
        )
        st.session_state.label_mode = label_mode

        if label_mode == "threshold":
            target_columns = st.multiselect(
                "Target column(s) — numeric, compared to threshold",
                options=target_options,
                default=[c for c in st.session_state.target_columns if c in target_options]
                or ([TARGET_COLUMN] if TARGET_COLUMN in target_options else target_options[:1]),
                key="target_cols_threshold",
            )
            class_threshold = st.slider(
                "Class threshold (value ≥ → class 1)",
                0.0,
                10.0,
                float(st.session_state.class_threshold),
                0.05,
                key="class_threshold_slider",
            )
            multi_target_combine = st.radio(
                "Multi-target rule",
                options=list(MULTI_TARGET_COMBINE.keys()),
                format_func=lambda k: MULTI_TARGET_COMBINE[k],
                horizontal=True,
                key="multi_target_combine_radio",
                disabled=len(target_columns) <= 1,
            )
        else:
            target_columns = st.selectbox(
                "Target column — must be 0/1 or two unique values",
                options=target_options,
                index=target_options.index(st.session_state.target_columns[0])
                if st.session_state.target_columns
                and st.session_state.target_columns[0] in target_options
                else 0,
                key="target_col_direct",
            )
            target_columns = [target_columns]
            class_threshold = float(st.session_state.class_threshold)
            multi_target_combine = "any"

        if not target_columns:
            st.error("Select at least one target column.")
            st.stop()

        st.session_state.target_columns = target_columns
        st.session_state.class_threshold = class_threshold
        st.session_state.multi_target_combine = multi_target_combine

        st.caption(f"**Target(s):** {', '.join(target_columns)}")

        st.subheader("Feature columns (X)")
        selectable = get_selectable_feature_columns(raw, target_columns=target_columns)

        feature_mode = st.radio(
            "Feature selection mode",
            options=["exclude", "include"],
            format_func=lambda m: (
                "Use all allowed columns except those dropped"
                if m == "exclude"
                else "Use only the columns I select"
            ),
            horizontal=True,
        )

        default_exclude = [c for c in ["Longitude", "Latitude", "Year"] if c in selectable]
        spectral_default = [c for c in selectable if c not in default_exclude]

        if feature_mode == "exclude":
            drop_columns = st.multiselect(
                "Drop these columns",
                options=selectable,
                default=default_exclude,
                key="feat_drop",
            )
            use_columns = None
            st.caption(f"Training with **{len(selectable) - len(drop_columns)}** features.")
        else:
            if "feat_include" not in st.session_state:
                st.session_state["feat_include"] = spectral_default
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("All features", key="btn_all_feat"):
                    st.session_state["feat_include"] = selectable
                    st.rerun()
            with col_b:
                if st.button("Spectral only", key="btn_spectral"):
                    skip = {"Soil_class", "LU_class", "Longitude", "Latitude", "Year"}
                    st.session_state["feat_include"] = [c for c in selectable if c not in skip]
                    st.rerun()
            use_columns = st.multiselect(
                "Use only these columns",
                options=selectable,
                default=st.session_state["feat_include"],
                key="feat_include",
            )
            drop_columns = []
            st.caption(f"Training with **{len(use_columns)}** selected features.")

        try:
            X, y, _ = prepare_dataset(
                data_path,
                target_columns=target_columns,
                label_mode=label_mode,
                class_threshold=class_threshold,
                multi_target_combine=multi_target_combine,
                drop_columns=drop_columns if feature_mode == "exclude" else None,
                use_columns=use_columns if feature_mode == "include" else None,
            )
            st.success(f"Ready: {len(y)} rows, {X.shape[1]} features")
            st.write(list(X.columns))
        except Exception as exc:
            st.error(str(exc))

# ── Hyperparameters tab ─────────────────────────────────────────────────────
with tab_params:
    if compare_all:
        st.info("Compare-all mode uses **default** hyperparameters for each algorithm.")
    else:
        grid_presets = render_param_grid(algorithm)

# ── Run tab ─────────────────────────────────────────────────────────────────
with tab_run:
    run_btn = st.button("Run experiment", type="primary", use_container_width=True)

    if run_btn:
        if not data_path:
            st.error("No data path.")
            st.stop()

        if feature_mode == "include" and not use_columns:
            st.error("Select at least one feature column.")
            st.stop()

        if not compare_all and not grid_presets:
            st.error("Fix hyperparameter grid values in the Hyperparameters tab.")
            st.stop()

        seed = None if random_seed == 0 else int(random_seed)
        smote_k_neighbors = None if smote_k == 0 else int(smote_k)
        algorithms = list(algo_labels.keys()) if compare_all else [algorithm]
        comparison_rows = []

        for algo in algorithms:
            from algo_params import default_params

            base_config = RunConfig(
                data_path=data_path,
                target_columns=target_columns,
                label_mode=label_mode,
                class_threshold=class_threshold,
                multi_target_combine=multi_target_combine,
                decision_threshold=decision_threshold,
                drop_columns=drop_columns if feature_mode == "exclude" else [],
                use_columns=use_columns if feature_mode == "include" else None,
                algorithm=algo,
                sampling=sampling,
                smote_k_neighbors=smote_k_neighbors,
                normalize=normalize,
                n_runs=3 if compare_all else int(n_runs),
                test_size=float(test_size),
                random_seed=seed,
                evaluation_mode=evaluation_mode,
                cv_folds=int(cv_folds),
                auto_tune_threshold=auto_tune_threshold,
                tune_metric=tune_metric,
            )

            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(current, total, label=algo):
                progress.progress(current / total)
                status.text(f"{label}: step {current} of {total}")

            try:
                with st.spinner(f"Running {algo}…"):
                    if compare_all:
                        from param_presets import ParamPreset
                        presets = [ParamPreset(name="default", params=default_params(algo))]
                        multi = run_multi_preset_experiment(base_config, presets, on_progress)
                        row = {
                            "algorithm": algo,
                            "recall_mean": multi.preset_results[0][2].summary.get("recall_mean"),
                            "f1_mean": multi.preset_results[0][2].summary.get("f1_mean"),
                            "balanced_accuracy_mean": multi.preset_results[0][2].summary.get(
                                "balanced_accuracy_mean"
                            ),
                            "roc_auc_mean": multi.preset_results[0][2].summary.get("roc_auc_mean"),
                        }
                        comparison_rows.append(row)
                    elif len(grid_presets) == 1:
                        cfg = base_config
                        cfg.model_params = dict(grid_presets[0].params)
                        cfg.preset_name = grid_presets[0].name
                        result = run_experiment(cfg, progress_callback=on_progress)
                        _cache_single_result(
                            algo, grid_presets[0].name, grid_presets[0].params, result
                        )
                    else:
                        multi = run_multi_preset_experiment(
                            base_config, grid_presets, on_progress
                        )
                        _cache_grid_results(algo, multi)
            except Exception as exc:
                st.error(f"{algo}: {exc}")
                continue
            finally:
                progress.empty()
                status.empty()

        if compare_all and comparison_rows:
            _cache_algo_compare(comparison_rows)
            st.success(f"Finished — compared {len(comparison_rows)} algorithms")

        elif st.session_state.get("last_results"):
            n = len(st.session_state.last_results.get("rows", []))
            st.success(f"Training complete — {n} configuration(s) cached.")

    _display_cached_results()


with tab_help:
    st.markdown(
        """
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

### 2) Features tab (target Y + feature columns)
**Target Y**
- Choose **one or more target columns** (default: `Ece mean`).
- **Threshold mode**: numeric column(s); class 1 when value ≥ threshold.
  - Multi-target: **ANY** (saline if any column meets rule) or **ALL** (all must meet rule).
- **Direct mode**: one column already coded as 0/1 (or two unique values mapped to 0/1).

Target columns are **automatically excluded** from features X (no leakage).

**Feature columns X**
You have two modes:
- **Exclude mode**: use all allowed columns except the ones you drop.
- **Include mode**: use **only** the selected columns (best when you want full control).

Notes:
- ID columns and leakage columns (like `Ece mean`) are automatically blocked from training.
- “Spectral only” is a fast way to avoid geographic memorization (longitude/latitude/year) and categorical soil/LU columns.

### 3) Hyperparameters tab (grid search)
- Enter **comma-separated values** in each field (e.g. trees: `300,200,150,50`, depth: `10,8,5`).
- The app tests **every combination** automatically (4 × 3 = 12 configs).
- Each combination runs for your chosen **Runs** × **CV folds** (if using stratified CV).
- Results are ranked; change **Sort / pick best by** on the Run tab **without retraining**.
- The heading updates automatically (e.g. **Best F1 score: 0.412**).
- Save/load the grid as JSON.

### 4) Run tab (evaluation + export)
Evaluation options:
- **Repeated random hold-out**: repeatedly split train/validation and average metrics.
- **Stratified k-fold CV**: splits data into k folds with similar class ratios; better when positives are rare.

Decision threshold options:
- **Fixed threshold**: predicts saline when \(P(saline)\ge\) threshold (default 0.5).
- **Auto-tune threshold**: searches probability cutoffs on each validation fold to maximize **recall** or **F1**.

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
    )
