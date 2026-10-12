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
from history_store import delete_run, get_run, list_runs, save_run
from models import list_algorithms
from param_presets import rank_rows
from pipeline import RunConfig, config_to_dict, run_experiment, run_multi_preset_experiment
from result_figures import render_result_figures
from ui_controls import render_param_grid

st.set_page_config(page_title="Salinity ML Lab", layout="wide", initial_sidebar_state="expanded")

if "algo_params_store" not in st.session_state:
    st.session_state.algo_params_store = {}

st.session_state.setdefault("target_columns", [TARGET_COLUMN])
st.session_state.setdefault("label_mode", "threshold")
st.session_state.setdefault("class_threshold", float(DEFAULT_SALINITY_THRESHOLD))
st.session_state.setdefault("multi_target_combine", "any")

grid_presets: list = []


def _apply_history_settings(saved_config: dict) -> None:
    config = saved_config["run_config"]
    algorithm = config.get("algorithm", "random_forest")
    targets = list(config.get("target_columns") or [TARGET_COLUMN])
    use_columns = config.get("use_columns")
    grid_text = saved_config.get("grid_text") or {}
    state = st.session_state

    state["use_upload"] = False
    state["data_file_path"] = config.get("data_path") or ""
    state["algorithm_select"] = algorithm
    state["compare_all"] = saved_config.get("run_type") == "compare_all"
    state["sampling_select"] = config.get("sampling", "smote")
    state["smote_k_neighbors"] = config.get("smote_k_neighbors") or 0
    state["normalize_features"] = config.get("normalize", True)
    state["evaluation_mode"] = config.get("evaluation_mode", "holdout")
    state["cv_folds"] = int(config.get("cv_folds", 5))
    state["n_runs"] = int(config.get("n_runs", DEFAULT_RUNS))
    state["test_size"] = float(config.get("test_size", DEFAULT_TEST_SIZE))
    state["auto_tune_threshold"] = config.get("auto_tune_threshold", False)
    state["tune_metric"] = config.get("tune_metric", "recall")
    state["decision_threshold"] = float(
        config.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    )
    state["random_seed"] = config.get("random_seed") or 0

    state["label_mode"] = config.get("label_mode", "threshold")
    state["label_mode_radio"] = state["label_mode"]
    state["target_columns"] = targets
    state["target_cols_threshold"] = targets
    state["target_col_direct"] = targets[0]
    state["class_threshold"] = float(
        config.get("class_threshold", DEFAULT_SALINITY_THRESHOLD)
    )
    state["class_threshold_slider"] = state["class_threshold"]
    state["multi_target_combine"] = config.get("multi_target_combine", "any")
    state["multi_target_combine_radio"] = state["multi_target_combine"]
    state["feature_mode"] = "include" if use_columns is not None else "exclude"
    state["feat_include"] = list(use_columns or [])
    state["feat_drop"] = list(config.get("drop_columns") or [])

    grid_store = dict(state.get("param_grid_store", {}))
    grid_store[algorithm] = dict(grid_text)
    state["param_grid_store"] = grid_store
    for name, value in grid_text.items():
        state[f"grid_{algorithm}_{name}"] = str(value)

    state["history_notice"] = (
        "Saved settings loaded. Review or edit them in the sidebar and tabs, then run the experiment."
    )
    state["history_view_id"] = None


def _delete_history_record(run_id: int) -> None:
    try:
        delete_run(run_id)
        st.session_state["history_notice"] = "Saved experiment deleted."
    except Exception as exc:
        st.session_state["history_notice"] = f"Could not delete saved experiment: {exc}"
    st.session_state.pop("history_selected_id", None)


def _save_history_record(
    *, title: str, run_type: str, algorithm: str, config: dict, results: dict
) -> int:
    return save_run(
        title=title,
        run_type=run_type,
        algorithm=algorithm,
        config=config,
        results=results,
    )


def _history_configuration(
    config: RunConfig,
    *,
    run_type: str,
    grid_text: dict,
    algorithms: list[str],
) -> dict:
    return {
        "run_config": config_to_dict(config),
        "run_type": run_type,
        "grid_text": dict(grid_text),
        "algorithms": list(algorithms),
    }


def _persist_completed_run(
    *, title: str, run_type: str, algorithm: str, config: dict, results: dict
) -> None:
    try:
        run_id = _save_history_record(
            title=title,
            run_type=run_type,
            algorithm=algorithm,
            config=config,
            results=results,
        )
        st.caption(f"Automatically saved to History as experiment {run_id}.")
    except Exception as exc:
        st.warning(f"The experiment completed, but could not be saved to History: {exc}")


def _metric_means(summary: dict) -> dict:
    return {
        k: v
        for k, v in summary.items()
        if k.endswith("_mean") or k.endswith("_std")
    }


def _best_heading(metric: str, score) -> str:
    title = BEST_RESULT_TITLES.get(metric, "Best result")
    if score is None:
        return title
    return f"{title}: {score:.3f}"


def _detail_from_result(name: str, params: dict, result) -> dict:
    return {
        "name": name,
        "params": params,
        "summary": result.summary,
        "per_run": result.summary_dataframe().to_dict(orient="records"),
        "feature_importance": result.feature_importance,
        "feature_names": result.feature_names,
        "class_balance": result.class_balance,
    }


def _cache_single_result(algorithm: str, name: str, params: dict, result) -> None:
    detail = _detail_from_result(name, params, result)
    st.session_state.last_results = {
        "mode": "single",
        "algorithm": algorithm,
        "rows": [{"combo_name": name, **_metric_means(result.summary)}],
        "details": [detail],
        "detail": detail,
    }


def _cache_grid_results(algorithm: str, multi) -> None:
    rows = []
    details = []
    for pname, pparams, pres in multi.preset_results:
        rows.append({"combo_name": pname, **_metric_means(pres.summary)})
        details.append(_detail_from_result(pname, pparams, pres))
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
                "recall_mean": r.get("recall_mean"),
                "recall_std": r.get("recall_std"),
                "f1_mean": r.get("f1_mean"),
                "f1_std": r.get("f1_std"),
                "train_f1_mean": r.get("train_f1_mean"),
                "train_f1_std": r.get("train_f1_std"),
                "balanced_accuracy_mean": r.get("balanced_accuracy_mean"),
                "balanced_accuracy_std": r.get("balanced_accuracy_std"),
                "roc_auc_mean": r.get("roc_auc_mean"),
                "roc_auc_std": r.get("roc_auc_std"),
            }
            for r in rows
        ],
    }


def _results_table(sorted_rows: list) -> pd.DataFrame:
    table = pd.DataFrame(sorted_rows)
    table = table.drop(columns=["params_summary"], errors="ignore")
    if "combo_name" in table.columns:
        table = table.rename(columns={"combo_name": "configuration"})
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "f1": "Validation F1",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "train_f1": "Training F1",
    }
    table = table.rename(
        columns={
            f"{metric}_{stat}": f"{label} {stat}"
            for metric, label in metric_labels.items()
            for stat in ("mean", "std")
        }
    )
    return table


def _find_detail(cached: dict, config_name: str) -> dict | None:
    for d in cached.get("details") or []:
        if d["name"] == config_name:
            return d
    if cached.get("detail") and cached["detail"].get("name") == config_name:
        return cached["detail"]
    return None


def _render_configuration_detail(detail: dict, show_figures: bool) -> None:
    s = detail["summary"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Recall", f"{s.get('recall_mean', 0):.3f}")
    m2.metric("Precision", f"{s.get('precision_mean', 0):.3f}")
    m3.metric(
        "Validation F1",
        f"{s.get('f1_mean', 0):.3f}",
        delta=f"± {s.get('f1_std', 0):.3f} SD",
        delta_color="off",
    )
    m4.metric(
        "Training F1",
        f"{s.get('train_f1_mean', 0):.3f}",
        delta=f"± {s.get('train_f1_std', 0):.3f} SD",
        delta_color="off",
    )
    m5.metric("Bal. accuracy", f"{s.get('balanced_accuracy_mean', 0):.3f}")
    roc = s.get("roc_auc_mean")
    m6.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "—")

    if s.get("tuned_threshold_mean") is not None:
        st.caption(
            f"Auto-tuned threshold: {s['tuned_threshold_mean']:.2f} "
            f"± {s.get('tuned_threshold_std', 0):.2f}"
        )

    if show_figures:
        st.subheader("Figures")
        render_result_figures(detail, detail["name"])

    st.subheader("Feature importance")
    if detail.get("feature_importance"):
        imp_df = (
            pd.DataFrame(
                list(detail["feature_importance"].items()),
                columns=["feature", "importance"],
            )
            .sort_values("importance", ascending=False)
        )
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
        st.bar_chart(imp_df.set_index("feature"))
    else:
        st.info("Feature importance is not available for this algorithm.")

    if detail.get("feature_names"):
        st.caption(f"**Features used ({len(detail['feature_names'])}):** {', '.join(detail['feature_names'])}")

    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "Validation F1",
        "train_f1": "Training F1",
        "roc_auc": "ROC-AUC",
    }
    summary_rows = [
        {
            "Metric": label,
            "Mean": s.get(f"{metric}_mean"),
            "Standard deviation": s.get(f"{metric}_std"),
        }
        for metric, label in metric_labels.items()
        if s.get(f"{metric}_mean") is not None
    ]
    st.subheader("Summary across runs / folds")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("Per-run / per-fold metrics")
    per_run = pd.DataFrame(detail["per_run"]).rename(
        columns={
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy",
            "recall": "Recall",
            "precision": "Precision",
            "f1": "Validation F1",
            "train_f1": "Training F1",
            "roc_auc": "ROC-AUC",
            "confusion_matrix": "Validation confusion matrix",
        }
    )
    st.dataframe(per_run, use_container_width=True, hide_index=True)


def _display_cached_results(cached: dict | None = None, *, view_id: str = "current") -> None:
    if cached is None:
        cached = st.session_state.get("last_results")
    if not cached:
        return

    key_suffix = "" if view_id == "current" else f"_{view_id}"

    st.divider()
    st.subheader("Results")
    st.caption("Change the sort metric below — no retraining needed.")

    sort_metric = st.selectbox(
        "Sort / pick best by",
        options=list(RANK_METRICS.keys()),
        format_func=lambda k: RANK_METRICS[k],
        key=f"results_sort_metric{key_suffix}",
    )

    rows = cached["rows"]
    sorted_rows, best_name = rank_rows(rows, metric=sort_metric)
    best_score = sorted_rows[0].get(sort_metric) if sorted_rows else None

    st.markdown(f"### {_best_heading(sort_metric, best_score)}")
    if best_name:
        st.markdown(f"**Settings:** `{best_name}`")

    display_table = _results_table(sorted_rows)
    st.caption("Click a row in the table to inspect that configuration.")

    table_response = st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"results_grid_table{key_suffix}",
    )

    selected_idx = 0
    if table_response.selection and table_response.selection.rows:
        selected_idx = table_response.selection.rows[0]

    selected_name = display_table.iloc[selected_idx]["configuration"]
    mode = cached["mode"]
    algo = cached.get("algorithm", "")

    show_figures = st.checkbox(
        "Show figures", value=False, key=f"show_result_figures{key_suffix}"
    )

    if mode in ("grid", "single"):
        detail = _find_detail(cached, selected_name)
        if detail is None and cached.get("details"):
            detail = cached["details"][0]
        elif detail is None:
            detail = cached.get("detail")

        if detail:
            st.markdown(f"#### Selected: `{selected_name}`")
            _render_configuration_detail(detail, show_figures)

            st.download_button(
                "Download selected configuration (JSON)",
                data=json.dumps(detail, indent=2, default=str),
                file_name="salinity_selected_run.json",
                mime="application/json",
                key=f"dl_selected{key_suffix}",
            )

        if mode == "grid":
            st.download_button(
                "Download full grid comparison (JSON)",
                data=json.dumps(
                    {
                        "algorithm": algo,
                        "sort_metric": sort_metric,
                        "best_configuration": best_name,
                        "comparison": sorted_rows,
                        "details": cached.get("details"),
                    },
                    indent=2,
                    default=str,
                ),
                file_name=f"{algo}_grid_comparison.json",
                mime="application/json",
                key=f"dl_grid_cached{key_suffix}",
            )


def _render_history_page() -> None:
    st.header("Experiment history")
    st.caption("Completed experiments are saved locally and remain available after restarting the app.")

    notice = st.session_state.pop("history_notice", None)
    if notice:
        st.info(notice)

    try:
        records = list_runs()
    except Exception as exc:
        st.error(f"Could not read experiment history: {exc}")
        return

    if not records:
        st.info("No saved experiments yet. Completed runs will appear here automatically.")
        return

    def _best_value(summary_rows: list[dict], field: str):
        values = [row.get(field) for row in summary_rows if row.get(field) is not None]
        return max(values) if values else None

    history_rows = []
    record_by_id = {record["id"]: record for record in records}
    for record in records:
        summary_rows = record["summary"]
        history_rows.append(
            {
                "Date": record["created_at"].replace("T", " "),
                "Experiment": record["title"],
                "Type": record["run_type"].replace("_", " ").title(),
                "Algorithm": record["algorithm"],
                "Configurations": len(summary_rows),
                "Best validation F1": _best_value(summary_rows, "f1_mean"),
                "Best training F1": _best_value(summary_rows, "train_f1_mean"),
            }
        )
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

    record_ids = list(record_by_id)
    selected_id = st.selectbox(
        "Select an experiment",
        options=record_ids,
        format_func=lambda run_id: (
            f"{record_by_id[run_id]['created_at'].replace('T', ' ')} · "
            f"{record_by_id[run_id]['title']}"
        ),
        key="history_selected_id",
    )
    selected = record_by_id[selected_id]
    saved_config = selected["config"]
    run_config = saved_config.get("run_config", {})

    st.markdown(f"#### {selected['title']}")
    st.caption(
        f"{selected['created_at'].replace('T', ' ')} · "
        f"{run_config.get('evaluation_mode', 'unknown')} · "
        f"{run_config.get('n_runs', 0)} run(s)"
    )
    data_path = run_config.get("data_path")
    if data_path:
        if Path(data_path).exists():
            st.caption(f"Data file: `{data_path}`")
        else:
            st.warning(f"The original data file is not currently available: {data_path}")

    load_col, view_col, delete_col = st.columns([1, 1, 1])
    with load_col:
        st.button(
            "Load settings into Run",
            key=f"history_load_{selected_id}",
            on_click=_apply_history_settings,
            args=(saved_config,),
            use_container_width=True,
        )
    with view_col:
        if st.button(
            "View saved results",
            key=f"history_view_{selected_id}",
            use_container_width=True,
        ):
            st.session_state["history_view_id"] = selected_id
    with delete_col:
        confirmed = st.checkbox(
            "Confirm deletion",
            key=f"history_confirm_delete_{selected_id}",
        )
        st.button(
            "Delete experiment",
            key=f"history_delete_{selected_id}",
            on_click=_delete_history_record,
            args=(selected_id,),
            disabled=not confirmed,
            use_container_width=True,
        )

    if st.session_state.get("history_view_id") == selected_id:
        saved_run = get_run(selected_id)
        if saved_run:
            _display_cached_results(
                saved_run["results"], view_id=f"history_{selected_id}"
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
    use_upload = st.checkbox("Upload Excel file", value=False, key="use_upload")
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
        data_path = st.text_input("Data file path", value=default_path, key="data_file_path")

    st.header("Algorithm")
    algo_labels = list_algorithms()
    algorithm = st.selectbox(
        "Model",
        options=list(algo_labels.keys()),
        format_func=lambda k: algo_labels[k],
        key="algorithm_select",
    )
    compare_all = st.checkbox("Compare all algorithms", value=False, key="compare_all")

    st.header("Training")
    sampling = st.selectbox(
        "Resampling",
        options=list(SAMPLING_CHOICES.keys()),
        format_func=lambda k: SAMPLING_CHOICES[k],
        key="sampling_select",
    )
    smote_k = st.number_input(
        "SMOTE / ADASYN neighbors (0 = auto)",
        min_value=0,
        max_value=10,
        value=0,
        help="Auto uses min(5, minority_count - 1).",
        key="smote_k_neighbors",
    )
    normalize = st.checkbox("Standardize features", value=True, key="normalize_features")

    evaluation_mode = st.selectbox(
        "Evaluation",
        options=["holdout", "stratified_cv"],
        format_func=lambda x: {
            "holdout": "Repeated random hold-out",
            "stratified_cv": "Stratified k-fold (per run)",
        }[x],
        key="evaluation_mode",
    )
    cv_folds = st.slider("CV folds", 2, 10, 5, key="cv_folds")
    n_runs = st.number_input("Runs", 1, 200, DEFAULT_RUNS, key="n_runs")
    test_size = st.slider(
        "Hold-out validation %", 0.1, 0.4, float(DEFAULT_TEST_SIZE), 0.05, key="test_size"
    )

    st.subheader("Decision threshold")
    auto_tune_threshold = st.checkbox(
        "Auto-tune on validation set",
        value=False,
        help="Scans probability thresholds each fold to maximize recall or F1.",
        key="auto_tune_threshold",
    )
    tune_metric = st.selectbox(
        "Optimize metric", ["recall", "f1"], disabled=not auto_tune_threshold, key="tune_metric"
    )
    decision_threshold = st.slider(
        "Fixed P(saline) threshold",
        0.05,
        0.95,
        float(DEFAULT_DECISION_THRESHOLD),
        0.05,
        disabled=auto_tune_threshold,
        key="decision_threshold",
    )

    random_seed = st.number_input(
        "Random seed (0 = random each run)", 0, 10_000, 0, key="random_seed"
    )

tab_data, tab_features, tab_params, tab_run, tab_history, tab_help = st.tabs(
    ["Data", "Features", "Hyperparameters", "Run", "History", "Guide"]
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
            key="feature_mode",
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
        history_grid_text = dict(
            st.session_state.get("param_grid_store", {}).get(algorithm, {})
        )
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
                            "recall_std": multi.preset_results[0][2].summary.get("recall_std"),
                            "f1_mean": multi.preset_results[0][2].summary.get("f1_mean"),
                            "f1_std": multi.preset_results[0][2].summary.get("f1_std"),
                            "train_f1_mean": multi.preset_results[0][2].summary.get(
                                "train_f1_mean"
                            ),
                            "train_f1_std": multi.preset_results[0][2].summary.get(
                                "train_f1_std"
                            ),
                            "balanced_accuracy_mean": multi.preset_results[0][2].summary.get(
                                "balanced_accuracy_mean"
                            ),
                            "balanced_accuracy_std": multi.preset_results[0][2].summary.get(
                                "balanced_accuracy_std"
                            ),
                            "roc_auc_mean": multi.preset_results[0][2].summary.get("roc_auc_mean"),
                            "roc_auc_std": multi.preset_results[0][2].summary.get("roc_auc_std"),
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
                        history_config = _history_configuration(
                            cfg,
                            run_type="single",
                            grid_text=history_grid_text,
                            algorithms=[algo],
                        )
                        _persist_completed_run(
                            title=f"{algo} · {evaluation_mode} · {n_runs} run(s)",
                            run_type="single",
                            algorithm=algo,
                            config=history_config,
                            results=st.session_state.last_results,
                        )
                    else:
                        multi = run_multi_preset_experiment(
                            base_config, grid_presets, on_progress
                        )
                        _cache_grid_results(algo, multi)
                        history_config = _history_configuration(
                            base_config,
                            run_type="grid",
                            grid_text=history_grid_text,
                            algorithms=[algo],
                        )
                        _persist_completed_run(
                            title=f"{algo} · {len(grid_presets)} configurations · {evaluation_mode}",
                            run_type="grid",
                            algorithm=algo,
                            config=history_config,
                            results=st.session_state.last_results,
                        )
            except Exception as exc:
                st.error(f"{algo}: {exc}")
                continue
            finally:
                progress.empty()
                status.empty()

        if compare_all and comparison_rows:
            _cache_algo_compare(comparison_rows)
            st.success(f"Finished — compared {len(comparison_rows)} algorithms")
            history_config = _history_configuration(
                base_config,
                run_type="compare_all",
                grid_text=history_grid_text,
                algorithms=algorithms,
            )
            _persist_completed_run(
                title=f"All algorithms · {evaluation_mode} · {base_config.n_runs} run(s)",
                run_type="compare_all",
                algorithm="all",
                config=history_config,
                results=st.session_state.last_results,
            )

        elif st.session_state.get("last_results"):
            n = len(st.session_state.last_results.get("rows", []))
            st.success(f"Training complete — {n} configuration(s) cached.")

    _display_cached_results()


with tab_history:
    _render_history_page()


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
