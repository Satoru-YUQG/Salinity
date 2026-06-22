from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RUNS,
    DEFAULT_SALINITY_THRESHOLD,
    DEFAULT_TEST_SIZE,
    TARGET_COLUMN,
)
from data import class_distribution, prepare_dataset
from evaluate import MetricBundle, evaluate_model
from models import get_model
from param_presets import ParamPreset, format_params_short, rank_presets
from threshold_search import find_best_threshold
from train import train_model, transform_for_model


@dataclass
class RunConfig:
    data_path: Optional[str] = None
    target_columns: Sequence[str] = field(default_factory=lambda: [TARGET_COLUMN])
    label_mode: str = "threshold"  # threshold | direct
    class_threshold: float = DEFAULT_SALINITY_THRESHOLD
    multi_target_combine: str = "any"  # any | all
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD
    drop_columns: Sequence[str] = field(default_factory=list)
    use_columns: Optional[Sequence[str]] = None
    algorithm: str = "random_forest"
    model_params: Dict[str, Any] = field(default_factory=dict)
    preset_name: Optional[str] = None
    sampling: str = "smote"
    smote_k_neighbors: Optional[int] = None
    normalize: bool = True
    n_runs: int = DEFAULT_RUNS
    test_size: float = DEFAULT_TEST_SIZE
    random_seed: Optional[int] = None
    evaluation_mode: str = "holdout"
    cv_folds: int = 5
    auto_tune_threshold: bool = False
    tune_metric: str = "recall"
    rank_metric: str = "recall_mean"


@dataclass
class ExperimentResult:
    config: RunConfig
    n_samples: int
    n_features: int
    class_balance: dict
    feature_names: List[str]
    run_metrics: List[MetricBundle]
    summary: Dict[str, float]
    tuned_thresholds: List[float] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None

    def summary_dataframe(self) -> pd.DataFrame:
        rows = [m.as_dict() for m in self.run_metrics]
        return pd.DataFrame(rows)


@dataclass
class MultiPresetResult:
    base_config: RunConfig
    preset_results: List[Tuple[str, Dict[str, Any], ExperimentResult]]
    comparison_rows: List[Dict[str, Any]]
    best_preset_name: Optional[str]
    rank_metric: str

    def comparison_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.comparison_rows)


def _aggregate_metrics(metrics: List[MetricBundle]) -> Dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "recall", "precision", "f1"]
    out: Dict[str, float] = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    roc_values = [m.roc_auc for m in metrics if m.roc_auc is not None]
    if roc_values:
        out["roc_auc_mean"] = float(np.mean(roc_values))
        out["roc_auc_std"] = float(np.std(roc_values))
    return out


def _safe_stratified_split(X, y, test_size, seed):
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y,
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None,
        )


def _feature_importance(model, feature_names: List[str]) -> Optional[Dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return {name: float(v) for name, v in zip(feature_names, vals)}
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).ravel()
        return {name: float(v) for name, v in zip(feature_names, coef)}
    return None


def _train_and_evaluate_fold(
    config: RunConfig,
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names: List[str],
    seed: int,
) -> tuple:
    model = get_model(config.algorithm, random_state=seed, params=config.model_params)
    model = train_model(
        model,
        X_train,
        y_train,
        sampling=config.sampling,
        normalize=config.normalize,
        random_state=seed,
        smote_k_neighbors=config.smote_k_neighbors,
    )

    X_val_t = transform_for_model(model, X_val)
    threshold = config.decision_threshold

    if config.auto_tune_threshold and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val_t)[:, 1]
        threshold, _ = find_best_threshold(y_val, proba, metric=config.tune_metric)

    metrics = evaluate_model(model, X_val_t, y_val, decision_threshold=threshold)
    importance = _feature_importance(model, feature_names)
    return metrics, threshold, importance


def _run_holdout_once(config: RunConfig, X, y, feature_names, seed) -> tuple:
    X_train, X_val, y_train, y_val = _safe_stratified_split(
        X, y, config.test_size, seed
    )
    return _train_and_evaluate_fold(
        config, X_train, X_val, y_train, y_val, feature_names, seed
    )


def _run_cv_once(config: RunConfig, X, y, feature_names, seed) -> List[tuple]:
    minority = int(y.value_counts().min())
    n_splits = min(config.cv_folds, minority, len(y))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        results.append(
            _train_and_evaluate_fold(
                config, X_train, X_val, y_train, y_val, feature_names, seed
            )
        )
    return results


def _load_xy(config: RunConfig):
    return prepare_dataset(
        path=config.data_path,
        target_columns=list(config.target_columns),
        label_mode=config.label_mode,
        class_threshold=config.class_threshold,
        multi_target_combine=config.multi_target_combine,
        drop_columns=config.drop_columns if not config.use_columns else None,
        use_columns=config.use_columns,
    )


def run_experiment(
    config: RunConfig,
    progress_callback: Optional[Callable[..., None]] = None,
) -> ExperimentResult:
    X, y, _meta = _load_xy(config)

    if y.nunique() < 2:
        raise ValueError(
            "Only one class present for the selected target/threshold. "
            "Adjust the threshold, target column(s), or label mode."
        )

    minority = int(y.value_counts().min())
    if minority < 2 and config.n_runs > 1 and config.evaluation_mode == "holdout":
        raise ValueError(
            f"Only {minority} sample(s) in the minority class. "
            "Use stratified CV, lower the threshold, or add more positive samples."
        )

    feature_names = list(X.columns)
    run_metrics: List[MetricBundle] = []
    tuned_thresholds: List[float] = []
    importances: List[Dict[str, float]] = []
    rng = np.random.default_rng(config.random_seed)

    for i in range(config.n_runs):
        if progress_callback:
            progress_callback(i + 1, config.n_runs, config.preset_name)

        seed = (
            int(rng.integers(0, 2**31 - 1))
            if config.random_seed is None
            else config.random_seed + i
        )

        if config.evaluation_mode == "stratified_cv":
            fold_results = _run_cv_once(config, X, y, feature_names, seed)
            for metrics, threshold, importance in fold_results:
                run_metrics.append(metrics)
                tuned_thresholds.append(threshold)
                if importance:
                    importances.append(importance)
        else:
            metrics, threshold, importance = _run_holdout_once(
                config, X, y, feature_names, seed
            )
            run_metrics.append(metrics)
            tuned_thresholds.append(threshold)
            if importance:
                importances.append(importance)

    summary = _aggregate_metrics(run_metrics)

    avg_importance: Optional[Dict[str, float]] = None
    if importances:
        keys = importances[0].keys()
        avg_importance = {
            k: float(np.mean([imp[k] for imp in importances])) for k in keys
        }

    if tuned_thresholds and config.auto_tune_threshold:
        summary["tuned_threshold_mean"] = float(np.mean(tuned_thresholds))
        summary["tuned_threshold_std"] = float(np.std(tuned_thresholds))

    return ExperimentResult(
        config=config,
        n_samples=len(y),
        n_features=X.shape[1],
        class_balance=class_distribution(y),
        feature_names=feature_names,
        run_metrics=run_metrics,
        summary=summary,
        tuned_thresholds=tuned_thresholds,
        feature_importance=avg_importance,
    )


def run_multi_preset_experiment(
    base_config: RunConfig,
    presets: Sequence[ParamPreset],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> MultiPresetResult:
    """Run the same experiment for multiple hyperparameter presets and compare."""
    if not presets:
        raise ValueError("Select at least one hyperparameter preset to run.")

    preset_results: List[Tuple[str, Dict[str, Any], ExperimentResult]] = []
    comparison_rows: List[Dict[str, Any]] = []
    total = len(presets) * base_config.n_runs
    global_step = 0

    for preset in presets:
        cfg = copy.deepcopy(base_config)
        cfg.model_params = dict(preset.params)
        cfg.preset_name = preset.name

        def _cb(run_i: int, run_n: int, pname: str = preset.name):
            nonlocal global_step
            global_step += 1
            if progress_callback:
                progress_callback(global_step, total, pname)

        result = run_experiment(cfg, progress_callback=_cb)
        preset_results.append((preset.name, preset.params, result))

        row = {
            "combo_name": preset.name,
            **{k: v for k, v in result.summary.items() if k.endswith("_mean")},
        }
        comparison_rows.append(row)

    sorted_rows, best = rank_presets(comparison_rows, metric=base_config.rank_metric)

    return MultiPresetResult(
        base_config=base_config,
        preset_results=preset_results,
        comparison_rows=sorted_rows,
        best_preset_name=best,
        rank_metric=base_config.rank_metric,
    )


def config_to_dict(config: RunConfig) -> dict:
    return asdict(config)
