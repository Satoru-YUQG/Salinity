"""Charts for experiment results (confusion matrix, metrics, feature importance)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate_confusion_matrix(per_run: List[dict]) -> np.ndarray:
    total = np.zeros((2, 2), dtype=int)
    for row in per_run:
        cm = row.get("confusion_matrix")
        if cm is not None:
            total += np.array(cm, dtype=int)
    return total


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion matrix (aggregated)"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i, j]}\n{cm[i, j]}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_metrics_bars(summary: dict, title: str = "Mean metrics"):
    keys = ["recall_mean", "precision_mean", "f1_mean", "balanced_accuracy_mean", "accuracy_mean"]
    labels = ["Recall", "Precision", "F1", "Bal. acc.", "Accuracy"]
    values = [summary.get(k, 0) or 0 for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#95a5a6"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_metric_stability(per_run: List[dict], title: str = "Metric spread across runs/folds"):
    if not per_run:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No run data", ha="center")
        return fig

    df = pd.DataFrame(per_run)
    metric_cols = [c for c in ["recall", "precision", "f1", "balanced_accuracy", "accuracy"] if c in df.columns]
    if not metric_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No metrics", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(7, 4))
    df[metric_cols].boxplot(ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: Dict[str, float], title: str = "Feature importance", top_n: int = 20):
    if not importance:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not available for this model", ha="center")
        return fig

    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.35)))
    ax.barh(names, vals, color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_result_figures(detail: dict, config_label: str) -> None:
    """Draw all standard result figures into the Streamlit container."""
    import streamlit as st

    per_run = detail.get("per_run") or []
    summary = detail.get("summary") or {}

    col1, col2 = st.columns(2)
    with col1:
        cm = aggregate_confusion_matrix(per_run)
        st.pyplot(plot_confusion_matrix(cm, f"Confusion matrix — {config_label[:40]}"))
        plt.close()
    with col2:
        st.pyplot(plot_metrics_bars(summary, f"Mean metrics — {config_label[:40]}"))
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_metric_stability(per_run, "Stability across runs/folds"))
        plt.close()
    with col4:
        st.pyplot(plot_feature_importance(detail.get("feature_importance"), "Feature importance"))
        plt.close()
