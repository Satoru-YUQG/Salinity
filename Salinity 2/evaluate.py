from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricBundle:
    accuracy: float
    balanced_accuracy: float
    recall: float
    precision: float
    f1: float
    roc_auc: Optional[float]
    confusion: np.ndarray
    train_f1: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion.tolist(),
            "train_f1": self.train_f1,
        }


def predict_with_threshold(model, X, decision_threshold: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return (proba >= decision_threshold).astype(int)
    return model.predict(X)


def evaluate_model(
    model,
    X_val,
    y_val,
    decision_threshold: float = 0.5,
) -> MetricBundle:
    y_pred = predict_with_threshold(model, X_val, decision_threshold)

    roc: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) > 1:
                roc = float(roc_auc_score(y_val, proba))
        except ValueError:
            roc = None

    return MetricBundle(
        accuracy=float(accuracy_score(y_val, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_val, y_pred)),
        recall=float(recall_score(y_val, y_pred, zero_division=0)),
        precision=float(precision_score(y_val, y_pred, zero_division=0)),
        f1=float(f1_score(y_val, y_pred, zero_division=0)),
        roc_auc=roc,
        confusion=confusion_matrix(y_val, y_pred, labels=[0, 1]),
    )
