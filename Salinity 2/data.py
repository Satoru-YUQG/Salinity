from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DATA_CANDIDATES,
    ID_COLUMNS,
    TARGET_COLUMN,
)


def resolve_data_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Data file not found: {candidate}")
        return candidate

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(
        "No data file found. Place Data.xlsx in this folder or pass a path.\n"
        f"Searched:\n  {searched}"
    )


def load_raw_dataframe(path: Optional[str | Path] = None) -> pd.DataFrame:
    data_path = resolve_data_path(path)
    return pd.read_excel(data_path)


def list_all_columns(path: Optional[str | Path] = None) -> list[str]:
    return list(load_raw_dataframe(path).columns)


def get_target_candidates(df: pd.DataFrame) -> List[str]:
    """Columns that can be used as target Y (everything except ID columns)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def get_selectable_feature_columns(
    df: pd.DataFrame,
    target_columns: Optional[Sequence[str]] = None,
    extra_exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Feature columns available for training (never includes targets or IDs)."""
    blocked = set(ID_COLUMNS) | set(target_columns or ()) | set(extra_exclude or ())
    return [c for c in df.columns if c not in blocked]


def list_feature_columns(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    extra_drop: Optional[Sequence[str]] = None,
    use_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    available = get_selectable_feature_columns(df, target_columns=target_columns)
    if use_columns is not None:
        chosen = [c for c in use_columns if c in available]
        if not chosen:
            raise ValueError("No valid feature columns selected.")
        return chosen
    drop = set(extra_drop or ())
    return [c for c in available if c not in drop]


def build_labels(
    df: pd.DataFrame,
    target_columns: Union[str, Sequence[str]],
    *,
    label_mode: str = "threshold",
    class_threshold: float = 3.0,
    multi_target_combine: str = "any",
) -> pd.Series:
    """
    Build binary labels (0/1).

    label_mode:
      - threshold: numeric column(s); positive when value >= class_threshold
      - direct: single column already coded as 0/1 (or two unique values mapped to 0/1)

    multi_target_combine (threshold mode only):
      - any: class 1 if ANY target column meets threshold
      - all: class 1 if ALL target columns meet threshold
    """
    if isinstance(target_columns, str):
        cols = [target_columns]
    else:
        cols = list(target_columns)

    if not cols:
        raise ValueError("Select at least one target column.")

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Target column '{col}' not in dataset.")

    if label_mode == "direct":
        if len(cols) != 1:
            raise ValueError("Direct label mode supports exactly one target column.")
        col = cols[0]
        series = df[col]
        uniques = sorted(series.dropna().unique())
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            return series.map(mapping).astype(int)
        if set(uniques).issubset({0, 1, 0.0, 1.0}):
            return series.astype(int)
        raise ValueError(
            f"Column '{col}' must be binary (0/1 or two distinct values). "
            f"Found: {uniques[:10]}"
        )

    if label_mode != "threshold":
        raise ValueError(f"Unknown label_mode: {label_mode}")

    masks = [df[col] >= class_threshold for col in cols]
    if multi_target_combine == "all":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
    return combined.astype(int)


def prepare_dataset(
    path: Optional[str | Path] = None,
    target_columns: Optional[Union[str, Sequence[str]]] = None,
    label_mode: str = "threshold",
    class_threshold: float = 3.0,
    multi_target_combine: str = "any",
    drop_columns: Optional[Iterable[str]] = None,
    use_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (X, y, meta)."""
    if use_columns is not None and drop_columns:
        raise ValueError("Use either use_columns or drop_columns, not both.")

    if target_columns is None:
        target_columns = [TARGET_COLUMN]

    if isinstance(target_columns, str):
        target_cols = [target_columns]
    else:
        target_cols = list(target_columns)

    df = load_raw_dataframe(path).copy()
    df = df.drop(columns=list(ID_COLUMNS), errors="ignore")

    feature_cols = list_feature_columns(
        df,
        target_columns=target_cols,
        extra_drop=drop_columns,
        use_columns=use_columns,
    )

    subset_cols = list(dict.fromkeys(target_cols + feature_cols))
    df = df.dropna(subset=subset_cols)

    y = build_labels(
        df,
        target_cols,
        label_mode=label_mode,
        class_threshold=class_threshold,
        multi_target_combine=multi_target_combine,
    )
    X = df[feature_cols].copy()

    meta = df[target_cols].copy()
    meta["Predicted_Class"] = y

    return X, y, meta


def class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    total = len(y)
    return {
        int(label): {
            "count": int(count),
            "percent": round(100.0 * count / total, 2) if total else 0.0,
        }
        for label, count in counts.items()
    }
