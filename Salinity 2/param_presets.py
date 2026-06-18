"""Parse comma-separated hyperparameter grids and build all combinations."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from algo_params import ParamSpec, default_params, get_param_schema


@dataclass
class ParamPreset:
    name: str
    params: Dict[str, Any]


def _format_default(spec: ParamSpec, value: Any) -> str:
    if spec.kind == "optional_int" and (value is None or int(value) <= 0):
        return "0"
    if spec.kind == "bool":
        return "true" if value else "false"
    return str(value)


def parse_value_token(token: str, spec: ParamSpec) -> Any:
    token = token.strip()
    if not token:
        raise ValueError(f"Empty value for {spec.name}")

    if spec.kind == "int":
        return int(token)
    if spec.kind == "optional_int":
        val = int(token)
        return None if val <= 0 else val
    if spec.kind == "float":
        return float(token)
    if spec.kind == "bool":
        low = token.lower()
        if low in ("1", "true", "yes", "y"):
            return True
        if low in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"{spec.name}: expected true/false, got '{token}'")
    if spec.kind == "choice":
        choices = spec.choices or []
        if token not in choices:
            raise ValueError(
                f"{spec.name}: '{token}' not in {choices}"
            )
        return token
    return token


def parse_value_list(text: str, spec: ParamSpec) -> List[Any]:
    """Parse '300,200,150,50' into typed values (deduplicated, order preserved)."""
    if not text or not str(text).strip():
        return [spec.default]

    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return [spec.default]

    seen = set()
    out: List[Any] = []
    for part in parts:
        val = parse_value_token(part, spec)
        key = val if isinstance(val, (int, float, str, bool)) else repr(val)
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


def count_grid_combinations(algorithm: str, grid_text: Dict[str, str]) -> int:
    total = 1
    for spec in get_param_schema(algorithm):
        n = len(parse_value_list(grid_text.get(spec.name, _format_default(spec, spec.default)), spec))
        total *= max(1, n)
    return total


def combo_label(params: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(params.keys()):
        val = params[key]
        if val is None:
            parts.append(f"{key}=∞")
        elif isinstance(val, bool):
            parts.append(f"{key}={'true' if val else 'false'}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def format_params_short(algorithm: str, params: Dict[str, Any]) -> str:
    return combo_label(params)


def build_param_grid(
    algorithm: str,
    grid_text: Dict[str, str],
) -> List[ParamPreset]:
    """
    Cartesian product of all comma-separated lists per hyperparameter.
    Example: trees 300,200 × depth 10,8 → 4 combinations.
    """
    schema = get_param_schema(algorithm)
    defaults = default_params(algorithm)

    value_lists: List[List[Any]] = []
    keys: List[str] = []

    for spec in schema:
        raw = grid_text.get(spec.name, _format_default(spec, defaults[spec.name]))
        values = parse_value_list(raw, spec)
        keys.append(spec.name)
        value_lists.append(values)

    presets: List[ParamPreset] = []
    for combo in itertools.product(*value_lists):
        params = dict(defaults)
        for name, val in zip(keys, combo):
            params[name] = val
        presets.append(ParamPreset(name=combo_label(params), params=params))

    if not presets:
        presets.append(ParamPreset(name=combo_label(defaults), params=defaults))
    return presets


def rank_rows(
    rows: List[Dict[str, Any]],
    metric: str = "recall_mean",
    name_key: str = "combo_name",
) -> Tuple[List[Dict[str, Any]], str | None]:
    if not rows:
        return [], None
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get(metric) is None, -(r.get(metric) or -1)),
    )
    best = sorted_rows[0].get(name_key)
    return sorted_rows, best


def rank_presets(
    rows: List[Dict[str, Any]],
    metric: str = "recall_mean",
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Backward-compatible alias."""
    key = "combo_name" if rows and "combo_name" in rows[0] else "preset_name"
    return rank_rows(rows, metric=metric, name_key=key)
