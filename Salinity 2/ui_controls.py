"""Streamlit widgets for hyperparameter grid search."""

from __future__ import annotations

import json
from typing import Dict, List

import streamlit as st

from algo_params import default_params, get_param_schema
from param_presets import (
    _format_default,
    build_param_grid,
    count_grid_combinations,
)
from param_presets import ParamPreset


def _ensure_grid_store(algorithm: str) -> None:
    if "param_grid_store" not in st.session_state:
        st.session_state.param_grid_store = {}
    if algorithm not in st.session_state.param_grid_store:
        defaults = default_params(algorithm)
        st.session_state.param_grid_store[algorithm] = {
            spec.name: _format_default(spec, defaults[spec.name])
            for spec in get_param_schema(algorithm)
        }


def render_param_grid(algorithm: str) -> List[ParamPreset]:
    """
    One text field per hyperparameter. Enter comma-separated values to sweep.
    All combinations are tested on Run (e.g. 300,200 × 10,8 → 4 configs).
    """
    _ensure_grid_store(algorithm)
    grid_text: Dict[str, str] = st.session_state.param_grid_store[algorithm]

    st.markdown(
        "Enter **comma-separated values** per parameter. "
        "The run will test **every combination** and rank results."
    )
    st.caption("Examples: `300,200,150,50` for trees · `10,8,5` for max depth · `0` = unlimited depth")

    toolbar = st.columns([1, 1, 2])
    with toolbar[0]:
        if st.button("↺ Reset to defaults", key=f"grid_reset_{algorithm}"):
            defaults = default_params(algorithm)
            st.session_state.param_grid_store[algorithm] = {
                spec.name: _format_default(spec, defaults[spec.name])
                for spec in get_param_schema(algorithm)
            }
            st.rerun()
    with toolbar[1]:
        uploaded = st.file_uploader(
            "Load grid JSON",
            type=["json"],
            key=f"grid_upload_{algorithm}",
        )
        if uploaded is not None:
            loaded = json.loads(uploaded.getvalue())
            if isinstance(loaded, dict):
                st.session_state.param_grid_store[algorithm].update(loaded)
                st.rerun()

    for spec in get_param_schema(algorithm):
        help_parts = []
        if spec.kind == "choice" and spec.choices:
            help_parts.append(f"Options: {', '.join(map(str, spec.choices))}")
        if spec.kind == "optional_int":
            help_parts.append("Use 0 for unlimited depth")
        if spec.kind == "bool":
            help_parts.append("Use true,false")

        grid_text[spec.name] = st.text_input(
            spec.label,
            value=grid_text.get(spec.name, _format_default(spec, spec.default)),
            help=" · ".join(help_parts) if help_parts else "Single value or comma-separated list",
            key=f"grid_{algorithm}_{spec.name}",
        )

    st.session_state.param_grid_store[algorithm] = grid_text

    try:
        n_combos = count_grid_combinations(algorithm, grid_text)
        presets = build_param_grid(algorithm, grid_text)
    except ValueError as exc:
        st.error(str(exc))
        return []

    st.info(f"**{n_combos}** hyperparameter combination(s) will be tested (× your run count / CV folds).")

    with st.expander("Preview combinations"):
        for p in presets[:20]:
            st.write(f"• {p.name}")
        if len(presets) > 20:
            st.caption(f"… and {len(presets) - 20} more")

    st.download_button(
        "Download grid (JSON)",
        data=json.dumps(grid_text, indent=2),
        file_name=f"{algorithm}_grid.json",
        mime="application/json",
    )

    return presets
