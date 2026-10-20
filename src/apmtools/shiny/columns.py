"""Column classification and default selections."""
from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES


def _default_x(data: pd.DataFrame) -> str:
    """Return ``time`` when available, otherwise the first column."""
    if "datetime" in data.columns:
        return "datetime"
    datetime_columns = [
        column
        for column in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[column])
    ]
    return str(datetime_columns[0] if datetime_columns else data.columns[0])



def _default_group(data: pd.DataFrame) -> str | None:
    """Return ``identifier`` when available."""
    return "identifier" if "identifier" in data.columns else None



def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric column names as strings."""
    return sorted([
        str(column)
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ])



def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return categorical column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if not pd.api.types.is_numeric_dtype(data[column])
            and not pd.api.types.is_datetime64_any_dtype(data[column])
        ],
        key=str.casefold,
    )



def _datetime_columns(data: pd.DataFrame) -> list[str]:
    """Return datetime column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if pd.api.types.is_datetime64_any_dtype(data[column])
        ],
        key=str.casefold,
    )



def _timedelta_columns(data: pd.DataFrame) -> list[str]:
    """Return timedelta column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if pd.api.types.is_timedelta64_dtype(data[column])
        ],
        key=str.casefold,
    )



def _alphabetical(values) -> list[str]:
    """Return string values sorted alphabetically, case-insensitively."""
    return sorted((str(value) for value in values), key=str.casefold)



def _normalise_y(data: pd.DataFrame, x: str, y: str | Sequence[str] | None) -> list[str]:
    """Validate and normalise an initial Y selection."""
    numeric = _numeric_columns(data)
    if y is None:
        return [column for column in numeric if column != x][:1]
    values = [y] if isinstance(y, str) else list(y)
    missing = [column for column in values if column not in data.columns]
    if missing:
        raise ValueError(f"Unknown y column(s): {missing}")
    return [str(column) for column in values]

