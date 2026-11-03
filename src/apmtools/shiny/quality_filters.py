"""Reusable, Shiny-independent subset filtering for quality dashboards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from datetime import datetime, time, timedelta
from .data import _parse_datetime_filter, _subset_choices


def filter_quality_data(frame, settings):
    """Apply the same staged categorical, numeric and date filters as tables.

    The configured selected values are Shiny selector *keys*, resolved against
    the source dataframe. Missing numeric values remain unchanged, consistent
    with the existing table filtering policy.
    """
    current = frame.copy()
    for step in settings.get('subset_chain', []):
        column = step.get('column')
        if column not in current.columns:
            continue
        _, value_map = _subset_choices(frame, column)
        values = [value_map[key] for key in step.get('values', []) if key in value_map]
        if not values:
            continue
        style = step.get('style', 'all')
        if hasattr(current, 'subset'):
            current = current.subset({column: values}, filter_style=style)
        else:
            matches = current[column].isin([v for v in values if v is not None])
            if any(v is None for v in values):
                matches |= current[column].isna()
            if style in ('negative all', 'negative any'):
                matches = ~matches
            current = current.loc[matches].copy()
    # Derived plot/table frames are often ordinary pandas DataFrames (without
    # Summary.dtfilter). Filter their actual datetime column without assuming
    # that the source column is literally named "datetime".
    time_col = settings.get('time_col')
    if time_col not in current.columns or not pd.api.types.is_datetime64_any_dtype(current[time_col]):
        time_col = next((col for col in current.columns
                         if pd.api.types.is_datetime64_any_dtype(current[col])), None)
    if time_col is not None:
        timestamps = pd.to_datetime(current[time_col], errors='coerce')
        mask = pd.Series(True, index=current.index)
        for field, lower in (('datetime_start', True), ('datetime_end', False)):
            kind, value = _parse_datetime_filter(settings.get(field))
            if kind == 'date':
                bound = datetime.combine(value, time.min)
                if lower:
                    mask &= timestamps.ge(bound)
                else:
                    mask &= timestamps.lt(bound + timedelta(days=1))
            elif kind == 'datetime':
                mask &= timestamps.ge(value) if lower else timestamps.lt(value)
            elif kind == 'time':
                times = timestamps.dt.time
                mask &= times.ge(value) if lower else times.lt(value)
        current = current.loc[mask.fillna(False)].copy()
    for column in settings.get('numeric_subset_columns', []):
        if column not in current.columns or not pd.api.types.is_numeric_dtype(current[column]):
            continue
        values = pd.to_numeric(current[column], errors='coerce')
        for comparison in ('gt', 'lt'):
            raw = settings.get('numeric_subset_bounds', {}).get(column, {}).get(comparison, '')
            try:
                bound = float(str(raw).strip())
            except (ValueError, TypeError):
                continue
            if not np.isfinite(bound):
                continue
            mask = values.gt(bound) if comparison == 'gt' else values.lt(bound)
            current = current.loc[mask.fillna(False) | values.isna()].copy()
            values = values.loc[current.index]
    return current
