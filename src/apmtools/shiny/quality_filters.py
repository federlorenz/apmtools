"""Reusable, Shiny-independent subset filtering for quality dashboards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from datetime import datetime, time, timedelta
from .data import _parse_datetime_filter, _subset_choices
from .datetime_intervals import saved_datetime_intervals


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
        intervals = saved_datetime_intervals(settings)
        positives, negatives = [], []
        for window in intervals:
            start_kind, start = _parse_datetime_filter(window['start'])
            end_kind, end = _parse_datetime_filter(window['end'])
            mask = pd.Series(True, index=current.index)
            active = False
            if start_kind in ('date', 'datetime'):
                bound = datetime.combine(start, time.min) if start_kind == 'date' else start
                mask &= timestamps.ge(bound)
                active = True
            if end_kind in ('date', 'datetime'):
                bound = (datetime.combine(end + timedelta(days=1), time.min)
                         if end_kind == 'date' else end)
                mask &= timestamps.lt(bound)
                active = True
            if start_kind == 'time' or end_kind == 'time':
                begin = start if start_kind == 'time' else None
                finish = end if end_kind == 'time' else None
                times = timestamps.dt.time
                if begin is not None and finish is not None:
                    mask &= ((times >= begin) | (times < finish)) if begin > finish else (
                        (times >= begin) & (times < finish))
                elif begin is not None:
                    mask &= times >= begin
                else:
                    mask &= times < finish
                active = True
            if active:
                (negatives if window['type'] == 'negative' else positives).append(mask.fillna(False))
        keep = pd.Series(False if positives else True, index=current.index)
        for mask in positives:
            keep |= mask
        for mask in negatives:
            keep &= ~mask
        current = current.loc[keep.to_numpy()].copy()
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
