"""Replay a table's active view as the source for a derived plot.

Only transformation settings, never an in-memory table snapshot, are retained
in saved plot configurations.  Each plot gets its own immutable source lineage.
"""
from __future__ import annotations

import copy
import pandas as pd

from .columns import (_alphabetical, _categorical_columns, _datetime_columns,
                      _numeric_columns, _timedelta_columns, _default_x)
from .config import FREQUENCIES
from .inherited_source import source_frame


def plot_source(summary, settings):
    """Replay the source lineage for a plot, or use its owning Summary."""
    steps = (settings or {}).get('source_steps') or []
    if not steps:
        return summary
    frame = source_frame(summary, steps)
    # Preserve Summary methods required by the Shiny filtering pipeline.
    if not hasattr(frame, 'subset'):
        frame = summary._constructor(frame)
    return frame


def plot_source_context(summary, settings):
    """Classify the *actual* data available to a plot's selectors."""
    frame = plot_source(summary, settings)
    columns = _alphabetical(frame.columns)
    col_map = {str(c): c for c in frame.columns}
    numeric = _alphabetical(_numeric_columns(frame))
    categorical = _alphabetical(_categorical_columns(frame) + [name for name in FREQUENCIES.values()
        if name is not None and name in frame.columns])
    datetime = _datetime_columns(frame)
    timedelta = _timedelta_columns(frame)
    virtual_time = [f for f in FREQUENCIES.values() if f is not None and f not in col_map]
    physical_groups = _alphabetical([c for c in columns
        if not pd.api.types.is_numeric_dtype(frame[col_map[c]])]
        + [f for f in FREQUENCIES.values() if f is not None and f in col_map])
    for name in virtual_time:
        col_map[name] = name
    return dict(summary=frame, columns=columns, column_map=col_map,
                numeric_columns=numeric, categorical_columns=categorical,
                datetime_columns=datetime, timedelta_columns=timedelta,
                time_group_columns=virtual_time, physical_group_columns=physical_groups,
                group_columns=physical_groups, subset_columns=columns,
                max_subset_stages=max(2, len(columns) + 1))


def table_plot_settings(summary, parent_steps, parent_settings):
    """Create editable plotting defaults on top of a table's current view."""
    from .derived_views import prepare_table_source
    frame, steps = prepare_table_source(summary, parent_kind='table',
                                        parent_steps=parent_steps,
                                        parent_settings=parent_settings)
    if frame.empty or not len(frame.columns):
        raise ValueError('The selected table contains no data to plot')
    datetime = _datetime_columns(frame)
    timedelta = _timedelta_columns(frame)
    numeric = _numeric_columns(frame)
    x = _default_x(frame)
    # Prefer an already-aggregated mean over its auxiliary count/variance
    # columns when selecting the initial outcome.
    preferred = [c for c in numeric if c.endswith('_mean')]
    y = (preferred or [c for c in numeric if c != x])[:1]
    groups = ['identifier'] if 'identifier' in frame.columns else []
    if datetime or timedelta:
        kind, x_value, y_value = 'line', (datetime + timedelta)[0], y
    elif categorical := [c for c in _categorical_columns(frame) if c not in groups]:
        kind, x_value, y_value = 'bar', [categorical[0]], y
    elif len(numeric) >= 2:
        kind, x_value, y_value = 'scatter', [numeric[0]], [numeric[1]]
    elif numeric:
        kind, x_value, y_value = 'histogram', [numeric[0]], []
    else:
        kind, x_value, y_value = 'bar', [x], []
    return {'source_steps': copy.deepcopy(steps), 'plot_type': kind,
            'x': x_value, 'y': y_value, 'group_by': groups,
            'frequency': 'None', 'aggregate': [], 'only_complete': False,
            'error_bands': False}
