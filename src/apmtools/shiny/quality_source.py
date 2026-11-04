"""Construct quality sources from raw observations and inherited filters only.

Unlike a derived table, a quality view must never inherit aggregation,
column projection or plot-specific row selection: it needs the original
identifier, timestamp and measurement columns to assess data coverage.
"""
from __future__ import annotations

import copy

from .inherited_source import source_frame


_FILTER_FIELDS = ('subset_chain', 'datetime_start', 'datetime_end',
                  'numeric_subset_columns', 'numeric_subset_bounds')


def quality_filter_settings(settings, columns):
    """Copy only filters valid against raw columns, excluding derived columns."""
    settings = settings or {}
    available = set(columns)
    result = {key: copy.deepcopy(settings[key]) for key in _FILTER_FIELDS if key in settings}
    result['subset_chain'] = [item for item in result.get('subset_chain', [])
                              if item.get('column') in available]
    result['numeric_subset_columns'] = [name for name in result.get('numeric_subset_columns', [])
                                         if name in available]
    result['numeric_subset_bounds'] = {
        name: bounds for name, bounds in result.get('numeric_subset_bounds', {}).items()
        if name in result['numeric_subset_columns']
    }
    return result


def quality_filter_lineage(steps, columns):
    """Convert saved plot/table lineage into filter-only steps on raw data."""
    return [quality_filter_settings(step, columns) for step in (steps or [])]


def prepare_quality_source(summary, *, parent_steps=None, restored_settings=None):
    """Return raw-column dataframe with only inherited filters replayed.

    Quality's own editable filters live in its tab settings, and are applied
    separately. The source lineage contains only earlier table steps.
    Legacy saved quality views are accepted; all aggregation/projection stages
    are removed when loading them.
    """
    steps = quality_filter_lineage(
        (restored_settings or {}).get('source_steps', []) if restored_settings is not None
        else parent_steps, summary.columns)
    frame = source_frame(summary, steps).reset_index(drop=True)
    frame.columns = [str(column) for column in frame.columns]
    return frame, steps
