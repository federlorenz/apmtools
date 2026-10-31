"""Replay configuration-only source transformations for derived tables."""
from __future__ import annotations
import copy
import pandas as pd
import numpy as np
from .data import _subset_choices, _apply_filters, _aggregate
from .config import AGGREGATES, FREQUENCIES
from .columns import _numeric_columns


def _plot_columns(current, step):
    """Resolve the plot's source columns without guessing aggregation suffixes."""
    kind = step.get('plot_type', 'line')
    x = step.get('x') or []
    y = step.get('y') or []
    x = [x] if isinstance(x, str) else list(x)
    y = [y] if isinstance(y, str) else list(y)
    values = x if kind == 'histogram' else y
    values = [str(v) for v in values if str(v) in current.columns]
    groups = step.get('group_by') or []
    groups = [groups] if isinstance(groups, str) else list(groups)
    groups = [str(g) for g in groups if g not in ('variables_x', 'variables_y') and str(g) in current.columns]
    x_columns = [str(v) for v in x if str(v) in current.columns]
    if kind in ('bar', 'box') and AGGREGATES.get(step.get('aggregate')) is not None:
        groups = list(dict.fromkeys(groups + x_columns))
    if kind == 'scatter' and AGGREGATES.get(step.get('aggregate')) is not None:
        values = list(dict.fromkeys(x_columns + values))
    return kind, values, groups, x_columns


def _plot_source(current, step):
    """Reconstruct the data contributing to a plot, with its exact aggregation."""
    kind, values, groups, x_columns = _plot_columns(current, step)
    if not values:
        # Legacy plot steps did not capture axis selections.
        return current
    # Exclude observations that cannot contribute to any selected outcome.
    current = current.loc[current[values].notna().any(axis=1)].copy()
    if current.empty:
        return current
    aggregate = AGGREGATES.get(step.get('aggregate'))
    frequency = FREQUENCIES.get(step.get('frequency'))
    if aggregate is not None:
        if kind == 'line' and x_columns and pd.api.types.is_timedelta64_dtype(current[x_columns[0]]):
            return pd.DataFrame()
        if frequency is None and not groups:
            return pd.DataFrame()
        result = _aggregate(current, values, frequency, groups, aggregate,
                            only_complete=bool(step.get('only_complete', True)) if frequency is not None else True,
                            flooring_timestamp=bool(step.get('flooring_timestamp', True)) if frequency is not None else True)
        if result is None or result.empty:
            return pd.DataFrame()
        # group_stat already emits the grouping and frequency columns.
        return result
    keep = list(dict.fromkeys(x_columns + groups + values))
    if 'datetime' in current.columns and 'datetime' not in keep:
        keep.insert(0, 'datetime')
    return current.loc[:, keep].copy()


def source_frame(summary, steps):
    """Rebuild a table source from its owning Summary and ordered settings.

    Each step is a configuration snapshot, never a DataFrame snapshot.
    """
    current = summary.copy(deep=False)
    for step in steps or []:
        for item in step.get('subset_chain', []):
            name = item.get('column')
            if name not in current.columns:
                raise ValueError(f'Inherited source column is unavailable: {name}')
            _, mapping = _subset_choices(current, name)
            selected = [mapping[k] for k in item.get('values', []) if k in mapping]
            if selected:
                current = current.subset({name: selected}, filter_style=item.get('style', 'all'))
        current = _apply_filters(current, {}, step.get('datetime_start'), step.get('datetime_end'))
        for name in step.get('numeric_subset_columns', []):
            if name not in current.columns:
                raise ValueError(f'Inherited numeric column is unavailable: {name}')
            bounds = step.get('numeric_subset_bounds', {}).get(name, {})
            for op in ('gt', 'lt'):
                try:
                    threshold = float(bounds.get(op, ''))
                except (ValueError, TypeError):
                    continue
                if np.isfinite(threshold):
                    values = pd.to_numeric(current[name], errors='coerce')
                    matches = values.gt(threshold) if op == 'gt' else values.lt(threshold)
                    current = current.loc[matches.fillna(False) | values.isna()].copy()
        # Both plot and table transformations are replayed from the source.
        # A plot-derived table must contain the actual plotted measurements,
        # not all numeric columns from the unprocessed Summary.
        if step.get('_source_kind') == 'plot':
            current = _plot_source(current, step)
        elif step.get('_source_kind') == 'table':
            aggregate = AGGREGATES.get(step.get('aggregate'))
            if aggregate is not None and not current.empty:
                frequency = FREQUENCIES.get(step.get('frequency'))
                groups = list(step.get('group_by') or [])
                numeric = [c for c in _numeric_columns(current) if c not in groups and current[c].notna().any()]
                if numeric:
                    current = _aggregate(current, numeric, frequency, groups, aggregate,
                        only_complete=bool(step.get('only_complete', True)) if frequency is not None else False,
                        flooring_timestamp=bool(step.get('flooring_timestamp', True)))
        current = summary._constructor(current) if not hasattr(current, 'subset') else current
    return current


def source_steps(parent_steps, parent_settings, kind):
    step = copy.deepcopy(parent_settings)
    step['_source_kind'] = kind
    return copy.deepcopy(parent_steps or []) + [step]
