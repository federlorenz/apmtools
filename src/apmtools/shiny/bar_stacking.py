"""Stacked-bar grouping, with series identity independent of panel facets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def selected_stack_columns(groups, stacking, panels=()):
    """Retain requested stack dimensions that are grouped and not faceted."""
    requested = [stacking] if isinstance(stacking, str) else (stacking or ())
    panels = [panels] if isinstance(panels, str) else (panels or ())
    return [name for name in dict.fromkeys(requested)
            if name in groups and name not in panels]


def stacked_bar_legend_key(series_key, stack_key):
    """Legend identity for one combination of adjacent and stacked groups.

    Keep the existing stack-only name when no adjacent grouping remains.  Y
    variables distinguish adjacent bar positions, not grouping combinations.
    """
    adjacent_group = series_key[0]
    return f"{adjacent_group} | {stack_key}" if adjacent_group else stack_key


def stacked_bar_segments(spec, stacking):
    """Yield (x category, adjacent series, stack key, value), in data order.

    Aggregate duplicate source rows by summation, matching ordinary barplots.
    Adjacent series are combinations of non-stacking grouping columns and Y
    variables; selected stack dimensions identify the segments within a bar.
    """
    frame = spec['data']
    selected = [c for c in selected_stack_columns(spec.get('groups', ()), stacking)
                if c in frame.columns]
    if not selected or spec.get('kind') != 'bar' or frame.empty:
        return []
    remaining = [c for c in spec['groups'] if c not in selected and c in frame.columns]
    cols = [spec['x'], *remaining, *selected]
    ys = [y for y in spec['ys'] if y in frame.columns]
    if not ys:
        return []
    result = []
    for keys, group in frame.groupby(cols[0] if len(cols) == 1 else cols,
                                     sort=False, dropna=False, observed=True):
        vals = keys if isinstance(keys, tuple) else (keys,)
        category = str(vals[0])
        sides = tuple('None' if pd.isna(v) else str(v) for v in vals[1:1+len(remaining)])
        stacks = tuple('None' if pd.isna(v) else str(v) for v in vals[1+len(remaining):])
        side_label = ' | '.join(sides)
        stack_label = ' | '.join(stacks)
        for y in ys:
            numeric = pd.to_numeric(group[y], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
            valid = numeric[np.isfinite(numeric)]
            if valid.size:
                series = (side_label, str(y) if len(ys) > 1 else '')
                result.append((category, series, stack_label, float(valid.sum())))
    return result
