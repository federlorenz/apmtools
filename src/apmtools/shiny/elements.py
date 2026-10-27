"""Selectable plotted elements, independent of the rendering backend."""
import pandas as pd


def _group_labels(spec):
    frame = spec['data']
    if spec.get('groups') and '__plot_group' in frame:
        return frame['__plot_group'].astype(str)
    return pd.Series('All observations', index=frame.index)


def element_keys(spec):
    """Selectable element identifiers with at least one valid plotted value.

    Elements with exclusively missing observations are not selectable, rather
    than merely being unchecked by default.
    """
    return default_visible_keys(spec)


def filter_elements(spec, selected):
    """Remove hidden elements without changing the underlying Summary or filters.

    For multi-Y lines, individual group/variable combinations are removed by
    masking only their Y values. Panels with no remaining values disappear.
    """
    if 'error' in spec or selected is None:
        return spec
    allowed = set(selected)
    if spec.get('hist_variables'):
        frame = spec['data'].loc[spec['data']['__hist_series'].isin(allowed)].copy()
        return {**spec, 'data': frame, '_visible_elements': allowed}
    if spec.get('multi_axis'):
        from .multi_axis import records
        filtered = {**spec, '_visible_elements': allowed}
        keep = pd.Index([])
        for _, _, _, _, _, rows in records(filtered):
            keep = keep.union(rows.index, sort=False)
        return {**filtered, 'data': spec['data'].loc[spec['data'].index.isin(keep)].copy()}
    frame = spec['data'].copy()
    kind = spec['kind']
    groups = _group_labels(spec)
    if kind == 'line':
        keep_any = pd.Series(False, index=frame.index)
        for y in spec['ys']:
            keys = groups.map(lambda g: f'{g} | {y}' if len(spec['ys']) > 1 else g)
            permitted = keys.isin(allowed)
            frame.loc[~permitted, y] = float('nan')
            keep_any |= permitted & frame[y].notna()
        frame = frame.loc[keep_any].copy()
    elif kind in ('box', 'bar'):
        keys = [f'{c} | {g}' if spec['groups'] else c
                for c, g in zip(frame[spec['x']].astype(str), groups)]
        frame = frame.loc[pd.Series(keys, index=frame.index).isin(allowed)].copy()
    else:
        frame = frame.loc[groups.isin(allowed)].copy()
    return {**spec, 'data': frame, '_visible_elements': allowed}


def default_visible_keys(spec):
    """Elements with at least one nonmissing plotted value (not merely rows)."""
    if 'error' in spec:
        return []
    if spec.get('multi_axis'):
        from .multi_axis import all_keys
        return all_keys(spec, nonmissing=True)
    frame, kind, x = spec['data'], spec['kind'], spec['x']
    if spec.get('hist_variables'):
        return list(dict.fromkeys(frame.loc[frame[x].notna(), '__hist_series'].astype(str)))
    groups = _group_labels(spec)
    if kind == 'line':
        return list(dict.fromkeys(
            f'{group} | {y}' if len(spec['ys']) > 1 else group
            for y in spec['ys'] if y in frame
            for group in groups[frame[y].notna()]
        ))
    if kind in ('box', 'bar'):
        y = next((y for y in spec['ys'] if y in frame), None)
        if y is None:
            return []
        valid = frame[y].notna()
        return list(dict.fromkeys(
            f'{category} | {group}' if spec['groups'] else str(category)
            for category, group in zip(frame.loc[valid, x].astype(str), groups.loc[valid])
        ))
    # Scatter requires both coordinates; histogram requires its numeric x.
    if kind == 'scatter':
        valid = frame[x].notna()
        for y in spec['ys']:
            if y in frame:
                valid &= frame[y].notna()
    else:
        valid = frame[x].notna()
    return list(dict.fromkeys(groups.loc[valid]))


