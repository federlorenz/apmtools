"""Multi-variable categorical and Cartesian X/Y scatter rendering.

Each trace is identified by a stable element key, independently of the backend.
"""
from __future__ import annotations
import pandas as pd


def records(spec):
    """Yield (key, group, y_variable, x_variable, category, rows) with valid values."""
    df, kind = spec['data'], spec['kind']
    groups = df['__plot_group'].astype(str) if spec['groups'] and '__plot_group' in df else pd.Series('All observations', index=df.index)
    xvars = spec.get('x_variables', [spec['x']])
    ys = spec['ys']
    allowed = spec.get('_visible_elements')
    for group in pd.unique(groups):
        part = df.loc[groups == group]
        original_groups = (part['__plot_global_group'].astype(str)
                           if spec.get('_original_groups') and '__plot_global_group' in part
                           else pd.Series(group, index=part.index))
        for y in ys:
            for xv in (xvars if kind == 'scatter' else [spec['x']]):
                if kind == 'scatter':
                    valid = part[xv].notna() & part[y].notna()
                    if allowed is not None:
                        keys = original_groups.map(lambda g: f'{g} | {xv} × {y}' if spec.get('_original_groups', spec['groups']) else f'{xv} × {y}')
                        valid &= keys.isin(allowed)
                    rows = part.loc[valid]
                    key = f'{group} | {xv} × {y}' if spec['groups'] else f'{xv} × {y}' 
                    if not rows.empty:
                        yield key, group, y, xv, None, rows
                else:
                    for category, cat_rows in part.groupby(spec['x'], sort=False, dropna=False):
                        key = f'{category} | {group} | {y}' if spec['groups'] else f'{category} | {y}'
                        valid = cat_rows[y].notna()
                        if allowed is not None:
                            original = original_groups.reindex(cat_rows.index)
                            keys = original.map(lambda g: f'{category} | {g} | {y}' if spec.get('_original_groups', spec['groups']) else f'{category} | {y}')
                            valid &= keys.isin(allowed)
                        rows = cat_rows.loc[valid]
                        if not rows.empty:
                            yield key, group, y, xv, str(category), rows


def all_keys(spec, nonmissing=False):
    """Return element choices; optionally omit elements with only missing values."""
    if 'error' in spec:
        return []
    if nonmissing:
        return list(dict.fromkeys(r[0] for r in records(spec)))
    df, kind = spec['data'], spec['kind']
    groups = (df['__plot_group'].astype(str) if spec['groups'] and '__plot_group' in df
              else pd.Series('All observations', index=df.index))
    keys = []
    for group in pd.unique(groups):
        part = df.loc[groups == group]
        for y in spec['ys']:
            if kind == 'scatter':
                for xv in spec.get('x_variables', [spec['x']]):
                    keys.append(f'{group} | {xv} × {y}' if spec['groups'] else f'{xv} × {y}')
            else:
                for cat in pd.unique(part[spec['x']].astype(str)):
                    keys.append(f'{cat} | {group} | {y}' if spec['groups'] else f'{cat} | {y}')
    return list(dict.fromkeys(keys))


def style_maps(spec, mode, palette, overrides):
    from .config import _MPL_PALETTES
    from .multi_y import multi_y_styles
    from .appearance import _parse_label_map
    import matplotlib.colors as mcolors
    groups = (list(pd.unique(spec['data']['__plot_group'].astype(str)))
              if spec['groups'] and '__plot_group' in spec['data'] else [])
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES['Default'])
    colors, dashes, _, _ = multi_y_styles(spec['ys'], groups, palette_values, mode,
                                         _parse_label_map(overrides), {},
                                         aliases=spec.get("legend_aliases", {}))
    return colors, dashes, groups
