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
                                         _parse_label_map(overrides), {})
    return colors, dashes, groups


def plotly_multi_axis(spec, *, mode='group', palette='Default', legend_labels='', trace_colors=''):
    import plotly.graph_objects as go
    from .appearance import _parse_label_map
    from .config import _MPL_PALETTES
    from .appearance import _axis_label
    from .multi_y import multi_y_styles
    df, kind = spec['data'], spec['kind']
    entries = list(records(spec))
    fig = go.Figure()
    colors, dashes, groups = style_maps(spec, mode, palette, trace_colors)
    rename = _parse_label_map(legend_labels)
    # Discrete hatching distinguishes variables when colours are assigned by group.
    patterns = ['', '/', '\\', 'x', '.', '+', '-']
    yvars = list(spec['ys'])
    xvars = list(spec.get('x_variables', [spec['x']]))
    active = set()
    for key, group, y, xv, cat, rows in entries:
        g = group if groups else None
        color = colors[(y, g)]
        active.add((y, g))
        if kind == 'scatter':
            symbol = ['circle', 'square', 'diamond', 'cross', 'triangle-up'][xvars.index(xv) % 5]
            fig.add_trace(go.Scatter(x=rows[xv], y=rows[y], mode='markers', name=key,
                                     marker=dict(color=color, symbol=symbol, opacity=.8),
                                     showlegend=False, customdata=[[key]] * len(rows), hovertemplate='%{x}, %{y}<extra>'+key+'</extra>'))
        elif kind == 'bar':
            # Match the original Matplotlib bar semantics: sum repeated observations.
            value = pd.to_numeric(rows[y], errors='coerce').sum()
            fig.add_trace(go.Bar(x=[cat], y=[value], name=key, showlegend=False,
                                 marker=dict(color=color, pattern=dict(shape=patterns[yvars.index(y) % len(patterns)]
                                     if mode == 'group' else '' if mode == 'none' else patterns[groups.index(g) % len(patterns)] if groups else ''))))
        else:
            fig.add_trace(go.Box(x=[cat] * len(rows), y=rows[y], name=key, showlegend=False,
                                 marker_color=color, line_color=color, boxpoints='outliers'))
    if kind == 'bar':
        fig.update_layout(barmode='group')
    if kind == 'box':
        fig.update_layout(boxmode='group')
    if not active:
        fig.update_layout(showlegend=False)
        return fig
    # Legend entries are proxies, avoiding repetition for each category.
    def proxy(label, color, symbol='circle', blank=False):
        if kind == 'scatter':
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=rename.get(label, label),
                                     marker=dict(color=color, symbol=symbol), showlegend=True, hoverinfo='skip'))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name=rename.get(label, label),
                                     line=dict(color=color, width=0 if blank else 3), showlegend=True, hoverinfo='skip'))
    if mode == 'none':
        for g in groups:
            if any((y, g) in active for y in yvars):
                proxy(g, colors[(yvars[0], g)])
    elif mode == 'both':
        for g in (groups or [None]):
            for y in yvars:
                if (y, g) in active:
                    proxy(f'{g}, {y}' if g is not None else y, colors[(y, g)])
    else:
        for y in yvars:
            pairs = [(y, g) for g in (groups or [None]) if (y, g) in active]
            if pairs:
                proxy(y, colors[pairs[0]])
        if groups:
            proxy(' ', 'rgba(0,0,0,0)', blank=True)
            for g in groups:
                pairs = [(y, g) for y in yvars if (y, g) in active]
                if pairs:
                    proxy(g, colors[pairs[0]])
    fig.update_layout(showlegend=bool(fig.data and any(t.showlegend for t in fig.data)))
    return fig
