"""Headless data-quality summaries and heatmap preparation.

The dashboard never infers completeness independently of the Stage 4A engine.
"""
from __future__ import annotations

import pandas as pd
from ..completeness import analyse_completeness

QUALITY_FREQUENCIES = ('minute', 'hour', 'day', 'week')


def quality_report(data, settings):
    """Recompute per-identifier, interval and variable coverage from source data."""
    values = list(settings.get('value_cols', []))
    if not values:
        raise ValueError('Select at least one measurement variable')
    freq = settings.get('frequency', 'hour')
    if freq not in QUALITY_FREQUENCIES:
        raise ValueError(f'Unsupported dashboard frequency: {freq}')
    return analyse_completeness(
        data, values, freq,
        id_col=settings.get('id_col', 'identifier'),
        time_col=settings.get('time_col', 'datetime'),
        min_completeness=float(settings.get('min_completeness', 1.0)),
        include_empty_intervals=bool(settings.get('include_empty_intervals', True)),
    )


def quality_summary(report):
    """Per-monitor and variable coverage, including zero-observation bins."""
    names = ['identifier', 'variable', 'intervals', 'complete_intervals',
             'observed', 'expected', 'missing', 'coverage_percent',
             'complete_percent']
    if report.empty:
        return pd.DataFrame(columns=names)
    key = report.columns[0]
    stats = (report.groupby([key, 'variable'], dropna=False, sort=True)
             .agg(intervals=('is_complete', 'size'),
                  complete_intervals=('is_complete', 'sum'),
                  observed=('observed', 'sum'),
                  expected=('expected', 'sum'),
                  missing=('missing', 'sum')).reset_index())
    stats['coverage_percent'] = (100 * stats['observed'] / stats['expected']).where(stats['expected'] > 0)
    stats['complete_percent'] = 100 * stats['complete_intervals'] / stats['intervals']
    return stats.rename(columns={key: 'identifier'})[names]


def selected_heatmap_variable(value_cols, selected):
    """Return a heatmap variable present in the coverage report.

    Changing the selected measurements cannot leave the heatmap looking for a
    numeric column that was not included in the report.
    """
    choices = list(value_cols or [])
    return selected if selected in choices else (choices[0] if choices else None)


def quality_heatmap_data(report, variable, *, mode='coverage'):
    """Return interval-by-identifier heatmap values on a 0–1 scale.

    ``mode='coverage'`` shows the measured observed/expected fraction, without
    reference to the completeness threshold. ``mode='threshold'`` is the
    historical binary classification (1=complete, 0=incomplete).
    ``mode='threshold_coverage'`` shows 1 for bins meeting the threshold, and
    the *actual* fraction for bins failing it. This retains the continuous
    shades for incomplete bins without losing the threshold classification.
    Undefined expected counts remain NaN in every mode.
    """
    if mode not in ('coverage', 'threshold', 'threshold_coverage'):
        raise ValueError("heatmap mode must be 'coverage', 'threshold', or 'threshold_coverage'")
    if report.empty:
        return pd.DataFrame()
    key = report.columns[0]
    portion = report.loc[report['variable'].eq(variable)].copy()
    if portion.empty:
        return pd.DataFrame()
    if mode == 'threshold':
        portion['_heatmap_value'] = portion['is_complete'].astype(float).where(
            portion['expected'].notna())
    elif mode == 'threshold_coverage':
        portion['_heatmap_value'] = portion['completeness'].where(
            ~portion['is_complete'].astype(bool), 1.0)
        portion['_heatmap_value'] = portion['_heatmap_value'].where(
            portion['expected'].notna())
    else:
        portion['_heatmap_value'] = portion['completeness']
    return portion.pivot_table(index=key, columns='interval',
                               values='_heatmap_value', aggfunc='mean', dropna=False)


def missing_interval_runs(report):
    """Group consecutive zero-observation bins into gap episodes per monitor/variable.

    A new episode begins whenever an observed bin separates missing bins.
    The report must include empty intervals to expose fully absent time bins.
    """
    fields = ['identifier', 'variable', 'first_missing_interval', 'last_missing_interval',
              'missing_intervals', 'expected_observations']
    if report.empty:
        return pd.DataFrame(columns=fields)
    key = report.columns[0]
    rows = []
    for (identifier, variable), group in report.groupby([key, 'variable'], sort=False, dropna=False):
        ordered = group.sort_values('interval')
        run = []
        def finish():
            if run:
                rows.append((identifier, variable, run[0]['interval'], run[-1]['interval'],
                             len(run), sum(record['expected'] for record in run)))
        for record in ordered.to_dict('records'):
            if record['observed'] == 0:
                run.append(record)
            else:
                finish()
                run = []
        finish()
    return pd.DataFrame(rows, columns=fields)


def quality_grouped_heatmaps(data, settings):
    """Return ``(group title, coverage matrix)`` pairs for selected groupings.

    Grouping is performed on raw filtered observations, independently of any
    plot/table aggregation. The Stage 4A engine is reused for each subset.
    """
    group_cols = list(dict.fromkeys(settings.get('group_by', []) or []))
    missing = [name for name in group_cols if name not in data.columns]
    if missing:
        raise ValueError(f"Grouping columns are absent: {missing}")
    if any(name == settings.get('time_col', 'datetime') for name in group_cols):
        raise ValueError('The timestamp column cannot be used for grouping')
    variable = selected_heatmap_variable(settings.get('value_cols', []), settings.get('heatmap_variable'))
    if data.empty or not variable:
        return []
    if not group_cols:
        return [('All observations', quality_heatmap_data(
            quality_report(data, settings), variable, mode='threshold_coverage'))]
    results = []
    for key, subset in data.groupby(group_cols, dropna=False, sort=True, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        title = ', '.join(f'{name}: {value}' for name, value in zip(group_cols, keys))
        matrix = quality_heatmap_data(quality_report(subset, settings), variable,
                                      mode='threshold_coverage')
        results.append((title, matrix))
    return results


def quality_heatmap_height(matrix):
    """Responsive per-group output height in CSS pixels (one figure per group).

    Expand with the number of monitor rows, without any shared fixed-height
    viewport or cap that would compress larger groups.
    """
    return max(360, 270 + 32 * len(matrix.index))


def quality_heatmap_figure(title, matrix, settings):
    """Draw exactly one independent, titled completeness heatmap.

    The Matplotlib figure and colourbar belong only to the supplied group.
    ``quality_grouped_heatmaps`` computes matrices once upstream and the
    Shiny output renders each independently.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    height = quality_heatmap_height(matrix)
    fig, ax = plt.subplots(figsize=(10, height / 100))
    if matrix.empty:
        ax.text(0.5, 0.5, "No data for selected measurement",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(str(title))
    else:
        colors = plt.get_cmap("viridis").copy()
        colors.set_bad("lightgray")
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto",
                          interpolation="nearest", cmap=colors,
                          norm=Normalize(vmin=0, vmax=1))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([str(value) for value in matrix.index], fontsize=8)
        count = len(matrix.columns)
        step = max(1, count // 8)
        ticks = sorted(set([0, count - 1] + list(range(0, count, step))))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(matrix.columns[i]) for i in ticks],
                           rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Aggregation interval")
        ax.set_ylabel(str(settings.get("id_col", "identifier")))
        ax.set_title(
            f"{title} — {settings.get('heatmap_variable')} "
            f"(complete at ≥ {float(settings.get('min_completeness', 1)):.0%})"
        )
        fig.colorbar(image, ax=ax, ticks=[0, .25, .5, .75, 1],
                     label="Coverage (passing bins displayed as 100%)")
    fig.tight_layout()
    return fig
