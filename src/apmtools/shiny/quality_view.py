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


def quality_heatmap_data(report, variable):
    """Interval coverage matrix (identifiers x bins); missing data remain NaN."""
    if report.empty:
        return pd.DataFrame()
    key = report.columns[0]
    portion = report.loc[report['variable'].eq(variable), [key, 'interval', 'completeness']].copy()
    if portion.empty:
        return pd.DataFrame()
    return portion.pivot_table(index=key, columns='interval', values='completeness', aggfunc='mean', dropna=False)


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
