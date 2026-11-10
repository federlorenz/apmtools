"""Shiny-independent, serializable transformation plans for Summary views.

The unified executor is the default for derived-table reconstruction. Other
Shiny plot calculations retain their established processing paths.
Plans describe operations, not materialized DataFrame snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .columns import _numeric_columns
from .config import AGGREGATES, FREQUENCIES
from .data import _aggregate, _apply_filters, _subset_choices
from .inherited_source import _plot_source
from .view_aggregation import normalize_statistics


class TransformError(ValueError):
    """Invalid or non-executable transformation plan."""


_OPERATIONS = frozenset({'subset', 'subset_values', 'datetime', 'numeric', 'select',
                         'plot_source', 'table_source', 'aggregate', 'reaggregate'})


@dataclass(frozen=True)
class TransformStep:
    operation: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        if self.operation not in _OPERATIONS:
            raise TransformError(f'Unsupported transformation operation: {self.operation!r}')
        if not isinstance(self.parameters, Mapping):
            raise TransformError('Step parameters must be a mapping')
        object.__setattr__(self, 'parameters', copy.deepcopy(dict(self.parameters)))

    def to_dict(self) -> dict:
        return {'operation': self.operation, 'parameters': copy.deepcopy(self.parameters)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'TransformStep':
        if not isinstance(payload, Mapping):
            raise TransformError('A transformation step must be a mapping')
        return cls(payload.get('operation'), payload.get('parameters', {}))


@dataclass(frozen=True)
class DataViewSpec:
    steps: tuple[TransformStep, ...] = ()
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise TransformError(f'Unsupported transformation schema version: {self.schema_version}')
        object.__setattr__(self, 'steps', tuple(self.steps))
        if not all(isinstance(step, TransformStep) for step in self.steps):
            raise TransformError('All steps must be TransformStep instances')

    def to_dict(self) -> dict:
        return {'schema_version': self.schema_version,
                'steps': [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'DataViewSpec':
        if not isinstance(payload, Mapping):
            raise TransformError('A view specification must be a mapping')
        return cls(tuple(TransformStep.from_dict(step) for step in payload.get('steps', [])),
                   payload.get('schema_version', 1))


def from_legacy_source_steps(steps: list[dict] | None) -> DataViewSpec:
    """Compile saved source_steps without modifying the legacy dictionaries.

    Numeric/categorical UI option keys are kept unchanged: option-key stability
    is a separate compatibility concern that must be addressed at migration time.
    """
    operations = []
    for settings in steps or []:
        for item in settings.get('subset_chain', []):
            operations.append(TransformStep('subset', item))
        operations.append(TransformStep('datetime', {
            'start': settings.get('datetime_start'),
            'end': settings.get('datetime_end'),
            'intervals': settings.get('datetime_intervals')}))
        for name in settings.get('numeric_subset_columns', []):
            operations.append(TransformStep('numeric', {
                'column': name,
                'bounds': (settings.get('numeric_subset_bounds') or {}).get(name, {})}))
        kind = settings.get('_source_kind')
        if kind == 'plot':
            operations.append(TransformStep('plot_source', settings))
        elif kind == 'table':
            operations.append(TransformStep('table_source', settings))
    return DataViewSpec(tuple(operations))



def _reaggregate_materialized(current, p, *, explicit=False):
    """Summarize pre-aggregated rows, not the original underlying observations.

    Each incoming row has equal weight. In particular, mean(hourly means)
    is NOT a reading-weighted mean; reconstructing the latter requires counts
    or access to raw records. Completeness cannot be re-inferred from means.
    """
    from .view_aggregation import aggregate_view, ViewAggregationError, normalize_statistics
    groups = list(p.get('group_by') or [])
    values = list(p.get('value_columns') or []) if explicit else [
        c for c in (p.get('aggregate_columns') if p.get('aggregate_columns') is not None
                    else _numeric_columns(current))
        if c in current.columns and c not in groups and current[c].notna().any()]
    if not values or not normalize_statistics(p.get('aggregate')) or current.empty:
        return current
    if explicit and p.get('only_complete', False):
        raise TransformError('Completeness cannot be re-evaluated on aggregated rows')
    try:
        return aggregate_view(current, values, p.get('frequency'), groups,
                              p.get('aggregate'),
                              only_complete=(bool(p.get('only_complete', False)) if explicit else False),
                              time_column=p.get('time_column'))
    except ViewAggregationError as exc:
        raise TransformError(str(exc)) from exc


def execute_view(summary, spec: DataViewSpec | Mapping[str, Any]):
    """Execute an ordered view plan against a Summary without mutating it.

    Keeps Summary methods and the established aggregation implementation as the
    authoritative numerical semantics. Returns a new working frame.
    """
    if not isinstance(spec, DataViewSpec):
        spec = DataViewSpec.from_dict(spec)
    current = summary.copy(deep=False)
    for step in spec.steps:
        p = step.parameters
        if step.operation == 'subset':
            name = p.get('column')
            if name not in current.columns:
                raise TransformError(f'Inherited source column is unavailable: {name}')
            _, mapping = _subset_choices(current, name)
            selected = [mapping[key] for key in p.get('values', []) if key in mapping]
            if selected:
                current = current.subset({name: selected}, filter_style=p.get('style', 'all'))
        elif step.operation == 'subset_values':
            # New plans use typed category values rather than positional UI
            # keys. The legacy subset operation remains unchanged.
            name = p.get('column')
            if name not in current.columns:
                raise TransformError(f'Selected category column unavailable: {name}')
            values = list(p.get('values') or [])
            if values:
                current = current.subset({name: values}, filter_style=p.get('style', 'all'))
        elif step.operation == 'datetime':
            current = _apply_filters(current, {}, p.get('start'), p.get('end'),
                                     datetime_intervals=p.get('intervals'))
        elif step.operation == 'numeric':
            name = p.get('column')
            if name not in current.columns:
                raise TransformError(f'Inherited numeric column is unavailable: {name}')
            bounds = p.get('bounds') or {}
            for op in ('gt', 'lt'):
                try:
                    threshold = float(bounds.get(op, ''))
                except (ValueError, TypeError):
                    continue
                if np.isfinite(threshold):
                    values = pd.to_numeric(current[name], errors='coerce')
                    matches = values.gt(threshold) if op == 'gt' else values.lt(threshold)
                    current = current.loc[matches.fillna(False) | values.isna()].copy()
        elif step.operation == 'select':
            columns = list(p.get('columns', []))
            missing = [c for c in columns if c not in current.columns]
            if missing:
                raise TransformError(f'Selected columns unavailable: {missing}')
            current = current.loc[:, columns].copy()
        elif step.operation == 'plot_source':
            current = _plot_source(current, p)
        elif step.operation == 'reaggregate':
            current = _reaggregate_materialized(current, p, explicit=True)
        elif step.operation in ('table_source', 'aggregate'):
            aggregate = normalize_statistics(p.get('aggregate'))
            if aggregate and not current.empty:
                frequency = FREQUENCIES.get(p.get('frequency'))
                groups = list(p.get('group_by') or [])
                if step.operation == 'aggregate':
                    numeric = list(p.get('value_columns') or [])
                    missing = [c for c in numeric if c not in current.columns]
                    if missing:
                        raise TransformError(f'Aggregation columns unavailable: {missing}')
                else:
                    chosen = p.get('aggregate_columns')
                    numeric = [c for c in (chosen if chosen is not None else _numeric_columns(current))
                               if c in current.columns and c not in groups and current[c].notna().any()]
                if numeric:
                    if step.operation == 'table_source' and 'datetime' not in current.columns:
                        # New-engine compatibility repair: the previous table
                        # may already be aggregated. Never fabricate timestamps
                        # or silently re-apply raw-observation completeness.
                        current = _reaggregate_materialized(current, p)
                    else:
                        current = _aggregate(current, numeric, frequency, groups, aggregate,
                            only_complete=bool(p.get('only_complete', True)) if frequency is not None else False,
                            flooring_timestamp=bool(p.get('flooring_timestamp', True)))
        current = summary._constructor(current) if not hasattr(current, 'subset') else current
    return current
