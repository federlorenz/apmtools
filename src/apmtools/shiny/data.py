"""Filtering, aggregation and temporal segmentation."""
from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES


def _composite_labels(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build readable composite labels without dropping or coercing missing groups.

    In pandas 3, DataFrame.agg(" | ".join, axis=1) can receive a float
    NaN even after astype(str), so convert every scalar at the join boundary.
    The original columns are left untouched for filtering and aggregation.
    """
    if not columns:
        raise ValueError("At least one label column is required")
    def label(value):
        return "None" if pd.isna(value) else str(value)
    if len(columns) == 1:
        return data[columns[0]].map(label)
    return pd.Series(
        (" | ".join(label(value) for value in row)
         for row in data[columns].itertuples(index=False, name=None)),
        index=data.index,
        dtype="str",
    )



def _subset_choices(data: pd.DataFrame, column: str) -> tuple[dict[str, str], dict[str, object]]:
    """Build stable Shiny choices and their corresponding original values."""
    values = sorted(pd.unique(data[column].dropna()), key=lambda value: str(value).casefold())
    # Missing values receive a distinct selector key, not the literal string "None".
    has_missing = bool(data[column].isna().any())
    choices: dict[str, str] = {}
    value_map: dict[str, object] = {}
    for i, value in enumerate(values):
        key = str(i)
        choices[key] = str(value)
        value_map[key] = value
    if has_missing:
        key = str(len(values))
        choices[key] = "None (missing)"
        value_map[key] = None
    return choices, value_map



def _parse_datetime_filter(value):
    """Parse an optional date, time or datetime without accepting partial input.

    Returns ``(kind, parsed)``; incomplete/invalid text is marked invalid so
    it cannot raise an exception while a user is still typing.
    """
    import re

    value = (value or "").strip()
    if not value:
        return "empty", None
    try:
        date_pattern = r"(\d{4})-(\d{1,2})-(\d{1,2})"
        match = re.fullmatch(date_pattern, value)
        if match:
            return "date", date(*(int(part) for part in match.groups()))
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?", value):
            return "time", time.fromisoformat(value)
        match = re.fullmatch(date_pattern + r"[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?", value)
        if match:
            year, month, day, hour, minute, second, microsecond = match.groups()
            return "datetime", datetime(
                int(year), int(month), int(day), int(hour), int(minute),
                int(second or 0), int((microsecond or "0").ljust(6, "0")),
            )
    except ValueError:
        pass
    return "invalid", None



def _apply_filters(data, subset_filters, datetime_start=None, datetime_end=None):
    """Apply Summary filters, accepting dates, times and full datetimes.

    Invalid or partially entered values are ignored until they become valid,
    avoiding exceptions during interactive text entry.  Valid inputs are
    delegated to ``Summary.dtfilter`` with the appropriate keyword arguments.
    """
    filtered = data
    if subset_filters:
        # Summary.subset may not treat None as matching NaN/pd.NA consistently.
        # Match missing entries explicitly, retaining AND across filter columns.
        mask = pd.Series(True, index=filtered.index)
        for column, selected in subset_filters.items():
            if column not in filtered.columns:
                continue
            wanted = [v for v in selected if not pd.isna(v)]
            matches = filtered[column].isin(wanted)
            if any(pd.isna(v) for v in selected):
                matches = matches | filtered[column].isna()
            mask &= matches
        filtered = filtered.loc[mask].copy()

    start_kind, start = _parse_datetime_filter(datetime_start)
    end_kind, end = _parse_datetime_filter(datetime_end)
    kwargs = {}
    if start_kind in ("date", "datetime"):
        kwargs["date_start"] = datetime.combine(start, time.min) if start_kind == "date" else start
    elif start_kind == "time":
        kwargs["time_start"] = start
    if end_kind in ("date", "datetime"):
        # dtfilter uses an exclusive upper bound. A date-only end bound
        # therefore means the end of that calendar day, not its midnight.
        kwargs["date_end"] = datetime.combine(end + timedelta(days=1), time.min) if end_kind == "date" else end
    elif end_kind == "time":
        kwargs["time_end"] = end
    if kwargs:
        filtered = filtered.dtfilter(**kwargs)
    return filtered



def _aggregate(data, value_columns, frequency, grouping_by, aggregate, *,
               only_complete=True, flooring_timestamp=True):
    """Aggregate through ``Summary.group_stat``.

    Grouping by categorical columns is supported even when no time frequency
    is selected, matching Summary.group_stat(frequency=None, grouping_by=...).
    This function is called only when aggregation is explicitly requested.
    """
    if frequency is None and not grouping_by:
        raise ValueError("Select at least one categorical grouping variable for categorical-only aggregation.")

    result = data.group_stat(
        value_cols=value_columns,
        frequency=frequency,
        grouping_by=grouping_by,
        aggregate=[aggregate],
        **({"only_complete": only_complete, "flooring_timestamp": flooring_timestamp}
           if frequency is not None else {}),
    )
    if result is None:
        raise ValueError(
            "Summary.group_stat() returned None. Check the selected variables, "
            "grouping columns, frequency, and completeness settings."
        )
    return result



def _aggregated_value_columns(data, value_columns, aggregate):
    """Return the columns produced by Summary.group_stat()."""
    columns = []
    for column in value_columns:
        aggregated = f"{column}_{aggregate}"
        if aggregated in data.columns:
            columns.append(aggregated)
    return columns



def _frequency_timedelta(frequency: str) -> pd.Timedelta | None:
    """Convert a timestamp aggregation frequency to its expected spacing."""
    if not frequency or frequency == "weekday":
        return None

    parts = frequency.split()
    if len(parts) == 1:
        multiplier = 1.0
        unit = parts[0]
    elif len(parts) == 2:
        try:
            multiplier = float(parts[0])
        except ValueError:
            return None
        unit = parts[1]
    else:
        return None

    units = {
        "second": "s",
        "minute": "min",
        "hour": "h",
        "day": "d",
        "week": "W",
    }
    if unit not in units:
        return None
    return pd.to_timedelta(multiplier, unit=units[unit])



def _add_source_sections(data: pd.DataFrame, *, x_column: str,
                         identifier_column: str = "identifier",
                         section_column: str = "__line_section") -> pd.DataFrame:
    """Split filtered line series wherever source observations were removed.

    ``__source_row_order`` is assigned *before* filtering and increments within
    each identifier. A jump in that ordinal means an intervening source row
    was filtered out. Each run receives its own line-group key.
    """
    if "__source_row_order" not in data.columns:
        return data
    result = data.sort_values([identifier_column, x_column] if identifier_column in data.columns
                              else [x_column], kind="stable").copy()
    groups = [identifier_column] if identifier_column in result.columns else []
    if groups:
        differences = result.groupby(groups, dropna=False)["__source_row_order"].diff()
        starts = differences.ne(1) | differences.isna()
        run = starts.astype(int).groupby([result[c] for c in groups], dropna=False).cumsum()
        result[section_column] = _composite_labels(result, groups) + " | run " + run.astype(str)
    else:
        differences = result["__source_row_order"].diff()
        run = (differences.ne(1) | differences.isna()).astype(int).cumsum()
        result[section_column] = "run " + run.astype(str)
    return result


def _add_gap_sections(
    data: pd.DataFrame,
    *,
    x_column: str,
    group_columns: list[str],
    frequency: str,
    section_column: str = "__line_section",
) -> pd.DataFrame:
    """Assign separate line sections when aggregated observations have gaps.

    Rows remain in the same section while adjacent timestamps are no farther
    apart than the selected aggregation frequency.  Section numbering restarts
    independently inside every selected plotting group.

    This is primarily used when ``identifier`` was not included in
    ``group_stat(grouping_by=...)`` and is therefore no longer available after
    aggregation.
    """
    result = data.copy()
    if result.empty:
        result[section_column] = pd.Series(dtype="string")
        return result

    step = _frequency_timedelta(frequency)
    if x_column not in result.columns:
        result[section_column] = "0"
        return result

    # Non-floored aggregation returns frequency *components* (for example,
    # hours 0..23), not timestamps. Compare their numerical spacing instead
    # of converting them to epoch datetimes.
    if pd.api.types.is_numeric_dtype(result[x_column]):
        unit = frequency.split()[-1] if frequency else ""
        base = {"second": "s", "minute": "min", "hour": "h",
                "day": "D", "week": "W"}.get(unit)
        if step is None or base is None:
            result[section_column] = "0"
            return result
        numeric_step = float(step / pd.to_timedelta(1, unit=base))
        if not np.isfinite(numeric_step) or numeric_step <= 0:
            result[section_column] = "0"
            return result
        sort_columns = [*group_columns, x_column]
        result = result.sort_values(sort_columns, kind="stable").copy()
        if group_columns:
            gap = result.groupby(group_columns, dropna=False)[x_column].diff()
            new_section = gap.gt(numeric_step) | gap.isna()
            section_number = (new_section.astype(int)
                              .groupby([result[c] for c in group_columns], dropna=False)
                              .cumsum().sub(1))
            group_label = _composite_labels(result, group_columns)
            result[section_column] = group_label + " | section " + section_number.astype(str)
        else:
            gap = result[x_column].diff()
            section_number = (gap.gt(numeric_step) | gap.isna()).astype(int).cumsum().sub(1)
            result[section_column] = "section " + section_number.astype(str)
        return result

    if step is None:
        result[section_column] = "0"
        return result

    result[x_column] = pd.to_datetime(result[x_column], errors="coerce")
    sort_columns = [*group_columns, x_column]
    result = result.sort_values(sort_columns, kind="stable").copy()

    if group_columns:
        gap = result.groupby(group_columns, dropna=False)[x_column].diff()
        new_section = gap.gt(step) | gap.isna()
        section_number = (
            new_section.astype(int)
            .groupby([result[c] for c in group_columns], dropna=False)
            .cumsum()
            .sub(1)
        )
        group_label = _composite_labels(result, group_columns)
        result[section_column] = group_label + " | section " + section_number.astype(str)
    else:
        gap = result[x_column].diff()
        section_number = (gap.gt(step) | gap.isna()).astype(int).cumsum().sub(1)
        result[section_column] = "section " + section_number.astype(str)

    return result



def _box_whisker_limits(data, value_column, group_columns):
    """Return padded Tukey-whisker bounds, excluding outlying observations.

    Each box is calculated independently using the same 1.5 IQR convention
    as Matplotlib/Plotly. Explicit user axis limits take precedence elsewhere.
    """
    import numpy as np
    import pandas as pd

    groups = data.groupby(group_columns, dropna=False, observed=True, sort=False)
    minimum, maximum = np.inf, -np.inf
    for _, frame in groups:
        values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        inliers = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        if len(inliers):
            minimum = min(minimum, float(inliers.min()))
            maximum = max(maximum, float(inliers.max()))
    if not np.isfinite(minimum):
        return None
    padding = (maximum - minimum) * .05 if maximum > minimum else max(abs(minimum) * .05, .5)
    return [minimum - padding, maximum + padding]
