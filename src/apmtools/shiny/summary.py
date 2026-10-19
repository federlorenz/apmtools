"""Interactive Plotly/Shiny application for :class:`apmtools.Summary`."""

from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..units import upas_units, lascar_units, purple_units

# Units are looked up by the original sensor variable name. Later mappings
# take precedence in the unlikely event of overlapping dictionary keys.
VARIABLE_UNITS = {**upas_units, **lascar_units, **purple_units}


def _axis_label(variable: str) -> str:
    """Return a variable label with units, including aggregated columns.

    Aggregation produces names such as ``pm25_mean`` or ``CO(ppm)_max``.
    Strip only a *recognized* aggregation suffix, and only when the full
    column name does not already have its own units entry.
    """
    name = str(variable)
    unit = VARIABLE_UNITS.get(name)
    if unit is None:
        for suffix in ("count", "max", "mean", "median", "min", "std", "sum"):
            marker = f"_{suffix}"
            if name.endswith(marker):
                unit = VARIABLE_UNITS.get(name[:-len(marker)])
                if unit is not None:
                    break
    return f"{name} [{unit}]" if unit else name


def _label_plot_axes(fig, plot_type, x_column, y_columns, *,
                     category_columns=None):
    """Label plot axes while retaining special histogram and facet labels."""
    if plot_type == "histogram":
        fig.update_xaxes(title_text=_axis_label(x_column))
    elif plot_type in {"box", "bar"}:
        categories = category_columns or [x_column]
        fig.update_xaxes(title_text=" | ".join(_axis_label(c)
                         for c in categories))
        fig.update_yaxes(title_text=_axis_label(y_columns[0]))
    elif plot_type == "line" and len(y_columns) > 1:
        fig.update_xaxes(title_text=_axis_label(x_column))
        # The values axis is shared by multiple series; facet headings carry
        # the variable-specific unit rather than misleadingly naming one unit.
        fig.update_yaxes(title_text="Value")
        for annotation in fig.layout.annotations:
            label = str(annotation.text)
            if label.startswith("variable="):
                annotation.text = _axis_label(label.split("=", 1)[1])
    else:
        fig.update_xaxes(title_text=_axis_label(x_column))
        if y_columns:
            fig.update_yaxes(title_text=_axis_label(y_columns[0]))
    return fig


def _parse_label_map(value):
    """Parse semicolon-separated original=replacement entries."""
    result = {}
    for entry in (value or "").split(";"):
        if "=" in entry:
            key, replacement = entry.split("=", 1)
            if key.strip() and replacement.strip():
                result[key.strip()] = replacement.strip()
    return result


def _numeric_axis_controls(kind, box_orientation="vertical"):
    """Return axes on which numeric/date tick-density controls apply."""
    if kind in ("line", "scatter", "histogram"):
        return True, True
    if kind == "box" and box_orientation == "horizontal":
        return True, False
    return False, kind in ("box", "bar")


def _matplotlib_tick_controls(ax, *, kind, box_orientation="vertical", show_grid=True,
                              x_major_ticks=6, y_major_ticks=6,
                              x_minor_ticks=False, y_minor_ticks=False):
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    from matplotlib.dates import AutoDateLocator
    use_x, use_y = _numeric_axis_controls(kind, box_orientation)
    for active, axis, count, minor in ((use_x, ax.xaxis, x_major_ticks, x_minor_ticks),
                                       (use_y, ax.yaxis, y_major_ticks, y_minor_ticks)):
        if not active:
            continue
        count = max(2, min(30, int(count or 6)))
        if axis.get_scale() == "linear":
            if isinstance(axis.get_major_locator(), AutoDateLocator):
                axis.set_major_locator(AutoDateLocator(
                    minticks=max(2, count - 2), maxticks=count))
            else:
                axis.set_major_locator(MaxNLocator(nbins=count))
        if minor:
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(__import__(
                "matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
        axis.grid(bool(show_grid and minor), which="minor")


def _apply_plot_appearance(fig, *, title, x_label="", y_label="",
                           legend_labels="", trace_colors="", palette="Default",
                           x_min="", x_max="", y_min="", y_max="",
                           template="plotly", legend_position="right", legend_inside=False,
                           show_grid=True, font_size=12, line_width=2,
                           marker_size=7, line_alpha="", line_style="", scatter_size="", scatter_marker="", scatter_alpha="",
                           hist_alpha="", box_showfliers=True, box_orientation="vertical",
                           title_font_size=14, axis_font_size=12, tick_font_size=10,
                           legend_font_size=12, panel_font_size=14, plot_kind=None,
                           x_major_ticks=6, y_major_ticks=6, x_minor_ticks=False, y_minor_ticks=False):
    """Apply per-plot appearance overrides without changing the source data."""
    from plotly.colors import qualitative

    fig.update_layout(title=title, template=template, font=dict(size=font_size),
                      title_font=dict(size=title_font_size),
                      legend=dict(font=dict(size=legend_font_size), title_font=dict(size=legend_font_size)))
    fig.update_xaxes(title_font=dict(size=axis_font_size),
                     tickfont=dict(size=tick_font_size))
    fig.update_yaxes(title_font=dict(size=axis_font_size),
                     tickfont=dict(size=tick_font_size))
    if x_label.strip():
        fig.update_xaxes(title_text=x_label.strip())
    if y_label.strip():
        fig.update_yaxes(title_text=y_label.strip())
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)
    use_x, use_y = _numeric_axis_controls(plot_kind, box_orientation)
    for active, axis_name, count, minor in ((use_x, "x", x_major_ticks, x_minor_ticks),
                                            (use_y, "y", y_major_ticks, y_minor_ticks)):
        if not active:
            continue
        update = getattr(fig, f"update_{axis_name}axes")
        update(nticks=max(2, min(30, int(count or 6))), minor=dict(
            ticks="inside" if minor else "", showgrid=bool(show_grid and minor)))
    for axis, lower, upper in (("x", x_min, x_max), ("y", y_min, y_max)):
        datetime_axis = axis == "x" and any(
            getattr(trace, "type", "") == "scatter" and
            any(isinstance(v, (pd.Timestamp, datetime, np.datetime64))
                for v in (trace.x[:3] if trace.x is not None else []))
            for trace in fig.data
        )
        low = _parse_axis_limit(lower, datetime_axis=datetime_axis)
        high = _parse_axis_limit(upper, datetime_axis=datetime_axis)
        if low is not None or high is not None:
            getattr(fig, f"update_{axis}axes")(
                range=[low, high], autorange=False)
    positions = {
        "right": dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        "left": dict(x=-0.1, y=1, xanchor="right", yanchor="top"),
        "top": dict(x=0.5, y=1.13, xanchor="center", yanchor="bottom", orientation="h"),
        "bottom": dict(x=0.5, y=-0.2, xanchor="center", yanchor="top", orientation="h"),
    }
    legend_settings = positions.get(legend_position, positions["right"]).copy()
    if legend_inside:
        legend_settings = {
            "right": dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
            "left": dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            "top": dict(x=0.5, y=0.99, xanchor="center", yanchor="top", orientation="h"),
            "bottom": dict(x=0.5, y=0.01, xanchor="center", yanchor="bottom", orientation="h"),
        }.get(legend_position, {})
        legend_settings = {**legend_settings,
                           "bgcolor": "rgba(255,255,255,0.7)"}
    fig.update_layout(legend={**legend_settings, "title": {"text": "Legend", "font": {
                      "size": legend_font_size}}, "font": {"size": legend_font_size}})
    labels = _parse_label_map(legend_labels)
    colors = _parse_label_map(trace_colors)
    alphas, styles, sizes, markers, marker_alphas = _series_style_maps(
        line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha)
    histogram_alphas = _parse_label_map(hist_alpha)
    palette_values = getattr(qualitative, palette,
                             None) if palette != "Default" else None
    # Color keys can refer to either original or renamed trace labels.
    for i, trace in enumerate(fig.data):
        original = str(trace.name or "")
        renamed = labels.get(original, original)
        if renamed != original:
            trace.name = renamed
            if getattr(trace, "legendgroup", None) == original:
                trace.legendgroup = renamed
        color = colors.get(original, colors.get(renamed))
        if color is None and palette_values:
            color = palette_values[i % len(palette_values)]
        if color:
            if getattr(trace, "line", None) is not None:
                trace.line.color = color
            if getattr(trace, "marker", None) is not None:
                trace.marker.color = color
            if getattr(trace, "fillcolor", None) is not None and trace.type == "histogram":
                trace.marker.color = color
        if plot_kind == "histogram":
            trace.opacity = _bounded_float(
                histogram_alphas, original, 0.65, 0, 1)
        if plot_kind == "box":
            trace.boxpoints = "outliers" if box_showfliers else False
            if box_orientation == "horizontal":
                old_x, old_y = trace.x, trace.y
                trace.x, trace.y = old_y, old_x
                trace.orientation = "h"
        if getattr(trace, "line", None) is not None and plot_kind == "line":
            trace.line.width = line_width
            if plot_kind == "line":
                trace.line.dash = {"-": "solid", ":": "dot",
                                   "--": "dash"}[_line_style(styles, original)]
                trace.opacity = _bounded_float(alphas, original, 1, 0, 1)
        if getattr(trace, "marker", None) is not None and trace.type in ("scatter", "scattergl"):
            if plot_kind == "scatter":
                trace.marker.size = _bounded_float(
                    sizes, original, 7, 0.1, 100)
                trace.marker.symbol = {"o": "circle", "x": "x"}[
                    _scatter_marker(markers, original)]
                trace.opacity = _bounded_float(
                    marker_alphas, original, 1, 0, 1)
            else:
                trace.marker.size = marker_size
    if plot_kind == "box" and box_orientation == "horizontal":
        x_title = fig.layout.xaxis.title.text
        y_title = fig.layout.yaxis.title.text
        fig.update_xaxes(title_text=y_title, type="linear", autorange=True)
        fig.update_yaxes(title_text=x_title, type="category", autorange=True)
    return fig


def _series_style_maps(line_alpha="", line_style="", scatter_size="", scatter_marker="", scatter_alpha=""):
    """Parse per-trace styles, keyed by the original legend/series name."""
    return tuple(_parse_label_map(v) for v in (line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha))


def _bounded_float(mapping, name, default, low, high):
    try:
        value = float(mapping.get(name, default))
        return value if np.isfinite(value) and low <= value <= high else float(default)
    except (ValueError, TypeError):
        return float(default)


def _line_style(mapping, name):
    return {"solid": "-", "dotted": ":", "dashed": "--", "-": "-", ":": ":", "--": "--"}.get(
        mapping.get(name, "solid").strip().lower(), "-")


def _scatter_marker(mapping, name):
    return {"dot": "o", "cross": "x", "o": "o", "x": "x"}.get(
        mapping.get(name, "dot").strip().lower(), "o")


PLOT_TYPES = {
    "bar": "Bar",
    "box": "Boxplot",
    "histogram": "Histogram",
    "line": "Line",
    "scatter": "Scatter",
}

# ``None`` means row-level plotting.  The other values are passed directly to
# Summary.group_stat(frequency=...).
FREQUENCIES = {
    "None": None,
    "Day": "day",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
    "Week": "week",
    "Weekday": "weekday",
}

# UI choices must use string values: Python None as a choice value is not a reliable HTML selection.
AGGREGATES = {
    "None": None,
    "Count": "count",
    "Maximum": "max",
    "Mean": "mean",
    "Median": "median",
    "Minimum": "min",
    "Standard deviation": "std",
    "Sum": "sum",
}


def _default_x(data: pd.DataFrame) -> str:
    """Return ``time`` when available, otherwise the first column."""
    if "datetime" in data.columns:
        return "datetime"
    datetime_columns = [
        column
        for column in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[column])
    ]
    return str(datetime_columns[0] if datetime_columns else data.columns[0])


def _default_group(data: pd.DataFrame) -> str | None:
    """Return ``identifier`` when available."""
    return "identifier" if "identifier" in data.columns else None


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric column names as strings."""
    return sorted([
        str(column)
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ])


def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return categorical column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if not pd.api.types.is_numeric_dtype(data[column])
            and not pd.api.types.is_datetime64_any_dtype(data[column])
        ],
        key=str.casefold,
    )


def _datetime_columns(data: pd.DataFrame) -> list[str]:
    """Return datetime column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if pd.api.types.is_datetime64_any_dtype(data[column])
        ],
        key=str.casefold,
    )


def _timedelta_columns(data: pd.DataFrame) -> list[str]:
    """Return timedelta column names as strings."""
    return sorted(
        [
            str(column)
            for column in data.columns
            if pd.api.types.is_timedelta64_dtype(data[column])
        ],
        key=str.casefold,
    )


def _alphabetical(values) -> list[str]:
    """Return string values sorted alphabetically, case-insensitively."""
    return sorted((str(value) for value in values), key=str.casefold)


def _normalise_y(data: pd.DataFrame, x: str, y: str | Sequence[str] | None) -> list[str]:
    """Validate and normalise an initial Y selection."""
    numeric = _numeric_columns(data)
    if y is None:
        return [column for column in numeric if column != x][:1]
    values = [y] if isinstance(y, str) else list(y)
    missing = [column for column in values if column not in data.columns]
    if missing:
        raise ValueError(f"Unknown y column(s): {missing}")
    return [str(column) for column in values]


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
    values = sorted(pd.unique(data[column].dropna()),
                    key=lambda value: str(value).casefold())
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
            return "datetime", time.fromisoformat(value)
        match = re.fullmatch(
            date_pattern + r"[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?", value)
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
        kwargs["date_start"] = datetime.combine(
            start, time.min) if start_kind == "date" else start
    elif start_kind == "datetime":
        kwargs["time_start"] = start
    if end_kind in ("date", "datetime"):
        # dtfilter uses an exclusive upper bound. A date-only end bound
        # therefore means the end of that calendar day, not its midnight.
        kwargs["date_end"] = datetime.combine(
            end + timedelta(days=1), time.min) if end_kind == "date" else end
    elif end_kind == "datetime":
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
        raise ValueError(
            "Select at least one categorical grouping variable for categorical-only aggregation.")

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


def _histogram_figure(
    data: pd.DataFrame,
    x_column: str,
    *,
    group_column: str | None = None,
    bins: int = 30,
    density: bool = False,
    stacked: bool = False,
    histtype: str = "bar",
    cumulative: int = 0,
):
    """Build a Matplotlib-compatible histogram using shared bin boundaries.

    ``cumulative`` is 0 (off), 1 (forward), or -1 (reverse).
    Density for stacked data is normalized across all groups; otherwise each
    group's histogram is normalized independently.
    """
    import plotly.graph_objects as go

    if histtype not in {"bar", "barstacked", "step", "stepfilled"}:
        raise ValueError(f"Unknown histtype: {histtype}")
    if cumulative not in {-1, 0, 1}:
        raise ValueError("cumulative must be -1, 0 or 1")
    if bins < 1:
        raise ValueError("bins must be positive")

    stacked = bool(stacked or histtype == "barstacked")
    labels = (
        list(pd.unique(data[group_column].astype(str)))
        if group_column is not None
        else ["All observations"]
    )
    arrays = []
    for label in labels:
        rows = data.loc[data[group_column].astype(
            str) == str(label)] if group_column else data
        values = pd.to_numeric(
            rows[x_column], errors="coerce").to_numpy(dtype=float)
        arrays.append(values[np.isfinite(values)])

    all_values = np.concatenate(arrays) if arrays else np.array([])
    fig = go.Figure()
    if not len(all_values):
        fig.update_layout(title="No numerical observations")
        return fig

    edges = np.histogram_bin_edges(all_values, bins=bins)
    widths = np.diff(edges)
    counts = np.array([np.histogram(v, bins=edges)[0]
                      for v in arrays], dtype=float)
    totals = counts.sum(axis=1)
    grand_total = totals.sum()

    if density:
        if stacked:
            heights = counts / (grand_total * widths)
        else:
            heights = np.divide(
                counts, totals[:, None] * widths[None, :],
                out=np.zeros_like(counts),
                where=(totals[:, None] > 0),
            )
    else:
        heights = counts.copy()

    if cumulative:
        # Cumulative density is a CDF/survival curve, not the sum of
        # bin-wise probability densities. It must integrate bin mass.
        mass = counts / grand_total if (density and stacked) else counts.copy()
        if density and not stacked:
            mass = np.divide(
                counts, totals[:, None],
                out=np.zeros_like(counts),
                where=(totals[:, None] > 0),
            )
        heights = (
            np.cumsum(mass, axis=1)
            if cumulative == 1
            else np.flip(np.cumsum(np.flip(mass, axis=1), axis=1), axis=1)
        )

    for index, label in enumerate(labels):
        values = heights[index]
        if histtype in ("bar", "barstacked"):
            fig.add_trace(go.Bar(
                name=str(label),
                x=(edges[:-1] + edges[1:]) / 2,
                y=values,
                width=widths,
                marker_line_width=0,
            ))
        else:
            # Horizontal steps start/end at the shared bin boundaries.
            xx = np.repeat(edges, 2)[1:-1]
            yy = np.repeat(values, 2)
            fig.add_trace(go.Scatter(
                name=str(label),
                x=xx,
                y=yy,
                mode="lines",
                fill="tozeroy" if histtype == "stepfilled" else None,
                stackgroup="histogram" if stacked and histtype == "stepfilled" else None,
            ))

    fig.update_layout(
        barmode="stack" if stacked else "group",
        xaxis_title=x_column,
        yaxis_title=(
            "Cumulative probability" if cumulative and density
            else "Cumulative count" if cumulative
            else "Probability density" if density
            else "Count"
        ),
    )
    return fig


def _boxplot_figure(data, x_column, y_column, *, group_column=None):
    """Create a boxplot, preserving outlier-only point display."""
    import plotly.express as px
    # Explicit category order keeps ticks centred under their boxes.
    categories = list(dict.fromkeys(data[x_column].dropna().astype(str)))
    fig = px.box(
        data, x=x_column, y=y_column, color=group_column, points="outliers",
        category_orders={x_column: categories},
    )
    # A categorical axis ensures even spacing; wide boxes fill each slot.
    fig.update_xaxes(type="category", categoryorder="array",
                     categoryarray=categories)
    fig.update_layout(boxmode="group", boxgap=0.12, boxgroupgap=0.04)
    # Plotly Express can produce unusually thin boxes when many observations
    # share a category. Allocate most of each category slot explicitly.
    n_traces = max(len(fig.data), 1)
    box_width = 0.82 / n_traces
    for i, trace in enumerate(fig.data):
        trace.width = box_width
    # A single ungrouped box trace does not convey useful legend information.
    if group_column is None and len(fig.data) == 1:
        fig.update_traces(showlegend=False)
        fig.update_layout(showlegend=False)
    return fig


def _barplot_figure(data, x_column, y_column, *, group_column=None):
    """Create grouped bars from the already prepared plotting data."""
    import plotly.express as px
    return px.bar(data, x=x_column, y=y_column, color=group_column, barmode="group")


def _scatter_figure(data, x_column, y_column, *, group_column=None):
    """Create a scatter plot from the already prepared plotting data."""
    import plotly.express as px
    return px.scatter(data, x=x_column, y=y_column, color=group_column)


def _lineplot_figure(plot_df, x_column, y_columns, *, group_columns, frequency, column_map):
    """Create line plots with identifier and time-gap segmentation preserved."""
    import plotly.express as px
    # ``identifier`` identifies an individual input monitor.  It
    # must therefore be used as Plotly's line-group key even when
    # it is not one of the user-selected colour/group columns.
    # This prevents observations from two different monitors
    # being connected by a line merely because they share the
    # same selected group.  The identifier is deliberately NOT
    # used for colour, so monitors in the same selected group
    # retain the same colour.  After frequency aggregation,
    # identifier is only available if it was selected as a
    # grouping variable.  If it is absent, discontinuous runs are
    # inferred from gaps larger than the selected frequency.
    identifier_column = (
        column_map["identifier"]
        if (
            "identifier" in column_map
            and column_map["identifier"] in plot_df.columns
        )
        else None
    )

    section_column = None
    if frequency is not None and identifier_column is None:
        plot_df = _add_gap_sections(
            plot_df,
            x_column=x_column,
            group_columns=group_columns,
            frequency=frequency,
        )
        section_column = "__line_section"

    if group_columns:
        group_column = "__plot_group"
        line_columns = [x_column, group_column, *y_columns]
        line_group_column = identifier_column or section_column
        if line_group_column is not None:
            line_columns.append(line_group_column)

        # Keep whichever column controls line segmentation as an
        # id_var during melt(). Otherwise pandas drops it and
        # Plotly cannot resolve line_group (e.g. __line_section).
        id_vars = [x_column, group_column]
        if line_group_column is not None:
            id_vars.append(line_group_column)

        long_df = plot_df[line_columns].melt(
            id_vars=id_vars,
            var_name="variable",
            value_name="value",
        )
        fig = px.line(
            long_df,
            x=x_column,
            y="value",
            color=group_column,
            line_group=line_group_column,
            facet_row="variable" if len(y_columns) > 1 else None,
        )
    elif identifier_column is not None:
        # With no explicit grouping, retain the normal Y-variable
        # colouring while using identifier to keep each monitor's
        # time series separate.
        if len(y_columns) == 1:
            fig = px.line(
                plot_df,
                x=x_column,
                y=y_columns[0],
                line_group=identifier_column,
            )
        else:
            line_columns = [x_column, identifier_column, *y_columns]
            long_df = plot_df[line_columns].melt(
                id_vars=[x_column, identifier_column],
                var_name="variable",
                value_name="value",
            )
            fig = px.line(
                long_df,
                x=x_column,
                y="value",
                color="variable",
                line_group=identifier_column,
                facet_row="variable",
            )
    elif section_column is not None:
        if len(y_columns) == 1:
            fig = px.line(
                plot_df,
                x=x_column,
                y=y_columns[0],
                line_group=section_column,
            )
        else:
            line_columns = [x_column, section_column, *y_columns]
            long_df = plot_df[line_columns].melt(
                id_vars=[x_column, section_column],
                var_name="variable",
                value_name="value",
            )
            fig = px.line(
                long_df,
                x=x_column,
                y="value",
                color="variable",
                line_group=section_column,
                facet_row="variable",
            )
    else:
        fig = px.line(plot_df, x=x_column, y=y_columns)
    return fig


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
            result[section_column] = group_label + \
                " | section " + section_number.astype(str)
        else:
            gap = result[x_column].diff()
            section_number = (gap.gt(numeric_step) | gap.isna()
                              ).astype(int).cumsum().sub(1)
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
        result[section_column] = group_label + \
            " | section " + section_number.astype(str)
    else:
        gap = result[x_column].diff()
        section_number = (gap.gt(step) | gap.isna()
                          ).astype(int).cumsum().sub(1)
        result[section_column] = "section " + section_number.astype(str)

    return result


def _safe_csv_filename(value: str | None) -> str:
    """Sanitize a user-selected CSV filename for the download response."""
    import os
    name = os.path.basename((value or "").replace("\\", "/").strip())
    if name in ("", ".", ".."):
        name = "export.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


# Native Matplotlib renderer: never constructs or inspects a Plotly figure.
_MPL_PALETTES = {
    "Default": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
    "Plotly": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
    "D3": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "G10": ["#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"],
    "T10": ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"],
    "Set1": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"],
    "Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Pastel": ["#AEC6CF", "#FFB347", "#77DD77", "#CFCFC4", "#B39EB5", "#FF6961", "#FDFD96", "#CB99C9"],
    "Dark24": ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A", "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1", "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1", "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038"],
}


def _matplotlib_defaults(spec, *, palette="Default"):
    """Derive labels and series colours directly from prepared pandas data."""
    kind, df = spec["kind"], spec["data"]
    x, ys, groups = spec["x"], spec["ys"], spec["groups"]
    names = ([str(v) for v in pd.unique(df["__plot_group"])]
             if groups and "__plot_group" in df else [])
    if not names and kind == "line" and len(ys) > 1:
        names = [str(y) for y in ys]
    if not names:
        names = ["All observations"]
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    colors = {name: palette_values[i % len(
        palette_values)] for i, name in enumerate(names)}
    x_label = (" | ".join(_axis_label(c) for c in spec["categories"])
               if kind in ("box", "bar") else _axis_label(x))
    y_label = ("Probability density" if kind == "histogram" else
               "Value" if kind == "line" and len(ys) > 1 else
               _axis_label(ys[0]) if ys else "Count")
    return dict(x_label=x_label, y_label=y_label, names=names, colors=colors)


def _axis_bounds_from_spec(spec, *, bins=30, density=False, stacked=False, cumulative=0):
    """Compute editable axis bounds from the prepared data, without rendering.

    Datetime X bounds are ISO timestamps; categorical X bounds are intentionally
    omitted. Values are calculated only when a valid plot specification exists.
    """
    if "error" in spec:
        return {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}
    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    bounds = {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}

    def fill(axis, values):
        if pd.api.types.is_datetime64_any_dtype(values):
            valid = pd.to_datetime(values, errors="coerce").dropna()
            if len(valid):
                bounds[f"{axis}_min"] = valid.min().isoformat(sep=" ")
                bounds[f"{axis}_max"] = valid.max().isoformat(sep=" ")
        else:
            valid = pd.to_numeric(values, errors="coerce")
            valid = valid[np.isfinite(
                valid.to_numpy(dtype=float, na_value=np.nan))]
            if len(valid):
                bounds[f"{axis}_min"] = format(float(valid.min()), ".12g")
                bounds[f"{axis}_max"] = format(float(valid.max()), ".12g")

    if kind not in ("bar", "box"):
        fill("x", df[x])
    if kind == "histogram":
        values = pd.to_numeric(df[x], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if len(values):
            groups = ([pd.to_numeric(part[x], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                       for _, part in df.groupby("__plot_group", sort=False)]
                      if spec["groups"] and "__plot_group" in df else [values])
            edges = np.histogram_bin_edges(values, bins=max(1, int(bins)))
            histograms = np.array([np.histogram(v[np.isfinite(v)], bins=edges)[0]
                                   for v in groups], dtype=float)
            if density:
                total = histograms.sum()
                if total:
                    histograms /= total * np.diff(edges)
            if cumulative:
                histograms = histograms.cumsum(axis=1)
            ymax = (histograms.sum(axis=0).max() if stacked and histograms.size
                    else histograms.max() if histograms.size else 0)
            bounds["y_min"], bounds["y_max"] = "0", format(float(ymax), ".12g")
    elif kind == "bar":
        values = pd.to_numeric(df[ys[0]], errors="coerce")
        cols = [x, "__plot_group"] if spec["groups"] else [x]
        totals = (df.assign(__axis_value=values).groupby(cols, dropna=False)["__axis_value"]
                  .sum(min_count=1).dropna())
        if len(totals):
            bounds["y_min"] = format(float(min(0, totals.min())), ".12g")
            bounds["y_max"] = format(float(max(0, totals.max())), ".12g")
    else:
        values = pd.concat([pd.to_numeric(df[y], errors="coerce")
                           for y in ys], ignore_index=True)
        fill("y", values)
    return bounds


def _pad_axis_bounds(bounds, fraction=0.05):
    """Return bounds expanded by 5% of the data span on each side.

    Datetime X bounds retain their timestamp representation. Zero-width
    numeric ranges receive a small nonzero span to avoid singular axes.
    """
    result = dict(bounds)
    for axis in ("x", "y"):
        lo_key, hi_key = f"{axis}_min", f"{axis}_max"
        lo, hi = result.get(lo_key, ""), result.get(hi_key, "")
        if not lo or not hi:
            continue
        try:
            low, high = float(lo), float(hi)
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            span = high - low
            margin = fraction * (span if span > 0 else max(abs(low), 1.0))
            result[lo_key] = format(low - margin, ".12g")
            result[hi_key] = format(high + margin, ".12g")
        except (ValueError, TypeError):
            if axis != "x":
                continue
            try:
                low, high = pd.Timestamp(lo), pd.Timestamp(hi)
                span = high - low
                margin = span * \
                    fraction if span > pd.Timedelta(
                        0) else pd.Timedelta(seconds=1)
                result[lo_key] = (low - margin).isoformat(sep=" ")
                result[hi_key] = (high + margin).isoformat(sep=" ")
            except (ValueError, TypeError, OverflowError):
                continue
    return result


def _panel_columns(spec, panel_by):
    """Resolve one or more selected facet columns in their selected order."""
    selected = [panel_by] if isinstance(
        panel_by, str) else list(panel_by or [])
    return [column for column in selected
            if column in spec.get("groups", ()) and column in spec.get("data", ())]


def _panel_partitions(spec, panel_by):
    """Yield panel title, panel data and remaining colour-group columns."""
    columns = _panel_columns(spec, panel_by)
    if not columns:
        return
    remaining = [g for g in spec["groups"] if g not in columns]
    grouper = columns[0] if len(columns) == 1 else columns
    for value, part in spec["data"].groupby(grouper, sort=False, dropna=False):
        values = value if isinstance(value, tuple) else (value,)
        title = " | ".join("None" if pd.isna(v) else str(v) for v in values)
        if remaining:
            part = part.copy(deep=False)
            part["__plot_group"] = (_composite_labels(part, remaining))
        yield title, part, remaining


def _panel_trace_defaults(spec, panel_by, *, palette="Default"):
    """Appearance entries for the series actually drawn inside each panel.

    Facet columns are removed from the colour grouping.  When all grouping
    columns are facets, each panel contains an ungrouped "All observations"
    trace; when some grouping columns remain, their values label the traces.
    """
    names = []
    for _, part, remaining in _panel_partitions(spec, panel_by):
        panel_spec = {**spec, "data": part, "groups": remaining}
        for name in _matplotlib_defaults(panel_spec, palette=palette)["names"]:
            if name not in names:
                names.append(name)
    if not names:
        return _matplotlib_defaults(spec, palette=palette)
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    return {"names": names,
            "colors": {name: palette_values[i % len(palette_values)]
                       for i, name in enumerate(names)}}


def _panel_axis_bounds(spec, panel_by, **kwargs):
    """Per-facet data limits, keeping categorical X limits blank."""
    if "error" in spec:
        return {}
    result = {}
    for title, part, remaining in _panel_partitions(spec, panel_by):
        sub_spec = {**spec, "data": part, "groups": remaining}
        result[title] = _axis_bounds_from_spec(sub_spec, **kwargs)
    return result


def _panel_limit_value(raw, panel, fallback=""):
    """Read a panel-specific limit from `panel: value` lines or a scalar limit."""
    raw = str(raw or "").strip()
    if not raw:
        return fallback
    if "\n" not in raw and ": " not in raw:
        return raw
    for line in raw.replace("; ", "\n").splitlines():
        label, separator, value = line.partition(": ")
        if separator and label.strip() == str(panel):
            return value.strip()
    return fallback


def _format_panel_bounds(bounds, key):
    return "\n".join(f"{panel}: {limits[key]}" for panel, limits in bounds.items()
                     if limits.get(key, "") != "")


def _parse_axis_limit(value, *, datetime_axis=False):
    """Parse numeric bounds or an ISO timestamp for a datetime X axis."""
    if value is None or not str(value).strip():
        return None
    if datetime_axis:
        try:
            parsed = pd.to_datetime(value, errors="raise")
            if not pd.isna(parsed):
                return parsed
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _matplotlib_direct_figure(spec, *, title="", x_label="", y_label="",
                              legend_labels="", trace_colors="", palette="Default",
                              x_min="", x_max="", y_min="", y_max="",
                              template="plotly", legend_position="right",
                              legend_inside=False, show_grid=True, font_size=12,
                              line_width=2, marker_size=7, line_alpha="", line_style="",
                              scatter_size="", scatter_marker="", scatter_alpha="", hist_alpha="",
                              box_showfliers=True, box_orientation="vertical",
                              title_font_size=14, axis_font_size=12, tick_font_size=10,
                              legend_font_size=12, panel_font_size=14, bins=30, density=False,
                              stacked=False, histtype="bar", cumulative=0,
                              panel_by=None, independent_panel_limits=True, panel_titles="", _axes=None,
                              x_major_ticks=6, y_major_ticks=6, x_minor_ticks=False, y_minor_ticks=False):
    """Draw directly from the filtered/aggregated DataFrame, without Plotly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if _axes is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    else:
        ax = _axes
        fig = ax.figure
    if "error" in spec:
        ax.text(.5, .5, spec["error"], ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    # Facet only the native Matplotlib renderer.  Each panel receives the
    # same filtered/aggregated data subset and the same appearance settings.
    if panel_by and _axes is None:
        import math
        partitions = list(_panel_partitions(spec, panel_by))
        panel_title_map = _parse_label_map(panel_titles)
        if partitions:
            if len(partitions) > 36:
                ax.text(.5, .5, f"{len(partitions)} panels requested; filter the data to 36 or fewer panels.",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
                return fig
            plt.close(fig)
            ncols = (2 if len(partitions) == 4 else min(3, len(partitions)))
            nrows = math.ceil(len(partitions) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), dpi=100,
                                     squeeze=False, sharex=False, sharey=False)
            for idx, (panel_title, panel_df, other_groups) in enumerate(partitions):
                panel_spec = {**spec, "data": panel_df, "groups": other_groups,
                              "_hide_ungrouped_legend": not bool(other_groups)}
                panel_ax = axes.flat[idx]
                _matplotlib_direct_figure(panel_spec, title=panel_title_map.get(panel_title, panel_title), x_label=x_label,
                                          y_label=y_label, legend_labels=legend_labels, trace_colors=trace_colors,
                                          palette=palette,
                                          x_min=_panel_limit_value(x_min, panel_title), x_max=_panel_limit_value(x_max, panel_title),
                                          y_min=_panel_limit_value(y_min, panel_title), y_max=_panel_limit_value(y_max, panel_title),
                                          template=template, legend_position=legend_position,
                                          legend_inside=legend_inside, show_grid=show_grid, font_size=font_size,
                                          line_width=line_width, marker_size=marker_size, line_alpha=line_alpha, line_style=line_style,
                                          scatter_size=scatter_size, scatter_marker=scatter_marker, scatter_alpha=scatter_alpha,
                                          hist_alpha=hist_alpha, box_showfliers=box_showfliers, box_orientation=box_orientation,
                                          title_font_size=panel_font_size, axis_font_size=axis_font_size,
                                          tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                                          panel_font_size=panel_font_size,
                                          bins=bins, density=density,
                                          stacked=stacked, histtype=histtype, cumulative=cumulative, _axes=panel_ax,
                                          x_major_ticks=x_major_ticks, y_major_ticks=y_major_ticks,
                                          x_minor_ticks=x_minor_ticks, y_minor_ticks=y_minor_ticks)
            for extra in list(axes.flat)[len(partitions):]:
                extra.set_visible(False)
            fig.tight_layout()
            return fig
    group_col = "__plot_group" if spec["groups"] and "__plot_group" in df.columns else None
    defaults = _matplotlib_defaults(spec, palette=palette)
    rename = _parse_label_map(legend_labels)
    overrides = _parse_label_map(trace_colors)
    alphas, styles, sizes, markers, marker_alphas = _series_style_maps(
        line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha)
    histogram_alphas = _parse_label_map(hist_alpha)
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    labels = defaults["names"] if group_col else (
        [str(y) for y in ys] if len(ys) > 1 else ["All observations"])
    colors = {name: overrides.get(name, defaults["colors"].get(name, palette_values[i % len(palette_values)]))
              for i, name in enumerate(labels)}
    for name, color in list(colors.items()):
        if not mcolors.is_color_like(color):
            colors[name] = defaults["colors"].get(name, palette_values[0])

    def series():
        if group_col:
            for name in labels:
                yield name, df.loc[df[group_col] == name]
        else:
            yield "All observations", df

    if kind == "histogram":
        arrays = []
        for name, rows in series():
            vals = pd.to_numeric(
                rows[x], errors="coerce").to_numpy(dtype=float)
            arrays.append(vals[np.isfinite(vals)])
        all_vals = np.concatenate(arrays) if arrays else np.array([])
        if len(all_vals):
            edges = np.histogram_bin_edges(all_vals, bins=max(1, int(bins)))
            _, _, hist_patches = ax.hist(
                arrays, bins=edges, density=density,
                stacked=stacked or histtype == "barstacked",
                histtype=("bar" if histtype == "barstacked" else histtype),
                cumulative=cumulative if cumulative else False,
                color=[colors[n] for n in labels],
                label=[rename.get(n, n) for n in labels])
            # Matplotlib returns one patch container per group for bar and
            # one polygon per group for step/stepfilled.
            for name, group_patches in zip(labels, hist_patches):
                alpha = _bounded_float(histogram_alphas, name,
                                       1 if histtype == "step" else .65, 0, 1)
                if hasattr(group_patches, "set_alpha"):
                    group_patches.set_alpha(alpha)
                else:
                    for patch in group_patches:
                        patch.set_alpha(alpha)
        defaults["y_label"] = ("Cumulative probability" if cumulative and density else
                               "Cumulative count" if cumulative else
                               "Probability density" if density else "Count")
    elif kind == "scatter":
        for name, rows in series():
            color = colors.get(name, palette_values[0])
            ax.scatter(rows[x], rows[ys[0]],
                       s=_bounded_float(sizes, name, marker_size, 0.1, 100)**2,
                       marker=_scatter_marker(markers, name),
                       alpha=_bounded_float(marker_alphas, name, 1, 0, 1),
                       color=color, label=rename.get(name, name))
    elif kind == "line":
        line_df = df
        id_col = "identifier" if "identifier" in df.columns else None
        section_col = None
        if spec["frequency"] is not None and id_col is None:
            line_df = _add_gap_sections(
                df, x_column=x, group_columns=spec["groups"], frequency=spec["frequency"])
            section_col = "__line_section"
        for y in ys:
            if group_col:
                sets = [(n, line_df.loc[line_df[group_col] == n])
                        for n in labels]
            else:
                sets = [(str(y) if len(ys) > 1 else "All observations", line_df)]
            for name, rows in sets:
                segment_col = id_col or section_col
                parts = (rows.groupby(segment_col, dropna=False,
                         sort=False) if segment_col else [(None, rows)])
                for _, part in parts:
                    part = part.sort_values(x)
                    ax.plot(part[x], part[y], color=colors.get(name, palette_values[0]),
                            linewidth=float(line_width),
                            linestyle=_line_style(styles, name),
                            alpha=_bounded_float(alphas, name, 1, 0, 1),
                            label=rename.get(name, name))
    elif kind in ("box", "bar"):
        # Build categorical groups in a single pass. The former implementation
        # repeatedly scanned the full DataFrame for every category × group,
        # which became prohibitively expensive for many identifiers.
        from collections import defaultdict

        category_values = df[x].astype(str)
        cats = list(pd.unique(category_values))
        cat_idx = {category: i for i, category in enumerate(cats)}
        group_names = labels if group_col else ["All observations"]
        group_idx = {name: i for i, name in enumerate(group_names)}
        n = max(1, len(group_names))
        # A boxplot creates many artists per category/group (whiskers, caps,
        # median, fliers). Thousands of combinations can stall the render
        # thread for minutes. Fail fast with an actionable message rather than
        # blocking the plot output indefinitely. No data is silently omitted.
        observed_pairs = df[[x, group_col]].drop_duplicates(
        ).shape[0] if group_col else len(cats)
        limit = 250 if kind == "box" else 1500
        if observed_pairs > limit:
            ax.text(.5, .5,
                    f"Too many category/group combinations ({observed_pairs:,}) for a {kind} plot.\n"
                    f"Use the filters, reduce Group by cardinality, or aggregate the data\n"
                    f"(rendering limit: {limit:,} combinations).",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, wrap=True)
            ax.set_axis_off()
            return fig
        width = .82 / n
        values = pd.to_numeric(df[ys[0]], errors="coerce")
        valid = values.notna() & np.isfinite(
            values.to_numpy(dtype=float, na_value=np.nan))
        plot_groups = df[group_col].astype(str) if group_col else pd.Series(
            "All observations", index=df.index)
        # Use integer group indices, avoiding repeated string comparisons.
        ci = category_values.map(cat_idx).to_numpy()
        gi = plot_groups.map(group_idx).to_numpy()
        valid_array = valid.to_numpy() & pd.notna(gi)
        positions = ci[valid_array] + (gi[valid_array] - (n - 1) / 2) * width
        numeric = values.to_numpy(dtype=float, na_value=np.nan)[valid_array]
        names = gi[valid_array].astype(int)
        if kind == "bar":
            # One bar artist collection instead of one call per category/group.
            totals = defaultdict(float)
            for position, value in zip(positions, numeric):
                totals[float(position)] += float(value)
            xpos = list(totals)
            # Derive group colour from the position's group index.
            palette_for_positions = {}
            for category_index, group_index, position in zip(ci[valid_array], names, positions):
                palette_for_positions[float(position)] = colors.get(
                    group_names[group_index], palette_values[0])
            if xpos:
                ax.bar(xpos, [totals[pos] for pos in xpos], width=width * .92,
                       color=[palette_for_positions[pos] for pos in xpos])
        else:
            # Calculate group memberships once and make one boxplot call.
            buckets = defaultdict(list)
            bucket_colors = {}
            for position, value, group_index in zip(positions, numeric, names):
                key = float(position)
                buckets[key].append(float(value))
                bucket_colors[key] = colors.get(
                    group_names[group_index], palette_values[0])
            if buckets:
                box_positions = sorted(buckets)
                bp = ax.boxplot([buckets[pos] for pos in box_positions],
                                positions=box_positions, widths=width * .92,
                                manage_ticks=False, patch_artist=True,
                                showfliers=bool(box_showfliers),
                                vert=(box_orientation != "horizontal"))
                for patch, position in zip(bp["boxes"], box_positions):
                    patch.set_facecolor(bucket_colors[position])
                    patch.set_alpha(.7)
        if kind == "box" and box_orientation == "horizontal":
            ax.set_yticks(range(len(cats)), cats)
            ax.set_ylim(-.5, max(.5, len(cats) - .5))
        else:
            ax.set_xticks(range(len(cats)), cats)
            ax.set_xlim(-.5, max(.5, len(cats) - .5))
        # A boxplot with a single ungrouped series does not need a legend.
        # Keep legends for genuinely grouped boxplots and for bar plots.
        show_category_legend = not (
            kind == "box" and len(group_names) == 1
            and group_names[0] == "All observations"
        )
        if len(group_names) <= 40 and show_category_legend and not spec.get("_hide_ungrouped_legend", not bool(spec["groups"])):
            ax.legend(handles=[Patch(facecolor=colors[name], label=rename.get(name, name))
                               for name in group_names], title="Legend")
    else:
        raise ValueError(f"Unknown plot type: {kind}")

    if kind not in ("box", "bar"):
        handles, texts = ax.get_legend_handles_labels()
        unique = dict(zip(texts, handles))
        if unique and not spec.get("_hide_ungrouped_legend", not bool(spec["groups"])):
            ax.legend(unique.values(), unique.keys(), title="Legend")
    legend = ax.get_legend()
    if legend is not None:
        locs = {"right": "upper right", "left": "upper left",
            "top": "upper center", "bottom": "lower center"}
        if legend_inside:
            legend.set_loc(locs.get(legend_position, "upper right"))
            legend.set_bbox_to_anchor(None)
        else:
            anchors = {"right": (1.01, 1), "left": (-.01, 1),
                                 "top": (.5, 1.15), "bottom": (.5, -.2)}
            legend.set_loc(locs.get(legend_position, "upper right"))
            legend.set_bbox_to_anchor(anchors.get(legend_position, (1.01, 1)))
    if template == "plotly_dark":
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#222222")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    elif template == "ggplot2":
        ax.set_facecolor("#EBEBEB")
    elif template in ("plotly_white", "simple_white"):
        ax.set_facecolor("white")
    xlabel = x_label.strip() or defaults["x_label"]
    ylabel = y_label.strip() or defaults["y_label"]
    if kind == "box" and box_orientation == "horizontal":
        xlabel, ylabel = ylabel, xlabel
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.tick_params(labelsize=tick_font_size)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)
        legend.get_title().set_fontsize(legend_font_size)
    ax.grid(bool(show_grid))
    _matplotlib_tick_controls(ax, kind=kind, box_orientation=box_orientation,
                              show_grid=show_grid, x_major_ticks=x_major_ticks,
                              y_major_ticks=y_major_ticks, x_minor_ticks=x_minor_ticks,
                              y_minor_ticks=y_minor_ticks)
    for lo, hi, getter, setter in ((x_min, x_max, ax.get_xlim, ax.set_xlim),
                                   (y_min, y_max, ax.get_ylim, ax.set_ylim)):
        try:
            from matplotlib.dates import date2num
            is_datetime_x = getter == ax.get_xlim and pd.api.types.is_datetime64_any_dtype(
                df[x])
            lower = _parse_axis_limit(lo, datetime_axis=is_datetime_x)
            upper = _parse_axis_limit(hi, datetime_axis=is_datetime_x)
            if is_datetime_x:
                lower = date2num(lower.to_pydatetime()
                                 ) if lower is not None else None
                upper = date2num(upper.to_pydatetime()
                                 ) if upper is not None else None
            if lower is not None or upper is not None:
                current = getter()
                setter(
                    lower if lower is not None else current[0], upper if upper is not None else current[1])
        except (TypeError, ValueError):
            pass
    if _axes is None:
        fig.tight_layout()
    return fig


def _matplotlib_figure(plotly_figure):
    """Render the shared Plotly trace specification with Matplotlib (Agg).

    The Shiny controls, data selection, aggregation, and appearance settings
    are shared between backends. No browser-side Plotly widget is created.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    source = plotly_figure
    category_labels = list(
        getattr(source.layout.xaxis, "categoryarray", None) or [])
    categorical = (getattr(source.layout.xaxis, "type", None) == "category")
    category_map = {str(value): i for i, value in enumerate(category_labels)}
    legend_seen = set()
    legend_handles = []
    legend_texts = []

    def xvalues(values):
        vals = list(values) if values is not None else []
        if categorical:
            for val in vals:
                if str(val) not in category_map:
                    category_map[str(val)] = len(category_map)
            return [category_map[str(val)] for val in vals]
        return vals

    traces = list(source.data)
    box_traces = [t for t in traces if t.type == "box"]
    for index, trace in enumerate(traces):
        name = str(trace.name or "")
        color = (getattr(getattr(trace, "line", None), "color", None)
                 or getattr(getattr(trace, "marker", None), "color", None)
                 or f"C{index % 10}")
        if not isinstance(color, str) or not mcolors.is_color_like(color):
            color = f"C{index % 10}"
        label = name if name and name not in legend_seen and trace.showlegend is not False else "_nolegend_"
        if label != "_nolegend_":
            legend_seen.add(name)
        if trace.type in ("scatter", "scattergl"):
            xs = xvalues(trace.x)
            ys = list(trace.y)
            mode = str(trace.mode or "lines")
            if "lines" in mode:
                ax.plot(xs, ys, color=color, label=label,
                        linewidth=getattr(trace.line, "width", None) or 2,
                        marker="o" if "markers" in mode else None,
                        markersize=getattr(trace.marker, "size", None) or 5)
            elif "markers" in mode:
                ax.scatter(xs, ys, color=color, label=label,
                           s=(getattr(trace.marker, "size", None) or 7) ** 2)
        elif trace.type == "box":
            # Position boxes at category centres, distributing grouped boxes
            # within each category to avoid excessive whitespace.
            xs = xvalues(trace.x)
            ys = np.asarray(trace.y, dtype=float)
            categories = list(dict.fromkeys(xs))
            box_index = box_traces.index(trace)
            width = 0.8 / max(len(box_traces), 1)
            offset = (box_index - (len(box_traces) - 1) / 2) * width
            groups = [ys[np.asarray(xs) == cat] for cat in categories]
            groups = [g[np.isfinite(g)] for g in groups]
            valid = [(cat, group)
                     for cat, group in zip(categories, groups) if len(group)]
            if valid:
                result = ax.boxplot([group for _, group in valid],
                                    positions=[
                                        cat + offset for cat, _ in valid],
                                    widths=width * .92, patch_artist=True,
                                    manage_ticks=False, showfliers=True)
                for box in result["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(.65)
                if label != "_nolegend_":
                    legend_handles.append(Patch(facecolor=color, alpha=.65))
                    legend_texts.append(label)
        elif trace.type == "bar":
            xs = xvalues(trace.x)
            if not categorical and xs and isinstance(xs[0], str):
                categorical = True
                xs = xvalues(trace.x)
            width = .8 / max(sum(t.type == "bar" for t in traces), 1)
            offset = (index - (len(traces) - 1) / 2) * width
            ax.bar(np.asarray(xs, dtype=float) + offset, trace.y,
                   width=width * .95, color=color, label=label)
        elif trace.type == "histogram":
            xs = np.asarray(trace.x, dtype=float)
            xs = xs[np.isfinite(xs)]
            nbins = getattr(trace, "nbinsx", None) or 30
            histnorm = getattr(trace, "histnorm", None)
            cumulative = getattr(
                getattr(trace, "cumulative", None), "enabled", False)
            direction = getattr(
                getattr(trace, "cumulative", None), "direction", "increasing")
            histtype = "step" if getattr(
                trace, "visible", True) == "legendonly" else "bar"
            ax.hist(xs, bins=nbins, density=histnorm in ("probability density", "density"),
                    cumulative=(-1 if direction ==
                                "decreasing" else True) if cumulative else False,
                    histtype=histtype, alpha=.65, color=color, label=label)

    if category_map:
        ax.set_xticks(list(category_map.values()), list(category_map.keys()))
    ax.set_xlabel(
        getattr(getattr(source.layout.xaxis, "title", None), "text", "") or "")
    ax.set_ylabel(
        getattr(getattr(source.layout.yaxis, "title", None), "text", "") or "")
    ax.set_title(
        getattr(getattr(source.layout, "title", None), "text", "") or "")
    ax.grid(bool(getattr(source.layout.xaxis, "showgrid", True)), axis="x")
    ax.grid(bool(getattr(source.layout.yaxis, "showgrid", True)), axis="y")
    for axis, setter in ((source.layout.xaxis, ax.set_xlim), (source.layout.yaxis, ax.set_ylim)):
        bounds = getattr(axis, "range", None)
        if bounds and all(v is not None for v in bounds):
            try:
                setter(*bounds)
            except (ValueError, TypeError):
                pass
    handles, labels = ax.get_legend_handles_labels()
    handles += legend_handles
    labels += legend_texts
    if handles:
        legend = source.layout.legend
        orientation = getattr(legend, "orientation", "v")
        inside = (0 <= float(getattr(legend, "x", 1.02)) <= 1 and
                  0 <= float(getattr(legend, "y", 1)) <= 1)
        loc = "upper right" if inside else "upper left"
        kwargs = {"loc": loc, "title": "Legend", "ncol": max(
            1, len(labels)) if orientation == "h" else 1}
        if not inside:
            kwargs["bbox_to_anchor"] = (1.01, 1)
        ax.legend(handles, labels, **kwargs)
    fig.tight_layout()
    return fig


def _standalone_matplotlib_script(spec, options):
    """Create a self-contained plotting script using this renderer's source.

    The CSV holds the *prepared* data, after subsetting and aggregation.
    Only pandas, numpy and matplotlib are required to rerun the plot.
    """
    import inspect
    import pprint

    dependencies = (
        _axis_label, _parse_label_map, _series_style_maps, _bounded_float,
        _line_style, _scatter_marker, _composite_labels,
        _frequency_timedelta, _add_gap_sections, _matplotlib_defaults,
        _panel_columns, _panel_partitions, _panel_limit_value,
        _parse_axis_limit, _numeric_axis_controls, _matplotlib_tick_controls,
        _matplotlib_direct_figure,
    )
    source = "\n\n".join(inspect.getsource(fn) for fn in dependencies)
    metadata = {key: spec.get(key) for key in
                ("kind", "x", "ys", "groups", "categories", "frequency")}
    date_columns = [str(c) for c in spec["data"].columns
                    if pd.api.types.is_datetime64_any_dtype(spec["data"][c])]
    return (
        '# Generated by apmtools Summary.plot_app (Matplotlib backend).\n'
        '# Run: python plot_script.py\n'
        '# Dependencies: pip install pandas numpy matplotlib\n'
        'from pathlib import Path\n'
        'import pandas as pd\nimport numpy as np\n'
        'from datetime import datetime, date, timedelta\n\n'
        'VARIABLE_UNITS = ' +
        pprint.pformat(VARIABLE_UNITS, sort_dicts=True) + '\n'
        '_MPL_PALETTES = ' +
        pprint.pformat(_MPL_PALETTES, sort_dicts=True) + '\n\n'
        + source + '\n\n'
        + 'def main():\n'
        + '    csv_path = Path(__file__).resolve().parent / "plot_data.csv"\n'
        + '    frame = pd.read_csv(csv_path, parse_dates=' +
        repr(date_columns) + ')\n'
        + '    spec = ' + repr(metadata) + '\n'
        + '    spec["data"] = frame\n'
        + '    options = ' + repr(options) + '\n'
        + '    figure = _matplotlib_direct_figure(spec, **options)\n'
        + '    import matplotlib.pyplot as plt\n'
        + '    plt.savefig("plot.png")\n\n'
        + 'if __name__ == "__main__":\n    main()\n'
    )


def create_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    backend: str = "plotly",
):
    """Create, but do not start, an interactive Shiny app for a Summary."""
    from shiny import App, reactive, render, ui, module
    if backend not in ("plotly", "matplotlib"):
        raise ValueError("backend must be 'plotly' or 'matplotlib'")
    if backend == "plotly":
        from shinywidgets import output_widget, render_plotly
    # Keep the Summary object intact: subset(), dtfilter() and group_stat()
    # are methods on Summary and are intentionally used below.
    summary = data.copy(deep=False)
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame/Summary")
    if summary.empty:
        raise ValueError(
            "Cannot create a Shiny application from an empty Summary.")

    columns = _alphabetical(summary.columns)
    column_map = {str(c): c for c in summary.columns}
    numeric_columns = _alphabetical(_numeric_columns(summary))
    categorical_columns = _categorical_columns(summary)
    datetime_columns = _datetime_columns(summary)
    time_group_columns = [f for f in FREQUENCIES.values(
    ) if f is not None and f not in column_map]
    group_columns = _alphabetical(
        [c for c in columns if not pd.api.types.is_numeric_dtype(
            summary[column_map[c]])]
        + time_group_columns
    )
    for name in time_group_columns:
        column_map[name] = name

    x = _default_x(summary) if x is None else str(x)
    if x not in column_map:
        raise ValueError(f"Unknown x column: {x!r}")

    initial_y = _normalise_y(summary, x, y)
    if group_by is None:
        initial_groups = [_default_group(summary)] if _default_group(
            summary) is not None else []
    elif isinstance(group_by, str):
        initial_groups = [group_by]
    else:
        initial_groups = list(group_by)
    unknown_groups = [c for c in initial_groups if c not in column_map]
    if unknown_groups:
        raise ValueError(f"Unknown group_by column(s): {unknown_groups!r}")
    # Numeric columns are not offered by the Group by selector.
    initial_groups = [c for c in initial_groups if c in group_columns]

    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. Choose one of: {', '.join(PLOT_TYPES)}")

    subset_columns = columns
    # Stable choices are shared across tabs. Computing unique values from the
    # full dataset on every filter change was expensive for large summaries.
    subset_choice_cache = {}

    def cached_subset_choices(name):
        if name not in subset_choice_cache:
            subset_choice_cache[name] = _subset_choices(
                summary, column_map[name])
        return subset_choice_cache[name]

    @module.ui
    def plot_tab_ui(initial_filters=None):
        initial_filters = initial_filters or {}
        return ui.div(
            ui.layout_sidebar(
                ui.sidebar(
                    ui.card(ui.card_header("Plot filters"),
                            ui.input_selectize("subset_columns", "Subset columns",
                                               choices={
                                               c: c for c in subset_columns},
                            selected=[], multiple=True),
                            ui.output_ui("subset_filters"),
                            ui.hr(),
                            ui.input_text("datetime_start", "Start datetime",
                                          value="",
                                          placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                            ui.input_text("datetime_end", "End datetime",
                                          value="",
                                          placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                            ui.help_text(
                            "Filters affect this plot only. New plots start with empty selections."),
                            ),
                    ui.card(ui.card_header("Plot settings"),
                            ui.input_select(
                                "plot_type", "Plot type", choices=PLOT_TYPES, selected=plot_type),
                            ui.output_ui("x_control"),
                            ui.output_ui("y_control"),
                            ui.card(
                        ui.card_header("Aggregation"),
                        ui.input_selectize("group_by", "Group by", choices={
                                           c: c for c in group_columns}, selected=[], multiple=True),
                        ui.output_ui("panel_controls"),
                        ui.input_select("frequency", "Frequency / aggregation mode", choices={
                            key: key for key in FREQUENCIES}, selected="None"),
                        ui.input_select("aggregate", "Apply statistical function", choices={
                            key: key for key in AGGREGATES}, selected="None"),
                        ui.help_text(
                            "Choose a statistical function to aggregate by the selected groups, even when frequency is None."),
                        ui.output_ui("frequency_options"),
                    ),
                        ui.panel_conditional("input.plot_type === 'histogram'",
                                             ui.input_numeric(
                                                 "bins", "Histogram bins", value=30, min=1, max=200),
                                ui.input_select("hist_density", "Density", choices={
                                                "False": "False", "True": "True"}, selected="False"),
                                ui.input_select("hist_stacked", "Stacked", choices={
                                                "False": "False", "True": "True"}, selected="False"),
                                ui.input_select("hist_type", "Histogram type", choices={
                                                "bar": "bar", "barstacked": "barstacked", "step": "step", "stepfilled": "stepfilled"}, selected="bar"),
                                ui.input_select("hist_cumulative", "Cumulative", choices={
                                                "0": "False", "1": "True (ascending)", "-1": "True (descending)"}, selected="0"),
                                             ),
                    ),
                    title="Plot filters / settings / aggregation", position="left", open="open", width="320px",
                ),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_text("title", "Plot title", value=""),
                        ui.input_text("x_axis_label",
                                          "X-axis label", value=""),
                        ui.input_text("y_axis_label",
                                          "Y-axis label", value=""),
                        ui.input_text_area(
                                "legend_labels", "Legend labels (old=new; ...)", value="", rows=6),
                        (ui.panel_conditional("input.multi_panel === true",
                                                  ui.input_text_area("panel_titles", "Panel titles (old=new; ...)",
                                                                     value="", rows=6)) if backend == "matplotlib" else ui.div()),
                        ui.input_text_area(
                                "trace_colors", "Trace colors (name=#hex; ...)", value="", rows=6),
                        ui.input_select("palette", "Color palette", choices=[
                                            "Default", "Plotly", "D3", "G10", "T10", "Set1", "Set2", "Pastel", "Dark24"], selected="Default"),
                        ui.layout_columns(
                                (ui.input_text_area("x_min", "X minimum", value="", rows=4) if backend ==
                                 "matplotlib" else ui.input_text("x_min", "X minimum", value="")),
                                (ui.input_text_area("x_max", "X maximum", value="", rows=4) if backend ==
                                 "matplotlib" else ui.input_text("x_max", "X maximum", value="")),
                                col_widths=(6, 6),
                            ),
                        ui.layout_columns(
                                (ui.input_text_area("y_min", "Y minimum", value="", rows=4) if backend ==
                                 "matplotlib" else ui.input_text("y_min", "Y minimum", value="")),
                                (ui.input_text_area("y_max", "Y maximum", value="", rows=4) if backend ==
                                 "matplotlib" else ui.input_text("y_max", "Y maximum", value="")),
                                col_widths=(6, 6),
                            ),
                            ui.input_select("template", "Theme", choices={
                                            "plotly": "Light", "plotly_white": "White", "plotly_dark": "Dark", "simple_white": "Minimal", "ggplot2": "GGPlot"}, selected="plotly"),
                        ui.input_select("legend_position", "Legend position", choices={
                                            k: k.title() for k in ("right", "left", "top", "bottom")}, selected="right"),
                        ui.input_checkbox(
                                "legend_inside", "Place legend inside plot", value=False),
                        ui.input_checkbox(
                                "show_grid", "Show grid", value=True),
                        ui.panel_conditional("input.plot_type === 'line' || input.plot_type === 'scatter' || input.plot_type === 'histogram' || (input.plot_type === 'box' && input.box_orientation === 'horizontal')",
                                                 ui.input_numeric(
                                                     "x_major_ticks", "X-axis major ticks (approximate)", value=6, min=2, max=30, step=1),
                                                 ui.input_checkbox("x_minor_ticks", "X-axis minor ticks / gridlines", value=False)),
                        ui.panel_conditional("input.plot_type === 'line' || input.plot_type === 'scatter' || input.plot_type === 'histogram' || input.plot_type === 'bar' || (input.plot_type === 'box' && input.box_orientation !== 'horizontal')",
                                                 ui.input_numeric(
                                                     "y_major_ticks", "Y-axis major ticks (approximate)", value=6, min=2, max=30, step=1),
                                                 ui.input_checkbox("y_minor_ticks", "Y-axis minor ticks / gridlines", value=False)),
                        ui.input_numeric(
                                "title_font_size", "Plot title font size", value=14, min=6, max=72),
                        ui.input_numeric(
                                "axis_font_size", "Axis title font size", value=12, min=6, max=72),
                        ui.input_numeric(
                                "tick_font_size", "Axis tick font size", value=10, min=6, max=72),
                        ui.input_numeric(
                                "legend_font_size", "Legend font size", value=12, min=6, max=72),
                        ui.panel_conditional("input.plot_type === 'line'",
                                                 ui.input_numeric("line_width", "Line width", value=2, min=0.5, max=15, step=0.5)),
                        (ui.input_numeric("panel_font_size", "Panel title font size",
                             value=14, min=6, max=72) if backend == "matplotlib" else ui.div()),
                        ui.panel_conditional("input.plot_type === 'histogram'",
                                                 ui.input_text_area("hist_alpha", "Histogram alpha (name=0..1; ...)", value="", rows=4)),
                        ui.panel_conditional("input.plot_type === 'box'",
                                                 ui.input_checkbox(
                                                     "box_showfliers", "Show outliers (fliers)", value=True),
                                                 ui.input_select("box_orientation", "Box orientation", choices={"vertical": "Vertical", "horizontal": "Horizontal"}, selected="vertical")),
                        ui.panel_conditional("input.plot_type === 'line'",
                                                 ui.input_text_area(
                                                     "line_alpha", "Line alpha (name=0..1; ...)", value="", rows=4),
                                                 ui.input_text_area("line_style", "Line style (name=solid/dotted/dashed; ...)", value="", rows=4)),
                        ui.panel_conditional("input.plot_type === 'scatter'",
                                                 ui.input_text_area(
                                                     "scatter_size", "Marker size (name=size; ...)", value="", rows=4),
                                                 ui.input_text_area(
                                                     "scatter_marker", "Marker type (name=dot/cross; ...)", value="", rows=4),
                                                 ui.input_text_area("scatter_alpha", "Marker alpha (name=0..1; ...)", value="", rows=4)),
                        title="Plot appearance", position="right", open="closed", width="320px",
                    ),
                    ui.card(ui.card_header("Plot"),
                            ui.div((output_widget("plot", height="100%", width="100%") if backend == "plotly" else ui.output_plot("plot", height="100%", width="100%")),
                                       class_="summary-plot-container"),
                            ui.input_action_button(
                                    "export_csv", "Export plotted data as CSV", class_="btn-primary mt-3"),
                            ui.input_action_button(
                                    "export_plot", "Export plot", class_="btn-secondary mt-3"),
                            (ui.download_button("export_python", "Export as a Python file and associated data",
                                                    class_="btn-secondary mt-3")
                                 if backend == "matplotlib" else ui.div())),
                ),
            ),
        )

    @module.server
    def plot_tab_server(input, output, session, initial_filters=None):
        initial_filters = initial_filters or {}

        @render.ui
        def subset_filters():
            controls = []
            for i, name in enumerate(list(input.subset_columns())):
                if name not in column_map:
                    continue
                choices, _ = cached_subset_choices(name)
                controls.append(ui.input_selectize(
                    f"subset_values_{i}", f"{name} values", choices=choices,
                    selected=[],
                    multiple=True,
                ))
            return ui.div(*controls) if controls else ui.help_text("Select columns to filter this plot.")

        @reactive.calc
        def filter_state():
            names = list(input.subset_columns())
            values = {}
            for i, name in enumerate(names):
                if name in column_map:
                    values[name] = list(input[f"subset_values_{i}"]())
            return {"columns": names, "values": values,
                    "start": input.datetime_start().strip(),
                    "end": input.datetime_end().strip()}

        @reactive.calc
        def filtered_summary():
            state = filter_state()
            filters = {}
            for name, keys in state["values"].items():
                _, mapping = cached_subset_choices(name)
                selected = [mapping[key] for key in keys if key in mapping]
                if selected:
                    filters[column_map[name]] = selected
            return _apply_filters(summary, filters, state["start"], state["end"])

        @reactive.effect
        @reactive.event(input.export_csv)
        def show_export_dialog():
            ui.modal_show(ui.modal(
                ui.input_text("export_filename",
                                "CSV filename", value="export.csv"),
                ui.download_button(
                    "download_csv", "Download CSV", class_="btn-primary"),
                title="Export plotted dataframe",
                easy_close=True,
                footer=ui.modal_button("Cancel"),
            ))

        @reactive.effect
        @reactive.event(input.export_plot)
        def show_plot_export_dialog():
            ui.modal_show(ui.modal(
                ui.input_numeric(
                    "plot_export_width", "Width (inches)", value=10, min=1, max=100, step=0.5),
                ui.input_numeric(
                    "plot_export_height", "Height (inches)", value=5, min=1, max=100, step=0.5),
                ui.input_numeric(
                    "plot_export_dpi", "Resolution (DPI)", value=150, min=30, max=1200, step=25),
                ui.download_button(
                    "download_plot_png", "Download plot.png", class_="btn-primary"),
                title="Export plot", easy_close=True,
                footer=ui.modal_button("Cancel"),
            ))

        @render.download(filename="plot.png", media_type="image/png")
        def download_plot_png():
            import io
            width = float(input.plot_export_width())
            height = float(input.plot_export_height())
            dpi = int(input.plot_export_dpi())
            if not (1 <= width <= 100 and 1 <= height <= 100 and 30 <= dpi <= 1200):
                raise ValueError(
                    "Invalid plot export dimensions or resolution")
            if width * dpi * height * dpi > 100_000_000:
                raise ValueError(
                    "Export exceeds 100 megapixels; reduce size or DPI")
            if backend == "matplotlib":
                import matplotlib.pyplot as plt
                fig = _matplotlib_direct_figure(
                    prepared_plot(), **matplotlib_options())
                try:
                    fig.set_size_inches(width, height, forward=True)
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=dpi)
                    yield buffer.getvalue()
                finally:
                    plt.close(fig)
            else:
                # Plotly static image export requires kaleido.
                # Use the same styled figure as the interactive plot.
                fig = styled_plotly_figure()
                try:
                    yield fig.to_image(format="png", width=round(width * dpi),
                                        height=round(height * dpi), scale=1)
                except Exception as exc:
                    raise RuntimeError(
                        'Plotly PNG export requires kaleido: pip install kaleido'
                    ) from exc

        @render.download(filename=lambda: _safe_csv_filename(input.export_filename()))
        def download_csv():
            # Use the same post-filter/post-aggregation dataframe as the plot.
            # The original dataframe is never mutated.
            frame = pd.DataFrame(selected_data()).reset_index(drop=True)
            yield frame.to_csv(index=False).encode("utf-8-sig")

        @render.ui
        def x_control():
            kind = input.plot_type()
            if kind in ("box", "bar"):
                choices = categorical_columns
                multiple = True
            elif kind == "line":
                choices = datetime_columns
                multiple = False
            else:
                choices = numeric_columns
                multiple = False

            if not choices:
                return ui.help_text(f"No compatible X variables for {kind}.")
            selected = []
            if multiple:
                return ui.input_selectize(
                    "x", "X-axis / categories",
                    choices={c: c for c in choices},
                    selected=selected, multiple=True,
                )
            return ui.input_select(
                "x", "X-axis / variable",
                choices={"": "None", **{c: c for c in choices}},
                selected="",
            )

        @render.ui
        def y_control():
            kind = input.plot_type()
            if kind == "histogram":
                return ui.div()
            if not numeric_columns:
                return ui.help_text("No numeric Y variables available.")
            selected = ""
            if kind in ("box", "bar", "scatter"):
                return ui.input_select(
                    "y", "Y-axis / variable",
                    choices={"": "None", **
                                {c: c for c in numeric_columns}},
                    selected="",
                )
            return ui.input_selectize(
                "y", "Y-axis / variables",
                choices={c: c for c in numeric_columns},
                selected=[],
                multiple=True,
            )

        def _optional_input(name, default=None):
            """Read a conditionally mounted Shiny input safely.

            Dynamic controls do not exist before their UI has been mounted.
            Their absence must not prevent the first plot from rendering.
            """
            try:
                value = getattr(input, name)()
            except (AttributeError, KeyError, TypeError):
                return default
            return default if value is None else value

        @render.ui
        def panel_controls():
            if backend != "matplotlib" or not list(input.group_by() or []):
                return ui.div()
            selected = list(input.group_by() or [])
            controls = [ui.input_checkbox("multi_panel", "Separate panels by group", value=False),
                        ui.panel_conditional("input.multi_panel === true",
                                                ui.input_checkbox("independent_panel_limits",
                                                                "Independent axis limits for each panel", value=True))]
            if len(selected) > 1:
                controls.append(ui.panel_conditional(
                    "input.multi_panel === true",
                    ui.input_selectize("panel_by", "Panel grouping variables",
                                        choices={
                                            name: name for name in selected},
                                        selected=selected, multiple=True)))
            return ui.div(*controls)

        @render.ui
        def frequency_options():
            if FREQUENCIES.get(input.frequency()) is None:
                return ui.div()
            return ui.div(
                ui.input_checkbox(
                    "only_complete", "Only complete intervals", value=True),
                ui.input_checkbox("flooring_timestamp",
                                    "Floor timestamps", value=True),
            )

        @reactive.calc
        def selected_data():
            current = filtered_summary()
            kind = input.plot_type()
            x_input = input.x()
            selected_x = ([x_input] if isinstance(
                x_input, str) else list(x_input or []))
            if not selected_x or any(c not in column_map for c in selected_x):
                return pd.DataFrame()

            if kind == "histogram":
                value_names = selected_x
            else:
                y_input = input.y()
                value_names = ([y_input] if isinstance(
                    y_input, str) else list(y_input or []))
            if not value_names or any(c not in column_map for c in value_names):
                return pd.DataFrame()

            selected_groups = list(input.group_by() or [])
            # Derive hour/day/etc. through group_stat without aggregation.
            # The returned index column identifies source rows.
            for component in selected_groups:
                if component not in time_group_columns:
                    continue
                generated = current.group_stat(
                    value_cols=value_names,
                    frequency=component,
                    grouping_by=None,
                    aggregate=None,
                    only_complete=False,
                    flooring_timestamp=False,
                )
                if generated is None or component not in generated.columns:
                    return pd.DataFrame()
                current = current.copy()
                if "index" in generated.columns and current.index.is_unique:
                    values = generated.set_index("index")[component]
                    current[component] = values.reindex(current.index)
                else:
                    valid = current["datetime"].notna().to_numpy()
                    current[component] = pd.NA
                    if len(generated) == int(valid.sum()):
                        current.loc[valid,
                                    component] = generated[component].to_numpy()
            grouping_by = [column_map[c] for c in selected_groups]
            frequency = FREQUENCIES.get(input.frequency())
            should_aggregate = AGGREGATES.get(
                input.aggregate()) is not None
            aggregate = AGGREGATES.get(input.aggregate())
            value_columns = [column_map[c] for c in value_names]

            if should_aggregate and kind in ("box", "bar"):
                for c in selected_x:
                    if column_map[c] not in grouping_by:
                        grouping_by.append(column_map[c])
            if kind == "scatter" and should_aggregate:
                # Both numerical axes must be aggregated on the same rows.
                value_columns = list(dict.fromkeys(
                    [column_map[selected_x[0]], *value_columns]
                ))
            if should_aggregate:
                current = _aggregate(
                    current, value_columns, frequency, grouping_by, aggregate,
                    only_complete=(bool(input.only_complete())
                                    if frequency is not None else True),
                    flooring_timestamp=(
                        bool(input.flooring_timestamp()) if frequency is not None else True),
                )
            return current

        @reactive.calc
        def prepared_plot():
            plot_df = selected_data()
            selected_plot = input.plot_type()
            x_input = input.x()
            selected_x = ([x_input] if isinstance(
                x_input, str) else list(x_input or []))
            y_input = None if selected_plot == "histogram" else input.y()
            selected_y = ([y_input] if isinstance(
                y_input, str) else list(y_input or []))
            selected_groups = list(input.group_by() or [])
            frequency = FREQUENCIES.get(input.frequency())
            should_aggregate = AGGREGATES.get(
                input.aggregate()) is not None

            if plot_df.empty or not selected_x or (
                selected_plot != "histogram" and not selected_y
            ):
                return {"error": "No data for the selected filters"}

            aggregate = AGGREGATES.get(input.aggregate())

            def mapped(column):
                original = column_map[column]
                return f"{original}_{aggregate}" if should_aggregate else original

            if selected_plot == "histogram":
                x_column = mapped(selected_x[0])
                y_columns = []
            elif selected_plot in ("box", "bar"):
                x_columns = [column_map[c] for c in selected_x]
                x_column = x_columns[0]
                y_columns = [mapped(selected_y[0])]
            elif selected_plot == "scatter":
                x_column = mapped(selected_x[0])
                y_columns = [mapped(selected_y[0])]
            else:
                x_column = frequency if frequency is not None else column_map[selected_x[0]]
                y_columns = [mapped(c) for c in selected_y]

            if selected_plot == "line" and should_aggregate and frequency is None:
                return {"error": "Categorical-only aggregation has no datetime axis; use a bar plot or select a frequency"}

            required = [x_column, *y_columns]
            if selected_plot in ("box", "bar"):
                required.extend(x_columns)
            if any(c not in plot_df.columns for c in required):
                return {"error": "Selected variables are unavailable after aggregation"}

            if selected_plot in ("box", "bar"):
                plot_df = plot_df.copy(deep=False)
                plot_df["__plot_category"] = _composite_labels(
                    plot_df, x_columns)
                x_column = "__plot_category"

            # Multiple grouping columns are supported by group_stat(). For
            # plotting, Plotly needs one color/category column, so combine the
            # selected grouping columns into a readable composite label.
            group_columns = [column_map[c] for c in selected_groups
                                if column_map[c] in plot_df.columns]
            if group_columns:
                plot_df = plot_df.copy(deep=False)
                plot_df["__plot_group"] = (
                    _composite_labels(plot_df, group_columns)
                )

            return {"data": plot_df, "kind": selected_plot, "x": x_column,
                    "ys": y_columns, "groups": group_columns,
                    "categories": x_columns if selected_plot in ("box", "bar") else None,
                    "frequency": frequency}

        @reactive.calc
        def base_figure():
            import plotly.express as px
            spec = prepared_plot()
            if "error" in spec:
                return px.scatter(title=spec["error"])
            plot_df = spec["data"]
            selected_plot = spec["kind"]
            x_column = spec["x"]
            y_columns = spec["ys"]
            group_columns = spec["groups"]
            frequency = spec["frequency"]
            x_columns = spec["categories"]
            if selected_plot == "histogram":
                fig = _histogram_figure(
                    plot_df, x_column,
                    group_column="__plot_group" if group_columns else None,
                    bins=int(input.bins()),
                    density=input.hist_density() == "True",
                    stacked=input.hist_stacked() == "True",
                    histtype=input.hist_type(),
                    cumulative=int(input.hist_cumulative()),
                )
            elif selected_plot == "box":
                fig = _boxplot_figure(
                    plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
            elif selected_plot == "bar":
                fig = _barplot_figure(
                    plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
            elif selected_plot == "scatter":
                fig = _scatter_figure(
                    plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
            else:
                fig = _lineplot_figure(
                    plot_df, x_column, y_columns, group_columns=group_columns, frequency=frequency, column_map=column_map)

            if not group_columns:
                # Retain trace names/colours for appearance editing, but
                # suppress the redundant ungrouped legend.
                fig.update_traces(showlegend=False)
                fig.update_layout(showlegend=False)

            _label_plot_axes(
                fig, selected_plot, x_column, y_columns,
                category_columns=x_columns if selected_plot in (
                    "box", "bar") else None,
            )
            fig.update_layout(
                hovermode="x unified" if selected_plot == "line" else "closest",
                margin=dict(l=60, r=30, t=70, b=60),
                autosize=True,
            )
            return fig

        # Populate appearance controls only after a valid plot has been
        # prepared.  The effect depends on the plot specification, not on
        # appearance inputs, so editing a label or colour cannot reset it.
        @reactive.effect
        @reactive.event(prepared_plot, lambda: _optional_input("multi_panel", False) if input.group_by() else False,
                        lambda: tuple(_optional_input("panel_by", []) or []) if input.group_by(
        ) and len(input.group_by()) > 1 else (),
            lambda: _optional_input("independent_panel_limits", True) if input.group_by() and _optional_input("multi_panel", False) else True, ignore_none=False)
        def populate_appearance_from_plot():
            spec = prepared_plot()
            if "error" in spec:
                return
            defaults = _matplotlib_defaults(spec, palette="Default")
            names = defaults["names"]
            selected_groups_for_legend = list(input.group_by() or [])
            if (backend == "matplotlib" and selected_groups_for_legend
                    and bool(_optional_input("multi_panel", False))):
                panel_variables_for_legend = (
                    (list(_optional_input("panel_by", []) or [])
                        or selected_groups_for_legend)
                    if len(selected_groups_for_legend) > 1
                    else selected_groups_for_legend
                )
                if panel_variables_for_legend:
                    trace_defaults = _panel_trace_defaults(
                        spec, panel_variables_for_legend, palette="Default"
                    )
                    names = trace_defaults["names"]
                    defaults["colors"] = trace_defaults["colors"]
            ui.update_text("title", value="Summary plot")
            ui.update_text("x_axis_label", value=defaults["x_label"])
            ui.update_text("y_axis_label", value=defaults["y_label"])
            limits = _axis_bounds_from_spec(
                spec,
                bins=int(
                    input.bins()) if spec["kind"] == "histogram" else 30,
                density=input.hist_density(
                ) == "True" if spec["kind"] == "histogram" else False,
                stacked=(input.hist_stacked() == "True" or input.hist_type(
                ) == "barstacked") if spec["kind"] == "histogram" else False,
                cumulative=int(input.hist_cumulative()
                                ) if spec["kind"] == "histogram" else 0,
            )
            limits = _pad_axis_bounds(limits)
            selected_groups = list(input.group_by() or [])
            if (backend == "matplotlib" and selected_groups and bool(_optional_input("multi_panel", False))
                    and bool(_optional_input("independent_panel_limits", True))):
                panel_variable = ((list(_optional_input("panel_by", []) or []) or selected_groups) if len(selected_groups) > 1
                                    else selected_groups)
                panel_bounds = _panel_axis_bounds(
                    spec, panel_variable,
                    bins=int(
                        input.bins()) if spec["kind"] == "histogram" else 30,
                    density=input.hist_density(
                    ) == "True" if spec["kind"] == "histogram" else False,
                    stacked=(input.hist_stacked() == "True" or input.hist_type(
                    ) == "barstacked") if spec["kind"] == "histogram" else False,
                    cumulative=int(input.hist_cumulative()
                                    ) if spec["kind"] == "histogram" else 0,
                )
                if panel_bounds:
                    panel_bounds = {name: _pad_axis_bounds(
                        bounds) for name, bounds in panel_bounds.items()}
                    limits = {key: _format_panel_bounds(
                        panel_bounds, key) for key in limits}
            for field, value in limits.items():
                if backend == "matplotlib":
                    ui.update_text_area(field, value=value)
                else:
                    ui.update_text(field, value=value)
            ui.update_text_area("legend_labels", value="; ".join(
                f"{name}={name}" for name in names))
            ui.update_text_area("trace_colors", value="; ".join(
                f"{name}={defaults['colors'][name]}" for name in names))
            kind = spec["kind"]
            if kind == "line":
                ui.update_text_area("line_alpha", value="; ".join(
                    f"{name}=1" for name in names))
                ui.update_text_area("line_style", value="; ".join(
                    f"{name}=solid" for name in names))
            elif kind == "histogram":
                ui.update_text_area("hist_alpha", value="; ".join(
                    f"{name}=0.65" for name in names))
            elif kind == "scatter":
                ui.update_text_area("scatter_size", value="; ".join(
                    f"{name}=7" for name in names))
                ui.update_text_area("scatter_marker", value="; ".join(
                    f"{name}=dot" for name in names))
                ui.update_text_area("scatter_alpha", value="; ".join(
                    f"{name}=1" for name in names))
            if backend == "matplotlib":
                panel_variables = ((list(_optional_input("panel_by", []) or []) or selected_groups) if len(selected_groups) > 1
                                    else selected_groups)
                panel_names = ([name for name, _, _ in _panel_partitions(spec, panel_variables)]
                                if selected_groups and bool(_optional_input("multi_panel", False)) else [])
                ui.update_text_area("panel_titles", value="; ".join(
                    f"{name}={name}" for name in panel_names))

        @reactive.effect
        @reactive.event(input.plot_type, ignore_init=True)
        def reset_on_plot_type_change():
            # Reset every plot-specific selection when the plot type changes.
            ui.update_selectize("subset_columns", selected=[])
            ui.update_text("datetime_start", value="")
            ui.update_text("datetime_end", value="")
            ui.update_selectize("group_by", selected=[])
            ui.update_checkbox("multi_panel", value=False)
            ui.update_checkbox("independent_panel_limits", value=True)
            ui.update_select("frequency", selected="None")
            ui.update_select("aggregate", selected="None")
            ui.update_text("title", value="")
            for field in ("x_axis_label", "y_axis_label"):
                ui.update_text(field, value="")
            for field in ("x_min", "x_max", "y_min", "y_max"):
                if backend == "matplotlib":
                    ui.update_text_area(field, value="")
                else:
                    ui.update_text(field, value="")
            for field in ("legend_labels", "trace_colors"):
                ui.update_text_area(field, value="")
            ui.update_select("palette", selected="Default")
            ui.update_select("template", selected="plotly")
            ui.update_select("legend_position", selected="right")
            ui.update_checkbox("legend_inside", value=False)
            ui.update_checkbox("show_grid", value=True)
            for field, value in (("title_font_size", 14), ("axis_font_size", 12),
                                    ("tick_font_size", 10), ("legend_font_size", 12),
                                    ("panel_font_size", 14)):
                ui.update_numeric(field, value=value)
            ui.update_numeric("line_width", value=2)
            ui.update_checkbox("box_showfliers", value=True)
            ui.update_select("box_orientation", selected="vertical")
            ui.update_text_area("hist_alpha", value="")
            for field in ("line_alpha", "line_style", "scatter_size", "scatter_marker", "scatter_alpha"):
                ui.update_text_area(field, value="")
            ui.update_numeric("bins", value=30)
            ui.update_select("hist_density", selected="False")
            ui.update_select("hist_stacked", selected="False")
            ui.update_select("hist_type", selected="bar")
            ui.update_select("hist_cumulative", selected="0")
            ui.update_checkbox("only_complete", value=True)
            ui.update_checkbox("flooring_timestamp", value=True)

        if backend == "plotly":
            @reactive.calc
            def styled_plotly_figure():
                import plotly.graph_objects as go
                fig = go.Figure(base_figure())
                return _apply_plot_appearance(
                    fig, title=input.title(), x_label=input.x_axis_label(),
                    y_label=input.y_axis_label(), legend_labels=input.legend_labels(),
                    trace_colors=input.trace_colors(), palette=input.palette(),
                    x_min=input.x_min(), x_max=input.x_max(),
                    y_min=input.y_min(), y_max=input.y_max(),
                    template=input.template(), legend_position=input.legend_position(),
                    legend_inside=input.legend_inside(), show_grid=input.show_grid(),
                    x_major_ticks=input.x_major_ticks(), y_major_ticks=input.y_major_ticks(),
                    x_minor_ticks=input.x_minor_ticks(), y_minor_ticks=input.y_minor_ticks(),
                    line_width=(input.line_width()
                                if input.plot_type() == "line" else 2),
                    hist_alpha=input.hist_alpha(), box_showfliers=input.box_showfliers(),
                    box_orientation=input.box_orientation(),
                    title_font_size=input.title_font_size(), axis_font_size=input.axis_font_size(),
                    tick_font_size=input.tick_font_size(), legend_font_size=input.legend_font_size(),
                    line_alpha=input.line_alpha(), line_style=input.line_style(),
                    scatter_size=input.scatter_size(), scatter_marker=input.scatter_marker(),
                    scatter_alpha=input.scatter_alpha(), plot_kind=input.plot_type(),
                )

            @render_plotly
            def plot():
                return styled_plotly_figure()
        else:
            @reactive.calc
            def matplotlib_options():
                # Shared with the export so that all appearance options,
                # including per-panel overrides, are reproduced exactly.
                return dict(title=input.title(),
                            x_label=input.x_axis_label(), y_label=input.y_axis_label(),
                            legend_labels=input.legend_labels(), trace_colors=input.trace_colors(),
                            palette=input.palette(), x_min=input.x_min(), x_max=input.x_max(),
                            y_min=input.y_min(), y_max=input.y_max(), template=input.template(),
                            legend_position=input.legend_position(), legend_inside=input.legend_inside(),
                            show_grid=input.show_grid(),
                            x_major_ticks=input.x_major_ticks(), y_major_ticks=input.y_major_ticks(),
                            x_minor_ticks=input.x_minor_ticks(), y_minor_ticks=input.y_minor_ticks(),
                            title_font_size=input.title_font_size(), axis_font_size=input.axis_font_size(),
                            tick_font_size=input.tick_font_size(), legend_font_size=input.legend_font_size(),
                            panel_font_size=input.panel_font_size(), hist_alpha=input.hist_alpha(),
                            box_showfliers=input.box_showfliers(), box_orientation=input.box_orientation(),
                            line_width=(input.line_width() if input.plot_type() == "line" else 2), line_alpha=input.line_alpha(),
                            line_style=input.line_style(), scatter_size=input.scatter_size(),
                            scatter_marker=input.scatter_marker(), scatter_alpha=input.scatter_alpha(),
                            bins=int(input.bins()) if input.plot_type(
                    ) == "histogram" else 30,
                            density=input.hist_density() == "True" if input.plot_type() == "histogram" else False,
                            stacked=input.hist_stacked() == "True" if input.plot_type() == "histogram" else False,
                            histtype=input.hist_type() if input.plot_type() == "histogram" else "bar",
                            cumulative=int(input.hist_cumulative(
                            )) if input.plot_type() == "histogram" else 0,
                            panel_by=(
                    ((list(_optional_input("panel_by", []) or []) or list(input.group_by() or [])) if len(list(input.group_by() or [])) > 1
                        else list(input.group_by() or []))
                    if list(input.group_by() or []) and bool(_optional_input("multi_panel", False)) else None
                ),
                    independent_panel_limits=(bool(_optional_input("independent_panel_limits", True))
                                                if input.group_by() and _optional_input("multi_panel", False) else True),
                    panel_titles=(_optional_input("panel_titles", "") if input.group_by() and _optional_input("multi_panel", False) else ""),)

            @render.download(filename="plot_export.zip", media_type="application/zip")
            def export_python():
                import io
                import zipfile
                spec = prepared_plot()
                if "error" in spec:
                    raise ValueError(spec["error"])
                frame = pd.DataFrame(spec["data"])
                script = _standalone_matplotlib_script(
                    spec, matplotlib_options())
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(
                        "plot_data.csv", frame.to_csv(index=False))
                    archive.writestr("plot_script.py", script)
                yield buffer.getvalue()

            @render.plot
            def plot():
                return _matplotlib_direct_figure(prepared_plot(), **matplotlib_options())

        return filter_state

    app_ui = ui.page_fluid(
        ui.tags.style("""
            .container, .container-fluid {max-width: none !important;}
            .summary-plot-container {width:100%; height:auto; aspect-ratio: 2 / 1; min-width:0; position:relative;}
            .summary-plot-container > * {width:100% !important;}
            .summary-plot-container .shiny-plot-output {min-height:400px;}
        """),
        ui.tags.script("""
        (() => {
          // A Plotly relayout mutates its DOM. Observing all descendants and
          // triggering another relayout causes a self-sustaining render loop.
          const observed = new WeakSet();
          const install = () => document.querySelectorAll('.summary-plot-container').forEach(el => {
            if (observed.has(el)) return;
            observed.add(el);
            let lastWidth = -1;
            let pending = false;
            const resize = () => {
              if (pending) return;
              pending = true;
              requestAnimationFrame(() => {
                pending = false;
                const width = Math.round(el.getBoundingClientRect().width);
                if (!width) return;
                const height = Math.round(width / 2);
                if (width === lastWidth && el.style.height === height + 'px') return;
                lastWidth = width;
                el.style.height = height + 'px';
                const graph = el.querySelector('.js-plotly-plot');
                if (graph && window.Plotly) window.Plotly.relayout(graph, {width, height});
              });
            };
            new ResizeObserver(resize).observe(el);
            resize();
          });
          // Only watch tab/container insertions, never individual Plotly nodes.
          let installPending = false;
          new MutationObserver(() => {
            if (installPending) return;
            installPending = true;
            requestAnimationFrame(() => {installPending = false; install();});
          }).observe(document.body, {childList:true, subtree:true});
          document.addEventListener('shown.bs.tab', () => requestAnimationFrame(install));
          install();
        })();
        """),
        ui.div(
            ui.div(
                ui.input_action_button(
                    "add_tab", "+ New plot", class_="btn-primary"),
                ui.input_action_button(
                    "close_tab", "Close selected plot", class_="btn-outline-danger"),
                class_="d-flex gap-2 mb-3",
            ),
            ui.navset_tab(ui.nav_panel("Plot 1", plot_tab_ui(
                "plot_1", {}), value="plot_1"), id="plot_tabs"),
        ),
        title="Summary plots",
    )

    def server(input, output, session):
        next_tab = reactive.value(1)
        active_tabs = reactive.value(("plot_1",))
        tab_filters = {"plot_1": plot_tab_server("plot_1", {})}

        @reactive.effect
        @reactive.event(input.add_tab)
        def add_plot():
            number = next_tab.get() + 1
            next_tab.set(number)
            tab_id = f"plot_{number}"
            # Register module before inserting its UI.
            selected = input.plot_tabs()
            # New tabs intentionally start with no inherited selections.
            inherited = {}
            tab_filters[tab_id] = plot_tab_server(tab_id, inherited)
            active_tabs.set((*active_tabs.get(), tab_id))
            ui.insert_nav_panel("plot_tabs", ui.nav_panel(
                f"Plot {number}", plot_tab_ui(tab_id, inherited), value=tab_id), select=True)

        @reactive.effect
        @reactive.event(input.close_tab)
        def close_plot():
            selected = input.plot_tabs()
            tabs = active_tabs.get()
            if selected not in tabs or len(tabs) <= 1:
                return
            ui.remove_nav_panel("plot_tabs", target=selected)
            active_tabs.set(tuple(t for t in tabs if t != selected))
            # Shiny does not provide a public API to unregister an output module.
            # The removed tab cannot be selected again and tab IDs are never reused.

    return App(app_ui, server)


def run_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    host: str = "127.0.0.1",
    port: int = 0,
    launch_browser: bool = True,
    daemon: bool = True,
    backend: str = "plotly",
):
    """Create and start the Shiny app in a background thread."""
    from shiny import run_app

    app = create_summary_app(data, x=x, y=y, group_by=group_by,
                             plot_type=plot_type, title=title, backend=backend)

    def _run() -> None:
        run_app(app, host=host, port=port,
                launch_browser=launch_browser, dev_mode=False)

    thread = threading.Thread(
        target=_run, name="apmtools-shiny-summary", daemon=daemon)
    thread.start()
    return thread
