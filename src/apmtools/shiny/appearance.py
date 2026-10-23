"""Figure labels and appearance controls."""
from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES


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
                     category_columns=None, x_label_override=None):
    """Label plot axes while retaining special histogram and facet labels."""
    if plot_type == "histogram":
        fig.update_xaxes(title_text=_axis_label(x_column))
    elif plot_type in {"box", "bar"}:
        categories = category_columns or [x_column]
        fig.update_xaxes(title_text=" | ".join(_axis_label(c) for c in categories))
        fig.update_yaxes(title_text=_axis_label(y_columns[0]))
    elif plot_type == "line" and len(y_columns) > 1:
        fig.update_xaxes(title_text=_axis_label(x_column))
        fig.update_yaxes(title_text="<br>".join(_axis_label(y) for y in y_columns))
        for annotation in fig.layout.annotations:
            label = str(annotation.text)
            if label.startswith("variable="):
                annotation.text = _axis_label(label.split("=", 1)[1])
    else:
        fig.update_xaxes(title_text=_axis_label(x_column))
        if y_columns:
            fig.update_yaxes(title_text=_axis_label(y_columns[0]))
    if x_label_override is not None:
        fig.update_xaxes(title_text=x_label_override)
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
                axis.set_major_locator(AutoDateLocator(minticks=max(2, count - 2), maxticks=count))
            else:
                axis.set_major_locator(MaxNLocator(nbins=count))
        if minor:
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(__import__("matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
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
    fig.update_xaxes(title_font=dict(size=axis_font_size), tickfont=dict(size=tick_font_size))
    fig.update_yaxes(title_font=dict(size=axis_font_size), tickfont=dict(size=tick_font_size))
    if x_label.strip():
        fig.update_xaxes(title_text=x_label.strip())
    if y_label.strip():
        fig.update_yaxes(title_text=y_label.strip().replace("\n", "<br>"))
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)
    use_x, use_y = _numeric_axis_controls(plot_kind, box_orientation)
    for active, axis_name, count, minor in ((use_x, "x", x_major_ticks, x_minor_ticks),
                                            (use_y, "y", y_major_ticks, y_minor_ticks)):
        if not active:
            continue
        update = getattr(fig, f"update_{axis_name}axes")
        update(nticks=max(2, min(30, int(count or 6))), minor=dict(ticks="inside" if minor else "", showgrid=bool(show_grid and minor)))
    for axis, lower, upper in (("x", x_min, x_max), ("y", y_min, y_max)):
        datetime_axis = axis == "x" and any(
            getattr(trace, "type", "") == "scatter" and
            any(isinstance(v, (pd.Timestamp, datetime, np.datetime64)) for v in (trace.x[:3] if trace.x is not None else []))
            for trace in fig.data
        )
        low = _parse_axis_limit(lower, datetime_axis=datetime_axis)
        high = _parse_axis_limit(upper, datetime_axis=datetime_axis)
        if low is not None or high is not None:
            getattr(fig, f"update_{axis}axes")(range=[low, high], autorange=False)
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
        legend_settings = {**legend_settings, "bgcolor": "rgba(255,255,255,0.7)"}
    fig.update_layout(legend={**legend_settings, "title": {"text": "Legend", "font": {"size": legend_font_size}}, "font": {"size": legend_font_size}})
    labels = _parse_label_map(legend_labels)
    colors = _parse_label_map(trace_colors)
    alphas, styles, sizes, markers, marker_alphas = _series_style_maps(
        line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha)
    histogram_alphas = _parse_label_map(hist_alpha)
    palette_values = getattr(qualitative, palette, None) if palette != "Default" else None
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
            trace.opacity = _bounded_float(histogram_alphas, original, 0.65, 0, 1)
        if plot_kind == "box":
            trace.boxpoints = "outliers" if box_showfliers else False
            if box_orientation == "horizontal":
                old_x, old_y = trace.x, trace.y
                trace.x, trace.y = old_y, old_x
                trace.orientation = "h"
        if getattr(trace, "line", None) is not None and plot_kind == "line":
            trace.line.width = line_width
            if plot_kind == "line":
                trace.line.dash = {"-": "solid", ":": "dot", "--": "dash"}[_line_style(styles, original)]
                trace.opacity = _bounded_float(alphas, original, 1, 0, 1)
        if getattr(trace, "marker", None) is not None and trace.type in ("scatter", "scattergl"):
            if plot_kind == "scatter":
                trace.marker.size = _bounded_float(sizes, original, 7, 0.1, 100)
                trace.marker.symbol = {"o": "circle", "x": "x"}[_scatter_marker(markers, original)]
                trace.opacity = _bounded_float(marker_alphas, original, 1, 0, 1)
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

