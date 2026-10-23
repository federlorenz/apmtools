"""Plotly figure construction."""
from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES
from .data import _add_gap_sections, _add_source_sections


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
        rows = data.loc[data[group_column].astype(str) == str(label)] if group_column else data
        values = pd.to_numeric(rows[x_column], errors="coerce").to_numpy(dtype=float)
        arrays.append(values[np.isfinite(values)])

    all_values = np.concatenate(arrays) if arrays else np.array([])
    fig = go.Figure()
    if not len(all_values):
        fig.update_layout(title="No numerical observations")
        return fig

    edges = np.histogram_bin_edges(all_values, bins=bins)
    widths = np.diff(edges)
    counts = np.array([np.histogram(v, bins=edges)[0] for v in arrays], dtype=float)
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
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=categories)
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
    if "__source_row_order" in plot_df.columns:
        plot_df = _add_source_sections(plot_df, x_column=x_column,
                                       identifier_column=identifier_column or "identifier")
        section_column = "__line_section"
    elif frequency is not None and identifier_column is None:
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
        line_group_column = section_column or identifier_column
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
            line_dash="variable" if len(y_columns) > 1 else None,
        )
    elif section_column is not None:
        # Preserve filtered-out gaps even when no colour grouping is selected.
        if len(y_columns) == 1:
            fig = px.line(plot_df, x=x_column, y=y_columns[0], line_group=section_column)
        else:
            long_df = plot_df[[x_column, section_column, *y_columns]].melt(
                id_vars=[x_column, section_column], var_name="variable", value_name="value")
            fig = px.line(long_df, x=x_column, y="value", color="variable",
                          line_group=section_column)
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
                line_dash="variable",
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
                line_dash="variable",
            )
    else:
        fig = px.line(plot_df, x=x_column, y=y_columns)
    return fig



def _style_multi_y_legend(fig, spec, mode, palette, trace_colors, line_style, legend_labels):
    """Give multi-Y line plots separate variable/group legend keys."""
    from .appearance import _parse_label_map
    from .multi_y import multi_y_styles
    from plotly.colors import qualitative
    import plotly.graph_objects as go

    variables = [str(v) for v in spec["ys"]]
    groups = [str(v) for v in spec["data"]["__plot_group"].dropna().unique()] if spec["groups"] else []
    palettes = getattr(qualitative, palette, None) if palette != "Default" else None
    colors, dashes, variables, groups = multi_y_styles(variables, groups, palettes or qualitative.Plotly,
                                                        mode, _parse_label_map(trace_colors), _parse_label_map(line_style))
    rename = _parse_label_map(legend_labels)
    for trace in fig.data:
        if trace.type not in ("scatter", "scattergl"):
            continue
        name = str(trace.name or "")
        # Plotly Express names traces "group, variable" when both encodings are present.
        group = None
        variable = None
        if groups:
            for g in groups:
                for v in variables:
                    if name == f"{g}, {v}" or name == f"{v}, {g}":
                        group, variable = g, v
                        break
                if variable is not None:
                    break
        else:
            variable = name if name in variables else next((v for v in variables if name.endswith(v)), None)
        if variable is None:
            continue
        key = (variable, group)
        trace.line.color = colors[key]
        trace.line.dash = {"-":"solid", "--":"dash", ":":"dot", "-.":"dashdot"}.get(dashes[key], dashes[key])
        trace.showlegend = False
        trace.legendgroup = f"series:{group}:{variable}"
    if mode == "none":
        for group in groups:
            key = (variables[0], group)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", name=rename.get(group, group),
                line=dict(color=colors[key], dash={"-":"solid", "--":"dash", ":":"dot", "-.":"dashdot"}.get(dashes[key], "solid")),
                legendgroup=f"group:{group}", showlegend=True, hoverinfo="skip",
            ))
        fig.update_layout(showlegend=bool(groups))
        return
    if mode == "both":
        for group in (groups or [None]):
            for variable in variables:
                key = (variable, group)
                label = f"{group}, {variable}" if group is not None else variable
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines", name=rename.get(label, label),
                    line=dict(color=colors[key], dash="solid"),
                    legendgroup=f"series:{group}:{variable}", showlegend=True, hoverinfo="skip",
                ))
        fig.update_layout(showlegend=True)
        return
    # Legend-only proxy traces avoid duplicating real observations and preserve hover data.
    added_section = False
    for heading, items, by_variable in (("Y variable", variables, True), ("Grouping", groups, False)):
        if not items:
            continue
        if added_section:
            # A blank legend row separating the variable and grouping keys.
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=" ",
                                     line=dict(color="rgba(0,0,0,0)", width=0),
                                     showlegend=True, hoverinfo="skip", legendgroup="separator"))
        added_section = True
        for index, item in enumerate(items):
            key = (item, groups[0] if groups else None) if by_variable else (variables[0], item)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=rename.get(item, item),
                                     line=dict(color=colors[key], dash={"-":"solid", "--":"dash", ":":"dot", "-.":"dashdot"}.get(dashes[key], "solid")),
                                     legendgroup=heading, legendgrouptitle_text=heading if index == 0 else None,
                                     showlegend=True, hoverinfo="skip"))
    fig.update_layout(showlegend=True)
