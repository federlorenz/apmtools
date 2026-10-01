"""Interactive Plotly/Shiny application for apmtools Summary objects."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pandas as pd


PLOT_TYPES = {
    "line": "Line",
    "bar": "Bar",
    "histogram": "Histogram",
    "box": "Boxplot",
}

# ``None`` means row-level plotting.  The other values are passed directly to
# Summary.group_stat(frequency=...).
FREQUENCIES = {
    "None (raw data)": None,
    "Second": "second",
    "Minute": "minute",
    "Hour": "hour",
    "Day": "day",
    "Week": "week",
    "Weekday": "weekday",
}

AGGREGATES = {
    "Mean": "mean",
    "Median": "median",
    "Sum": "sum",
    "Minimum": "min",
    "Maximum": "max",
    "Count": "count",
    "Standard deviation": "std",
}

def _default_x(data: pd.DataFrame) -> str:
    """Return the first datetime column, or otherwise the first column."""
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
    return [
        str(column)
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ]


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


def _subset_choices(data: pd.DataFrame, column: str) -> tuple[dict[str, str], dict[str, object]]:
    """Build stable Shiny choices and their corresponding original values."""
    values = pd.unique(data[column].dropna())
    choices: dict[str, str] = {}
    value_map: dict[str, object] = {}
    for i, value in enumerate(values):
        key = str(i)
        choices[key] = str(value)
        value_map[key] = value
    return choices, value_map


def _apply_filters(data, subset_filters, datetime_start=None, datetime_end=None):
    """Apply multiple ``Summary.subset()`` filters and datetime filtering.

    Parameters
    ----------
    subset_filters : dict
        Mapping of column names to the original values that should be retained.
        Conditions for different columns are combined with ``filter_style="all"``.
        Values within one column are OR'd by :meth:`Summary.subset`.
    """
    filtered = data

    if subset_filters:
        # Summary.subset() handles non-contiguous DataFrame indices
        # positionally, so no index manipulation is required here.
        filtered = filtered.subset(subset_filters, filter_style="all")

    if datetime_start or datetime_end:
        filtered = filtered.dtfilter(
            date_start=datetime_start or None,
            date_end=datetime_end or None,
        )

    return filtered


def _aggregate(data, value_columns, frequency, grouping_by, aggregate):
    """Aggregate through ``Summary.group_stat``.

    ``group_by`` is a plotting/grouping choice for raw observations.  It only
    becomes a statistical grouping when a frequency is selected.
    """
    if frequency is None:
        return data

    result = data.group_stat(
        value_cols=value_columns,
        frequency=frequency,
        grouping_by=grouping_by,
        aggregate=[aggregate],
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
    if step is None or x_column not in result.columns:
        # Frequencies such as weekday do not preserve enough chronological
        # information to infer discontinuities reliably.
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
        group_label = result[group_columns].astype(str).agg(" | ".join, axis=1)
        result[section_column] = group_label + \
            " | section " + section_number.astype(str)
    else:
        gap = result[x_column].diff()
        section_number = (gap.gt(step) | gap.isna()
                          ).astype(int).cumsum().sub(1)
        result[section_column] = "section " + section_number.astype(str)

    return result


def create_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
):
    """Create, but do not start, an interactive Shiny app for a Summary."""
    from shiny import App, reactive, render, ui
    from shinywidgets import output_widget, render_plotly
    import plotly.express as px

    # Keep the Summary object intact: subset(), dtfilter() and group_stat()
    # are methods on Summary and are intentionally used below.
    summary = data.copy(deep=True)
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame/Summary")
    if summary.empty:
        raise ValueError(
            "Cannot create a Shiny application from an empty Summary.")

    columns = [str(c) for c in summary.columns]
    column_map = {str(c): c for c in summary.columns}
    numeric_columns = _numeric_columns(summary)

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

    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. Choose one of: {', '.join(PLOT_TYPES)}")

    subset_columns = columns

    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.h4("Plot controls"),
            ui.input_select("plot_type", "Plot type",
                            choices=PLOT_TYPES, selected=plot_type),
            ui.input_select("x", "X-axis / category",
                            choices=columns, selected=x),
            ui.input_selectize(
                "y", "Y-axis / variable", choices=numeric_columns,
                selected=initial_y, multiple=True,
            ),
            ui.input_selectize(
                "group_by", "Group by",
                choices={c: c for c in columns},
                selected=initial_groups,
                multiple=True,
                options={"placeholder": "Select one or more grouping columns"},
            ),
            ui.hr(),
            ui.h5("Aggregation"),
            ui.input_select(
                "frequency", "Frequency", choices=FREQUENCIES,
                selected="None (raw data)",
            ),
            ui.input_select(
                "aggregate", "Statistic", choices=AGGREGATES,
                selected="Mean",
            ),
            ui.hr(),
            ui.h5("Subset"),
            ui.input_selectize(
                "subset_columns", "Columns", choices={c: c for c in subset_columns},
                selected=[], multiple=True,
                options={"placeholder": "Select one or more columns"},
            ),
            ui.output_ui("subset_filters"),
            ui.hr(),
            ui.h5("Datetime filter"),
            ui.input_text(
                "datetime_start", "Start datetime",
                value="", placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
            ),
            ui.input_text(
                "datetime_end", "End datetime",
                value="", placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
            ),
            ui.input_numeric("bins", "Histogram bins",
                             value=30, min=1, max=200, step=1),
            ui.input_text("title", "Plot title", value=title),
            ui.help_text(
                "Subset and datetime filters are applied before aggregation. "
                "When a frequency is selected, aggregation is performed with "
                "Summary.group_stat()."
            ),
        ),
        ui.card(
            ui.card_header("Interactive plot"),
            output_widget("plot", height="75vh"),
        ),
    )

    def server(input, output, session):
        @render.ui
        def subset_filters():
            """Render one value selector for every selected subset column."""
            selected_columns = list(input.subset_columns())
            if not selected_columns:
                return ui.help_text(
                    "Select one or more columns above to choose their values."
                )

            controls = []
            for i, selected_column in enumerate(selected_columns):
                if selected_column not in column_map:
                    continue
                choices, _ = _subset_choices(
                    summary, column_map[selected_column])
                input_id = f"subset_values_{i}"
                controls.append(
                    ui.input_selectize(
                        input_id,
                        f"{selected_column} values",
                        choices=choices,
                        selected=[],
                        multiple=True,
                        options={
                            "placeholder": f"Select {selected_column} values"},
                    )
                )
            return ui.div(*controls)

        @reactive.calc
        def filtered_summary():
            selected_columns = list(input.subset_columns())
            subset_filters = {}

            for i, selected_column in enumerate(selected_columns):
                if selected_column not in column_map:
                    continue
                selected_keys = list(input[f"subset_values_{i}"]())
                if not selected_keys:
                    # An explicitly selected column with no selected values
                    # should not remove all data; it simply contributes no
                    # filtering condition until values are chosen.
                    continue
                _, value_map = _subset_choices(
                    summary, column_map[selected_column])
                values = [value_map[key]
                          for key in selected_keys if key in value_map]
                if values:
                    subset_filters[column_map[selected_column]] = values

            start = input.datetime_start().strip()
            end = input.datetime_end().strip()
            return _apply_filters(summary, subset_filters, start, end)

        @reactive.calc
        def selected_data():
            current = filtered_summary()
            selected_y = list(input.y())
            if not selected_y:
                return pd.DataFrame()

            y_columns = [column_map[c] for c in selected_y]
            selected_groups = list(input.group_by())
            grouping_by = [column_map[c] for c in selected_groups]
            frequency = FREQUENCIES[input.frequency()]
            aggregate = AGGREGATES[input.aggregate()]

            # All selected grouping columns are passed directly to
            # Summary.group_stat() when a frequency is selected.  With raw
            # data they remain plotting-only grouping columns.
            if frequency is not None:
                current = _aggregate(current, y_columns,
                                     frequency, grouping_by, aggregate)

            return current

        @render_plotly
        def plot():
            plot_df = selected_data()
            selected_plot = input.plot_type()
            selected_x = input.x()
            selected_y = list(input.y())
            selected_groups = list(input.group_by())
            frequency = FREQUENCIES[input.frequency()]

            if plot_df.empty or not selected_y:
                return px.scatter(title="No data for the selected filters")

            # Keep the original selected Y columns separate from the columns
            # produced by group_stat().  For example, selecting ``pm25`` and
            # hourly means produces ``pm25_mean`` in the aggregated result.
            selected_y_columns = [column_map[c] for c in selected_y]

            # Summary.group_stat() creates the frequency column itself.
            # For example, frequency="hour" creates an ``hour`` column,
            # while an aggregate such as mean creates ``pm25_mean``.
            # Therefore the aggregated frequency column replaces the
            # originally selected X-axis column.
            if frequency is not None:
                aggregate = AGGREGATES[input.aggregate()]
                y_columns = _aggregated_value_columns(
                    plot_df, selected_y_columns, aggregate
                )
                x_column = frequency

                if x_column not in plot_df.columns:
                    return px.scatter(
                        title=(
                            f"Aggregation did not produce the expected "
                            f"'{x_column}' column"
                        )
                    )
            else:
                y_columns = [
                    c for c in selected_y_columns if c in plot_df.columns
                ]
                x_column = column_map[selected_x]

            if not y_columns:
                return px.scatter(
                    title="Selected variables are not available after aggregation"
                )

            # Multiple grouping columns are supported by group_stat(). For
            # plotting, Plotly needs one color/category column, so combine the
            # selected grouping columns into a readable composite label.
            group_columns = [column_map[c] for c in selected_groups
                             if column_map[c] in plot_df.columns]
            if group_columns:
                plot_df = plot_df.copy()
                plot_df["__plot_group"] = (
                    plot_df[group_columns].astype(str).agg(" | ".join, axis=1)
                )

            if selected_plot == "histogram":
                if group_columns:
                    # With grouping, overlay one histogram per group on the
                    # same axes.  ``group`` is the color dimension rather
                    # than the selected variable, so each group gets its own
                    # histogram.  Transparency makes overlapping groups
                    # directly comparable.
                    group_column = "__plot_group"
                    long_df = plot_df[[group_column, *y_columns]].melt(
                        id_vars=[group_column],
                        var_name="variable",
                        value_name="value",
                    )
                    fig = px.histogram(
                        long_df,
                        x="value",
                        color=group_column,
                        nbins=int(input.bins()),
                        barmode="overlay",
                        opacity=0.45,
                        facet_row=("variable" if len(y_columns) > 1 else None),
                    )
                    # Keep the groups as the legend entries rather than the
                    # selected Y variables.
                    fig.update_traces(marker_line_width=0)
                else:
                    long_df = plot_df[y_columns].melt(
                        var_name="variable", value_name="value"
                    )
                    fig = px.histogram(
                        long_df,
                        x="value",
                        color="variable",
                        nbins=int(input.bins()),
                        barmode="overlay" if len(
                            y_columns) > 1 else "relative",
                        opacity=0.65 if len(y_columns) > 1 else 1.0,
                    )
            elif selected_plot == "box":
                if not group_columns:
                    long_df = plot_df[y_columns].melt(
                        var_name="variable", value_name="value")
                    fig = px.box(long_df, x="variable",
                                 y="value", points="outliers")
                else:
                    group_column = "__plot_group"
                    long_df = plot_df[[group_column, *y_columns]].melt(
                        id_vars=[group_column], var_name="variable", value_name="value",
                    )
                    fig = px.box(long_df, x="variable", y="value",
                                 color=group_column, points="outliers")
            elif selected_plot == "bar":
                if not group_columns:
                    fig = px.bar(plot_df, x=x_column,
                                 y=y_columns, barmode="group")
                else:
                    group_column = "__plot_group"
                    long_df = plot_df[[x_column, group_column, *y_columns]].melt(
                        id_vars=[x_column, group_column], var_name="variable", value_name="value",
                    )
                    fig = px.bar(
                        long_df, x=x_column, y="value", color=group_column,
                        facet_row="variable" if len(y_columns) > 1 else None, barmode="group",
                    )
            else:
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
                        line_columns = [
                            x_column, identifier_column, *y_columns]
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

            fig.update_layout(
                title=input.title(),
                hovermode="x unified" if selected_plot == "line" else "closest",
                margin=dict(l=60, r=30, t=70, b=60),
            )
            return fig

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
):
    """Create and start the Shiny app in a background thread."""
    from shiny import run_app

    app = create_summary_app(
        data, x=x, y=y, group_by=group_by, plot_type=plot_type, title=title)

    def _run() -> None:
        run_app(app, host=host, port=port,
                launch_browser=launch_browser, dev_mode=False)

    thread = threading.Thread(
        target=_run, name="apmtools-shiny-summary", daemon=daemon)
    thread.start()
    return thread
