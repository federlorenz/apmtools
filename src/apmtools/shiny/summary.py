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


def _default_x(data: pd.DataFrame) -> str:
    """Return the first datetime column, or otherwise the first column."""
    datetime_columns = [
        column
        for column in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[column])
    ]
    return str(datetime_columns[0] if datetime_columns else data.columns[0])


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric column names as strings."""
    return [
        str(column)
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ]


def _normalise_y(
    data: pd.DataFrame,
    x: str,
    y: str | Sequence[str] | None,
) -> list[str]:
    """Validate and normalise an initial Y selection."""
    if y is None:
        return [column for column in _numeric_columns(data) if column != x][:1]

    values = [y] if isinstance(y, str) else list(y)
    missing = [column for column in values if column not in data.columns]
    if missing:
        raise ValueError(f"Unknown y column(s): {missing}")
    return [str(column) for column in values]


def create_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
):
    """
    Create, but do not start, a Shiny application for a Summary/DataFrame.

    Supported plot types are ``line``, ``bar``, ``histogram`` and ``box``.
    """
    from shiny import App, reactive, render, ui
    from shinywidgets import output_widget, render_plotly
    import plotly.express as px

    df = pd.DataFrame(data).copy()
    if df.empty:
        raise ValueError("Cannot create a Shiny application from an empty Summary.")

    original_columns = list(df.columns)
    column_map = {str(column): column for column in original_columns}
    columns = list(column_map)
    numeric_columns = _numeric_columns(df)

    if x is None:
        x = _default_x(df)
    else:
        x = str(x)
    if x not in column_map:
        raise ValueError(f"Unknown x column: {x!r}")

    initial_y = _normalise_y(df, column_map[x], y)

    if group_by is not None:
        group_by = str(group_by)
        if group_by not in column_map:
            raise ValueError(f"Unknown group_by column: {group_by!r}")

    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. "
            f"Choose one of: {', '.join(PLOT_TYPES)}"
        )

    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.h4("Plot controls"),

            ui.input_select(
                "plot_type",
                "Plot type",
                choices=PLOT_TYPES,
                selected=plot_type,
            ),

            ui.input_select(
                "x",
                "X-axis / category",
                choices=columns,
                selected=x,
            ),

            ui.input_selectize(
                "y",
                "Y-axis / variable",
                choices=numeric_columns,
                selected=initial_y,
                multiple=True,
            ),

            ui.input_select(
                "group_by",
                "Group by",
                choices={"None": "None", **{c: c for c in columns}},
                selected=group_by if group_by is not None else "None",
            ),

            ui.input_numeric(
                "bins",
                "Histogram bins",
                value=30,
                min=1,
                max=200,
                step=1,
            ),

            ui.input_text(
                "title",
                "Plot title",
                value=title,
            ),

            ui.help_text(
                "Line: ordered X values and one or more Y variables. "
                "Bar: categorical X and numeric Y. "
                "Histogram: distribution of numeric variables. "
                "Boxplot: distribution of numeric variables, optionally grouped."
            ),
        ),

        ui.card(
            ui.card_header("Interactive plot"),
            output_widget("plot", height="75vh"),
        ),
    )

    def server(input, output, session):
        @reactive.calc
        def selected_data() -> pd.DataFrame:
            selected_plot = input.plot_type()
            selected_x = input.x()
            selected_y = list(input.y())
            selected_group = input.group_by()

            if selected_plot == "histogram":
                if not selected_y:
                    return pd.DataFrame()
                use_columns = [column_map[c] for c in selected_y]
            else:
                if not selected_y:
                    return pd.DataFrame()
                use_columns = [
                    column_map[selected_x],
                    *(column_map[c] for c in selected_y),
                ]

                if selected_group != "None":
                    group_column = column_map[selected_group]
                    if group_column not in use_columns:
                        use_columns.append(group_column)

            return df.loc[:, use_columns].copy()

        @render_plotly
        def plot():
            plot_df = selected_data()
            selected_plot = input.plot_type()
            selected_x = input.x()
            selected_y = list(input.y())
            selected_group = input.group_by()

            if plot_df.empty or not selected_y:
                return px.scatter(title="Select at least one numeric variable")

            y_columns = [column_map[c] for c in selected_y]

            if selected_plot == "histogram":
                # Plotly accepts multiple columns through y with a long-form
                # transformation; melt gives consistent legends.
                long_df = plot_df[y_columns].melt(
                    var_name="variable",
                    value_name="value",
                )
                fig = px.histogram(
                    long_df,
                    x="value",
                    color="variable",
                    nbins=int(input.bins()),
                    barmode="overlay" if len(y_columns) > 1 else "relative",
                )

            elif selected_plot == "box":
                if selected_group == "None":
                    long_df = plot_df[y_columns].melt(
                        var_name="variable",
                        value_name="value",
                    )
                    fig = px.box(
                        long_df,
                        x="variable",
                        y="value",
                        points="outliers",
                    )
                else:
                    group_column = column_map[selected_group]
                    long_df = plot_df[
                        [group_column, *y_columns]
                    ].melt(
                        id_vars=[group_column],
                        var_name="variable",
                        value_name="value",
                    )
                    fig = px.box(
                        long_df,
                        x="variable",
                        y="value",
                        color=group_column,
                        points="outliers",
                    )

            elif selected_plot == "bar":
                x_column = column_map[selected_x]
                if selected_group == "None":
                    fig = px.bar(
                        plot_df,
                        x=x_column,
                        y=y_columns,
                        barmode="group",
                    )
                else:
                    group_column = column_map[selected_group]
                    long_df = plot_df.melt(
                        id_vars=[x_column, group_column],
                        value_vars=y_columns,
                        var_name="variable",
                        value_name="value",
                    )
                    fig = px.bar(
                        long_df,
                        x=x_column,
                        y="value",
                        color=group_column,
                        facet_row="variable" if len(y_columns) > 1 else None,
                        barmode="group",
                    )

            else:  # line
                x_column = column_map[selected_x]
                if selected_group == "None":
                    fig = px.line(
                        plot_df,
                        x=x_column,
                        y=y_columns,
                    )
                else:
                    group_column = column_map[selected_group]
                    long_df = plot_df.melt(
                        id_vars=[x_column, group_column],
                        value_vars=y_columns,
                        var_name="variable",
                        value_name="value",
                    )
                    fig = px.line(
                        long_df,
                        x=x_column,
                        y="value",
                        color=group_column,
                        facet_row="variable" if len(y_columns) > 1 else None,
                    )

            fig.update_layout(
                title=input.title(),
                hovermode="x unified"
                if selected_plot == "line"
                else "closest",
                margin=dict(l=60, r=30, t=70, b=60),
            )
            return fig

    return App(app_ui, server)


def run_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    host: str = "127.0.0.1",
    port: int = 0,
    launch_browser: bool = True,
    daemon: bool = True,
):
    """
    Create and start the Shiny app in a background thread.

    ``port=0`` asks the operating system to select an available port.
    """
    from shiny import run_app

    app = create_summary_app(
        data,
        x=x,
        y=y,
        group_by=group_by,
        plot_type=plot_type,
        title=title,
    )

    def _run() -> None:
        run_app(
            app,
            host=host,
            port=port,
            launch_browser=launch_browser,
            dev_mode=False,
        )

    thread = threading.Thread(
        target=_run,
        name="apmtools-shiny-summary",
        daemon=daemon,
    )
    thread.start()
    return thread
