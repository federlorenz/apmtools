"""Convert multi-statistic wide results to renderer-compatible plot data."""
from __future__ import annotations

import pandas as pd


def prepare_multi_stat_plot(frame, sources, statistics, *, split_statistics=False,
                            split_variables=False):
    """Return a plot frame and Y columns with optional virtual dimensions.

    When split_statistics is disabled, every source/statistic combination is
    displayed as a separate Y series. When enabled, the statistics dimension
    becomes a categorical grouping key, while series remain source-specific.
    Missing source/statistic combinations stay as all-NA columns.
    """
    sources = list(dict.fromkeys(sources))
    statistics = list(dict.fromkeys(statistics))
    columns = [f"{source}_{stat}" for source in sources for stat in statistics]
    work = pd.DataFrame(frame).copy()
    for col in columns:
        if col not in work:
            work[col] = float('nan')
    if not split_statistics and not split_variables:
        return work, columns

    # Always copy Y values before melting so that scatter X columns which
    # overlap selected Y variables remain available to the renderer.
    temporaries = []
    names = {}
    for i, source in enumerate(sources):
        for stat in statistics:
            temp = f"__multi_stat_y_{i}_{stat}"
            while temp in work:
                temp += "_"
            work[temp] = work[f"{source}_{stat}"]
            temporaries.append(temp)
            names[temp] = (source, stat)
    work = work.melt(id_vars=[c for c in work if c not in temporaries],
                     value_vars=temporaries, var_name="__stat_source", value_name="__stat_value")
    work["variables_y"] = work["__stat_source"].map(lambda k: str(names[k][0]))
    work["statistics"] = work["__stat_source"].map(lambda k: str(names[k][1]))
    work = work.drop(columns=["__stat_source"])
    # Restrict the groups if only one of the two virtual dimensions is selected
    # by combining the other dimension into the plot series label.
    if split_statistics and split_variables:
        return work, ["__stat_value"]
    if split_statistics:
        if len(sources) == 1:
            return work, ["__stat_value"]
        # Multiple sources must remain distinct series under a statistic group.
        # Group composition is handled by the caller.
        return work, ["__stat_value"]
    return work, ["__stat_value"]
