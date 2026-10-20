from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

VARIABLE_UNITS = {**upas_units, **lascar_units, **purple_units}


PLOT_TYPES = {
    "bar": "Bar",
    "box": "Boxplot",
    "histogram": "Histogram",
    "line": "Line",
    "scatter": "Scatter",
}


FREQUENCIES = {
    "None": None,
    "Day": "day",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
    "Week": "week",
    "Weekday": "weekday",
}


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

