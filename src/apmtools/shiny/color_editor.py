"""Stable colour-editor keys for the actual prepared plotting data."""
from __future__ import annotations
import re
import pandas as pd

_HEX = re.compile(r"#[0-9a-fA-F]{6}\Z")

def valid_hex_color(value):
    return isinstance(value, str) and _HEX.fullmatch(value.strip()) is not None


def active_color_keys(spec, *, mode="group", panel_by=()):
    """Return colour dimensions with nonempty plotted observations.

    Faceting removes its grouping columns. For multi-Y plots, 'both' uses
    combinations, while group/variable/none use the corresponding colour
    dimension (the same keys accepted by multi_y_styles).
    """
    if "error" in spec or spec["data"].empty:
        return ()
    if spec.get("hist_variables"):
        return tuple(dict.fromkeys(spec["data"].loc[
            spec["data"][spec["x"]].notna(), "__hist_series"].astype(str)))
    from .matplotlib_renderer import _panel_partitions
    from .multi_axis import records
    parts = (_panel_partitions(spec, panel_by) if panel_by
             else [(None, spec["data"], spec["groups"])])
    keys = []
    for _, frame, remaining in parts:
        if frame.empty:
            continue
        sub = {**spec, "data": frame, "groups": remaining,
               "_original_groups": spec["groups"]} if panel_by else {**spec, "data": frame, "groups": remaining}
        if spec.get("multi_axis"):
            pairs = [(str(y), str(g) if remaining else None)
                     for _, g, y, _, _, _ in records(sub)]
        else:
            ys = list(sub["ys"])
            groups = (list(pd.unique(frame["__plot_group"].astype(str)))
                      if remaining and "__plot_group" in frame else [None])
            pairs = []
            for g in groups:
                rows = (frame.loc[frame["__plot_group"].astype(str) == g]
                        if g is not None else frame)
                for y in ys:
                    if y in rows and rows[y].notna().any():
                        pairs.append((str(y), g))
        multi = bool(spec.get("multi_axis") or len(sub["ys"]) > 1)
        for y, g in pairs:
            if not multi or mode in ("group", "none"):
                key = g if g is not None else y
            elif mode == "variable":
                key = y
            else:
                key = f"{g}, {y}" if g is not None else y
            if key not in keys and key != "All observations":
                keys.append(key)
    return tuple(keys)


def color_defaults(spec, keys, palette="Default"):
    from .config import _MPL_PALETTES
    values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    return {key: values[i % len(values)] for i, key in enumerate(keys)}
