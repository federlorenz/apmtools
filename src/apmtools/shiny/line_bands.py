"""Per-X descriptive intervals and t confidence intervals for line plots.

Intervals describe the observed rows of the *prepared* plot specification.
They are not cluster-robust, and already-aggregated rows cannot recover
uncertainty from observations discarded by upstream aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAND_METHODS = frozenset(("ci", "sd", "se", "percentile", "custom"))


def calculate_line_band(frame, x, y, *, method="ci", confidence=95,
                        lower_column="", upper_column=""):
    """Return X, mean/central Y, lower, upper, and N for each unique X.

    In custom mode, every finite lower/upper pair for the same X must agree;
    conflicting rows are invalid rather than silently being averaged. SD/SE
    and CI are undefined for singletons. Percentiles require >=2 observations.
    The returned series are sorted by X and preserve datetime X values.
    """
    if method not in BAND_METHODS:
        raise ValueError(f"Unknown line band method: {method}")
    confidence = float(confidence)
    if not 0 < confidence < 100:
        raise ValueError("Confidence level must lie strictly between 0 and 100")
    required = [x, y] + ([lower_column, upper_column] if method == "custom" else [])
    missing = [col for col in required if not col or col not in frame]
    if missing:
        raise ValueError(f"Error band columns missing from plot data: {missing}")
    values = frame[required].copy()
    values[y] = pd.to_numeric(values[y], errors="coerce")
    values = values.loc[values[x].notna() & np.isfinite(values[y].to_numpy(dtype=float))]
    if values.empty:
        return pd.DataFrame(columns=[x, "center", "lower", "upper", "n"])
    if method == "custom":
        for col in (lower_column, upper_column):
            values[col] = pd.to_numeric(values[col], errors="coerce")
        result = []
        for coordinate, part in values.groupby(x, sort=True, dropna=True):
            lo = part[lower_column].dropna().unique()
            hi = part[upper_column].dropna().unique()
            if len(lo) > 1 or len(hi) > 1:
                raise ValueError(f"Conflicting custom interval bounds at X={coordinate}")
            lower = float(lo[0]) if len(lo) else np.nan
            upper = float(hi[0]) if len(hi) else np.nan
            if np.isfinite(lower) and np.isfinite(upper) and lower > upper:
                raise ValueError(f"Lower error band bound exceeds upper at X={coordinate}")
            result.append((coordinate, float(part[y].mean()), lower, upper, len(part)))
        return pd.DataFrame(result, columns=[x, "center", "lower", "upper", "n"])

    grouped = values.groupby(x, sort=True, dropna=True)[y]
    stats = grouped.agg(["mean", "std", "count"])
    center, deviation, count = (stats[name].to_numpy(dtype=float, copy=True) for name in ("mean", "std", "count"))
    if method == "sd":
        spread = deviation
        lower, upper = center - spread, center + spread
    elif method in ("se", "ci"):
        spread = deviation / np.sqrt(count)
        if method == "ci":
            from scipy.stats import t
            spread *= t.ppf((1 + confidence / 100) / 2, count - 1)
        lower, upper = center - spread, center + spread
    else:
        percent = (100 - confidence) / 2
        lower = grouped.quantile(percent / 100).to_numpy(dtype=float, copy=True)
        upper = grouped.quantile(1 - percent / 100).to_numpy(dtype=float, copy=True)
        lower[count < 2] = np.nan
        upper[count < 2] = np.nan
    result = pd.DataFrame({x: stats.index.to_numpy(), "center": center,
                           "lower": lower, "upper": upper, "n": count.astype(int)})
    return result


def resolved_band_options(frame, y, *, method, confidence=95,
                          lower_column="", upper_column=""):
    """Prefer uncertainty retained during group_stat aggregation.

    Only mean series can consume the stored mean-based uncertainty bounds.
    A previously materialized table with matching bound columns also works.
    """
    if method != "custom" and str(y).endswith("_mean"):
        root = str(y)[:-len("_mean")]
        lo = f"{root}_{method}_lower"
        hi = f"{root}_{method}_upper"
        if lo in frame.columns and hi in frame.columns:
            return {"method": "custom", "confidence": confidence,
                    "lower_column": lo, "upper_column": hi}
    return {"method": method, "confidence": confidence,
            "lower_column": lower_column, "upper_column": upper_column}


def calculate_resolved_line_band(frame, x, y, *, method="ci", confidence=95,
                                 lower_column="", upper_column=""):
    """Use pre-aggregation uncertainty when present, with original sample N."""
    options = resolved_band_options(frame, y, method=method,
        confidence=confidence, lower_column=lower_column,
        upper_column=upper_column)
    band = calculate_line_band(frame, x, y, **options)
    if options["method"] == "custom" and method != "custom" and str(y).endswith("_mean"):
        original_count = str(y)[:-len("_mean")] + "_n"
        if original_count in frame and not band.empty:
            # Report underlying sample count, not the number of aggregate rows.
            valid = frame.loc[frame[x].notna(), [x, original_count]].copy()
            valid[original_count] = pd.to_numeric(valid[original_count], errors="coerce")
            counts = valid.groupby(x, sort=True, dropna=True)[original_count].sum(min_count=1)
            band["n"] = band[x].map(counts).fillna(0).astype(int)
    return band


def line_band_export(spec, *, method="ci", confidence=95,
                     lower_column="", upper_column=""):
    """Export exactly the estimated mean and band bounds for each group/X/Y."""
    df = spec["data"]
    group_col = "__plot_group" if spec.get("groups") and "__plot_group" in df else None
    parts = df.groupby(group_col, sort=False, dropna=False) if group_col else [("All observations", df)]
    results = []
    for label, part in parts:
        for y in spec["ys"]:
            band = calculate_resolved_line_band(part, spec["x"], y,
                    method=method, confidence=confidence,
                    lower_column=lower_column, upper_column=upper_column)
            if len(band):
                band.insert(0, "series", str(y))
                band.insert(0, "group", str(label))
                results.append(band)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["group", "series", spec["x"], "center", "lower", "upper", "n"])
