"""Long-format data preparation for overlaid multi-variable histograms."""
import pandas as pd
from .data import _composite_labels


def histogram_long_data(frame, variables, *, group_columns=()):
    """Create one histogram series per variable/group using a shared numeric axis.

    Non-histogram columns are retained so faceting, export, and filtering work.
    """
    variables = list(dict.fromkeys(variables))
    if not variables:
        raise ValueError("At least one histogram variable is required")
    missing = [name for name in variables if name not in frame]
    if missing:
        raise KeyError(f"Missing histogram variables: {missing}")
    groups = list(group_columns)
    for col in groups:
        if col not in frame:
            raise KeyError(col)
    chunks = []
    for name in variables:
        part = frame.copy()
        part["__hist_value"] = pd.to_numeric(part[name], errors="coerce")
        part["__hist_variable"] = name
        part["__hist_series"] = (name + " | " + _composite_labels(part, groups)
                                 if groups else name)
        part = part.loc[part["__hist_value"].notna()]
        chunks.append(part)
    if not chunks:
        return frame.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)
