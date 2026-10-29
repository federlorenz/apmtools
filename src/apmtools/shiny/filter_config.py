"""Pure transformations for canonical plot-filter configuration.

No Shiny reactive reads or writes belong here. Both the renderer-facing
filter state and explicit saved snapshots use this same mapping.
"""
from __future__ import annotations
import copy


def filter_config_fields(state):
    """Return independent JSON-compatible draft fields for effective filters.

    The stage order and selected numerical-column order are preserved.
    Copies prevent later UI mutations from altering a saved snapshot.
    """
    return {
        "subset_chain": copy.deepcopy(state["stages"]),
        "numeric_subset_bounds": copy.deepcopy(state["numeric_bounds"]),
        "numeric_subset_columns": list(state["numeric_bounds"]),
        "datetime_start": state["start"],
        "datetime_end": state["end"],
    }
