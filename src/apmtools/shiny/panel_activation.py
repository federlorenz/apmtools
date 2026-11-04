"""Explicit panel selection and stacking eligibility for Shiny plot views."""


def selected_panel_groups(groups, requested, excluded=()):
    """Return eligible, explicitly selected panel dimensions in user order.

    An empty selection never implies all available grouping columns.  The
    enabled/disabled state of the panel controls is handled separately by UI.
    """
    eligible = set(groups or ()) - set(excluded or ())
    return list(dict.fromkeys(name for name in (requested or ()) if name in eligible))
