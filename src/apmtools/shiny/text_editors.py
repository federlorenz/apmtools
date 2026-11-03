"""Safe, side-effect-free helpers for dynamically mounted label editors."""
from __future__ import annotations


def update_label_mapping(mapping, changes):
    """Merge mounted edits without publishing incomplete empty text values.

    Clearing an editor is an intermediate typing state, not a new series/tick
    identity.  Keep the previously rendered label until the user types a
    non-empty replacement, preventing Shiny from tearing down the input while
    its contents are being edited.

    Returns a fresh mapping and whether a meaningful change was made.
    """
    updated = dict(mapping)
    changed = False
    for key, value in changes.items():
        if value is None or not isinstance(value, str) or not value.strip():
            continue
        if updated.get(key, key) != value:
            updated[key] = value
            changed = True
    return updated, changed
