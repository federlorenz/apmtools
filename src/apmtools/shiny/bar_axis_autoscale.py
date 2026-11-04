"""Distinguish auto-populated bar-axis input text from explicit bounds.

The Shiny UI historically writes data-derived limits into the same controls
used for manual overrides.  A browser update can lag one render behind a
change in stacking.  Recognising both previous and current generated values
prevents that stale text being interpreted as an explicit clipping limit.
"""
from __future__ import annotations


def remember_auto_limits(history, limits):
    """Record the displayed default bounds for comparison across redraws."""
    for field in ('y_min', 'y_max'):
        text = str(limits.get(field, '') or '').strip()
        if text:
            seen = history.setdefault(field, [])
            if text not in seen:
                seen.append(text)
                # Avoid retaining every value ever produced by past datasets.
                if len(seen) > 8:
                    del seen[:-8]


def release_automatic_bar_limits(options, history, *, protected=()):
    """Let Matplotlib autoscale limits that the UI filled automatically.

    Bounds different from the generated values are user edits and remain
    explicit.  Restored explicit saved limits are also protected.
    """
    resolved = dict(options)
    for field in ('y_min', 'y_max'):
        if field in protected:
            continue
        text = str(options.get(field, '') or '').strip()
        if text in history.get(field, ()):
            resolved[field] = ''
    return resolved
