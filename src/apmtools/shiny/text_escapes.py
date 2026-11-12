"""Decode explicit user-facing plot label escapes without evaluating Python code.

Persist the literal input (e.g. ``r'first\nsecond'``) in saved plot
settings; apply this only at the Matplotlib/formatter display boundary.
Unknown backslash escapes (including Matplotlib mathtext commands) are left
unchanged. A doubled backslash quotes the following backslash.
"""
from __future__ import annotations

import re

def decode_plot_text(value):
    if value is None:
        return ""
    raw = str(value)
    pattern = re.compile(r"\\(?:\\|n|r|t|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})")

    def replacement(match):
        token = match.group()[1:]
        if token == "\\":
            return "\\"
        if token == "n":
            return "\n"
        if token == "r":
            return "\r"
        if token == "t":
            return "\t"
        return chr(int(token[1:], 16))

    return pattern.sub(replacement, raw)
