"""Legend identities based on original columns, not inferred name suffixes."""


def variable_aliases(source_variables=(), aggregate=None):
    """Return processed-column -> source-column identities from plot metadata.

    Never strip suffixes from an arbitrary label: a source column may itself
    legitimately be named ``temperature_mean``.
    """
    return {f"{source}_{aggregate}": str(source) for source in source_variables
            if aggregate is not None and str(source)}


def canonical_legend_key(key, aliases=None):
    return (aliases or {}).get(str(key), str(key))


def canonical_legend_map(mapping, preferred_suffix=None, aliases=None):
    """Consolidate known aliases while preserving unrelated group identities.

    ``preferred_suffix`` is retained for compatibility with older callers;
    canonicalization requires explicit aliases to avoid guessing from names.
    """
    aliases = aliases or {}
    groups = {}
    for key, value in mapping.items():
        base = canonical_legend_key(key, aliases)
        groups.setdefault(base, []).append((str(key), value))
    result = {}
    for base, entries in groups.items():
        preferred = next((value for key, value in entries if key in aliases), None)
        direct = next((value for key, value in entries if key == base), None)
        chosen = preferred if preferred is not None else direct if direct is not None else entries[0][1]
        if chosen != base:
            result[base] = chosen
    return result


def legend_label(mapping, key, aliases=None):
    key = str(key)
    return mapping.get(canonical_legend_key(key, aliases), mapping.get(key, key))


def original_legend_keys(displayed, mapping, known_keys=(), preferred_suffix=None, aliases=None):
    mapping = canonical_legend_map(mapping, aliases=aliases)
    reverse = {}
    for original, replacement in mapping.items():
        reverse.setdefault(replacement, set()).add(original)
    for original in known_keys:
        base = canonical_legend_key(original, aliases)
        reverse.setdefault(mapping.get(base, base), set()).add(base)
    keys = []
    for label in displayed:
        matches = reverse.get(label, set())
        original = next(iter(matches)) if len(matches) == 1 else label
        if original not in keys:
            keys.append(original)
    return tuple(keys)
