"""Independent categorical colour and line-style encodings for multi-Y lines."""


def multi_y_styles(variables, groups, palette, mode="group", colors=None, styles=None):
    variables = [str(v) for v in variables]
    groups = [str(g) for g in groups] if groups else []
    colors = colors or {}
    styles = styles or {}
    palette = list(palette) or ["#636EFA", "#EF553B", "#00CC96"]
    combinations = [(var, group) for group in (groups or [None]) for var in variables]
    if mode == "none":
        # Restore original single-Y group encoding across all Y variables.
        color_map = {}
        dash_map = {}
        for var in variables:
            for group in (groups or [None]):
                idx = groups.index(group) if group is not None else 0
                color_map[(var, group)] = colors.get(group, palette[idx % len(palette)]) if group is not None else palette[0]
                style = styles.get(group, "-") if group is not None else "-"
                dash_map[(var, group)] = {"solid":"-", "dashed":"--", "dotted":":", "dashdot":"-."}.get(style, style)
        return color_map, dash_map, variables, groups
    primary = groups if mode == "group" and groups else variables
    secondary = variables if mode == "group" and groups else groups
    dash_values = ["-", "--", ":", "-."]
    color_map, dash_map = {}, {}
    for var in variables:
        for group in (groups or [None]):
            p = group if mode == "group" and groups else var
            q = var if mode == "group" and groups else group
            if mode == "both":
                index = combinations.index((var, group))
                if index < len(palette):
                    default_color = palette[index]
                else:
                    import colorsys
                    hue = (index * 0.618033988749895) % 1.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
                    default_color = "#{:02x}{:02x}{:02x}".format(*(round(255 * c) for c in rgb))
                combined = f"{group}, {var}" if group is not None else var
                color_map[(var, group)] = colors.get(combined, default_color)
                dash_map[(var, group)] = styles.get(combined, "-")
                continue
            color_map[(var, group)] = colors.get(p, colors.get(f"{group}, {var}", palette[primary.index(p) % len(palette)]))
            # For colour-by-variable, the grouping dimension must control dashes.
            # Do not let a variable-level style override erase that distinction.
            default_dash = dash_values[secondary.index(q) % len(dash_values)] if q in secondary else "-"
            # The Shiny appearance panel automatically populates each group with
            # "solid".  Those generated defaults must not collapse the
            # group-specific dash cycle in colour-by-variable mode.
            override = styles.get(q)
            if mode == "variable" and groups and override in ("solid", "-"):
                override = None
            dash_map[(var, group)] = override if override is not None else default_dash
            if dash_map[(var, group)] in ("solid", "dashed", "dotted", "dashdot"):
                dash_map[(var, group)] = {"solid":"-", "dashed":"--", "dotted":":", "dashdot":"-."}[dash_map[(var, group)]]
    return color_map, dash_map, variables, groups
