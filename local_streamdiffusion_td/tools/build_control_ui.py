"""Build a reusable TouchDesigner control dashboard for Local StreamDiffusion.

Usage
-----
1. Put this file in the COMP that owns the custom parameters as a Text DAT.
2. Rename the Text DAT to ``build_control_ui`` (optional, but recommended).
3. Right-click the DAT and choose ``Run Script``.

The script only replaces a child named ``ui_root``.  Existing operators and
custom parameters are left untouched.  Parameter COMPs are used deliberately:
they stay connected to the original custom parameters without extra callbacks
or duplicated state.
"""


# Leave blank when this Text DAT is inside the COMP that owns the parameters.
# If it is stored elsewhere, use an absolute TouchDesigner operator path such
# as "/project1/streamdiffusion_control".
SOURCE_OP_PATH = ""

UI_ROOT_NAME = "ui_root"
UI_WIDTH = 1366
UI_HEIGHT = 768

# These must match the custom page names shown in the parameter dialog.
PAGE_LIVE = "realtime parameter"
PAGE_AI = "AI setting parameter"
PAGE_OSC = "osc setting"


COLORS = {
    "canvas": (0.035, 0.047, 0.063, 1.0),
    "header": (0.047, 0.063, 0.086, 1.0),
    "card": (0.065, 0.086, 0.115, 1.0),
    "panel": (0.043, 0.059, 0.080, 1.0),
    "accent": (0.310, 0.890, 0.690, 1.0),
    "accent_dim": (0.105, 0.260, 0.215, 1.0),
    "warm": (0.945, 0.745, 0.345, 1.0),
    "text": (0.900, 0.940, 0.975, 1.0),
    "muted": (0.480, 0.555, 0.640, 1.0),
    "line": (0.125, 0.165, 0.215, 1.0),
}


def _par(operator, name):
    """Return a parameter when it exists, otherwise None."""
    try:
        return getattr(operator.par, name)
    except Exception:
        return None


def _set(operator, name, value):
    """Set a parameter while remaining compatible across TD builds."""
    parameter = _par(operator, name)
    if parameter is None:
        return False
    try:
        parameter.val = value
        return True
    except Exception:
        return False


def _rgba(operator, stem, color):
    """Set a TouchDesigner RGBA parameter group such as bgcolor/fontcolor."""
    r, g, b, a = color
    _set(operator, stem + "r", r)
    _set(operator, stem + "g", g)
    _set(operator, stem + "b", b)
    _set(operator, stem.replace("color", "") + "alpha", a)


def _layout(operator, x, y, width, height):
    _set(operator, "hmode", "fixed")
    _set(operator, "vmode", "fixed")
    _set(operator, "x", x)
    _set(operator, "y", y)
    _set(operator, "w", width)
    _set(operator, "h", height)
    _set(operator, "display", True)
    _set(operator, "enable", True)


def _container(parent_op, name, x, y, width, height, color):
    panel = parent_op.create(containerCOMP, name)
    _layout(panel, x, y, width, height)
    _rgba(panel, "bgcolor", color)
    _set(panel, "border", 0)
    return panel


def _text(
    parent_op,
    name,
    text,
    x,
    y,
    width,
    height,
    size=14,
    color=None,
    background=None,
    align="left",
):
    item = parent_op.create(textCOMP, name)
    _layout(item, x, y, width, height)
    _set(item, "text", text)
    _set(item, "fontsize", size)
    _set(item, "alignx", align)
    _set(item, "aligny", "center")
    _set(item, "wordwrap", True)
    _set(item, "fit", False)
    _rgba(item, "fontcolor", color or COLORS["text"])
    _rgba(item, "bgcolor", background or (0.0, 0.0, 0.0, 0.0))
    return item


def _parameter_panel(parent_op, name, source, page_name, x, y, width, height):
    panel = parent_op.create(parameterCOMP, name)
    _layout(panel, x, y, width, height)

    # Parameter COMP directly controls the source custom parameters.  Page
    # names containing spaces must be quoted in the page-scope pattern.
    _set(panel, "op", source.path)
    _set(panel, "header", False)
    _set(panel, "pagenames", False)
    _set(panel, "labels", True)
    _set(panel, "separators", True)
    _set(panel, "allowexpand", False)
    _set(panel, "allowexpend", False)  # compatibility with older builds
    _set(panel, "builtin", False)
    _set(panel, "custom", True)
    _set(panel, "combinescopes", "all")
    _set(panel, "pagescope", "'{}'".format(page_name))
    _set(panel, "parscope", "*")
    _set(panel, "scopeorder", True)
    _set(panel, "compress", True)
    _set(panel, "vscrollbar", "auto")
    _rgba(panel, "bgcolor", COLORS["panel"])
    return panel


def _card(parent_op, name, title, kicker, x, y, width, height, accent):
    card = _container(parent_op, name, x, y, width, height, COLORS["card"])
    _container(card, "accent", 0, height - 4, width, 4, accent)
    _text(
        card,
        "kicker",
        kicker.upper(),
        16,
        height - 38,
        width - 32,
        18,
        size=10,
        color=accent,
    )
    _text(
        card,
        "title",
        title,
        16,
        height - 66,
        width - 32,
        28,
        size=18,
        color=COLORS["text"],
    )
    return card


def _available_page_names(source):
    try:
        return [page.name for page in source.customPages]
    except Exception:
        return []


def build():
    source = op(SOURCE_OP_PATH) if SOURCE_OP_PATH else parent()
    if source is None:
        raise RuntimeError("Parameter owner COMP could not be found.")

    old_root = source.op(UI_ROOT_NAME)
    if old_root is not None:
        old_root.destroy()

    # Keep the source COMP at a predictable 16:9 dashboard size.
    _set(source, "hmode", "fixed")
    _set(source, "vmode", "fixed")
    _set(source, "w", UI_WIDTH)
    _set(source, "h", UI_HEIGHT)

    root = _container(
        source,
        UI_ROOT_NAME,
        0,
        0,
        UI_WIDTH,
        UI_HEIGHT,
        COLORS["canvas"],
    )
    root.nodeX = 0
    root.nodeY = -250

    # Header
    header = _container(root, "header", 18, 686, 1330, 64, COLORS["header"])
    _container(header, "brand_line", 0, 0, 5, 64, COLORS["accent"])
    _text(
        header,
        "brand",
        "LOCAL STREAMDIFFUSION",
        22,
        26,
        520,
        28,
        size=20,
    )
    _text(
        header,
        "subtitle",
        "REALTIME IMAGE GENERATION CONTROL",
        23,
        8,
        430,
        18,
        size=10,
        color=COLORS["muted"],
    )
    _text(
        header,
        "mode_badge",
        "LIVE CONTROL",
        970,
        16,
        132,
        32,
        size=11,
        color=COLORS["accent"],
        background=COLORS["accent_dim"],
        align="center",
    )
    _text(
        header,
        "osc_badge",
        "OSC  127.0.0.1 : 13001",
        1112,
        16,
        196,
        32,
        size=11,
        color=COLORS["text"],
        background=COLORS["panel"],
        align="center",
    )

    # The live page gets the largest, left-most area because it is used during
    # performance.  Configuration is kept separate to prevent accidental
    # model reloads while operating live controls.
    live = _card(
        root,
        "live_card",
        "Realtime Parameters",
        "Performance controls",
        18,
        18,
        445,
        650,
        COLORS["accent"],
    )
    _parameter_panel(
        live,
        "live_parameters",
        source,
        PAGE_LIVE,
        12,
        14,
        421,
        560,
    )

    ai = _card(
        root,
        "ai_card",
        "AI / Spout Configuration",
        "Apply before performance",
        477,
        18,
        565,
        650,
        COLORS["warm"],
    )
    _parameter_panel(
        ai,
        "ai_parameters",
        source,
        PAGE_AI,
        12,
        14,
        541,
        560,
    )

    connection = _card(
        root,
        "connection_card",
        "OSC Connection",
        "Local control link",
        1056,
        458,
        292,
        210,
        COLORS["accent"],
    )
    _parameter_panel(
        connection,
        "osc_parameters",
        source,
        PAGE_OSC,
        12,
        14,
        268,
        120,
    )

    guide = _card(
        root,
        "guide_card",
        "Operation Flow",
        "Show sequence",
        1056,
        18,
        292,
        426,
        COLORS["muted"],
    )
    _text(
        guide,
        "step_1_num",
        "01",
        18,
        294,
        42,
        34,
        size=20,
        color=COLORS["accent"],
    )
    _text(
        guide,
        "step_1",
        "CONFIGURE\nModel / backend / resolution",
        64,
        280,
        204,
        58,
        size=12,
    )
    _container(guide, "rule_1", 18, 266, 250, 1, COLORS["line"])

    _text(
        guide,
        "step_2_num",
        "02",
        18,
        210,
        42,
        34,
        size=20,
        color=COLORS["warm"],
    )
    _text(
        guide,
        "step_2",
        "APPLY\nReload the inference engine",
        64,
        196,
        204,
        58,
        size=12,
    )
    _container(guide, "rule_2", 18, 182, 250, 1, COLORS["line"])

    _text(
        guide,
        "step_3_num",
        "03",
        18,
        126,
        42,
        34,
        size=20,
        color=COLORS["accent"],
    )
    _text(
        guide,
        "step_3",
        "RUN\nOperate from realtime controls",
        64,
        112,
        204,
        58,
        size=12,
    )
    _text(
        guide,
        "hint",
        "LIVE values update without model reload.",
        18,
        34,
        250,
        46,
        size=10,
        color=COLORS["muted"],
        background=COLORS["panel"],
        align="center",
    )

    # Make the dashboard the visible panel of the parameter-owner COMP.
    _set(source, "opviewer", root.path)
    try:
        root.viewer = True
    except Exception:
        pass

    available = _available_page_names(source)
    missing = [
        name for name in (PAGE_LIVE, PAGE_AI, PAGE_OSC) if name not in available
    ]
    if missing:
        print(
            "[streamdiffusion UI] Built, but custom page name(s) were not found: {}. "
            "Available pages: {}".format(", ".join(missing), ", ".join(available))
        )
    else:
        print("[streamdiffusion UI] Dashboard built at {}".format(root.path))

    return root


# A Text DAT's Run Script command executes this file from top to bottom.
build()
