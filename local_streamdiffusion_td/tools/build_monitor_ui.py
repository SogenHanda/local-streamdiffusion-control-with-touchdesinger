"""Build a live OSC monitoring dashboard inside a TouchDesigner Container.

This builder supports the current OSC In DAT layout where each row contains a
single message string such as::

    /streamdiffusion/monitor/input_fps 17.93354
    /streamdiffusion/monitor/state "実行中"

It also accepts an OSC In DAT whose address and value are split into columns.

Usage
-----
1. Put this file in the Container that should own the monitor UI as a Text DAT.
2. Keep ``OSC_IN_DAT_PATH`` blank for automatic discovery of an OSC In DAT on
   port 9001, or enter its absolute TouchDesigner path.
3. Right-click the Text DAT and choose ``Run Script``.

Only a child named ``monitor_ui_root`` is replaced when the script is rerun.
"""


SOURCE_CONTAINER_PATH = ""
OSC_IN_DAT_PATH = ""
OSC_MONITOR_PORT = 9001
OSC_PREFIX = "/streamdiffusion/monitor/"

UI_ROOT_NAME = "monitor_ui_root"
UI_WIDTH = 1366
UI_HEIGHT = 768


COLORS = {
    "canvas": (0.035, 0.047, 0.063, 1.0),
    "header": (0.055, 0.075, 0.105, 1.0),
    "card": (0.075, 0.100, 0.135, 1.0),
    "card_alt": (0.065, 0.087, 0.118, 1.0),
    "line": (0.135, 0.175, 0.225, 1.0),
    "text": (0.925, 0.955, 0.985, 1.0),
    "muted": (0.510, 0.610, 0.710, 1.0),
    "green": (0.315, 0.890, 0.690, 1.0),
    "blue": (0.325, 0.690, 0.965, 1.0),
    "yellow": (0.945, 0.735, 0.325, 1.0),
    "purple": (0.690, 0.500, 0.960, 1.0),
    "red": (0.925, 0.330, 0.350, 1.0),
}


METRICS = (
    # key, label, formatter, accent, optional normalized bar maximum
    ("input_fps", "INPUT FPS", "number", "green", None),
    ("source_fps", "SOURCE FPS", "source_fps", "green", None),
    ("output_fps", "OUTPUT FPS", "number", "green", None),
    ("inference_fps", "INFER FPS", "number", "green", None),

    ("inference_ms", "INFERENCE", "milliseconds", "blue", None),
    ("end_to_end_ms", "END TO END", "milliseconds", "blue", None),
    ("active_backend", "BACKEND", "raw", "blue", None),
    ("gpu_utilization", "GPU LOAD", "percent_value", "yellow", 100.0),

    ("vram", "VRAM", "vram", "yellow", "vram"),
    ("gpu_temperature", "GPU TEMP", "temperature", "yellow", 100.0),
    ("input_resolution", "INPUT SIZE", "raw", "blue", None),
    ("output_resolution", "OUTPUT SIZE", "raw", "blue", None),

    ("spout_receive_ms", "SPOUT IN", "milliseconds", "blue", None),
    ("spout_send_ms", "SPOUT OUT", "milliseconds", "blue", None),
    ("input_age_ms", "INPUT AGE", "milliseconds", "blue", None),
    ("preprocess_ms", "PREPROCESS", "milliseconds", "blue", None),

    ("vae_encode_ms", "VAE ENCODE", "milliseconds", "purple", None),
    ("unet_ms", "UNET", "milliseconds", "purple", None),
    ("vae_decode_ms", "VAE DECODE", "milliseconds", "purple", None),
    ("postprocess_ms", "POSTPROCESS", "milliseconds", "purple", None),

    ("motion_score", "MOTION", "ratio_percent_1", "purple", 1.0),
    ("temporal_feedback", "INPUT HOLD", "ratio_percent_0", "purple", 1.0),
    ("latent_morph", "LATENT MORPH", "ratio_percent_0", "purple", 1.0),
)


def _par(operator, name):
    try:
        return getattr(operator.par, name)
    except Exception:
        return None


def _set(operator, name, value):
    parameter = _par(operator, name)
    if parameter is None:
        return False
    try:
        parameter.val = value
        return True
    except Exception:
        return False


def _expr(operator, name, expression):
    parameter = _par(operator, name)
    if parameter is None:
        return False
    try:
        parameter.expr = expression
        return True
    except Exception:
        return False


def _rgba(operator, stem, color):
    red, green, blue, alpha = color
    _set(operator, stem + "r", red)
    _set(operator, stem + "g", green)
    _set(operator, stem + "b", blue)
    _set(operator, stem.replace("color", "") + "alpha", alpha)


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
    _set(item, "wordwrap", False)
    _set(item, "editmode", "locked")
    _set(item, "fit", False)
    _rgba(item, "fontcolor", color or COLORS["text"])
    _rgba(item, "bgcolor", background or (0.0, 0.0, 0.0, 0.0))
    return item


def _dynamic_text(parent_op, name, expression, *args, **kwargs):
    item = _text(parent_op, name, "-", *args, **kwargs)
    _expr(item, "text", expression)
    return item


def _port_value(operator):
    for name in ("port", "localport", "networkport"):
        parameter = _par(operator, name)
        if parameter is None:
            continue
        try:
            return int(parameter.eval())
        except Exception:
            continue
    return None


def _looks_like_osc_in(operator):
    descriptions = (
        str(getattr(operator, "type", "")),
        str(getattr(operator, "OPType", "")),
        str(getattr(operator, "name", "")),
    )
    description = " ".join(descriptions).lower()
    return "osc" in description and "in" in description


def _contains_monitor_messages(operator):
    """Return True when a DAT already contains our monitor OSC addresses."""
    try:
        row_count = int(operator.numRows)
        column_count = int(operator.numCols)
    except Exception:
        return False
    if row_count <= 0 or column_count <= 0:
        return False

    # Search only the FIFO tail. OSC In DAT normally keeps at most 100 rows.
    first_row = max(0, row_count - 128)
    try:
        for row in range(first_row, row_count):
            for column in range(column_count):
                if OSC_PREFIX in operator[row, column].val:
                    return True
    except Exception:
        return False
    return False


def _find_osc_in(owner):
    if OSC_IN_DAT_PATH:
        result = op(OSC_IN_DAT_PATH)
        if result is None:
            raise RuntimeError(
                "OSC_IN_DAT_PATH was not found: {}".format(OSC_IN_DAT_PATH)
            )
        return result

    candidates = []
    try:
        candidates.append(owner)
        candidates.extend(owner.findChildren(maxDepth=8))
    except Exception:
        pass
    try:
        candidates.extend(op("/").findChildren(maxDepth=16))
    except Exception:
        pass

    seen = set()
    matching_type = []
    matching_port = []
    for candidate in candidates:
        path = getattr(candidate, "path", "")
        if not path or path in seen:
            continue
        seen.add(path)

        # The received monitor address is the strongest possible match and is
        # independent of operator naming and TouchDesigner build differences.
        if _contains_monitor_messages(candidate):
            return candidate

        if not _looks_like_osc_in(candidate):
            continue
        matching_type.append(candidate)
        if _port_value(candidate) == OSC_MONITOR_PORT:
            matching_port.append(candidate)

    if len(matching_port) == 1:
        return matching_port[0]

    if len(matching_type) == 1:
        return matching_type[0]

    detected = ", ".join(
        "{} (port {})".format(
            getattr(candidate, "path", "?"),
            _port_value(candidate),
        )
        for candidate in matching_type
    ) or "none"
    raise RuntimeError(
        "OSC In DAT on port {} could not be selected automatically. "
        "Detected OSC-like operators: {}. Set OSC_IN_DAT_PATH at the top "
        "of build_monitor_ui.py.".format(
            OSC_MONITOR_PORT,
            detected,
        )
    )


def _model_code(osc_path):
    # This code runs as the module of the generated _monitor_model Text DAT.
    return '''import ast

SOURCE_PATH = {source_path!r}
PREFIX = {prefix!r}


def _source():
    return op(SOURCE_PATH)


def _unquote(value):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return str(ast.literal_eval(value))
        except Exception:
            return value[1:-1]
    return value


def raw(key, default='-'):
    dat = _source()
    if dat is None:
        return default
    address = PREFIX + key

    # Search newest to oldest. This works with the current one-column
    # "address value" rows and also with split address/value columns.
    for row in range(dat.numRows - 1, -1, -1):
        cells = [dat[row, col].val.strip() for col in range(dat.numCols)]
        for index, cell in enumerate(cells):
            if cell == address:
                for candidate in cells[index + 1:]:
                    if candidate and not candidate.startswith(','):
                        return _unquote(candidate)
            elif cell.startswith(address + ' '):
                return _unquote(cell[len(address):].strip())
    return default


def _float(key, default=None):
    value = raw(key, '')
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def number(key, digits=1, suffix='', zero_as_dash=False):
    value = _float(key)
    if value is None or (zero_as_dash and value <= 0.0):
        return '-'
    return '{{:.{{}}f}}'.format(value, int(digits)) + suffix


def ratio_percent(key, digits=0):
    value = _float(key)
    if value is None:
        return '-'
    return '{{:.{{}}f}} %'.format(value * 100.0, int(digits))


def vram():
    used = _float('vram_used_mb')
    total = _float('vram_total_mb')
    if used is None or total is None or total <= 0.0:
        return '-'
    return '{{:.1f}}/{{:.1f}} GB'.format(used / 1024.0, total / 1024.0)


def is_running():
    value = _float('running', 0.0)
    return bool(value and value > 0.5)


def normalized(key, maximum=1.0):
    value = _float(key, 0.0) or 0.0
    maximum = max(float(maximum), 1e-9)
    return max(0.0, min(1.0, value / maximum))


def vram_ratio():
    used = _float('vram_used_mb', 0.0) or 0.0
    total = _float('vram_total_mb', 0.0) or 0.0
    return max(0.0, min(1.0, used / total)) if total > 0.0 else 0.0
'''.format(source_path=osc_path, prefix=OSC_PREFIX)


def _refresh_dependency(expression):
    # absTime keeps the panel values fresh even when the OSC In DAT's FIFO
    # table reuses rows rather than changing its shape.
    return "({}) + ('' if absTime.frame >= 0 else '')".format(expression)


def _value_expression(model_path, key, formatter):
    module = "op({!r}).module".format(model_path)
    if formatter == "raw":
        expression = "{}.raw({!r})".format(module, key)
    elif formatter == "source_fps":
        expression = "{}.number({!r}, 1, '', True)".format(module, key)
    elif formatter == "milliseconds":
        expression = "{}.number({!r}, 1, ' ms')".format(module, key)
    elif formatter == "percent_value":
        expression = "{}.number({!r}, 0, ' %')".format(module, key)
    elif formatter == "temperature":
        expression = "{}.number({!r}, 0, ' °C', True)".format(module, key)
    elif formatter == "ratio_percent_1":
        expression = "{}.ratio_percent({!r}, 1)".format(module, key)
    elif formatter == "ratio_percent_0":
        expression = "{}.ratio_percent({!r}, 0)".format(module, key)
    elif formatter == "vram":
        expression = "{}.vram()".format(module)
    else:
        expression = "{}.number({!r}, 1)".format(module, key)
    return _refresh_dependency(expression)


def _metric_card(parent_op, index, spec, model_path, x, y, width, height):
    key, label, formatter, accent_name, bar_maximum = spec
    accent = COLORS[accent_name]
    card = _container(
        parent_op,
        "metric_{:02d}_{}".format(index + 1, key),
        x,
        y,
        width,
        height,
        COLORS["card"] if index % 2 == 0 else COLORS["card_alt"],
    )
    _container(card, "accent", 0, 0, 4, height, accent)
    _text(
        card,
        "label",
        label,
        16,
        height - 33,
        width - 30,
        20,
        size=10,
        color=COLORS["muted"],
    )
    _dynamic_text(
        card,
        "value",
        _value_expression(model_path, key, formatter),
        16,
        23 if bar_maximum is not None else 15,
        width - 30,
        38,
        size=18,
        color=COLORS["text"],
    )

    if bar_maximum is not None:
        _container(card, "bar_track", 16, 12, width - 32, 4, COLORS["line"])
        fill = _container(card, "bar_value", 16, 12, 0, 4, accent)
        module = "op({!r}).module".format(model_path)
        if bar_maximum == "vram":
            ratio = "{}.vram_ratio()".format(module)
        else:
            ratio = "{}.normalized({!r}, {})".format(
                module,
                key,
                float(bar_maximum),
            )
        _expr(
            fill,
            "w",
            "max(0, (parent().width - 32) * ({})) + (0 * absTime.frame)".format(
                ratio
            ),
        )
    return card


def build():
    owner = op(SOURCE_CONTAINER_PATH) if SOURCE_CONTAINER_PATH else parent()
    if owner is None:
        raise RuntimeError("The monitor owner Container could not be found.")
    osc_in = _find_osc_in(owner)

    old_root = owner.op(UI_ROOT_NAME)
    if old_root is not None:
        old_root.destroy()

    _set(owner, "hmode", "fixed")
    _set(owner, "vmode", "fixed")
    _set(owner, "w", UI_WIDTH)
    _set(owner, "h", UI_HEIGHT)

    root = _container(
        owner,
        UI_ROOT_NAME,
        0,
        0,
        UI_WIDTH,
        UI_HEIGHT,
        COLORS["canvas"],
    )
    root.nodeX = 0
    root.nodeY = -260

    model = root.create(textDAT, "_monitor_model")
    model.text = _model_code(osc_in.path)
    model.nodeX = -240
    model.nodeY = -120

    header = _container(root, "header", 18, 684, 1330, 66, COLORS["header"])
    _container(header, "brand_line", 0, 0, 5, 66, COLORS["green"])

    dot = _container(header, "status_dot", 20, 25, 14, 14, COLORS["green"])
    running_expression = (
        "((1 if op({!r}).module.is_running() else 0) + (0 * absTime.frame))".format(
            model.path
        )
    )
    _expr(dot, "bgcolorr", "0.315 if ({}) else 0.925".format(running_expression))
    _expr(dot, "bgcolorg", "0.890 if ({}) else 0.330".format(running_expression))
    _expr(dot, "bgcolorb", "0.690 if ({}) else 0.350".format(running_expression))

    _dynamic_text(
        header,
        "state",
        _refresh_dependency("op({!r}).module.raw('state')".format(model.path)),
        46,
        31,
        190,
        24,
        size=17,
        color=COLORS["text"],
    )
    _dynamic_text(
        header,
        "status",
        _refresh_dependency("op({!r}).module.raw('status')".format(model.path)),
        46,
        10,
        420,
        20,
        size=10,
        color=COLORS["muted"],
    )
    _text(
        header,
        "title",
        "LOCAL STABLE DIFFUSION / MONITOR",
        525,
        24,
        500,
        28,
        size=18,
        color=COLORS["text"],
        align="center",
    )
    _text(
        header,
        "connection",
        "OSC  127.0.0.1 : 9001",
        1090,
        17,
        218,
        32,
        size=11,
        color=COLORS["green"],
        background=(0.070, 0.160, 0.145, 1.0),
        align="center",
    )

    columns = 4
    card_width = 326
    card_height = 101
    gap_x = 8
    gap_y = 8
    start_x = 18
    top_y = 668

    for index, spec in enumerate(METRICS):
        column = index % columns
        row = index // columns
        x = start_x + column * (card_width + gap_x)
        y = top_y - (row + 1) * card_height - row * gap_y
        _metric_card(
            root,
            index,
            spec,
            model.path,
            x,
            y,
            card_width,
            card_height,
        )

    # Fill the final empty cell so the 4-column grid remains visually balanced.
    empty = _container(root, "footer_info", 1020, 18, 328, 101, COLORS["card_alt"])
    _text(
        empty,
        "label",
        "MONITOR SOURCE",
        16,
        66,
        294,
        20,
        size=10,
        color=COLORS["muted"],
    )
    _text(
        empty,
        "value",
        osc_in.path,
        16,
        19,
        294,
        44,
        size=10,
        color=COLORS["blue"],
    )

    _set(owner, "opviewer", root.path)
    try:
        root.viewer = True
    except Exception:
        pass

    print(
        "[streamdiffusion monitor UI] Built {} from {}".format(
            root.path,
            osc_in.path,
        )
    )
    return root


build()
