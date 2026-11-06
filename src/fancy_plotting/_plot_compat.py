"""Deprecated plotting spellings, isolated from the Matplotlib interface."""
from numbers import Integral
import warnings

import numpy as np

_ALIASES = {"dcol": "color", "dsym": "marker", "dstyle": "linestyle",
            "decol": "ecolor", "d_width": "linewidth", "oplt": "overplot",
            "error_color": "ecolor", "xrange": "xlim", "yrange": "ylim"}
_COLORS = {0: "white", 1: "black", 2: "red", 3: "green", 4: "blue", 5: "cyan",
           6: "magenta", 7: "yellow", 8: "orange", 9: "yellowgreen", 10: "springgreen",
           11: "deepskyblue", 12: "blueviolet", 13: "deeppink", 14: "dimgray", 15: "lightgray"}
_MARKERS = {0: "None", 1: ".", 2: "+", 3: "*", 4: "o", 5: "x", 6: "s", 7: "^", 8: "*"}
_LINES = {0: "none", 1: "-", 2: "--", 3: "-.", 4: ":", 5: (0, (6, 2, 1, 2, 1, 2))}


def _translate_number(value, table, name):
    if isinstance(value, Integral):
        if value not in table:
            raise ValueError(f"unsupported legacy {name} index {value}")
        return table[value]
    if isinstance(value, (list, np.ndarray)) or (isinstance(value, tuple) and all(isinstance(v, Integral) for v in value)):
        return [_translate_number(v, table, name) for v in value]
    return value


def translate_legacy(options):
    result = {}
    for key, value in options.items():
        target = _ALIASES.get(key, key)
        if key in _ALIASES or key == "sum_exp":
            replacement = "combine='exposure' or 'mean_exposure'" if key == "sum_exp" else target
            warnings.warn(f"{key!r} is deprecated; use {replacement}.",
                          DeprecationWarning, stacklevel=4)
        if key in ("dcol", "decol"):
            value = _translate_number(value, _COLORS, key)
        elif key == "dsym":
            value = _translate_number(value, _MARKERS, key)
        elif key == "dstyle":
            value = _translate_number(value, _LINES, key)
        if target in result:
            raise ValueError(f"option {target!r} was supplied twice through aliases")
        result[target] = value
    return result
