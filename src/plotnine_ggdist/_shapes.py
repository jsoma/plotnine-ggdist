"""
Point shapes: matplotlib marker names, or R's numeric ``pch`` codes

R's codes 0-25 are translated to the nearest matplotlib marker, with R's
colouring rules: 0-14 are outlines in the colour, 15-20 are solid in the
colour, and 21-25 are filled with the fill colour and outlined in the
colour.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = ("resolve_shape",)

Mode = Literal["fill", "solid", "open", "line"]

# pch -> (matplotlib marker, colouring, relative size)
_PCH: dict[int, tuple[Any, Mode, float]] = {
    0: ("s", "open", 1.0),
    1: ("o", "open", 1.0),
    2: ("^", "open", 1.0),
    3: ("+", "line", 1.0),
    4: ("x", "line", 1.0),
    5: ("D", "open", 1.0),
    6: ("v", "open", 1.0),
    7: ("s", "open", 1.0),  # square with x: no exact matplotlib marker
    8: ((6, 2, 0), "line", 1.0),  # asterisk
    9: ("D", "open", 1.0),  # diamond with +
    10: ("o", "open", 1.0),  # circle with +
    11: ((6, 1, 0), "line", 1.0),  # two triangles: six-pointed star
    12: ("s", "open", 1.0),  # square with +
    13: ("o", "open", 1.0),  # circle with x
    14: ("s", "open", 1.0),  # square with triangle
    15: ("s", "solid", 1.0),
    16: ("o", "solid", 1.0),
    17: ("^", "solid", 1.0),
    18: ("d", "solid", 1.0),
    19: ("o", "solid", 1.0),
    20: ("o", "solid", 2 / 3),
    21: ("o", "fill", 1.0),
    22: ("s", "fill", 1.0),
    23: ("D", "fill", 1.0),
    24: ("^", "fill", 1.0),
    25: ("v", "fill", 1.0),
}


def resolve_shape(shape: Any) -> tuple[Any, Mode, float]:
    """
    (matplotlib marker, colouring mode, relative size) for a shape

    Numbers (including floats such as 21.0) are R ``pch`` codes; anything
    else is a matplotlib marker, filled with the fill colour and outlined
    in the colour.
    """
    if isinstance(shape, (bool, np.bool_)):
        raise ValueError(f"Unknown shape {shape!r}")
    if isinstance(shape, (int, float, np.integer, np.floating)) and not isinstance(shape, bool):
        code = int(shape)
        if code != shape or code not in _PCH:
            raise ValueError(f"Numeric shapes are R pch codes 0-25, not {shape!r}")
        return _PCH[code]
    if isinstance(shape, str) and shape.isdigit():
        return resolve_shape(int(shape))
    return shape, "fill", 1.0


def colours_for(mode: Mode, fill: Any, colour: Any) -> tuple[Any, Any]:
    """
    (face colour, edge colour) for a colouring mode
    """
    if mode == "fill":
        return fill, colour
    if mode == "solid":
        return colour, colour
    return (0.0, 0.0, 0.0, 0.0), colour
