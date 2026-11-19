"""
Orientation helpers

The stats and geoms do their work in "horizontal" coordinates: values
(and the slab function's domain) along x, groups along y. Vertical layers
are flipped on the way in and flipped back on the way out.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from plotnine.exceptions import PlotnineError

HORIZONTAL = "horizontal"
VERTICAL = "vertical"

_ALIASES = {
    "horizontal": HORIZONTAL,
    "y": HORIZONTAL,
    "vertical": VERTICAL,
    "x": VERTICAL,
}

_SWAP = {
    "x": "y",
    "xmin": "ymin",
    "xmax": "ymax",
    "xend": "yend",
    "xdist": "ydist",
    "xintercept": "yintercept",
}
_SWAP |= {v: k for k, v in _SWAP.items()}


def normalize_orientation(value: Any) -> str | None:
    """
    Resolve "horizontal"/"y" and "vertical"/"x"; None means detect
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return _ALIASES[str(value).lower()]
    except KeyError as e:
        raise PlotnineError(
            f"orientation must be 'horizontal' (or 'y'), 'vertical' (or 'x') "
            f"or None, not {value!r}"
        ) from e


def flip(data: pd.DataFrame, orientation: str) -> pd.DataFrame:
    """
    Swap x/y columns when `orientation` is vertical (an involution)
    """
    if orientation != VERTICAL:
        return data
    return data.rename(columns={c: _SWAP[c] for c in data.columns if c in _SWAP})


def flip_scales(scales: Any, orientation: str) -> tuple[Any, Any]:
    """
    Return (value_scale, position_scale) for an orientation
    """
    if orientation == VERTICAL:
        return scales.y, scales.x
    return scales.x, scales.y
