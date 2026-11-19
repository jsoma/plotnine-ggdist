"""
unit(): sizes in physical units, for dot plot bin widths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ("unit",)

_PER_POINT = {
    "pt": 1.0,
    "points": 1.0,
    "mm": 72.27 / 25.4,
    "cm": 72.27 / 2.54,
    "in": 72.27,
    "inches": 72.27,
}


@dataclass(frozen=True)
class unit:
    """
    A size in physical units or as a fraction of the panel

    Use for the `binwidth` of the dots geoms, e.g. ``unit(1.5, "mm")`` or
    ``unit((0, 0.1), "npc")`` (at most 10% of the panel along the value
    axis). Like R's ``grid::unit()``.

    Parameters
    ----------
    value :
        A number, or a pair (min, max).
    units :
        ``"mm"``, ``"cm"``, ``"in"``, ``"pt"`` or ``"npc"`` (fraction of the
        panel's extent along the value axis).
    """

    value: Any
    units: str = "mm"

    def __post_init__(self):
        if self.units not in _PER_POINT and self.units != "npc":
            raise ValueError(f"Unknown unit {self.units!r}; use one of {sorted(_PER_POINT)} or 'npc'")

    def to_pixels(self, dpi: float, axis_px: float) -> np.ndarray:
        """
        The value(s) in pixels
        """
        v = np.asarray(self.value, dtype=float)
        if self.units == "npc":
            return v * axis_px
        # grid's points are 1/72.27 in
        return v * _PER_POINT[self.units] / 72.27 * dpi
