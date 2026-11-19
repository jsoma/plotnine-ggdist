"""
stat_lineribbon and stat_ribbon

For each value of ``x``, summarise the samples (or distribution) of ``y``
into a point and intervals: a line through the points and one ribbon per
interval width.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from plotnine.exceptions import PlotnineError
from plotnine.mapping.evaluation import after_stat
from plotnine.stats.stat import stat

from ._core.dist import is_dist
from ._core.intervals import parse_point_interval, summarise
from ._orientation import HORIZONTAL, VERTICAL, flip, normalize_orientation
from ._utils import compute_panel_by_group
from .stat_slabinterval import as_level, width_levels

if TYPE_CHECKING:
    from plotnine.facets.layout import Layout
    from plotnine.iapi import pos_scales

__all__ = ("stat_lineribbon", "stat_ribbon")


class stat_lineribbon(stat):
    """
    Line through point summaries with ribbons for intervals

    For each ``x``, summarises the draws of ``y`` (or the distribution in
    ``ydist``) with `point_interval`. Port of `ggdist::stat_lineribbon()`.

    Parameters
    ----------
    mapping, data, geom, position, na_rm, inherit_aes, show_legend :
        As for any plotnine stat.
    point_interval :
        As for [](`~plotnine_ggdist.stat_slabinterval`). Default
        ``"median_qi"``.
    width :
        Interval widths. Default ``(0.5, 0.8, 0.95)``.
    orientation :
        ``"vertical"`` (default unless ``xdist`` is mapped): ribbons span
        y at each x. ``"horizontal"``: ribbons span x at each y.

    Computed variables
    ------------------
    x, y :
        Position and point summary.
    ymin, ymax :
        Interval ends.
    .width, level :
        Interval width, numeric and as an ordered categorical (widest
        first).
    """

    REQUIRED_AES: set[str] = set()
    DEFAULT_AES = {"fill": after_stat("level"), "weight": None}
    CREATES = {"ymin", "ymax", ".width", "level", "orientation"}
    DEFAULT_PARAMS = {
        "geom": "lineribbon",
        "position": "identity",
        "na_rm": False,
        "orientation": None,
        "point_interval": "median_qi",
        "width": (0.5, 0.8, 0.95),
        # geom parameter, copied to the geom
        "show_line": True,
    }

    def compute_layer(self, data: pd.DataFrame, layout: Layout) -> pd.DataFrame:
        orientation = normalize_orientation(self.params.get("orientation"))
        if orientation is None:
            orientation = HORIZONTAL if "xdist" in data.columns else VERTICAL
        # work with values along y (vertical); flip horizontal layers
        flipped = orientation == HORIZONTAL
        data = flip(data, VERTICAL) if flipped else data
        if "y" not in data.columns and "ydist" not in data.columns:
            raise PlotnineError("stat_lineribbon needs y (samples) or ydist (distributions)")
        if "x" not in data.columns:
            raise PlotnineError("stat_lineribbon needs a position aesthetic (x for vertical ribbons)")
        result = super().compute_layer(data, layout)
        if len(result):
            result["orientation"] = orientation
        return flip(result, VERTICAL) if flipped else result

    def compute_panel(self, data: pd.DataFrame, scales: pos_scales) -> pd.DataFrame:
        return compute_panel_by_group(data, scales, self.compute_group)

    def compute_group(self, data: pd.DataFrame, scales: pos_scales) -> pd.DataFrame:
        point_fn, interval_fn, _, _ = parse_point_interval(self.params["point_interval"])
        widths = self.params["width"]
        levels = width_levels(widths)
        pieces = []
        if "ydist" in data.columns:
            for _, row in data.iterrows():
                d = row["ydist"]
                if not is_dist(d):
                    continue
                s = summarise(d, widths, point_fn, interval_fn)
                s["x"] = row["x"]
                pieces.append(s)
        else:
            has_w = "weight" in data.columns and data["weight"].notna().any()
            for x, sub in data.groupby("x", sort=True):
                w = sub["weight"].to_numpy(dtype=float) if has_w else None
                s = summarise(sub["y"].to_numpy(dtype=float), widths, point_fn, interval_fn, weights=w)
                s["x"] = x
                pieces.append(s)
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True)[["x", "y", "ymin", "ymax", ".width"]]
        out["level"] = as_level(out[".width"], levels)
        return out


class stat_ribbon(stat_lineribbon):
    """
    Ribbons for intervals, without the line

    Port of `ggdist::stat_ribbon()`. See
    [](`~plotnine_ggdist.stat_lineribbon`).
    """

    DEFAULT_PARAMS = stat_lineribbon.DEFAULT_PARAMS | {"show_line": False}
