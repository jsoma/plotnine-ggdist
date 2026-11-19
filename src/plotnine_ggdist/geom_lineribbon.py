"""
geom_lineribbon: a line with any number of uncertainty ribbons
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from plotnine._utils import SIZE_FACTOR
from plotnine.geoms.geom import geom

from ._utils import width_aware_layer
from ._orientation import HORIZONTAL, VERTICAL, flip, normalize_orientation
from .geom_slabinterval import _MM_TO_PLOTNINE, _linestyle, _missing, _rgba

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import DrawingArea
    from plotnine.iapi import panel_view
    from plotnine.layer import layer

__all__ = ("geom_lineribbon",)


def _orientation(data: pd.DataFrame, param: Any) -> str:
    o = normalize_orientation(param)
    if o is not None:
        return o
    if "orientation" in data.columns and data["orientation"].notna().any():
        return normalize_orientation(data["orientation"].dropna().iloc[0]) or VERTICAL
    cols = set(data.columns)
    if {"xmin", "xmax"} <= cols and not ({"ymin", "ymax"} <= cols):
        return HORIZONTAL
    return VERTICAL


class geom_lineribbon(geom):
    """
    Line plus ribbons

    Draws one ribbon per interval (``ymin``/``ymax``; widest underneath)
    and a line through ``y``. Use with
    [](`~plotnine_ggdist.stat_lineribbon`), or directly with output of
    [](`~plotnine_ggdist.point_interval`) (map ``ymin=".lower"``,
    ``ymax=".upper"``). With a ``.width`` column and no ``fill`` mapping,
    ribbons are filled by ``level`` (``.width``, widest first), as in
    ggdist. Port of `ggdist::geom_lineribbon()`.

    Parameters
    ----------
    orientation :
        ``"vertical"`` (default; ribbons span y) or ``"horizontal"``
        (ribbons span x, using ``xmin``/``xmax``).
    show_line :
        Draw the line. Default `True`.

    Aesthetics
    ----------
    **x**, **y**, **ymin**, **ymax** (or xmin, xmax when horizontal), alpha,
    color (line), fill (ribbons), group, linetype, linewidth (line width),
    .width (orders the ribbons; wider ones are drawn first).
    """

    DEFAULT_AES = {
        "alpha": 1,
        "color": "black",
        "fill": "gray",
        "linetype": "solid",
        "linewidth": 1.25 * _MM_TO_PLOTNINE,
    }
    REQUIRED_AES = {"x", "y"}
    DEFAULT_PARAMS = {
        "stat": "width_identity",
        "position": "identity",
        "na_rm": False,
        "orientation": None,
        "show_line": True,
    }
    legend_key_size = staticmethod(lambda data, min_size, lyr: min_size)

    def to_layer(self):
        # carry an unmapped `.width` column through, as ggdist does
        return width_aware_layer(super().to_layer())

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        orientation = _orientation(data, self.params.get("orientation"))
        return data.assign(orientation=orientation)

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Any, ax: Axes):
        params = self.params
        orientation = _orientation(data, params.get("orientation"))
        # work in vertical coordinates: x along, ribbons over y
        v = flip(data, VERTICAL) if orientation == HORIZONTAL else data
        zorder, raster = params["zorder"], params.get("raster", False)

        def disp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            df = pd.DataFrame({"x": x, "y": y})
            if orientation == HORIZONTAL:
                df = flip(df, VERTICAL)
            d = coord.transform(df, panel_params)
            return np.column_stack([d["x"], d["y"]])

        if {"ymin", "ymax"} <= set(v.columns):
            rib = v[v["ymin"].notna() & v["ymax"].notna()]
            if ".width" in rib.columns:
                rib = rib.assign(_w=rib[".width"].astype(float))
            else:
                rib = rib.assign(_w=-(rib["ymax"] - rib["ymin"]).groupby(rib["group"]).transform("mean"))
            polys, colors = [], []
            # widest ribbons first so narrower ones sit on top
            for (_, _), r in sorted(
                rib.groupby(["group", "_w"], sort=False),
                key=lambda kv: -kv[0][1],
            ):
                r = r.sort_values("x", kind="stable")
                x = r["x"].to_numpy(dtype=float)
                top = disp(x, r["ymax"].to_numpy(dtype=float))
                bot = disp(x[::-1], r["ymin"].to_numpy(dtype=float)[::-1])
                polys.append(np.vstack([top, bot]))
                colors.append(_rgba(r["fill"].iloc[0], r["alpha"].iloc[0]))
            if polys:
                ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none", zorder=zorder, rasterized=raster))

        if params["show_line"]:
            lines, cols, lws, lss = [], [], [], []
            for _, g in v.groupby("group", sort=False):
                g = g.drop_duplicates("x").sort_values("x", kind="stable")
                g = g[g["y"].notna()]
                if len(g) < 2:
                    continue
                lines.append(disp(g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)))
                cols.append(_rgba(g["color"].iloc[0], g["alpha"].iloc[0]))
                lws.append(float(g["linewidth"].iloc[0]) * SIZE_FACTOR)
                lss.append(_linestyle(g["linetype"].iloc[0]))
            if lines:
                ax.add_collection(LineCollection(lines, colors=cols, linewidths=lws, linestyles=lss, zorder=zorder, rasterized=raster))

    @staticmethod
    def draw_legend(data: pd.Series, da: DrawingArea, lyr: layer) -> DrawingArea:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        w, h = da.width, da.height
        fill = data.get("fill")
        if not _missing(fill):
            da.add_artist(Rectangle((0, 0), w, h, facecolor=_rgba(fill, data.get("alpha", 1)), edgecolor="none"))
        # the line goes in the key only if one of its aesthetics is mapped
        line_aes = {"color", "colour", "linetype", "linewidth"}
        if lyr.geom.params.get("show_line", True) and (line_aes & set(lyr.mapping.keys()) or _missing(fill)):
            color = data.get("color")
            lw = data.get("linewidth")
            if not _missing(color) and not _missing(lw):
                da.add_artist(Line2D([0, w], [h / 2, h / 2], color=_rgba(color, data.get("alpha", 1)), linewidth=float(lw) * SIZE_FACTOR, linestyle=_linestyle(data.get("linetype"))))
        return da


