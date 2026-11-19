"""
geom_slabinterval and the geoms built on it

A composite geom with three sub-geometries: a *slab* (a function, such as
a density, drawn as a filled shape whose thickness varies along the value
axis), *intervals* (lines), and a *point* summary. Each has its own
aesthetics (``slab_fill``, ``interval_color``, ``point_size``, ...) that
fall back to the shared ones (``fill``, ``color``, ``alpha``, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from plotnine._utils import SIZE_FACTOR, resolution
from plotnine.exceptions import PlotnineError
from plotnine.geoms.geom import geom

from ._utils import width_aware_layer
from ._orientation import HORIZONTAL, VERTICAL, flip, normalize_orientation
from ._shapes import colours_for, resolve_shape

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import DrawingArea
    from plotnine.iapi import panel_view
    from plotnine.layer import layer

__all__ = (
    "geom_slabinterval",
    "geom_slab",
    "geom_pointinterval",
    "geom_interval",
)

# ggplot2 line widths are nominally mm, but ggplot2 converts them with
# .pt = 72.27 / 25.4 into R's lwd, and R draws lwd in 1/96 inch, so a
# ggplot2 linewidth of 1 is drawn 72.27 / 25.4 * 72 / 96 = 2.13 points wide.
# plotnine line widths are `value * sqrt(pi)` points. This factor converts a
# ggdist default into plotnine units that draw at the same physical width
# (checked against ggdist's rendered reference images).
_MM_TO_PLOTNINE = (72.27 / 25.4) * (72 / 96) / SIZE_FACTOR
# ggdist's interval_size_range = c(0.6, 1.4)
DEFAULT_INTERVAL_SIZE_RANGE = (0.6 * _MM_TO_PLOTNINE, 1.4 * _MM_TO_PLOTNINE)
# Point size = thickest interval * fatten_point * this, so the default point
# (fatten_point = 1.8) has ggdist's overall diameter (about 7 pt, measured from
# its reference images; plotnine's point outline adds to the marker size)
_POINT_SCALE = 0.98

_TOP = {"top", "right", "topright"}
_BOTTOM = {"bottom", "left", "bottomleft"}
_SIDES = _TOP | _BOTTOM | {"both", "topleft", "bottomright"}


def _missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return False


def _col(data: pd.DataFrame, name: str, fallback: str | None = None, default: Any = None) -> pd.Series:
    """
    A sub-geometry aesthetic, falling back to a shared one where missing
    """
    if name in data.columns:
        s = data[name]
        if fallback is not None and fallback in data.columns:
            s = s.where(~s.map(_missing), data[fallback])
        elif default is not None:
            s = s.where(~s.map(_missing), default)
        return s
    if fallback is not None and fallback in data.columns:
        return data[fallback]
    return pd.Series([default] * len(data), index=data.index)


def _rgba(color: Any, alpha: Any) -> tuple[float, float, float, float]:
    if _missing(color) or (isinstance(color, str) and color.lower() == "none"):
        return (0.0, 0.0, 0.0, 0.0)
    a = 1.0 if _missing(alpha) else float(alpha)
    r, g, b, a0 = to_rgba(color)
    return (r, g, b, a0 * a)


_LINETYPES = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdot": "dashdot",
    "longdash": (0, (8, 4)),
    "twodash": (0, (4, 2, 8, 2)),
    "blank": "None",
}


def _linestyle(lt: Any) -> Any:
    if _missing(lt):
        return "solid"
    return _LINETYPES.get(lt, lt) if isinstance(lt, str) else lt


def _resolve_side(side: Any, orientation: str) -> str:
    side = "topright" if _missing(side) else str(side)
    if side not in _SIDES:
        raise PlotnineError(f"Unknown side {side!r}; use one of {sorted(_SIDES)}")
    if side == "topleft":
        return "top" if orientation == HORIZONTAL else "bottom"
    if side == "bottomright":
        return "bottom" if orientation == HORIZONTAL else "top"
    if side in _TOP:
        return "top"
    if side in _BOTTOM:
        return "bottom"
    return "both"


def resolve_sides(data: pd.DataFrame, orientation: str) -> pd.DataFrame:
    """
    Resolve `side` to "top", "bottom" or "both"

    Needed at draw time too: plotnine re-applies set aesthetics (e.g.
    ``side="left"``) after `setup_data`.
    """
    if "side" not in data.columns:
        return data
    return data.assign(side=[_resolve_side(v, orientation) for v in data["side"]])


def detect_orientation(data: pd.DataFrame, param: Any, hint: str | None = None) -> str:
    """
    Orientation from (in order): the parameter, the stat's `orientation`
    column, which of xmin/xmax or ymin/ymax are present, which position
    scale is discrete (`hint`); otherwise horizontal
    """
    o = normalize_orientation(param)
    if o is not None:
        return o
    if "orientation" in data.columns:
        o = normalize_orientation(data["orientation"].dropna().iloc[0]) if data["orientation"].notna().any() else None
        if o is not None:
            return o
    cols = set(data.columns)
    if {"xmin", "xmax"} & cols and not ({"ymin", "ymax"} & cols):
        return HORIZONTAL
    if {"ymin", "ymax"} & cols and not ({"xmin", "xmax"} & cols):
        return VERTICAL
    return hint or HORIZONTAL


class geom_slabinterval(geom):
    """
    Slab + point + interval meta-geom

    The geom behind [](`~plotnine_ggdist.stat_halfeye`) and friends. Use it
    directly with precomputed data: rows with ``datatype="slab"`` give the
    slab function (``x`` and ``thickness``, for a horizontal layer) and rows
    with ``datatype="interval"`` give the intervals (``x``, ``xmin``,
    ``xmax``). Port of `ggdist::geom_slabinterval()`.

    Parameters
    ----------
    mapping, data, stat, position, na_rm, inherit_aes, show_legend :
        As for any plotnine geom.
    orientation :
        ``"horizontal"``/``"y"``, ``"vertical"``/``"x"`` or ``None`` (from
        the stat, or detected from which of ``xmin``/``ymin`` are present).
    normalize :
        How to scale ``thickness`` so its maximum is 1: ``"all"`` (default,
        across the layer), ``"panels"``, ``"xy"`` (per position on the group
        axis), ``"groups"`` (per group) or ``"none"``.
    fill_type :
        How to fill a slab whose fill or alpha varies along it:
        ``"segments"`` (default; one polygon per run of equal colours) or
        ``"gradient"`` (a smooth image clipped to the slab).
    interval_size_range :
        Line widths (plotnine units) of the widest and narrowest intervals
        when ``linewidth`` is not set. The default draws them as wide as
        ggdist's (its ``interval_size_range = c(0.6, 1.4)``).
    fatten_point :
        Point size relative to the thickest interval, when ``point_size``
        is not set. Default 1.8, which draws points the size of ggdist's.
    show_slab, show_point, show_interval :
        Which sub-geometries to draw.

    Aesthetics
    ----------
    **x**, **y**, xmin, xmax, ymin, ymax, thickness, datatype, alpha, color,
    fill, linetype, linewidth (interval line width), shape, size (point
    size), stroke, group;
    side (``"top"``/``"right"``/``"topright"``, ``"bottom"``/``"left"``/
    ``"bottomleft"``, ``"topleft"``, ``"bottomright"``, ``"both"``),
    scale (fraction of the space between groups the slab may use,
    default 0.9), height (that space, in data units; default the
    resolution of the group axis, i.e. 1 for discrete axes), justification (where the interval sits within the slab's
    space, 0-1; default from `side`);
    slab_color, slab_fill, slab_alpha, slab_linewidth, slab_linetype,
    interval_color, interval_alpha, interval_linetype,
    point_color, point_fill, point_alpha, point_size.
    """

    DEFAULT_AES = {
        "alpha": 1,
        "color": "black",
        "fill": "#A6A6A6",  # R's gray65, ggdist's default slab fill
        "linetype": "solid",
        "linewidth": None,
        "shape": "o",
        "size": None,
        "stroke": 0.5,
        "thickness": np.nan,
        "side": "topright",
        "scale": 0.9,
        "height": None,
        "justification": None,
        "datatype": None,
        "slab_color": "none",
        "slab_fill": None,
        "slab_alpha": None,
        "slab_linewidth": 1.0 * _MM_TO_PLOTNINE,  # ggdist's slab outline: linewidth 1
        "slab_linetype": None,
        "interval_color": None,
        "interval_alpha": None,
        "interval_linetype": None,
        "point_color": None,
        "point_fill": None,
        "point_alpha": None,
        "point_size": None,
    }
    REQUIRED_AES = {"x", "y"}
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "orientation": None,
        "normalize": "all",
        "fill_type": "segments",
        "interval_size_range": DEFAULT_INTERVAL_SIZE_RANGE,
        "fatten_point": 1.8,
        "show_slab": True,
        "show_point": True,
        "show_interval": True,
    }
    legend_key_size = staticmethod(lambda data, min_size, lyr: min_size)

    def to_layer(self):
        # carry an unmapped `.width` column through, as ggdist does
        return width_aware_layer(super().to_layer())
    # use DEFAULT_AES["linewidth"] for intervals instead of sizing by .width
    _FIXED_LINEWIDTH = False

    # Data preparation -----------------------------------------------------

    def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:
        # slab rows have no xmin/xmax, interval rows no thickness: only
        # rows without a position are dropped
        keep = data["x"].notna() & data["y"].notna() if {"x", "y"} <= set(data.columns) else slice(None)
        return data[keep].reset_index(drop=True)

    def _aes(self, data: pd.DataFrame, name: str) -> pd.Series:
        """
        An aesthetic's values at setup time

        `setup_data` runs before plotnine fills in set (``side="both"``) and
        default aesthetics, so look them up here: mapped column, then set
        value, then default.
        """
        if name in data.columns:
            return data[name]
        if name in self.aes_params:
            return pd.Series([self.aes_params[name]] * len(data), index=data.index, dtype=object)
        return pd.Series([self.DEFAULT_AES.get(name)] * len(data), index=data.index, dtype=object)

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        params = self.params
        orientation = detect_orientation(data, params.get("orientation"), getattr(self, "_scale_orientation", None))
        data = flip(data.copy(), orientation)
        data["orientation"] = orientation

        thickness = pd.to_numeric(self._aes(data, "thickness"), errors="coerce")
        if "datatype" not in data.columns or data["datatype"].isna().all():
            data["datatype"] = np.where(thickness.notna(), "slab", "interval")
        is_slab = (data["datatype"] == "slab").to_numpy()
        is_iv = ~is_slab

        # thickness normalisation
        if is_slab.any():
            data["thickness"] = thickness.astype(float)
            data.loc[is_slab, "thickness"] = self._normalize(data.loc[is_slab], params["normalize"])

        # space allotted to each slab along the group axis
        res = resolution(data["y"].to_numpy(dtype=float), zero=False) if len(data) else 1.0
        height = pd.to_numeric(self._aes(data, "height"), errors="coerce").fillna(res)
        data["side"] = [_resolve_side(v, orientation) for v in self._aes(data, "side")]
        default_just = data["side"].map({"top": 0.0, "bottom": 1.0, "both": 0.5})
        data["justification"] = [
            d if _missing(j) else float(j)
            for j, d in zip(self._aes(data, "justification"), default_just, strict=True)
        ]
        data["scale"] = pd.to_numeric(self._aes(data, "scale"), errors="coerce").fillna(0.9)
        if is_slab.any() and params["show_slab"]:
            span = data["scale"] * height
            data["ymin"] = data["y"] - data["justification"] * span
            data["ymax"] = data["y"] + (1 - data["justification"]) * span

        # interval line widths and point sizes
        if is_iv.any():
            data = self._interval_sizes(data, is_iv)

        return flip(data, orientation)

    @staticmethod
    def _normalize(slab: pd.DataFrame, how: str) -> pd.Series:
        t = slab["thickness"].astype(float)
        if how == "none":
            return t
        if how == "all":
            keys: list[str] = []
        elif how == "panels":
            keys = ["PANEL"]
        elif how == "xy":
            keys = ["PANEL", "y"]
        elif how == "groups":
            keys = ["PANEL", "y", "group"]
        else:
            raise PlotnineError(
                f"normalize must be 'all', 'panels', 'xy', 'groups' or 'none', not {how!r}"
            )
        finite = t.where(np.isfinite(t))
        if keys:
            keys = [k for k in keys if k in slab.columns]
            mx = finite.groupby([slab[k] for k in keys]).transform("max")
        else:
            mx = pd.Series(finite.max(), index=t.index)
        mx = mx.where(mx > 0, 1.0)
        out = t / mx
        # infinite spikes (a single repeated value) become full height
        return out.where(np.isfinite(out), 1.0)

    def _interval_sizes(self, data: pd.DataFrame, is_iv: np.ndarray) -> pd.DataFrame:
        lo, hi = self.params["interval_size_range"]
        lw = data["linewidth"] if "linewidth" in data.columns else self._aes(data, "linewidth")
        if "linewidth" not in data.columns and "linewidth" not in self.aes_params and not self._FIXED_LINEWIDTH:
            lw = pd.Series(np.nan, index=data.index)
        lw = pd.to_numeric(lw, errors="coerce")
        if lw[is_iv].isna().all():
            if ".width" in data.columns and data.loc[is_iv, ".width"].notna().any():
                w = data[".width"].astype(float)
                wmin, wmax = w[is_iv].min(), w[is_iv].max()
                if wmax > wmin:
                    # narrower intervals are thicker
                    lw = hi - (w - wmin) / (wmax - wmin) * (hi - lo)
                else:
                    lw = pd.Series((lo + hi) / 2, index=data.index)
            else:
                lw = pd.Series((lo + hi) / 2, index=data.index)
        data["linewidth"] = lw
        ps = pd.to_numeric(self._aes(data, "point_size"), errors="coerce")
        size = pd.to_numeric(self._aes(data, "size"), errors="coerce")
        ps = ps.where(ps.notna(), size)
        if ps[is_iv].isna().any():
            keys = [k for k in ("PANEL", "group", "x", "y") if k in data.columns]
            thickest = data.loc[is_iv].groupby(keys, dropna=False)["linewidth"].transform("max")
            fattened = pd.Series(np.nan, index=data.index)
            fattened[is_iv] = thickest * self.params["fatten_point"] * _POINT_SCALE
            ps = ps.where(ps.notna(), fattened)
        data["point_size"] = ps
        return data

    # Drawing ----------------------------------------------------------------

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: panel_view,
        coord: Any,
        ax: Axes,
    ):
        params = self.params
        orientation = detect_orientation(data, params.get("orientation"))
        data = resolve_sides(data, orientation)
        h = flip(data, orientation)
        is_slab = (h["datatype"] == "slab").to_numpy()
        zorder = params["zorder"]
        raster = params.get("raster", False)

        # panel range along the value axis, for infinite interval ends
        # (coord_flip swaps the panel's x and y)
        from plotnine.coords import coord_flip

        value_is_y = (orientation == VERTICAL) != isinstance(coord, coord_flip)
        prange = panel_params.y.range if value_is_y else panel_params.x.range

        def to_display(df: pd.DataFrame) -> pd.DataFrame:
            return coord.transform(flip(df, orientation), panel_params)

        if params["show_slab"] and is_slab.any():
            slabs = h[is_slab]
            group_keys = [k for k in ("group", "y") if k in slabs.columns]
            groups = [g for _, g in slabs.groupby(group_keys, sort=False, dropna=False)]
            # when slabs overlap (e.g. ridgelines), draw those further along
            # the side they grow towards first, so nearer ones sit on top
            def order(g: pd.DataFrame) -> float:
                y = float(g["y"].iloc[0])
                return y if g["side"].iloc[0] == "bottom" else -y

            for g in sorted(groups, key=order):
                self._draw_slab(g.sort_values("x", kind="stable"), to_display, ax, zorder, raster)

        iv = h[~is_slab]
        if len(iv) and (params["show_interval"] or params["show_point"]):
            if params["show_interval"] and {"xmin", "xmax"} <= set(iv.columns):
                self._draw_intervals(iv, prange, to_display, ax, zorder, raster)
            if params["show_point"]:
                self._draw_points(iv, to_display, ax, zorder, raster)

    def _draw_slab(self, s: pd.DataFrame, to_display, ax: Axes, zorder: float, raster: bool):
        x = s["x"].to_numpy(dtype=float)
        t = s["thickness"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(t)
        s, x, t = s[ok], x[ok], t[ok]
        if len(s) < 2:
            return
        ymin = s["ymin"].to_numpy(dtype=float)
        ymax = s["ymax"].to_numpy(dtype=float)
        span = ymax - ymin
        side = s["side"].iloc[0]
        if side == "top":
            lo, hi = ymin, ymin + t * span
        elif side == "bottom":
            lo, hi = ymax - t * span, ymax
        else:
            mid = (ymin + ymax) / 2
            lo, hi = mid - t * span / 2, mid + t * span / 2

        fill = _col(s, "slab_fill", "fill")
        alpha = _col(s, "slab_alpha", "alpha")
        colors = [_rgba(f, a) for f, a in zip(fill, alpha, strict=True)]

        def poly(i0: int, i1: int, left_mid: bool, right_mid: bool) -> np.ndarray:
            """
            Polygon over points i0..i1, optionally starting/ending half way
            to the neighbouring point
            """
            xs, los, his = list(x[i0 : i1 + 1]), list(lo[i0 : i1 + 1]), list(hi[i0 : i1 + 1])
            if left_mid:
                xs.insert(0, (x[i0 - 1] + x[i0]) / 2)
                los.insert(0, (lo[i0 - 1] + lo[i0]) / 2)
                his.insert(0, (hi[i0 - 1] + hi[i0]) / 2)
            if right_mid:
                xs.append((x[i1] + x[i1 + 1]) / 2)
                los.append((lo[i1] + lo[i1 + 1]) / 2)
                his.append((hi[i1] + hi[i1 + 1]) / 2)
            px = np.r_[xs, xs[::-1]]
            py = np.r_[his, los[::-1]]
            d = to_display(pd.DataFrame({"x": px, "y": py}))
            return np.column_stack([d["x"], d["y"]])

        uniq = len(set(colors))
        if uniq == 1:
            if colors[0][3] > 0:
                coll = PolyCollection([poly(0, len(x) - 1, False, False)], facecolors=[colors[0]], edgecolors="none", zorder=zorder, rasterized=raster)
                ax.add_collection(coll)
        elif self.params["fill_type"] == "gradient":
            self._draw_gradient(x, lo, hi, colors, poly(0, len(x) - 1, False, False), to_display, ax, zorder)
        else:
            # one polygon per run of equal colours, split half way between
            # points where the colour changes
            polys, cols = [], []
            start = 0
            for i in range(1, len(x) + 1):
                if i == len(x) or colors[i] != colors[start]:
                    polys.append(poly(start, i - 1, start > 0, i < len(x)))
                    cols.append(colors[start])
                    start = i
            keep = [k for k, c in enumerate(cols) if c[3] > 0]
            opaque = all(cols[k][3] == 1 for k in keep)
            # opaque segments: a thin edge in the face colour hides the
            # antialiasing seams between them; translucent ones would show
            # the overlap, so they get no edge
            coll = PolyCollection(
                [polys[k] for k in keep],
                facecolors=[cols[k] for k in keep],
                edgecolors="face" if opaque else "none",
                linewidths=0.3 if opaque else 0,
                zorder=zorder,
                rasterized=raster,
            )
            ax.add_collection(coll)

        # outline along the slab function (not the baseline)
        edge = _col(s, "slab_color", None, "none").iloc[0]
        if not (_missing(edge) or (isinstance(edge, str) and edge.lower() == "none")):
            ea = _col(s, "slab_alpha", "alpha").iloc[0]
            lw = float(_col(s, "slab_linewidth", None, 1.0 * _MM_TO_PLOTNINE).iloc[0]) * SIZE_FACTOR
            ls = _linestyle(_col(s, "slab_linetype", "linetype").iloc[0])
            lines = []
            if side in ("top", "both"):
                d = to_display(pd.DataFrame({"x": x, "y": hi}))
                lines.append(np.column_stack([d["x"], d["y"]]))
            if side in ("bottom", "both"):
                d = to_display(pd.DataFrame({"x": x, "y": lo}))
                lines.append(np.column_stack([d["x"], d["y"]]))
            ax.add_collection(LineCollection(lines, colors=[_rgba(edge, ea)], linewidths=lw, linestyles=ls, zorder=zorder, rasterized=raster))

    def _draw_gradient(self, x, lo, hi, colors, outline, to_display, ax: Axes, zorder: float):
        from matplotlib.patches import Polygon

        # resample the colours onto an even grid along the value axis
        grid = np.linspace(x[0], x[-1], 1024)
        idx = np.clip(np.searchsorted(x, grid), 0, len(x) - 1)
        rgba = np.array(colors)[idx]
        y0, y1 = np.min(lo), np.max(hi)
        corners = to_display(pd.DataFrame({"x": [x[0], x[-1]], "y": [y0, y1]}))
        cx, cy = corners["x"].to_numpy(), corners["y"].to_numpy()
        flipped_axes = self._display_flipped(to_display)
        img = rgba[None, :, :] if not flipped_axes else rgba[:, None, :]
        extent = (cx[0], cx[1], cy[0], cy[1])
        im = ax.imshow(img, extent=extent, aspect="auto", interpolation="bilinear", origin="lower", zorder=zorder)
        patch = Polygon(outline, closed=True, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        im.set_clip_path(patch)

    @staticmethod
    def _display_flipped(to_display) -> bool:
        """
        Whether the value axis ends up vertical on screen
        """
        d = to_display(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}))
        return bool(np.isclose(d["x"].iloc[0], d["x"].iloc[1]))

    def _draw_intervals(self, iv: pd.DataFrame, prange, to_display, ax: Axes, zorder: float, raster: bool):
        iv = iv.copy()
        lo = iv["xmin"].astype(float).replace([-np.inf, np.inf], [prange[0], prange[1]])
        hi = iv["xmax"].astype(float).replace([-np.inf, np.inf], [prange[0], prange[1]])
        iv = iv.assign(_lo=lo, _hi=hi)
        iv = iv[np.isfinite(iv["_lo"]) & np.isfinite(iv["_hi"])]
        # wider intervals underneath narrower (usually thicker) ones
        order = -iv[".width"].astype(float) if ".width" in iv.columns else pd.Series(0.0, index=iv.index)
        iv = iv.assign(_o=order.fillna(0.0)).sort_values(["_o", "linewidth"], kind="stable")
        color = _col(iv, "interval_color", "color")
        alpha = _col(iv, "interval_alpha", "alpha")
        linetype = _col(iv, "interval_linetype", "linetype")
        a = to_display(pd.DataFrame({"x": iv["_lo"].to_numpy(), "y": iv["y"].to_numpy()}))
        b = to_display(pd.DataFrame({"x": iv["_hi"].to_numpy(), "y": iv["y"].to_numpy()}))
        segs = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(a["x"], a["y"], b["x"], b["y"], strict=True)]
        coll = LineCollection(
            segs,
            colors=[_rgba(c, al) for c, al in zip(color, alpha, strict=True)],
            linewidths=iv["linewidth"].astype(float).to_numpy() * SIZE_FACTOR,
            linestyles=[_linestyle(lt) for lt in linetype],
            capstyle="butt",
            zorder=zorder,
            rasterized=raster,
        )
        ax.add_collection(coll)

    def _draw_points(self, iv: pd.DataFrame, to_display, ax: Axes, zorder: float, raster: bool):
        keys = [k for k in ("PANEL", "group", "x", "y") if k in iv.columns]
        pts = iv.sort_values("point_size", kind="stable").drop_duplicates(keys, keep="last")
        pts = pts[np.isfinite(pts["x"].astype(float))]
        if not len(pts):
            return
        d = to_display(pd.DataFrame({"x": pts["x"].to_numpy(), "y": pts["y"].to_numpy()}))
        color = _col(pts, "point_color", "color")
        alpha = _col(pts, "point_alpha", "alpha")
        fill = _col(pts, "point_fill", "point_color")
        fill = fill.where(~fill.map(_missing), color)
        # R's pch 21-25 use the fill aesthetic (falling back to the slab fill)
        pch_fill = _col(pts, "point_fill", "fill")
        size = pts["point_size"].astype(float).to_numpy()
        stroke = pts["stroke"].astype(float).to_numpy() if "stroke" in pts.columns else np.full(len(pts), 0.5)
        shapes = pts["shape"].to_numpy()
        keys = [str(sh) for sh in shapes]
        for key in dict.fromkeys(keys):
            i = np.flatnonzero(np.asarray(keys) == key)
            marker, mode, rel = resolve_shape(shapes[i[0]])
            numeric = marker is not shapes[i[0]]
            faces, edges = [], []
            for j in i:
                f = pch_fill.iloc[j] if numeric else fill.iloc[j]
                face, edge = colours_for(mode, _rgba(f, alpha.iloc[j]), _rgba(color.iloc[j], alpha.iloc[j]))
                faces.append(face)
                edges.append(edge)
            ax.scatter(
                d["x"].to_numpy()[i],
                d["y"].to_numpy()[i],
                s=((size[i] * rel + stroke[i]) ** 2) * np.pi,
                facecolor=faces,
                edgecolor=edges,
                linewidth=stroke[i] * SIZE_FACTOR,
                marker=marker,
                zorder=zorder,
                rasterized=raster,
            )

    # Legend -------------------------------------------------------------------

    @staticmethod
    def _slab_key(get: Any, da: DrawingArea) -> None:
        """
        Legend key for the slab: a filled rectangle
        """
        from matplotlib.patches import Rectangle

        fill = get("slab_fill", "fill", "gray")
        alpha = get("slab_alpha", "alpha", 1)
        da.add_artist(Rectangle((0, 0), da.width, da.height, facecolor=_rgba(fill, alpha), edgecolor="none"))

    @staticmethod
    def draw_legend(data: pd.Series, da: DrawingArea, lyr: layer) -> DrawingArea:
        """
        Slab key under an interval key (line + point)
        """
        from matplotlib.lines import Line2D

        params = lyr.geom.params
        w, h = da.width, da.height
        mapped = set(lyr.mapping.keys()) | {
            k for k, v in lyr.stat.DEFAULT_AES.items() if v is not None and hasattr(v, "__class__") and type(v).__name__ == "after_stat"
        }
        slab_aes = {"fill", "slab_fill", "slab_alpha", "slab_color", "alpha"}
        iv_aes = {"color", "colour", "interval_color", "linewidth", "linetype", "interval_alpha", "alpha", "point_color", "point_fill", "shape", "size", "point_size"}

        def get(name: str, fallback: str | None = None, default: Any = None) -> Any:
            v = data.get(name)
            if _missing(v) and fallback is not None:
                v = data.get(fallback)
            return default if _missing(v) else v

        show_slab = params.get("show_slab", True) and bool(mapped & slab_aes or not params.get("show_interval", True))
        show_iv = params.get("show_interval", True) and bool(mapped & iv_aes or not show_slab)
        show_pt = params.get("show_point", True) and show_iv

        if show_slab:
            lyr.geom._slab_key(get, da)

        lo, hi = params.get("interval_size_range", DEFAULT_INTERVAL_SIZE_RANGE)
        lw = get("linewidth", None, (lo + hi) / 2)
        if show_iv:
            color = get("interval_color", "color", "black")
            alpha = get("interval_alpha", "alpha", 1)
            line = Line2D([0, w], [h / 2, h / 2], color=_rgba(color, alpha), linewidth=min(float(lw) * SIZE_FACTOR, h), linestyle=_linestyle(get("interval_linetype", "linetype")), solid_capstyle="butt")
            da.add_artist(line)
        if show_pt:
            color = get("point_color", "color", "black")
            alpha = get("point_alpha", "alpha", 1)
            shape = get("shape", None, "o")
            marker, mode, rel = resolve_shape(shape)
            fill = get("point_fill", "fill", "gray") if marker is not shape else get("point_fill", "point_color", color)
            face, edge = colours_for(mode, _rgba(fill, alpha), _rgba(color, alpha))
            size = get("point_size", "size", float(lw) * params.get("fatten_point", 1.8) * _POINT_SCALE)
            stroke = get("stroke", None, 0.5)
            pt = Line2D([w / 2], [h / 2], marker=marker, markersize=(float(size) * rel + float(stroke)) * SIZE_FACTOR, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=float(stroke) * SIZE_FACTOR, linestyle="none")
            da.add_artist(pt)
        return da


class geom_slab(geom_slabinterval):
    """
    Slab only

    [](`~plotnine_ggdist.geom_slabinterval`) without the point and
    intervals. Port of `ggdist::geom_slab()`. Note that in ggdist,
    `geom_slab`'s `linewidth`/`color` style the slab outline; here use
    ``slab_linewidth``/``slab_color`` (or ``color``, which ``geom_slab``
    passes to the outline).
    """

    DEFAULT_PARAMS = geom_slabinterval.DEFAULT_PARAMS | {
        "show_point": False,
        "show_interval": False,
    }
    DEFAULT_AES = geom_slabinterval.DEFAULT_AES | {"color": "none"}

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "datatype" not in data.columns or data["datatype"].isna().all():
            data = data.assign(datatype="slab")
        if "thickness" not in data.columns or data["thickness"].isna().all():
            raise PlotnineError("geom_slab needs a thickness aesthetic")
        # color styles the slab outline
        color = self._aes(data, "color")
        sc = self._aes(data, "slab_color")
        use_color = sc.map(lambda v: _missing(v) or (isinstance(v, str) and v == "none"))
        data["slab_color"] = sc.where(~use_color, color)
        return super().setup_data(data)


class geom_pointinterval(geom_slabinterval):
    """
    Point summary with intervals

    [](`~plotnine_ggdist.geom_slabinterval`) without the slab: give
    ``x``, ``xmin``, ``xmax`` (horizontal) or ``y``, ``ymin``, ``ymax``
    (vertical), e.g. from [](`~plotnine_ggdist.median_qi`). Line widths
    follow ``.width`` when the data has it. Port of
    `ggdist::geom_pointinterval()`.
    """

    DEFAULT_PARAMS = geom_slabinterval.DEFAULT_PARAMS | {"show_slab": False}

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().setup_data(data.assign(datatype="interval"))


class geom_interval(geom_slabinterval):
    """
    Multiple intervals, typically coloured by ``level``

    [](`~plotnine_ggdist.geom_slabinterval`) with only the interval lines,
    drawn thick. With a ``.width`` column and no ``color`` mapping, the
    intervals are coloured by ``level`` (``.width``, widest first), as in
    ggdist. Port of `ggdist::geom_interval()`.
    """

    DEFAULT_PARAMS = geom_slabinterval.DEFAULT_PARAMS | {
        "stat": "width_identity_color",
        "show_slab": False,
        "show_point": False,
    }
    DEFAULT_AES = geom_slabinterval.DEFAULT_AES | {"linewidth": 4 * _MM_TO_PLOTNINE}  # ggdist: 4
    _FIXED_LINEWIDTH = True

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().setup_data(data.assign(datatype="interval"))
