"""
geom_dotsinterval and the dot plot geoms

Dots take the place of the slab in [](`~plotnine_ggdist.geom_slabinterval`).
How big the dots can be depends on the size of the panel, so the layout is
done when the plot is drawn: a matplotlib artist finds the bin width at
which the tallest stack fits the space each group is given, then bins and
stacks the dots (see ``_core/dots.py``).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import IdentityTransform
from plotnine._utils import SIZE_FACTOR
from plotnine.exceptions import PlotnineError

from ._core.dots import LAYOUTS, find_binwidth, layout_dots, layout_height
from ._core.smooth import get_smooth
from ._shapes import colours_for, resolve_shape
from ._orientation import flip
from ._units import unit
from .geom_slabinterval import _MM_TO_PLOTNINE, _col, _rgba, detect_orientation, geom_slabinterval, resolve_sides

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.offsetbox import DrawingArea
    from plotnine.iapi import panel_view

__all__ = ("geom_dotsinterval", "geom_dots", "geom_swarm", "geom_weave", "geom_blur_dots")

_GRAY65 = "#A6A6A6"


class _DotsArtist(Artist):
    """
    Lays out and draws the dots of one panel at draw time
    """

    def __init__(self, groups: list[dict[str, Any]], value_on_x: bool, params: dict[str, Any], zorder: float):
        super().__init__()
        self.groups = groups
        self.value_on_x = value_on_x
        self.params = params
        self.set_zorder(zorder)
        self.binwidth_px: float | None = None
        self._warned = False

    # helpers ---------------------------------------------------------------------

    def _to_px(self, values: np.ndarray, other: float) -> np.ndarray:
        """
        Pixel positions along the value axis
        """
        t = self.axes.transData
        pts = np.column_stack([values, np.full(len(values), other)]) if self.value_on_x else np.column_stack([np.full(len(values), other), values])
        return t.transform(pts)[:, 0 if self.value_on_x else 1]

    def _other_px(self, value: float, others: np.ndarray) -> np.ndarray:
        """
        Pixel positions along the other axis
        """
        t = self.axes.transData
        pts = np.column_stack([np.full(len(others), value), others]) if self.value_on_x else np.column_stack([others, np.full(len(others), value)])
        return t.transform(pts)[:, 1 if self.value_on_x else 0]

    def _axis_px(self) -> float:
        bbox = self.axes.bbox
        return float(bbox.width if self.value_on_x else bbox.height)

    def _binwidth_px(self, bw: Any, dpi: float, ref_value: float) -> np.ndarray:
        if isinstance(bw, unit):
            return np.atleast_1d(bw.to_pixels(dpi, self._axis_px()))
        bw = np.atleast_1d(np.asarray(bw, dtype=float))
        return np.abs(self._to_px(ref_value + bw, 0.0) - self._to_px(np.array([ref_value]), 0.0))

    # drawing -----------------------------------------------------------------------

    def draw(self, renderer: RendererBase) -> None:
        if not self.get_visible() or not self.groups:
            return
        p = self.params
        dpi = self.figure.dpi
        layout, stackratio, overlaps = p["layout"], float(p["stackratio"]), p["overlaps"]

        prepared = []
        for g in self.groups:
            v_px = self._to_px(g["values"], g["region"][0])
            lo_px, hi_px = self._other_px(g["values"][0] if len(g["values"]) else 0.0, np.array(g["region"]))
            prepared.append((g, v_px, lo_px, hi_px))

        # one bin width for the panel: the largest that fits every group
        bw = p["binwidth"]
        ref = next((g["values"][0] for g in self.groups if len(g["values"])), 0.0)
        fixed = None if bw is None else self._binwidth_px(bw, dpi, ref)
        if fixed is not None and len(fixed) == 1:
            w = float(fixed[0])
        else:
            fits = [
                find_binwidth(v_px, abs(hi_px - lo_px), stackratio, layout, g["side"])
                for g, v_px, lo_px, hi_px in prepared
                if len(v_px)
            ]
            w = min(fits) if fits else 1.0
            if fixed is not None:
                w = float(np.clip(w, fixed.min(), fixed.max()))
        self.binwidth_px = w
        if p.get("verbose"):
            data_w = abs(self._to_px(np.array([ref + 1.0]), 0)[0] - self._to_px(np.array([ref]), 0)[0])
            print(f"binwidth = {w / data_w:.6g} data units ({w / self._axis_px():.6g} npc)")

        offsets, faces, edges, lws, shapes, rels, sds = [], [], [], [], [], [], []
        for g, v_px, lo_px, hi_px in prepared:
            if not len(v_px):
                continue
            side = g["side"]
            x_px, off = layout_dots(v_px, w, layout, side, stackratio, overlaps, g["order"])
            maxh = abs(hi_px - lo_px)
            h = layout_height(off, w, side)
            if h > maxh * (1 + 1e-6) and maxh > 0:
                off = self._overflow(off, w, side, h, maxh)
            sign = 1.0 if hi_px >= lo_px else -1.0
            base = {"top": lo_px, "bottom": hi_px, "both": (lo_px + hi_px) / 2}[side]
            other = base + off * sign
            pts = np.column_stack([x_px, other]) if self.value_on_x else np.column_stack([other, x_px])
            offsets.append(pts)
            faces += g["faces"]
            edges += g["edges"]
            lws.append(g["linewidths"])
            shapes += g["shapes"]
            rels.append(g["rels"])
            if g.get("sds") is not None:
                # blur sd: data units -> pixels along the value axis
                sds.append(np.abs(self._to_px(g["values"] + g["sds"], g["region"][0]) - v_px))
        if not offsets:
            return
        pts = np.vstack(offsets)
        lw = np.concatenate(lws)
        rel = np.concatenate(rels)
        if self.params.get("blur") is not None:
            sd_px = np.concatenate(sds) if sds else np.zeros(len(pts))
            self._draw_blurred(renderer, pts, w * float(p["dotsize"]) * rel / 2, np.nan_to_num(sd_px), faces, shapes)
            return
        # the outline is drawn centred on the edge of the marker, so shrink
        # the marker by its width: the dot's outer diameter is binwidth * dotsize
        d_pt = np.maximum(w * float(p["dotsize"]) * 72 / dpi * rel - lw, 0.1)
        keys = np.asarray([str(sh) for sh in shapes], dtype=object)
        for key in pd.unique(keys):
            m = keys == key
            marker = MarkerStyle(shapes[int(np.flatnonzero(m)[0])])
            path = marker.get_path().transformed(marker.get_transform())
            coll = PathCollection(
                [path],
                sizes=d_pt[m] ** 2,
                offsets=pts[m],
                offset_transform=IdentityTransform(),
                facecolors=[f for f, k in zip(faces, m, strict=True) if k],
                edgecolors=[e for e, k in zip(edges, m, strict=True) if k],
                linewidths=lw[m],
                zorder=self.get_zorder(),
            )
            coll.set_figure(self.figure)
            coll.set_clip_box(self.get_clip_box())
            coll.set_clip_path(self.get_clip_path())
            coll.set_rasterized(self.get_rasterized())
            coll.draw(renderer)

    def _draw_blurred(self, renderer: RendererBase, pts: np.ndarray, radius: np.ndarray, sd: np.ndarray, faces: list, shapes: list) -> None:
        """
        Draw each dot as a radial gradient confined to a band one dot high

        Concentric discs, outermost first, whose stacked opacities reproduce
        the blur function, each clipped to the dot's height across the value
        axis: the dot itself looks soft-edged, and a wide blur becomes a
        horizontal smear rather than a large blob.
        """
        from matplotlib.collections import PolyCollection

        from ._core.blur import blur_reach, get_blur

        blur = get_blur(self.params["blur"])
        theta = np.linspace(0, 2 * np.pi, 49)[:-1]
        unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
        unit_square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        verts, colours = [], []
        for (cx, cy), r, s_, face, shape in zip(pts, radius, sd, faces, shapes, strict=True):
            reach = max(blur_reach(blur, r, s_), r)
            n = int(np.clip(np.ceil(reach / 0.75), 4, 80))
            rho = np.linspace(reach, 0, n, endpoint=False)  # outer to inner
            target = np.clip(np.asarray(blur(rho, r, s_), dtype=float), 0, 1) * face[3]
            target = np.maximum.accumulate(target)  # opacity can only grow inwards
            base = unit_square if shape in ("s", 15, 22, 0) else unit_circle
            # disc k covers radius <= rho[k]; transmission through discs
            # 0..k must be 1 - target[k]
            prev_t = 1.0
            for rk, tk in zip(rho, target, strict=True):
                if prev_t <= 1e-9:
                    break
                t = 1 - tk
                a = 1 - t / prev_t
                prev_t = t
                if a <= 1e-4:
                    continue
                shape_pts = base * rk
                shape_pts[:, 1] = np.clip(shape_pts[:, 1], -r, r)  # one dot high
                if self.value_on_x:
                    poly = shape_pts + (cx, cy)
                else:
                    poly = shape_pts[:, ::-1] + (cx, cy)
                verts.append(poly)
                colours.append((face[0], face[1], face[2], min(a, 1.0)))
        if not verts:
            return
        coll = PolyCollection(verts, facecolors=colours, edgecolors="none", linewidths=0, transform=IdentityTransform(), zorder=self.get_zorder())
        coll.set_figure(self.figure)
        coll.set_clip_box(self.get_clip_box())
        coll.set_clip_path(self.get_clip_path())
        coll.set_rasterized(self.get_rasterized())
        coll.draw(renderer)

    def _overflow(self, off: np.ndarray, w: float, side: str, h: float, maxh: float) -> np.ndarray:
        mode = self.params["overflow"]
        if mode == "compress":
            if side == "both":
                mid = (off.max() + off.min()) / 2
                span = np.ptp(off)
                return off if span == 0 else mid + (off - mid) * max(maxh - w, 0) / span
            reach = np.max(np.abs(off))
            return off if reach == 0 else off * max(maxh - w / 2, 0) / reach
        if mode == "warn" and not self._warned:
            self._warned = True
            warnings.warn(
                "The dots overflow the space for them (binwidth is too large). "
                "Use binwidth=None to fit the dots, a smaller binwidth, or "
                "overflow='compress' to squeeze the stacks, or set overflow='keep' "
                "to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        return off


class geom_dotsinterval(geom_slabinterval):
    """
    Dot plot + point + interval meta-geom

    Like [](`~plotnine_ggdist.geom_slabinterval`), with dots in place of the
    slab: rows with ``datatype="slab"`` are dots at ``x`` (horizontal) or
    ``y`` (vertical). The dots are binned and stacked in the space each
    group is given (``side``, ``scale``, ``justification``, ``height``),
    with a bin width chosen when the plot is drawn so the tallest stack
    fits. Port of `ggdist::geom_dotsinterval()`.

    Parameters
    ----------
    binwidth :
        ``None`` (default): fit the dots to the space. A number: the bin
        width in data units. A [](`~plotnine_ggdist.unit`) (e.g.
        ``unit(1.5, "mm")``): a physical size. A pair (number or `unit`):
        fit the dots, but keep the bin width within these bounds.
    dotsize :
        Dot diameter relative to the bin width. Default 1.07, so dots in
        neighbouring bins nearly touch.
    stackratio :
        Distance between dot centres in a stack, relative to the bin width.
        Default 1 (touching).
    layout :
        ``"bin"`` (default; Wilkinson dot-density bins), ``"weave"`` (rows
        as in "bin", dots at their own positions), ``"hex"`` (alternate rows
        offset, for hexagonal packing with ``stackratio`` < 1) or
        ``"swarm"`` (a compact beeswarm) or ``"bar"`` (for discrete data:
        each value's dots fill a rectangular bar).
    overlaps :
        ``"nudge"`` (default): move overlapping bins (or dots, for
        "weave") apart as little as possible. ``"keep"``: leave them.
    overflow :
        When a set `binwidth` makes the dots taller than their space:
        ``"warn"`` (default), ``"keep"`` (silently), or ``"compress"``
        (squeeze the stacks so the dots overlap but fit).
    smooth :
        Smoother applied to each group's dot positions before layout:
        ``None``/``"none"`` (default), ``"bounded"``, ``"unbounded"``,
        ``"discrete"``, ``"bar"``, or a function such as
        ``smooth_bounded(bounds=(0, 1))``. See
        [](`~plotnine_ggdist.smooth_bounded`).
    verbose :
        Print the bin width chosen.
    orientation, interval_size_range, fatten_point, show_slab, show_point, show_interval :
        As for [](`~plotnine_ggdist.geom_slabinterval`).

    Aesthetics
    ----------
    As for [](`~plotnine_ggdist.geom_slabinterval`), plus ``order`` (stacking
    order within bins for "bin", "hex" and "bar") and ``slab_shape`` (marker
    of the dots: a matplotlib marker such as ``"o"`` (default) or ``"s"``,
    or an R ``pch`` number 0-25). Dots use ``slab_fill`` (default
    `fill`), ``slab_color`` (outline; default R's gray65), ``slab_alpha``
    and ``slab_linewidth``.
    """

    DEFAULT_AES = geom_slabinterval.DEFAULT_AES | {
        "slab_color": _GRAY65,
        "slab_linewidth": 0.5 * _MM_TO_PLOTNINE,
        "slab_shape": None,
        "order": None,
    }
    DEFAULT_PARAMS = geom_slabinterval.DEFAULT_PARAMS | {
        "binwidth": None,
        "dotsize": 1.07,
        "stackratio": 1.0,
        "layout": "bin",
        "overlaps": "nudge",
        "overflow": "warn",
        "smooth": None,
        "verbose": False,
    }

    def setup_params(self, data: pd.DataFrame):
        p = self.params
        if p["layout"] not in LAYOUTS:
            raise PlotnineError(f"layout must be one of {LAYOUTS}, not {p['layout']!r}")
        if p["overlaps"] not in ("nudge", "keep"):
            raise PlotnineError(f"overlaps must be 'nudge' or 'keep', not {p['overlaps']!r}")
        if p["overflow"] not in ("warn", "keep", "compress"):
            raise PlotnineError(f"overflow must be 'warn', 'keep' or 'compress', not {p['overflow']!r}")

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # a dot plot of one variable: put it on a 0 baseline
        if "x" in data.columns and "y" not in data.columns:
            data = data.assign(y=0.0)
            if self.params.get("orientation") is None:
                self.params["orientation"] = "horizontal"
        elif "y" in data.columns and "x" not in data.columns:
            data = data.assign(x=0.0)
            if self.params.get("orientation") is None:
                self.params["orientation"] = "vertical"
        if "datatype" not in data.columns or data["datatype"].isna().all():
            if {"xmin", "xmax"} <= set(data.columns) or {"ymin", "ymax"} <= set(data.columns):
                data = data.assign(datatype="interval")
            else:
                data = data.assign(datatype="slab")
        # dots have no thickness; mark slab rows so the region is computed
        data = data.assign(thickness=np.where(data["datatype"] == "slab", 1.0, np.nan))
        return super().setup_data(data)

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Any, ax: Axes):
        params = self.params
        orientation = detect_orientation(data, params.get("orientation"))
        data = resolve_sides(data, orientation)
        is_slab = (data["datatype"] == "slab").to_numpy()
        if params["show_slab"] and is_slab.any():
            self._draw_dots(data[is_slab], orientation, panel_params, coord, ax)
        rest = data[~is_slab]
        if len(rest):
            super().draw_panel(rest, panel_params, coord, ax)

    def _draw_dots(self, dots: pd.DataFrame, orientation: str, panel_params: panel_view, coord: Any, ax: Axes):
        h = flip(dots, orientation)

        def to_screen(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = coord.transform(flip(pd.DataFrame({"x": x, "y": y}), orientation), panel_params)
            return d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float)

        # which screen axis carries the values
        sx, sy = to_screen(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        value_on_x = not np.isclose(sx[0], sx[1])

        fill = _col(h, "slab_fill", "fill")
        edge = _col(h, "slab_color", None, _GRAY65)
        alpha = _col(h, "slab_alpha", "alpha")
        lw = pd.to_numeric(_col(h, "slab_linewidth", None, 0.5 * _MM_TO_PLOTNINE), errors="coerce").fillna(0)
        # geom_dots (no point) also takes the dot shape from `shape`
        shape_fallback = None if self.params["show_point"] else "shape"
        resolved = [resolve_shape(sh) for sh in _col(h, "slab_shape", shape_fallback, "o")]
        colours = [
            colours_for(mode, _rgba(f, a), _rgba(e, a))
            for (_, mode, _), f, e, a in zip(resolved, fill, edge, alpha, strict=True)
        ]
        h = h.assign(
            _fill=[c[0] for c in colours],
            _edge=[c[1] for c in colours],
            _lw=lw.to_numpy() * SIZE_FACTOR,
            _shape=[r[0] for r in resolved],
            _rel=[r[2] for r in resolved],
        )
        smooth = get_smooth(self.params.get("smooth"))

        groups = []
        keys = [k for k in ("group", "y") if k in h.columns]
        for _, g in h.groupby(keys, sort=False, dropna=False):
            g = g[np.isfinite(g["x"].astype(float))]
            if not len(g):
                continue
            g = g.assign(x=np.asarray(smooth(g["x"].to_numpy(dtype=float)), dtype=float))
            vx, vy = to_screen(g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float))
            values = vx if value_on_x else vy
            # region along the group axis, in screen data coordinates
            r0x, r0y = to_screen(g["x"].to_numpy(dtype=float)[:1], g["ymin"].to_numpy(dtype=float)[:1])
            r1x, r1y = to_screen(g["x"].to_numpy(dtype=float)[:1], g["ymax"].to_numpy(dtype=float)[:1])
            region = (float(r0y[0]), float(r1y[0])) if value_on_x else (float(r0x[0]), float(r1x[0]))
            order = g["order"].to_numpy() if "order" in g.columns and g["order"].notna().any() else None
            groups.append(
                {
                    "values": values,
                    "region": region,
                    "side": g["side"].iloc[0],
                    "order": order,
                    "faces": list(g["_fill"]),
                    "edges": list(g["_edge"]),
                    "linewidths": g["_lw"].to_numpy(dtype=float),
                    "shapes": list(g["_shape"]),
                    "rels": g["_rel"].to_numpy(dtype=float),
                    "sds": pd.to_numeric(g["sd"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    if "sd" in g.columns
                    else None,
                }
            )
        artist = _DotsArtist(groups, value_on_x, dict(self.params), self.params["zorder"])
        if self.params.get("raster"):
            artist.set_rasterized(True)
        ax.add_artist(artist)

    @staticmethod
    def _slab_key(get: Any, da: DrawingArea) -> None:
        """
        Legend key for the dots: one dot
        """
        from matplotlib.lines import Line2D

        fill = get("slab_fill", "fill", _GRAY65)
        alpha = get("slab_alpha", "alpha", 1)
        edge = get("slab_color", None, _GRAY65)
        lw = float(get("slab_linewidth", None, 0.5 * _MM_TO_PLOTNINE)) * SIZE_FACTOR
        marker, mode, rel = resolve_shape(get("slab_shape", None, "o"))
        face, edge_c = colours_for(mode, _rgba(fill, alpha), _rgba(edge, alpha))
        size = 0.7 * min(da.width, da.height) * rel
        da.add_artist(
            Line2D(
                [da.width / 2], [da.height / 2],
                marker=marker, markersize=size,
                markerfacecolor=face, markeredgecolor=edge_c,
                markeredgewidth=lw, linestyle="none",
            )
        )  # fmt: skip


class geom_dots(geom_dotsinterval):
    """
    Dot plot

    [](`~plotnine_ggdist.geom_dotsinterval`) without the point and
    intervals: one dot per row of the data. Port of `ggdist::geom_dots()`.
    """

    DEFAULT_PARAMS = geom_dotsinterval.DEFAULT_PARAMS | {
        "show_point": False,
        "show_interval": False,
    }

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().setup_data(data.assign(datatype="slab"))


class geom_swarm(geom_dots):
    """
    Beeswarm plot

    [](`~plotnine_ggdist.geom_dots`) with ``layout="swarm"``, dots centred
    on the group (``side="both"``) and the size of
    [](`~plotnine.geom_point`)'s points (``binwidth=unit(1.5, "mm")``);
    ``overflow="compress"`` lets them overlap rather than shrink. Port of
    `ggdist::geom_swarm()`.
    """

    DEFAULT_AES = geom_dots.DEFAULT_AES | {"side": "both"}
    DEFAULT_PARAMS = geom_dots.DEFAULT_PARAMS | {
        "layout": "swarm",
        "binwidth": unit(1.5, "mm"),
        "overflow": "compress",
    }


class geom_weave(geom_dots):
    """
    Dot plot with dots at their own positions, centred on the group

    [](`~plotnine_ggdist.geom_dots`) with ``layout="weave"``,
    ``side="both"``, ``binwidth=unit(1.5, "mm")`` and
    ``overflow="compress"``. Port of `ggdist::geom_weave()`.
    """

    DEFAULT_AES = geom_dots.DEFAULT_AES | {"side": "both"}
    DEFAULT_PARAMS = geom_dots.DEFAULT_PARAMS | {
        "layout": "weave",
        "binwidth": unit(1.5, "mm"),
        "overflow": "compress",
    }



class geom_blur_dots(geom_dots):
    """
    Blurry dot plot

    [](`~plotnine_ggdist.geom_dots`) where each dot is blurred by the ``sd``
    aesthetic (in data units along the value axis): a radial blur, kept within
    the height of one dot, so wide blurs show as smears along the value axis.
    For example, to show the
    Monte Carlo error of quantiles (see
    [](`~plotnine_ggdist.stat_mcse_dots`)). Port of
    `ggdist::geom_blur_dots()`.

    Parameters
    ----------
    blur :
        ``"gaussian"`` (default; [](`~plotnine_ggdist.blur_gaussian`)),
        ``"interval"`` ([](`~plotnine_ggdist.blur_interval`)), or a function
        ``f(x, r, sd)`` giving the opacity at distance ``x`` from the centre
        of a dot of radius ``r``.
    **kwargs :
        As for [](`~plotnine_ggdist.geom_dots`).

    Notes
    -----
    Only circles (the default) and squares (``slab_shape="s"``) are
    supported, and the dots have no outline.
    """

    DEFAULT_AES = geom_dots.DEFAULT_AES | {"sd": 0.0, "slab_color": "none"}
    DEFAULT_PARAMS = geom_dots.DEFAULT_PARAMS | {"blur": "gaussian"}
