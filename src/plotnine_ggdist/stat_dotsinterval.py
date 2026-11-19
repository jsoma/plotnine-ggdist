"""
stat_dotsinterval and stat_dots: dot plots and quantile dot plots

Like [](`~plotnine_ggdist.stat_slabinterval`), but the slab rows are the
dots themselves: one row per draw, or per quantile for a quantile dot
plot (always, for analytical distributions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from plotnine.mapping.evaluation import after_stat

from ._core.dist import is_dist
from ._core.dots import dot_quantiles, ppoints
from ._core.mcse import mcse_quantile
from .stat_slabinterval import _slab_widths, _transform, stat_slabinterval

__all__ = ("stat_dotsinterval", "stat_dots", "stat_mcse_dots")

DEFAULT_DIST_QUANTILES = 100


class stat_dotsinterval(stat_slabinterval):
    """
    Dot plot with point summary and intervals

    Draws one dot per draw of sample data, or a quantile dot plot when
    `quantiles` is set (Kay et al. 2016). For analytical distributions
    (``xdist``/``ydist``) it always draws quantiles, 100 by default. Port of
    `ggdist::stat_dotsinterval()`.

    Parameters
    ----------
    quantiles :
        Number of quantiles to draw. ``None`` (default): every draw for
        sample data, 100 for distributions.
    point_interval, width, orientation :
        As for [](`~plotnine_ggdist.stat_slabinterval`).

    Dot layout parameters (``binwidth``, ``dotsize``, ``stackratio``,
    ``layout``, ``overlaps``, ``overflow``) belong to the geom; see
    [](`~plotnine_ggdist.geom_dotsinterval`).

    Computed variables
    ------------------
    x or y :
        Dots: the value of each dot. Interval: the point summary.
    xmin, xmax or ymin, ymax :
        Interval ends.
    n :
        Dots: number of draws (``inf`` for distributions).
    .width, level :
        As for [](`~plotnine_ggdist.stat_slabinterval`); for dots, the
        narrowest interval containing the dot, so
        ``aes(fill=after_stat("level"))`` colours dots by interval.
    """

    DEFAULT_AES = {"thickness": None, "weight": None}
    CREATES = {"datatype", "n", ".width", "level", "orientation"}
    DEFAULT_PARAMS = stat_slabinterval.DEFAULT_PARAMS | {
        "geom": "dotsinterval",
        "quantiles": None,
    }

    def _slab(
        self,
        x: Any,
        weights: np.ndarray | None,
        intervals: pd.DataFrame | None,
        trans: Any,
        scales: Any,
    ) -> pd.DataFrame:
        q = self.params["quantiles"]
        if is_dist(x):
            values = dot_quantiles(x, int(q) if q else DEFAULT_DIST_QUANTILES)
            n = np.inf
            if trans is not None:
                values = _transform(trans, values)
        else:
            vals = np.asarray(x, dtype=float)
            keep = np.isfinite(vals)
            vals = vals[keep]
            w = None if weights is None else weights[keep]
            n = float(len(vals)) if w is None else float(np.sum(w))
            if q:
                values = dot_quantiles(vals, int(q), w)
            else:
                values = vals
        values = values[np.isfinite(values)]
        dots = pd.DataFrame({"datatype": "slab", "x": values, "n": n})
        dots[".width"] = _slab_widths(dots["x"].to_numpy(), intervals, trans)
        return dots


class stat_dots(stat_dotsinterval):
    """
    Dot plot (or quantile dot plot) without intervals

    Port of `ggdist::stat_dots()`. See
    [](`~plotnine_ggdist.stat_dotsinterval`).
    """

    DEFAULT_PARAMS = stat_dotsinterval.DEFAULT_PARAMS | {
        "geom": "dots",
        "show_interval": False,
    }


class stat_mcse_dots(stat_dots):
    """
    Quantile dot plot with each dot blurred by its Monte Carlo error

    Computes the Monte Carlo standard error of each quantile shown
    ([](`~plotnine_ggdist.mcse_quantile`); the draws are taken as one chain,
    in data order) as the ``se`` variable, mapped to the ``sd`` aesthetic
    of [](`~plotnine_ggdist.geom_blur_dots`). Analytical distributions
    have ``se`` 0. Port of `ggdist::stat_mcse_dots()`.

    Parameters
    ----------
    quantiles :
        Number of quantiles (``None``, the default: one per draw).
    **kwargs :
        As for [](`~plotnine_ggdist.stat_dots`) and
        [](`~plotnine_ggdist.geom_blur_dots`) (e.g. ``blur="interval"``).

    Computed variables
    ------------------
    se :
        Monte Carlo standard error of each dot's quantile.
    """

    DEFAULT_AES = stat_dots.DEFAULT_AES | {"sd": after_stat("se")}
    CREATES = stat_dots.CREATES | {"se"}
    DEFAULT_PARAMS = stat_dots.DEFAULT_PARAMS | {"geom": "blur_dots"}

    def _slab(self, x, weights, intervals, trans, scales):
        dots = super()._slab(x, weights, intervals, trans, scales)
        if is_dist(x) or not len(dots):
            dots["se"] = 0.0
            return dots
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
        k = len(dots)
        p = ppoints(k)
        se_sorted = mcse_quantile(vals, p)
        se = np.empty(k)
        se[np.argsort(dots["x"].to_numpy(), kind="stable")] = se_sorted
        dots["se"] = se
        return dots
