"""
plotnine-ggdist: distribution and uncertainty visualisation for plotnine

An independent Python implementation of the R package ggdist
(https://mjskay.github.io/ggdist/) by Matthew Kay, for plotnine.
"""

from ._core.density import (
    Density,
    bandwidth_dpi,
    bandwidth_nrd,
    bandwidth_nrd0,
    bounder_cdf,
    bounder_cooke,
    bounder_range,
    density_bounded,
    density_histogram,
    density_unbounded,
    weighted_ecdf,
    weighted_quantile,
)
from ._core.blur import blur_gaussian, blur_interval
from ._core.dots import bin_dots, find_dotplot_binwidth
from ._core.mcse import ess_basic, mcse_quantile
from ._core.smooth import smooth_bar, smooth_bounded, smooth_discrete, smooth_none, smooth_unbounded
from ._core.intervals import SHORTCUTS as _SHORTCUTS
from ._core.intervals import Mode, hdci, hdi, ll, point_interval, qi, ul
from . import _stat_identity  # noqa: F401  (registers stat_width_identity)
from ._units import unit
from .geom_dotsinterval import geom_blur_dots, geom_dots, geom_dotsinterval, geom_swarm, geom_weave
from .geom_lineribbon import geom_lineribbon
from .geom_slabinterval import geom_interval, geom_pointinterval, geom_slab, geom_slabinterval
from .stat_dotsinterval import stat_dots, stat_dotsinterval, stat_mcse_dots
from .stat_lineribbon import stat_lineribbon, stat_ribbon
from .stat_slabinterval import (
    stat_ccdfinterval,
    stat_cdfinterval,
    stat_eye,
    stat_gradientinterval,
    stat_halfeye,
    stat_histinterval,
    stat_interval,
    stat_pointinterval,
    stat_slab,
    stat_slabinterval,
)
from .theme_ggdist import theme_ggdist

mean_qi = _SHORTCUTS["mean_qi"]
mean_hdi = _SHORTCUTS["mean_hdi"]
mean_hdci = _SHORTCUTS["mean_hdci"]
mean_ll = _SHORTCUTS["mean_ll"]
mean_ul = _SHORTCUTS["mean_ul"]
median_qi = _SHORTCUTS["median_qi"]
median_hdi = _SHORTCUTS["median_hdi"]
median_hdci = _SHORTCUTS["median_hdci"]
median_ll = _SHORTCUTS["median_ll"]
median_ul = _SHORTCUTS["median_ul"]
mode_qi = _SHORTCUTS["mode_qi"]
mode_hdi = _SHORTCUTS["mode_hdi"]
mode_hdci = _SHORTCUTS["mode_hdci"]
mode_ll = _SHORTCUTS["mode_ll"]
mode_ul = _SHORTCUTS["mode_ul"]

__all__ = (
    # stats
    "stat_slabinterval",
    "stat_halfeye",
    "stat_eye",
    "stat_slab",
    "stat_ccdfinterval",
    "stat_cdfinterval",
    "stat_gradientinterval",
    "stat_histinterval",
    "stat_pointinterval",
    "stat_interval",
    "stat_lineribbon",
    "stat_ribbon",
    "stat_dots",
    "stat_dotsinterval",
    "stat_mcse_dots",
    # geoms
    "geom_slabinterval",
    "geom_slab",
    "geom_pointinterval",
    "geom_interval",
    "geom_lineribbon",
    "geom_dots",
    "geom_dotsinterval",
    "geom_swarm",
    "geom_weave",
    "geom_blur_dots",
    # dot layout, smoothing and blur
    "bin_dots",
    "find_dotplot_binwidth",
    "unit",
    "smooth_none",
    "smooth_bounded",
    "smooth_unbounded",
    "smooth_discrete",
    "smooth_bar",
    "blur_gaussian",
    "blur_interval",
    "mcse_quantile",
    "ess_basic",
    # point_interval
    "point_interval",
    "qi",
    "hdi",
    "hdci",
    "ll",
    "ul",
    "Mode",
    "mean_qi",
    "mean_hdi",
    "mean_hdci",
    "mean_ll",
    "mean_ul",
    "median_qi",
    "median_hdi",
    "median_hdci",
    "median_ll",
    "median_ul",
    "mode_qi",
    "mode_hdi",
    "mode_hdci",
    "mode_ll",
    "mode_ul",
    # densities
    "Density",
    "density_bounded",
    "density_unbounded",
    "density_histogram",
    "bandwidth_dpi",
    "bandwidth_nrd0",
    "bandwidth_nrd",
    "bounder_cdf",
    "bounder_cooke",
    "bounder_range",
    "weighted_quantile",
    "weighted_ecdf",
    # theme
    "theme_ggdist",
)
