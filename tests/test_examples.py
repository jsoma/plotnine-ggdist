"""
Image tests: ports of ggdist's documentation examples

Each test named after a ggdist reference image (e.g. ``stat_halfeye-1``) is
a one-to-one port of that roxygen example: same plot, same parameters,
with the draws read from ``tests/data/`` (see ``make_data.py``) because
numpy cannot reproduce R's random numbers. ``tests/reference_r/render.R``
renders the same plots, from the same data, with R's ggdist into
``tests/reference_r/`` for side-by-side comparison.

The tests after the ports cover features the examples don't.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from plotnine import (
    aes,
    after_stat,
    coord_flip,
    expand_limits,
    geom_boxplot,
    facet_wrap,
    ggplot,
    position_dodge,
    scale_color_brewer,
    scale_fill_brewer,
    scale_x_log10,
    scale_y_continuous,
)
from scipy import stats

from plotnine_ggdist import (
    geom_blur_dots,
    mcse_quantile,
    smooth_bounded,
    smooth_discrete,
    smooth_unbounded,
    stat_mcse_dots,
    geom_dots,
    geom_dotsinterval,
    geom_swarm,
    geom_weave,
    stat_dots,
    stat_dotsinterval,
    unit,
    geom_interval,
    geom_lineribbon,
    geom_pointinterval,
    geom_slab,
    median_qi,
    stat_ccdfinterval,
    stat_cdfinterval,
    stat_eye,
    stat_gradientinterval,
    stat_halfeye,
    stat_histinterval,
    stat_interval,
    stat_lineribbon,
    stat_pointinterval,
    stat_ribbon,
    stat_slab,
    theme_ggdist,
)

DATA = Path(__file__).parent / "data"
df = pd.read_csv(DATA / "samples.csv")
ridge = pd.read_csv(DATA / "ridge.csv")
samples5 = pd.read_csv(DATA / "samples5.csv")
line = pd.read_csv(DATA / "line.csv")
ranks = pd.read_csv(DATA / "ranks.csv")
bimodal = pd.read_csv(DATA / "bimodal.csv")
dots = pd.read_csv(DATA / "dots.csv")
normal1000 = pd.read_csv(DATA / "normal1000.csv")
poisson = pd.read_csv(DATA / "poisson.csv")
beta = pd.read_csv(DATA / "beta.csv")
abcd = pd.read_csv(DATA / "abcd.csv")

# dist_df = data.frame(group = c("a", "b", "c"), mean = c(5, 7, 8), sd = c(1, 1.5, 1))
# aes(xdist = dist_normal(mean, sd))
dist_df = pd.DataFrame({"group": ["a", "b", "c"], "mean": [5, 7, 8], "sd": [1, 1.5, 1]})
dist_df["dist"] = [stats.norm(m, s) for m, s in zip(dist_df["mean"], dist_df["sd"], strict=True)]

# tibble(x = 1:10, sd = seq(1, 3, length.out = 10)); aes(ydist = dist_normal(x, sd))
line_dist = pd.DataFrame({"x": np.arange(1, 11), "sd": np.linspace(1, 3, 10)})
line_dist["dist"] = [stats.norm(x, s) for x, s in zip(line_dist["x"], line_dist["sd"], strict=True)]

# expand.grid(mean = 1:3, input = seq(-2, 6, length.out = 100)) |>
#   mutate(group = letters[4 - mean], density = dnorm(input, mean, 1))
grid = pd.DataFrame(
    [(m, v) for v in np.linspace(-2, 6, 100) for m in (1, 2, 3)], columns=["mean", "input"]
)
grid["group"] = [["a", "b", "c"][3 - m] for m in grid["mean"]]
grid["density"] = stats.norm.pdf(grid["input"], grid["mean"], 1)

T = theme_ggdist()  # the examples call theme_set(theme_ggdist())


def samples_plot():
    return ggplot(df, aes(x="value", y="group"))


def dist_plot():
    return ggplot(dist_df, aes(y="group", xdist="dist"))


# ports of the roxygen examples --------------------------------------------------------

SHORTCUTS = {
    "stat_halfeye": stat_halfeye,
    "stat_cdfinterval": stat_cdfinterval,
    "stat_eye": stat_eye,
    "stat_slab": stat_slab,
    "stat_pointinterval": stat_pointinterval,
    "stat_gradientinterval": stat_gradientinterval,
    "stat_histinterval": stat_histinterval,
}


@pytest.mark.parametrize("name", SHORTCUTS)
def test_shortcut_on_samples(name):
    p = samples_plot() + SHORTCUTS[name]() + T
    assert p == f"{name}-1"


@pytest.mark.parametrize("name", [n for n in SHORTCUTS if n != "stat_histinterval"])
def test_shortcut_on_distributions(name):
    p = dist_plot() + SHORTCUTS[name]() + T
    assert p == f"{name}-2"


def test_stat_slabinterval_1():
    p = ggplot(samples5, aes(x="group", y="value")) + stat_eye() + T
    assert p == "stat_slabinterval-1"


def test_stat_slabinterval_2():
    p = ggplot(samples5, aes(x="group", y="value")) + stat_eye(aes(thickness=after_stat("pdf * n"))) + T
    assert p == "stat_slabinterval-2"


def test_stat_slabinterval_3():
    d = pd.DataFrame(
        {
            "group": ["a", "b", "c", "c", "c"],
            "subgroup": ["h", "h", "h", "i", "j"],
            "mean": [5, 7, 8, 9, 7],
            "sd": [1, 1.5, 1, 1, 1],
        }
    )
    d["dist"] = [stats.norm(m, s) for m, s in zip(d["mean"], d["sd"], strict=True)]
    p = ggplot(d, aes(x="group", ydist="dist", fill="subgroup")) + stat_eye(position="dodge") + T
    assert p == "stat_slabinterval-3"


# stat_slabinterval-4 uses the dist = "norm", arg1 = ... interface (not ported)


def test_stat_slabinterval_5():
    # dist = "lnorm", arg1 = log(10), arg2 = 2 * log(10)
    d = pd.DataFrame({"dist": [stats.lognorm(2 * np.log(10), scale=10)]})
    p = (
        ggplot(d, aes(y=1, xdist="dist"))
        + stat_halfeye()
        + scale_x_log10(breaks=10.0 ** np.arange(-5, 8, 2))
        + T
    )
    assert p == "stat_slabinterval-5"


def test_stat_slab_3_ridges():
    p = (
        ggplot(ridge, aes(y="group", x="value"))
        + stat_slab(color="black", expand=True, trim=False, height=2)
        + T
    )
    assert p == "stat_slab-3"


def test_stat_interval_1():
    assert samples_plot() + stat_interval() + scale_color_brewer() + T == "stat_interval-1"


def test_stat_interval_2():
    assert dist_plot() + stat_interval() + scale_color_brewer() + T == "stat_interval-2"


def test_stat_ccdfinterval_1():
    assert samples_plot() + stat_ccdfinterval() + expand_limits(x=0) + T == "stat_ccdfinterval-1"


def test_stat_ccdfinterval_2():
    assert dist_plot() + stat_ccdfinterval() + expand_limits(x=0) + T == "stat_ccdfinterval-2"


def test_stat_lineribbon_1():
    p = ggplot(line, aes(x="x", y="y")) + stat_lineribbon() + scale_fill_brewer() + T
    assert p == "stat_lineribbon-1"


def test_stat_lineribbon_2():
    p = ggplot(line_dist, aes(x="x", ydist="dist")) + stat_lineribbon() + scale_fill_brewer() + T
    assert p == "stat_lineribbon-2"


def test_stat_ribbon_1():
    p = ggplot(line, aes(x="x", y="y")) + stat_ribbon() + scale_fill_brewer() + T
    assert p == "stat_ribbon-1"


def test_stat_ribbon_2():
    p = ggplot(line_dist, aes(x="x", ydist="dist")) + stat_ribbon() + scale_fill_brewer() + T
    assert p == "stat_ribbon-2"


def test_geom_lineribbon_1():
    summary = median_qi(line, "y", by="x", width=[0.5, 0.8, 0.95])
    p = (
        ggplot(summary, aes(x="x", y="y", ymin=".lower", ymax=".upper"))
        + geom_lineribbon()
        + scale_fill_brewer()
        + T
    )
    assert p == "geom_lineribbon-1"


def test_geom_pointinterval_1():
    summary = median_qi(ranks, "u_tau", by="i", width=[0.8, 0.95])
    p = ggplot(summary, aes(y="i", x="u_tau", xmin=".lower", xmax=".upper")) + geom_pointinterval() + T
    assert p == "geom_pointinterval-1"


def test_geom_pointinterval_2():
    summary = median_qi(ranks, "u_tau", by="i", width=[0.8, 0.95])
    p = ggplot(summary, aes(x="i", y="u_tau", ymin=".lower", ymax=".upper")) + geom_pointinterval() + T
    assert p == "geom_pointinterval-2"


def test_geom_interval_1():
    summary = median_qi(ranks, "u_tau", by="i", width=[0.5, 0.8, 0.95, 0.99])
    p = (
        ggplot(summary, aes(y="i", x="u_tau", xmin=".lower", xmax=".upper"))
        + geom_interval()
        + scale_color_brewer()
        + T
    )
    assert p == "geom_interval-1"


def test_geom_interval_2():
    summary = median_qi(ranks, "u_tau", by="i", width=[0.5, 0.8, 0.95, 0.99])
    p = (
        ggplot(summary, aes(x="i", y="u_tau", ymin=".lower", ymax=".upper"))
        + geom_interval()
        + scale_color_brewer()
        + T
    )
    assert p == "geom_interval-2"


def test_geom_slab_1():
    p = ggplot(grid, aes(y="group", x="input", thickness="density")) + geom_slab() + T
    assert p == "geom_slab-1"


def test_geom_slab_2():
    p = ggplot(grid, aes(x="group", y="input", thickness="density")) + geom_slab() + T
    assert p == "geom_slab-2"


def test_geom_slab_3():
    p = (
        ggplot(grid, aes(y="group", x="input", thickness="density"))
        + geom_slab(height=2, color="black")
        + T
    )
    assert p == "geom_slab-3"


# dot plots: ports of the roxygen examples ---------------------------------------------

DOT_GEOMS = {"geom_dots": geom_dots, "geom_dotsinterval": geom_dotsinterval, "geom_swarm": geom_swarm, "geom_weave": geom_weave}


@pytest.mark.parametrize("name", DOT_GEOMS)
def test_dot_geoms_horizontal(name):
    assert ggplot(dots, aes(x="value", y="g")) + DOT_GEOMS[name]() + T == f"{name}-1"


@pytest.mark.parametrize("name", DOT_GEOMS)
def test_dot_geoms_vertical(name):
    assert ggplot(dots, aes(y="value", x="g")) + DOT_GEOMS[name]() + T == f"{name}-2"


def test_geom_dotsinterval_3_quantile_dotplot():
    p = ggplot(ranks, aes(x="u_tau", y="factor(i)")) + stat_dots(quantiles=100) + T
    assert p == "geom_dotsinterval-3"


def test_geom_dotsinterval_4_fill_by_expression():
    # ggdist's example uses x > 6 for its RankCorr_u_tau data; x > 1 suits the stand-in
    p = ggplot(ranks, aes(x="u_tau", y="factor(i)", fill=after_stat("x > 1"))) + stat_dotsinterval(quantiles=100) + T
    assert p == "geom_dotsinterval-4"


def test_stat_dots_1():
    assert ggplot(line, aes(x="x", y="y")) + stat_dots() + T == "stat_dots-1"


def test_stat_dots_2():
    assert ggplot(line_dist, aes(x="x", ydist="dist")) + stat_dots(quantiles=50) + T == "stat_dots-2"


def test_stat_dotsinterval_1():
    assert ggplot(line, aes(x="x", y="y")) + stat_dotsinterval() + T == "stat_dotsinterval-1"


def test_stat_dotsinterval_2():
    assert ggplot(line_dist, aes(x="x", ydist="dist")) + stat_dotsinterval(quantiles=50) + T == "stat_dotsinterval-2"


# dot plots: beyond the examples ----------------------------------------------------------


def test_raincloud():
    small = df.head(180)  # 60 draws per group
    p = (
        ggplot(small, aes(x="group", y="value", fill="group"))
        + stat_slab(adjust=0.5, height=0.6, justification=-0.2)
        + geom_boxplot(width=0.12, outlier_shape="")
        + stat_dots(side="left", justification=1.1, binwidth=0.25)
        + T
    )
    assert p == "raincloud"


def test_quantile_dotplot_colored_by_interval():
    p = (
        dist_plot()
        + stat_dotsinterval(aes(fill=after_stat("level")), quantiles=50, slab_color="white")
        + scale_fill_brewer(na_value="#cccccc")
        + T
    )
    assert p == "quantile_dotplot_by_level"


@pytest.mark.parametrize("layout", ["bin", "weave", "hex", "swarm"])
def test_dot_layouts_side_both(layout):
    p = (
        ggplot(dots, aes(x="value", y="g"))
        + geom_dots(layout=layout, side="both", stackratio=0.9 if layout == "hex" else 1)
        + T
    )
    assert p == f"layout_{layout}"


def test_fixed_binwidth_compressed():
    p = ggplot(dots, aes(x="value", y="g")) + geom_dots(binwidth=unit(3, "mm"), overflow="compress") + T
    assert p == "fixed_binwidth_compressed"


def test_discrete_quantile_dotplot():
    d = pd.DataFrame({"g": ["poisson(4)"], "d": [stats.poisson(4)]})
    p = ggplot(d, aes(y="g", xdist="d")) + stat_dotsinterval(quantiles=40) + T
    assert p == "discrete_quantile_dotplot"


# smoothing, bar layout, blur: ports of the roxygen examples ------------------------------


def test_smooth_density_1():
    assert ggplot(normal1000, aes("x")) + geom_dots() + T == "smooth_density-1"


def test_smooth_density_2():
    assert ggplot(normal1000, aes("x")) + geom_dots(smooth="unbounded") + T == "smooth_density-2"


def test_smooth_density_3():
    p = ggplot(normal1000, aes("x")) + geom_dots(smooth=smooth_unbounded(kernel="triangular", adjust=0.5)) + T
    assert p == "smooth_density-3"


def test_smooth_density_4():
    p = ggplot(beta, aes("x_beta")) + geom_dots(smooth=smooth_bounded(bounds=(0, 1))) + T
    assert p == "smooth_density-4"


SMOOTH_DISCRETE = {
    "smooth_discrete-1": {},
    "smooth_discrete-2": {"layout": "bar"},
    "smooth_discrete-3": {"smooth": "discrete"},
    "smooth_discrete-4": {"smooth": "bar"},
    "smooth_discrete-5": {"smooth": smooth_discrete(kernel="epanechnikov", width=0.8), "side": "both"},
}


@pytest.mark.parametrize("name", SMOOTH_DISCRETE)
def test_smooth_discrete(name):
    assert ggplot(poisson, aes("x")) + geom_dots(**SMOOTH_DISCRETE[name]) + T == name


def _blur_data():
    from plotnine_ggdist._core.dots import ppoints

    x = normal1000["x"].to_numpy()
    p = ppoints(100)
    return pd.DataFrame({"q": np.quantile(x, p), "se": mcse_quantile(x, p)})


def test_geom_blur_dots_1():
    assert ggplot(_blur_data(), aes(x="q", sd="se")) + geom_blur_dots() + T == "geom_blur_dots-1"


def test_geom_blur_dots_2():
    assert ggplot(_blur_data(), aes(x="q", sd="se")) + geom_blur_dots(blur="interval") + T == "geom_blur_dots-2"


def test_stat_mcse_dots_1():
    assert ggplot(normal1000, aes(x="x")) + stat_mcse_dots(quantiles=100, layout="weave") + T == "stat_mcse_dots-1"


# smoothing, bar layout, shapes: beyond the examples ----------------------------------------


def test_bar_layout_stacked_fill():
    # vignette: geom_dots(aes(fill = g, order = g), layout = "bar", group = NA, color = NA);
    # group=1 is plotnine's way to lay out all the dots together
    p = (
        ggplot(abcd, aes(x="x", fill="g", order="g"))
        + geom_dots(layout="bar", group=1, slab_color="none")
        + scale_y_continuous(breaks=[])
        + T
    )
    assert p == "bar_layout_stacked_fill"


def test_pch_shapes():
    d = pd.DataFrame({"g": ["pch 1", "pch 16", "pch 21", "pch 22"], "s": [1, 16, 21, 22]})
    d["dist"] = [stats.norm(0, 1)] * 4
    p = (
        ggplot(d, aes(y="g", xdist="dist", slab_shape="s"))
        + stat_dotsinterval(quantiles=30, fill="#8FB8DE", slab_color="#1f4e79", shape=21)
        + T
    )
    assert p == "pch_shapes"


# beyond the examples ----------------------------------------------------------------------


def test_fill_by_level_and_slab_outline():
    p = dist_plot() + stat_halfeye(aes(fill=after_stat("level")), slab_color="gray") + scale_fill_brewer(na_value="#eeeeee") + T
    assert p == "fill_by_level"


def test_fill_by_expression():
    p = samples_plot() + stat_halfeye(aes(fill=after_stat("x > 7"))) + T
    assert p == "fill_by_expression"


def test_multimodal_hdi():
    p = ggplot(bimodal, aes(x="x")) + stat_halfeye(point_interval="mode_hdi") + T
    assert p == "multimodal_hdi"


def test_log_scale_distributions():
    d = pd.DataFrame({"g": ["sdlog 0.5", "sdlog 1"], "d": [stats.lognorm(0.5), stats.lognorm(1)]})
    p = ggplot(d, aes(y="g", xdist="d")) + stat_halfeye() + scale_x_log10() + T
    assert p == "log_scale_distributions"


def test_discrete_distributions():
    d = pd.DataFrame({"g": ["poisson(3)", "binom(10, 0.3)"], "d": [stats.poisson(3), stats.binom(10, 0.3)]})
    p = ggplot(d, aes(y="g", xdist="d")) + stat_halfeye() + T
    assert p == "discrete_distributions"


def test_facets_and_coord_flip():
    d = df.assign(half=np.where(np.arange(len(df)) < len(df) / 2, "first", "second"))
    p = ggplot(d, aes(x="value", y="group")) + stat_halfeye() + facet_wrap("half") + coord_flip() + T
    assert p == "facets_and_coord_flip"


def test_sides_and_colours():
    p = (
        samples_plot()
        + stat_halfeye(aes(fill="group"), side="bottom", interval_color="gray", point_color="black")
        + T
    )
    assert p == "side_bottom_and_subgeometry_colors"


def test_dodged_pointintervals():
    d = df.assign(cond=np.tile(["ctrl", "treat"], 750))
    p = ggplot(d, aes(x="group", y="value", color="cond")) + stat_pointinterval(position=position_dodge(width=0.4)) + T
    assert p == "dodged_pointintervals"


def test_gradient_segments():
    p = dist_plot() + stat_gradientinterval(fill_type="segments", fill="#2166ac") + T
    assert p == "gradient_segments"
