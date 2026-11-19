# %% [markdown]
# # A tour of plotnine-ggdist
#
# Run the whole thing with `uv run python demo/tour.py` (images go to
# `demo/tour_output/`), or open it in VS Code / Jupyter and run the `# %%`
# cells one at a time.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    after_stat,
    coord_flip,
    facet_wrap,
    geom_boxplot,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_brewer,
    scale_fill_brewer,
    scale_fill_manual,
    scale_x_log10,
    theme,
    theme_set,
)
from plotnine.composition import Compose
from scipy import stats

from plotnine_ggdist import (
    geom_blur_dots,
    geom_dots,
    geom_interval,
    geom_lineribbon,
    geom_pointinterval,
    geom_swarm,
    mcse_quantile,
    mean_hdi,
    median_qi,
    mode_hdci,
    point_interval,
    smooth_bounded,
    stat_ccdfinterval,
    stat_dots,
    stat_dotsinterval,
    stat_eye,
    stat_gradientinterval,
    stat_halfeye,
    stat_histinterval,
    stat_interval,
    stat_lineribbon,
    stat_mcse_dots,
    stat_pointinterval,
    stat_slab,
    theme_ggdist,
    unit,
)

OUT = Path(__file__).parent / "tour_output"
OUT.mkdir(exist_ok=True)
theme_set(theme_ggdist())


def show(p, name, size=(7, 4)):
    """Save a plot (or a composition) to demo/tour_output/<name>.png"""
    # `&` applies a theme to every plot in a composition, `+` to one plot
    p = p & theme(figure_size=size) if isinstance(p, Compose) else p + theme(figure_size=size)
    p.save(OUT / f"{name}.png", dpi=110, verbose=False)
    print("saved", name)


rng = np.random.default_rng(2026)

# sample data: draws for three groups (think posterior or bootstrap draws)
draws = pd.DataFrame(
    {
        "group": np.repeat(["a", "b", "c"], 1000),
        "value": np.r_[rng.normal(5, 1, 1000), rng.normal(7, 1.5, 1000), rng.gamma(9, 1, 1000)],
    }
)

# analytical distributions: a column of frozen scipy.stats distributions
dists = pd.DataFrame(
    {
        "name": ["normal(5, 1)", "t(3, loc=7)", "gamma(8)"],
        "dist": [stats.norm(5, 1), stats.t(3, loc=7), stats.gamma(8)],
    }
)

# %% [markdown]
# ## 1. Point + interval summaries as data frames

# %%
print(median_qi(draws, "value", by="group", width=[0.66, 0.95]))
print(draws.groupby("group").pipe(mean_hdi, "value"))
print(point_interval(stats.gamma(3), point="mode", interval="hdci", width=0.9))

# a bimodal sample: the HDI is two intervals, the HDCI one
bimodal = np.r_[rng.normal(0, 1, 2000), rng.normal(6, 1, 2000)]
print(point_interval(bimodal, point="mode", interval="hdi", width=0.8))
print(mode_hdci(bimodal, width=0.8))

# %% [markdown]
# ## 2. Half-eyes and eyes

# %%
show(ggplot(draws, aes("value", "group")) + stat_halfeye(), "02a_halfeye_samples")
show(ggplot(dists, aes(y="name", xdist="dist")) + stat_halfeye(), "02b_halfeye_distributions")
show(ggplot(draws, aes("group", "value")) + stat_eye(), "02c_eye_vertical")

# colour the slab by interval, or by any expression of the computed variables
show(
    ggplot(dists, aes(y="name", xdist="dist"))
    + stat_halfeye(aes(fill=after_stat("level")), slab_color="gray")
    + scale_fill_brewer(na_value="#eeeeee"),
    "02d_fill_by_level",
)
show(
    ggplot(draws, aes("value", "group"))
    + geom_vline(xintercept=7, linetype="dashed", color="gray")
    + stat_halfeye(aes(fill=after_stat("x > 7")))
    + scale_fill_manual(values=["#D1495B", "#8FB8DE"]),
    "02e_fill_by_expression",
)

# other point/interval functions and widths
show(
    ggplot(draws, aes("value", "group"))
    + stat_halfeye(point_interval="mode_hdi", width=[0.5, 0.8, 0.95]),
    "02f_mode_hdi_three_widths",
)

# layout: side, justification, scale
show(
    ggplot(draws, aes("value", "group")) + stat_halfeye(side="bottom", scale=0.6, justification=1.1),
    "02g_side_bottom",
)

# %% [markdown]
# ## 3. Other slab + interval plots

# %%
show(ggplot(draws, aes("value", "group")) + stat_pointinterval(), "03a_pointinterval")
show(ggplot(draws, aes("value", "group")) + stat_interval() + scale_color_brewer(), "03b_interval")
show(ggplot(dists, aes(y="name", xdist="dist")) + stat_ccdfinterval(), "03c_ccdf_bars")
show(ggplot(dists, aes(y="name", xdist="dist")) + stat_gradientinterval(), "03d_gradient")
show(ggplot(draws, aes("value", "group")) + stat_histinterval(), "03e_histogram")

# ridgelines
ridge = pd.DataFrame({"g": np.repeat(list("abcd"), 500), "v": rng.normal(np.repeat([3, 2, 1, 0], 500))})
show(
    ggplot(ridge, aes("v", "g")) + stat_slab(height=2, color="black", expand=True, trim=False),
    "03f_ridgelines",
)

# dodged intervals (vertical layouts)
two = draws.assign(cond=np.tile(["ctrl", "treat"], 1500))
show(
    ggplot(two, aes("group", "value", color="cond")) + stat_pointinterval(position=position_dodge(width=0.4)),
    "03g_dodged",
)

# %% [markdown]
# ## 4. Analytical distributions: discrete, transformed scales

# %%
discrete = pd.DataFrame(
    {"name": ["poisson(3)", "binom(10, 0.3)"], "dist": [stats.poisson(3), stats.binom(10, 0.3)]}
)
show(ggplot(discrete, aes(y="name", xdist="dist")) + stat_halfeye(), "04a_discrete")

lognormal = pd.DataFrame({"name": ["sdlog 0.5", "sdlog 1"], "dist": [stats.lognorm(0.5), stats.lognorm(1)]})
show(ggplot(lognormal, aes(y="name", xdist="dist")) + stat_halfeye() + scale_x_log10(), "04b_log_scale")

# %% [markdown]
# ## 5. Lines with ribbons

# %%
x = np.repeat(np.arange(1, 11), 200)
line = pd.DataFrame({"x": x, "y": rng.normal(np.sin(x / 2) * 3 + x, 1 + x / 5)})
show(ggplot(line, aes("x", "y")) + stat_lineribbon() + scale_fill_brewer(), "05a_lineribbon_samples")

forecast = pd.DataFrame({"month": np.arange(1, 13)})
forecast["dist"] = [stats.t(5, loc=10 + 0.6 * m, scale=0.5 + 0.15 * m) for m in forecast["month"]]
show(
    ggplot(forecast, aes(x="month", ydist="dist")) + stat_lineribbon(width=[0.5, 0.8, 0.95]) + scale_fill_brewer(),
    "05b_lineribbon_distributions",
)

# %% [markdown]
# ## 6. Precomputed summaries with the geoms
#
# `.width` in the data sizes the intervals and colours ribbons / intervals.

# %%
summary = median_qi(draws, "value", by="group", width=[0.66, 0.95])
show(
    ggplot(summary, aes(x="value", xmin=".lower", xmax=".upper", y="group")) + geom_pointinterval(),
    "06a_geom_pointinterval",
)
show(
    ggplot(median_qi(draws, "value", by="group", width=[0.5, 0.8, 0.95]),
           aes(y="value", ymin=".lower", ymax=".upper", x="group"))
    + geom_interval()
    + scale_color_brewer(),
    "06b_geom_interval",
)
show(
    ggplot(median_qi(line, "y", by="x", width=[0.5, 0.8, 0.95]), aes(x="x", y="y", ymin=".lower", ymax=".upper"))
    + geom_lineribbon()
    + scale_fill_brewer(),
    "06c_geom_lineribbon",
)

# %% [markdown]
# ## 7. Dot plots

# %%
small = draws.groupby("group").head(150)
show(ggplot(small, aes("value", "group")) + stat_dots(), "07a_dots")
show(ggplot(small, aes("value", "group")) + stat_dotsinterval(), "07b_dotsinterval")

# quantile dot plots (always, for distributions; 100 quantiles by default)
show(ggplot(dists, aes(y="name", xdist="dist")) + stat_dotsinterval(quantiles=50), "07c_quantile_dotplot")
show(
    ggplot(dists, aes(y="name", xdist="dist"))
    + stat_dotsinterval(aes(fill=after_stat("level")), quantiles=50, slab_color="white")
    + scale_fill_brewer(na_value="#cccccc"),
    "07d_quantile_dotplot_by_level",
)

# the layouts, side by side
layouts = [
    ggplot(small, aes("value", "group")) + geom_dots(layout=lay, side="both") + labs(title=f'layout="{lay}"')
    for lay in ["bin", "weave", "hex", "swarm"]
]
show((layouts[0] | layouts[1]) / (layouts[2] | layouts[3]), "07e_layouts", size=(11, 7))

# beeswarm with point-sized dots
show(ggplot(small, aes("group", "value")) + geom_swarm(), "07f_swarm")

# a fixed dot size: compress the stacks instead of overflowing
show(
    ggplot(draws, aes("value", "group")) + geom_dots(binwidth=unit(2, "mm"), overflow="compress"),
    "07g_fixed_binwidth",
)

# raincloud: slab + boxplot + dots
show(
    ggplot(draws.groupby("group").head(60), aes("group", "value", fill="group"))
    + stat_slab(adjust=0.5, height=0.6, justification=-0.2)
    + geom_boxplot(width=0.12, outlier_shape="")
    + stat_dots(side="left", justification=1.1, binwidth=0.25),
    "07h_raincloud",
)

# %% [markdown]
# ## 8. Discrete data: bar layout and smoothing

# %%
counts = pd.DataFrame({"k": rng.poisson(2, 1000), "g": np.tile(["a", "b"], 500)})
show(ggplot(counts, aes("k")) + geom_dots(), "08a_counts_default")
show(ggplot(counts, aes("k")) + geom_dots(layout="bar"), "08b_counts_bar")
show(
    ggplot(counts, aes(x="k", fill="g", order="g")) + geom_dots(layout="bar", group=1, slab_color="none"),
    "08c_counts_bar_stacked",
)
show(ggplot(counts, aes("k")) + geom_dots(smooth="discrete", side="both"), "08d_counts_smooth_discrete")

beta = pd.DataFrame({"p": rng.beta(0.5, 0.5, 1000)})
show(ggplot(beta, aes("p")) + geom_dots(smooth=smooth_bounded(bounds=(0, 1))), "08e_density_dotplot")

# %% [markdown]
# ## 9. Blurry dots: Monte Carlo error of quantiles

# %%
x1000 = pd.DataFrame({"x": rng.normal(size=1000)})
show(ggplot(x1000, aes("x")) + stat_mcse_dots(quantiles=100), "09a_mcse_dots")
show(ggplot(x1000, aes("x")) + stat_mcse_dots(quantiles=100, blur="interval"), "09b_mcse_interval")

p = np.linspace(0.01, 0.99, 60)
q = pd.DataFrame({"q": np.quantile(x1000["x"], p), "se": mcse_quantile(x1000["x"], p)})
show(ggplot(q, aes(x="q", sd="se")) + geom_blur_dots(), "09c_geom_blur_dots")

# %% [markdown]
# ## 10. Shapes (R pch codes or matplotlib markers), facets, coord_flip, composition

# %%
shapes = pd.DataFrame({"g": ["pch 1", "pch 16", "pch 21", '"s"'], "s": [1, 16, 21, "s"]})
shapes["dist"] = [stats.norm(0, 1)] * 4
show(
    ggplot(shapes, aes(y="g", xdist="dist", slab_shape="s"))
    + stat_dotsinterval(quantiles=30, fill="#8FB8DE", slab_color="#1f4e79"),
    "10a_shapes",
)

halves = draws.assign(half=np.where(np.arange(len(draws)) % 2 == 0, "even rows", "odd rows"))
show(
    ggplot(halves, aes("value", "group")) + stat_halfeye() + facet_wrap("half") + coord_flip(),
    "10b_facets_coord_flip",
    size=(9, 4),
)

left = ggplot(draws, aes("value", "group")) + stat_halfeye() + labs(title="samples")
right = ggplot(dists, aes(y="name", xdist="dist")) + stat_dotsinterval(quantiles=50) + labs(title="distributions")
show(left | right, "10c_side_by_side", size=(11, 4))

print(f"\nAll images are in {OUT}")
