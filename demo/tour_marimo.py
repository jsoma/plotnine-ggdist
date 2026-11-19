import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A tour of plotnine-ggdist

    Each cell draws one example. Open with `uv run --with marimo marimo edit demo/tour_marimo.py`.
    Set `SAVE_IMAGES = True` in the next cell to also write the plots to `demo/tour_output/`.
    """)
    return


@app.cell
def _():
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

    theme_set(theme_ggdist())

    # Set to True to also write each plot to demo/tour_output/<name>.png
    SAVE_IMAGES = False
    OUT = Path(__file__).parent / "tour_output"

    def fig(p, name, size=(7, 4)):
        """Size a plot (or composition) for display, optionally saving it"""
        # `&` applies a theme to every plot in a composition, `+` to one plot
        p = (
            p & theme(figure_size=size)
            if isinstance(p, Compose)
            else p + theme(figure_size=size)
        )
        if SAVE_IMAGES:
            OUT.mkdir(exist_ok=True)
            p.save(OUT / f"{name}.png", dpi=110, verbose=False)
        return p

    return (
        aes,
        after_stat,
        coord_flip,
        facet_wrap,
        fig,
        geom_blur_dots,
        geom_boxplot,
        geom_dots,
        geom_interval,
        geom_lineribbon,
        geom_pointinterval,
        geom_swarm,
        geom_vline,
        ggplot,
        labs,
        mcse_quantile,
        mean_hdi,
        median_qi,
        mode_hdci,
        np,
        pd,
        point_interval,
        position_dodge,
        scale_color_brewer,
        scale_fill_brewer,
        scale_fill_manual,
        scale_x_log10,
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
        stats,
        unit,
    )


@app.cell
def _(np, pd, stats):
    rng = np.random.default_rng(2026)

    # sample data: draws for three groups (think posterior or bootstrap draws)
    draws = pd.DataFrame(
        {
            "group": np.repeat(["a", "b", "c"], 1000),
            "value": np.r_[
                rng.normal(5, 1, 1000), rng.normal(7, 1.5, 1000), rng.gamma(9, 1, 1000)
            ],
        }
    )

    # analytical distributions: a column of frozen scipy.stats distributions
    dists = pd.DataFrame(
        {
            "name": ["normal(5, 1)", "t(3, loc=7)", "gamma(8)"],
            "dist": [stats.norm(5, 1), stats.t(3, loc=7), stats.gamma(8)],
        }
    )

    # more data used below
    small = draws.groupby("group").head(150)
    bimodal = np.r_[rng.normal(0, 1, 2000), rng.normal(6, 1, 2000)]
    ridge = pd.DataFrame(
        {
            "g": np.repeat(list("abcd"), 500),
            "v": rng.normal(np.repeat([3, 2, 1, 0], 500)),
        }
    )
    two_conditions = draws.assign(cond=np.tile(["ctrl", "treat"], 1500))
    x_line = np.repeat(np.arange(1, 11), 200)
    line = pd.DataFrame(
        {"x": x_line, "y": rng.normal(np.sin(x_line / 2) * 3 + x_line, 1 + x_line / 5)}
    )
    forecast = pd.DataFrame({"month": np.arange(1, 13)})
    forecast["dist"] = [
        stats.t(5, loc=10 + 0.6 * m, scale=0.5 + 0.15 * m) for m in forecast["month"]
    ]
    counts = pd.DataFrame({"k": rng.poisson(2, 1000), "g": np.tile(["a", "b"], 500)})
    beta = pd.DataFrame({"p": rng.beta(0.5, 0.5, 1000)})
    x1000 = pd.DataFrame({"x": rng.normal(size=1000)})
    return (
        beta,
        bimodal,
        counts,
        dists,
        draws,
        forecast,
        line,
        ridge,
        small,
        two_conditions,
        x1000,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Point + interval summaries as data frames
    """)
    return


@app.cell
def _(
    bimodal,
    draws,
    mean_hdi,
    median_qi,
    mo,
    mode_hdci,
    point_interval,
    stats,
):
    mo.vstack(
        [
            mo.md('**`median_qi(draws, "value", by="group", width=[0.66, 0.95])`**'),
            median_qi(draws, "value", by="group", width=[0.66, 0.95]),
            mo.md('**`draws.groupby("group").pipe(mean_hdi, "value")`**'),
            draws.groupby("group").pipe(mean_hdi, "value"),
            mo.md(
                '**A distribution: `point_interval(stats.gamma(3), point="mode", interval="hdci", width=0.9)`**'
            ),
            point_interval(stats.gamma(3), point="mode", interval="hdci", width=0.9),
            mo.md("**A bimodal sample: the HDI is two intervals, the HDCI one**"),
            point_interval(bimodal, point="mode", interval="hdi", width=0.8),
            mode_hdci(bimodal, width=0.8),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Half-eyes and eyes
    """)
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_halfeye):
    fig(ggplot(draws, aes("value", "group")) + stat_halfeye(), "02a_halfeye_samples")
    return


@app.cell
def _(aes, dists, fig, ggplot, stat_halfeye):
    fig(
        ggplot(dists, aes(y="name", xdist="dist")) + stat_halfeye(),
        "02b_halfeye_distributions",
    )
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_eye):
    fig(ggplot(draws, aes("group", "value")) + stat_eye(), "02c_eye_vertical")
    return


@app.cell
def _(aes, after_stat, dists, fig, ggplot, scale_fill_brewer, stat_halfeye):
    # colour the slab by interval
    fig(
        ggplot(dists, aes(y="name", xdist="dist"))
        + stat_halfeye(aes(fill=after_stat("level")), slab_color="gray")
        + scale_fill_brewer(na_value="#eeeeee"),
        "02d_fill_by_level",
    )
    return


@app.cell
def _(
    aes,
    after_stat,
    draws,
    fig,
    geom_vline,
    ggplot,
    scale_fill_manual,
    stat_halfeye,
):
    # ... or by any expression of the computed variables
    fig(
        ggplot(draws, aes("value", "group"))
        + geom_vline(xintercept=7, linetype="dashed", color="gray")
        + stat_halfeye(aes(fill=after_stat("x > 7")))
        + scale_fill_manual(values=["#D1495B", "#8FB8DE"]),
        "02e_fill_by_expression",
    )
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_halfeye):
    # other point/interval functions and widths
    fig(
        ggplot(draws, aes("value", "group"))
        + stat_halfeye(point_interval="mode_hdi", width=[0.5, 0.8, 0.95]),
        "02f_mode_hdi_three_widths",
    )
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_halfeye):
    # layout: side, justification, scale
    fig(
        ggplot(draws, aes("value", "group"))
        + stat_halfeye(side="bottom", scale=0.6, justification=1.1),
        "02g_side_bottom",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Other slab + interval plots
    """)
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_pointinterval):
    fig(
        ggplot(draws, aes("value", "group")) + stat_pointinterval(), "03a_pointinterval"
    )
    return


@app.cell
def _(aes, draws, fig, ggplot, scale_color_brewer, stat_interval):
    fig(
        ggplot(draws, aes("value", "group")) + stat_interval() + scale_color_brewer(),
        "03b_interval",
    )
    return


@app.cell
def _(aes, dists, fig, ggplot, stat_ccdfinterval):
    fig(
        ggplot(dists, aes(y="name", xdist="dist")) + stat_ccdfinterval(),
        "03c_ccdf_bars",
    )
    return


@app.cell
def _(aes, dists, fig, ggplot, stat_gradientinterval):
    fig(
        ggplot(dists, aes(y="name", xdist="dist")) + stat_gradientinterval(),
        "03d_gradient",
    )
    return


@app.cell
def _(aes, draws, fig, ggplot, stat_histinterval):
    fig(ggplot(draws, aes("value", "group")) + stat_histinterval(), "03e_histogram")
    return


@app.cell
def _(aes, fig, ggplot, ridge, stat_slab):
    # ridgelines
    fig(
        ggplot(ridge, aes("v", "g"))
        + stat_slab(height=2, color="black", expand=True, trim=False),
        "03f_ridgelines",
    )
    return


@app.cell
def _(aes, fig, ggplot, position_dodge, stat_pointinterval, two_conditions):
    # dodged intervals (vertical layouts)
    fig(
        ggplot(two_conditions, aes("group", "value", color="cond"))
        + stat_pointinterval(position=position_dodge(width=0.4)),
        "03g_dodged",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Analytical distributions: discrete, transformed scales
    """)
    return


@app.cell
def _(aes, fig, ggplot, pd, stat_halfeye, stats):
    _discrete = pd.DataFrame(
        {
            "name": ["poisson(3)", "binom(10, 0.3)"],
            "dist": [stats.poisson(3), stats.binom(10, 0.3)],
        }
    )
    fig(ggplot(_discrete, aes(y="name", xdist="dist")) + stat_halfeye(), "04a_discrete")
    return


@app.cell
def _(aes, fig, ggplot, pd, scale_x_log10, stat_halfeye, stats):
    _lognormal = pd.DataFrame(
        {
            "name": ["sdlog 0.5", "sdlog 1"],
            "dist": [stats.lognorm(0.5), stats.lognorm(1)],
        }
    )
    fig(
        ggplot(_lognormal, aes(y="name", xdist="dist"))
        + stat_halfeye()
        + scale_x_log10(),
        "04b_log_scale",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Lines with ribbons
    """)
    return


@app.cell
def _(aes, fig, ggplot, line, scale_fill_brewer, stat_lineribbon):
    fig(
        ggplot(line, aes("x", "y")) + stat_lineribbon() + scale_fill_brewer(),
        "05a_lineribbon_samples",
    )
    return


@app.cell
def _(aes, fig, forecast, ggplot, scale_fill_brewer, stat_lineribbon):
    fig(
        ggplot(forecast, aes(x="month", ydist="dist"))
        + stat_lineribbon(width=[0.5, 0.8, 0.95])
        + scale_fill_brewer(),
        "05b_lineribbon_distributions",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Precomputed summaries with the geoms

    `.width` in the data sizes the intervals and colours ribbons / intervals.
    """)
    return


@app.cell
def _(aes, draws, fig, geom_pointinterval, ggplot, median_qi):
    _summary = median_qi(draws, "value", by="group", width=[0.66, 0.95])
    fig(
        ggplot(_summary, aes(x="value", xmin=".lower", xmax=".upper", y="group"))
        + geom_pointinterval(),
        "06a_geom_pointinterval",
    )
    return


@app.cell
def _(aes, draws, fig, geom_interval, ggplot, median_qi, scale_color_brewer):
    _summary = median_qi(draws, "value", by="group", width=[0.5, 0.8, 0.95])
    fig(
        ggplot(_summary, aes(y="value", ymin=".lower", ymax=".upper", x="group"))
        + geom_interval()
        + scale_color_brewer(),
        "06b_geom_interval",
    )
    return


@app.cell
def _(aes, fig, geom_lineribbon, ggplot, line, median_qi, scale_fill_brewer):
    _summary = median_qi(line, "y", by="x", width=[0.5, 0.8, 0.95])
    fig(
        ggplot(_summary, aes(x="x", y="y", ymin=".lower", ymax=".upper"))
        + geom_lineribbon()
        + scale_fill_brewer(),
        "06c_geom_lineribbon",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Dot plots
    """)
    return


@app.cell
def _(aes, fig, ggplot, small, stat_dots):
    fig(ggplot(small, aes("value", "group")) + stat_dots(), "07a_dots")
    return


@app.cell
def _(aes, fig, ggplot, small, stat_dotsinterval):
    fig(ggplot(small, aes("value", "group")) + stat_dotsinterval(), "07b_dotsinterval")
    return


@app.cell
def _(aes, dists, fig, ggplot, stat_dotsinterval):
    # quantile dot plots (always, for distributions; 100 quantiles by default)
    fig(
        ggplot(dists, aes(y="name", xdist="dist")) + stat_dotsinterval(quantiles=50),
        "07c_quantile_dotplot",
    )
    return


@app.cell
def _(
    aes,
    after_stat,
    dists,
    fig,
    ggplot,
    scale_fill_brewer,
    stat_dotsinterval,
):
    fig(
        ggplot(dists, aes(y="name", xdist="dist"))
        + stat_dotsinterval(
            aes(fill=after_stat("level")), quantiles=50, slab_color="white"
        )
        + scale_fill_brewer(na_value="#cccccc"),
        "07d_quantile_dotplot_by_level",
    )
    return


@app.cell
def _(aes, fig, geom_dots, ggplot, labs, small):
    # the layouts, side by side
    _layouts = [
        ggplot(small, aes("value", "group"))
        + geom_dots(layout=_lay, side="both")
        + labs(title=f'layout="{_lay}"')
        for _lay in ["bin", "weave", "hex", "swarm"]
    ]
    fig(
        (_layouts[0] | _layouts[1]) / (_layouts[2] | _layouts[3]),
        "07e_layouts",
        size=(11, 7),
    )
    return


@app.cell
def _(aes, fig, geom_swarm, ggplot, small):
    # beeswarm with point-sized dots
    fig(ggplot(small, aes("group", "value")) + geom_swarm(), "07f_swarm")
    return


@app.cell
def _(aes, draws, fig, geom_dots, ggplot, unit):
    # a fixed dot size: compress the stacks instead of overflowing
    fig(
        ggplot(draws, aes("value", "group"))
        + geom_dots(binwidth=unit(2, "mm"), overflow="compress"),
        "07g_fixed_binwidth",
    )
    return


@app.cell
def _(aes, draws, fig, geom_boxplot, ggplot, stat_dots, stat_slab):
    # raincloud: slab + boxplot + dots
    fig(
        ggplot(draws.groupby("group").head(60), aes("group", "value", fill="group"))
        + stat_slab(adjust=0.5, height=0.6, justification=-0.2)
        + geom_boxplot(width=0.12, outlier_shape="")
        + stat_dots(side="left", justification=1.1, binwidth=0.25),
        "07h_raincloud",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Discrete data: bar layout and smoothing
    """)
    return


@app.cell
def _(aes, counts, fig, geom_dots, ggplot):
    fig(ggplot(counts, aes("k")) + geom_dots(), "08a_counts_default")
    return


@app.cell
def _(aes, counts, fig, geom_dots, ggplot):
    fig(ggplot(counts, aes("k")) + geom_dots(layout="bar"), "08b_counts_bar")
    return


@app.cell
def _(aes, counts, fig, geom_dots, ggplot):
    fig(
        ggplot(counts, aes(x="k", fill="g", order="g"))
        + geom_dots(layout="bar", group=1, slab_color="none"),
        "08c_counts_bar_stacked",
    )
    return


@app.cell
def _(aes, counts, fig, geom_dots, ggplot):
    fig(
        ggplot(counts, aes("k")) + geom_dots(smooth="discrete", side="both"),
        "08d_counts_smooth_discrete",
    )
    return


@app.cell
def _(aes, beta, fig, geom_dots, ggplot, smooth_bounded):
    fig(
        ggplot(beta, aes("p")) + geom_dots(smooth=smooth_bounded(bounds=(0, 1))),
        "08e_density_dotplot",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Blurry dots: Monte Carlo error of quantiles
    """)
    return


@app.cell
def _(aes, fig, ggplot, stat_mcse_dots, x1000):
    fig(ggplot(x1000, aes("x")) + stat_mcse_dots(quantiles=100), "09a_mcse_dots")
    return


@app.cell
def _(aes, fig, ggplot, stat_mcse_dots, x1000):
    fig(
        ggplot(x1000, aes("x")) + stat_mcse_dots(quantiles=100, blur="interval"),
        "09b_mcse_interval",
    )
    return


@app.cell
def _(aes, fig, geom_blur_dots, ggplot, mcse_quantile, np, pd, x1000):
    _p = np.linspace(0.01, 0.99, 60)
    _q = pd.DataFrame(
        {"q": np.quantile(x1000["x"], _p), "se": mcse_quantile(x1000["x"], _p)}
    )
    fig(ggplot(_q, aes(x="q", sd="se")) + geom_blur_dots(), "09c_geom_blur_dots")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Shapes (R pch codes or matplotlib markers), facets, coord_flip, composition
    """)
    return


@app.cell
def _(aes, fig, ggplot, pd, stat_dotsinterval, stats):
    _shapes = pd.DataFrame(
        {"g": ["pch 1", "pch 16", "pch 21", '"s"'], "s": [1, 16, 21, "s"]}
    )
    _shapes["dist"] = [stats.norm(0, 1)] * 4
    fig(
        ggplot(_shapes, aes(y="g", xdist="dist", slab_shape="s"))
        + stat_dotsinterval(quantiles=30, fill="#8FB8DE", slab_color="#1f4e79"),
        "10a_shapes",
    )
    return


@app.cell
def _(aes, coord_flip, draws, facet_wrap, fig, ggplot, np, stat_halfeye):
    _halves = draws.assign(
        half=np.where(np.arange(len(draws)) % 2 == 0, "even rows", "odd rows")
    )
    fig(
        ggplot(_halves, aes("value", "group"))
        + stat_halfeye()
        + facet_wrap("half")
        + coord_flip(),
        "10b_facets_coord_flip",
        size=(9, 4),
    )
    return


@app.cell
def _(aes, dists, draws, fig, ggplot, labs, stat_dotsinterval, stat_halfeye):
    _left = (
        ggplot(draws, aes("value", "group")) + stat_halfeye() + labs(title="samples")
    )
    _right = (
        ggplot(dists, aes(y="name", xdist="dist"))
        + stat_dotsinterval(quantiles=50)
        + labs(title="distributions")
    )
    fig(_left | _right, "10c_side_by_side", size=(11, 4))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
