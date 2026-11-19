"""
Tests for kernels, dot smoothers, the "bar" layout, R pch shapes, blurs and
Monte Carlo standard errors (no image comparison)
"""

import io

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, ggplot
from scipy import integrate, stats

from plotnine_ggdist import (
    blur_gaussian,
    blur_interval,
    density_bounded,
    density_unbounded,
    ess_basic,
    geom_blur_dots,
    geom_dots,
    mcse_quantile,
    smooth_bar,
    smooth_bounded,
    smooth_discrete,
    smooth_none,
    smooth_unbounded,
    stat_mcse_dots,
)
from plotnine_ggdist._core.dots import layout_dots, layout_height, find_binwidth
from plotnine_ggdist._core.kernels import KERNELS, get_kernel
from plotnine_ggdist._shapes import colours_for, resolve_shape

rng = np.random.default_rng(5)
X = rng.normal(size=2000)


# kernels -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(KERNELS))
def test_kernels_have_unit_mass_and_variance(name):
    k = KERNELS[name]
    pts = None if not np.isfinite(k.support) else [-k.support, k.support]
    assert integrate.quad(k.pdf, -12, 12, points=pts)[0] == pytest.approx(1, abs=1e-6)
    assert integrate.quad(lambda u: u * u * k.pdf(u), -12, 12, points=pts)[0] == pytest.approx(1, abs=1e-6)
    np.testing.assert_allclose(k.cdf(np.array([-20.0, 0.0, 20.0])), [0, 0.5, 1], atol=1e-12)


def test_partial_kernel_names():
    assert get_kernel("ep").name == "epanechnikov"
    assert get_kernel("rect").name == "rectangular"
    with pytest.raises(ValueError):
        get_kernel("nope")


@pytest.mark.parametrize("name", ["rectangular", "epanechnikov", "biweight"])
def test_kde_with_other_kernels(name):
    d = density_unbounded(X, kernel=name, trim=False)
    assert np.trapezoid(d.pdf, d.x) == pytest.approx(1, abs=0.01)
    b = density_bounded(np.abs(X), kernel=name, bounds=(0, None))
    assert np.trapezoid(b.pdf, b.x) == pytest.approx(1, abs=0.01)


# smoothers ---------------------------------------------------------------------------


def test_smooth_none_and_partials():
    np.testing.assert_array_equal(smooth_none(X), X)
    f = smooth_discrete(kernel="epanechnikov", width=0.8)
    assert callable(f)
    np.testing.assert_allclose(f(np.array([1.0, 1, 2])), smooth_discrete(np.array([1.0, 1, 2]), kernel="epanechnikov", width=0.8))


def test_smooth_density_keeps_rank_order_and_count():
    out = smooth_unbounded(X)
    assert len(out) == len(X)
    np.testing.assert_array_equal(np.argsort(out, kind="stable"), np.argsort(X, kind="stable"))
    # evenly spaced quantiles of a smooth estimate: no big gaps in the middle
    mid = np.sort(out)[500:1500]
    assert np.diff(mid).max() < 5 * np.median(np.diff(mid))


def test_smooth_bounded_respects_bounds():
    b = rng.beta(0.5, 0.5, 1000)
    out = smooth_bounded(b, bounds=(0, 1))
    assert out.min() >= 0 and out.max() <= 1


def test_smooth_discrete_and_bar_stay_near_values():
    k = rng.poisson(3, 500).astype(float)
    # width 0.7, resolution 1: within about 0.35 of the value
    assert np.all(np.abs(smooth_bar(k) - k) <= 0.35 + 1e-9)
    assert np.all(np.abs(smooth_discrete(k) - k) <= 0.4)
    bar = smooth_bar(np.array([2.0, 2, 2, 2]))
    np.testing.assert_allclose(bar, [2 - 0.2625, 2 - 0.0875, 2 + 0.0875, 2 + 0.2625])


def test_geom_dots_smooth_parameter_draws():
    df = pd.DataFrame({"x": rng.poisson(2, 300)})
    for smooth in ("bounded", "unbounded", "discrete", "bar", smooth_discrete(kernel="ep")):
        (ggplot(df, aes("x")) + geom_dots(smooth=smooth)).draw().savefig(io.BytesIO())
    with pytest.raises(ValueError, match="smooth"):
        (ggplot(df, aes("x")) + geom_dots(smooth="wobbly")).draw().savefig(io.BytesIO())


# bar layout ---------------------------------------------------------------------------


def test_bar_layout_fills_rectangles():
    v = np.repeat([0.0, 10.0], [7, 3])
    x, off = layout_dots(v, 1.0, "bar", "top")
    # 9 columns fit in 90% of the gap of 10; 7 dots: one row
    assert len(np.unique(off[v == 0])) == 1
    np.testing.assert_allclose(np.sort(x[v == 0]), np.arange(7) - 3)
    p = np.column_stack([x, off])
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 1 - 1e-9


def test_bar_layout_rows_and_order():
    v = np.r_[np.zeros(5), 2.0]  # resolution 2 -> one column (0.9 * 2 / 1.5 = 1.2)
    x, off = layout_dots(v, 1.5, "bar", "top", order=[4, 3, 2, 1, 0, 0])
    np.testing.assert_allclose(off[:5], [6.75, 5.25, 3.75, 2.25, 0.75])


def test_bar_binwidth_fits():
    v = rng.poisson(2, 1000).astype(float) * 10
    w = find_binwidth(v, 40, layout="bar")
    _, off = layout_dots(v, w, "bar", "top")
    assert layout_height(off, w, "top") <= 40 + 1e-9
    assert w <= 0.9 * 10


# shapes --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "marker", "mode"),
    [(1, "o", "open"), (16, "o", "solid"), (21, "o", "fill"), (22.0, "s", "fill"), ("19", "o", "solid"), ("D", "D", "fill")],
)
def test_resolve_shape(shape, marker, mode):
    m, md, _ = resolve_shape(shape)
    assert (m, md) == (marker, mode)


def test_pch_20_is_small_and_bad_codes_fail():
    assert resolve_shape(20)[2] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        resolve_shape(26)


def test_colours_for_modes():
    fill, col = (1, 0, 0, 1), (0, 0, 1, 1)
    assert colours_for("fill", fill, col) == (fill, col)
    assert colours_for("solid", fill, col) == (col, col)
    assert colours_for("open", fill, col)[0][3] == 0


# blurs ---------------------------------------------------------------------------------


def test_blur_gaussian_is_the_blurred_disc():
    x = np.array([0.0, 0.5, 1.0])
    expected = stats.norm.cdf((x + 0.5) / 0.2) - stats.norm.cdf((x - 0.5) / 0.2)
    np.testing.assert_allclose(blur_gaussian(x, 0.5, 0.2), expected)
    np.testing.assert_allclose(blur_gaussian(x, 0.5, 0), [1, 1, 0])


def test_blur_interval():
    f = blur_interval(width=0.95)
    np.testing.assert_allclose(f(np.array([0.2, 0.8, 2.5]), 0.5, 1.0), [1, 0.5, 0])


# MCSE ----------------------------------------------------------------------------------


def test_ess_iid_and_autocorrelated():
    assert ess_basic(rng.normal(size=4000)) == pytest.approx(4000, rel=0.15)
    ar = np.zeros(8000)
    e = rng.normal(size=8000)
    for t in range(1, 8000):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    # AR(1): ESS = n (1 - phi) / (1 + phi)
    assert ess_basic(ar) == pytest.approx(8000 * 0.2 / 1.8, rel=0.2)


def test_mcse_quantile_matches_theory_for_iid_normal():
    # MCSE estimates are themselves noisy: compare the average over many
    # quantiles with the asymptotic standard error of a sample quantile
    x = rng.normal(size=4000)
    p = np.linspace(0.2, 0.8, 13)
    theory = np.sqrt(p * (1 - p) / len(x)) / stats.norm.pdf(stats.norm.ppf(p))
    assert np.mean(mcse_quantile(x, p) / theory) == pytest.approx(1, abs=0.15)


def test_stat_mcse_dots_computes_se(layer_data):
    df = pd.DataFrame({"x": rng.normal(size=2000)})
    d = layer_data(ggplot(df, aes("x")) + stat_mcse_dots(quantiles=50))
    assert len(d) == 50
    assert (d["se"] > 0).all()
    np.testing.assert_allclose(d["sd"], d["se"])
    # tails are less certain than the middle
    s = d.sort_values("x")["se"].to_numpy()
    assert min(s[:3].mean(), s[-3:].mean()) > 1.5 * s[20:30].mean()


def test_geom_blur_dots_draws():
    df = pd.DataFrame({"q": np.linspace(-2, 2, 40), "se": np.linspace(0.01, 0.3, 40)})
    for blur in ("gaussian", "interval", blur_interval(width=0.5)):
        fig = (ggplot(df, aes(x="q", sd="se")) + geom_blur_dots(blur=blur)).draw()
        fig.savefig(io.BytesIO())
