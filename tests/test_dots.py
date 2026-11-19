"""
Tests for dot plots: the layout core, the stats and the geoms (no image
comparison; see test_examples.py for those)
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, after_stat, coord_flip, ggplot
from plotnine.exceptions import PlotnineError
from scipy import stats

from plotnine_ggdist import (
    bin_dots,
    find_dotplot_binwidth,
    geom_dots,
    geom_dotsinterval,
    geom_swarm,
    stat_dots,
    stat_dotsinterval,
    unit,
)
from plotnine_ggdist._core.dots import (
    layout_dots,
    layout_height,
    nudge_apart,
    ppoints,
    wilkinson_bins,
)

rng = np.random.default_rng(99)
NORMAL = rng.normal(size=300)


def min_gap(x, y):
    p = np.column_stack([x, y])
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min()


# core -------------------------------------------------------------------------------------


def test_ppoints_matches_r():
    # R: ppoints(5); ppoints(20)[1:2]
    np.testing.assert_allclose(ppoints(5), [0.1190476, 0.3095238, 0.5, 0.6904762, 0.8809524], atol=1e-7)
    np.testing.assert_allclose(ppoints(20)[:2], [0.025, 0.075])


def test_bins_are_within_one_width_and_symmetric():
    x = np.sort(stats.norm.ppf(ppoints(40)))
    bins, centers = wilkinson_bins(x, 0.4)
    for b in range(len(centers)):
        members = x[bins == b]
        assert np.ptp(members) < 0.4
    np.testing.assert_allclose(centers, -centers[::-1])


def test_nudge_apart_is_minimal_and_separates():
    p = np.array([0.0, 0.5, 0.6, 3.0])
    out = nudge_apart(p, 1.0)
    assert np.all(np.diff(out) >= 1 - 1e-12)
    # the far point does not move; the cluster spreads around its mean
    assert out[3] == 3.0
    assert np.mean(out[:3]) == pytest.approx(np.mean(p[:3]))


@pytest.mark.parametrize("layout", ["bin", "weave", "hex", "swarm"])
@pytest.mark.parametrize("side", ["top", "bottom", "both"])
def test_layouts_do_not_overlap(layout, side):
    x, off = layout_dots(NORMAL, 0.15, layout, side)
    assert min_gap(x, off) >= 0.15 * (1 - 1e-6)
    if side == "top":
        assert off.min() >= 0.075 - 1e-9
    elif side == "bottom":
        assert off.max() <= -0.075 + 1e-9


def test_bin_layout_of_symmetric_data_is_symmetric():
    v = stats.norm.ppf(ppoints(51))
    x, off = layout_dots(v, 0.3, "bin", "top")
    pts = sorted(zip(np.round(x, 9), np.round(off, 9), strict=True))
    mirrored = sorted(zip(np.round(-x, 9), np.round(off, 9), strict=True))
    assert pts == mirrored


def test_weave_keeps_values_when_there_is_room():
    v = np.array([0.0, 0.05, 1.0, 1.02, 2.0])
    x, _ = layout_dots(v, 0.1, "weave", "top")
    np.testing.assert_allclose(x, v)


def test_hex_offsets_alternate():
    v = np.zeros(4)
    x, off = layout_dots(v, 1.0, "hex", "top")
    np.testing.assert_allclose(sorted(x), [-0.25, -0.25, 0.25, 0.25])
    np.testing.assert_allclose(sorted(off), [0.5, 1.5, 2.5, 3.5])


def test_order_sets_stacking():
    v = np.zeros(3)
    _, off = layout_dots(v, 1.0, "bin", "top", order=[2, 0, 1])
    np.testing.assert_allclose(off, [2.5, 0.5, 1.5])


def test_stackratio():
    _, off = layout_dots(np.zeros(3), 1.0, "bin", "top", stackratio=0.5)
    np.testing.assert_allclose(sorted(off), [0.5, 1.0, 1.5])


@pytest.mark.parametrize("layout", ["bin", "swarm"])
def test_find_binwidth_fits(layout):
    w = find_dotplot_binwidth(NORMAL, maxheight=1.0, heightratio=1, layout=layout)
    x, off = layout_dots(NORMAL, w, layout, "top")
    assert layout_height(off, w, "top") <= 1.0 + 1e-9
    if layout == "bin":
        # and the next size up would not
        x2, off2 = layout_dots(NORMAL, w * 1.05, layout, "top")
        assert layout_height(off2, w * 1.05, "top") > 1.0


def test_bin_dots_api():
    x = stats.norm.ppf(ppoints(20))
    out = bin_dots(x, y=0, binwidth=0.5)
    assert out.columns.tolist() == ["x", "y", "bin"]
    assert (out["y"] > 0).all()
    vertical = bin_dots(x, y=1, binwidth=0.5, orientation="vertical", side="both")
    np.testing.assert_allclose(sorted(vertical["y"]), sorted(out["x"]))
    assert vertical["x"].mean() == pytest.approx(1)


def test_unit():
    assert unit(72.27, "pt").to_pixels(100, 500) == pytest.approx(100)
    assert unit(25.4, "mm").to_pixels(100, 500) == pytest.approx(100)
    assert unit(0.1, "npc").to_pixels(100, 500) == pytest.approx(50)
    with pytest.raises(ValueError):
        unit(1, "furlong")


# stats ----------------------------------------------------------------------------------

DF = pd.DataFrame({"g": np.tile(["a", "b"], 150), "v": rng.normal(np.tile([0.0, 3.0], 150))})


def test_stat_dots_one_dot_per_draw(layer_data):
    d = layer_data(ggplot(DF, aes("v", "g")) + stat_dots())
    assert (d["datatype"] == "slab").all()
    assert len(d) == len(DF)
    np.testing.assert_allclose(np.sort(d["x"]), np.sort(DF["v"]))


def test_quantile_dotplot_of_samples(layer_data):
    d = layer_data(ggplot(DF, aes("v", "g")) + stat_dots(quantiles=20))
    assert d.groupby("group").size().tolist() == [20, 20]
    a = DF.loc[DF["g"] == "a", "v"]
    np.testing.assert_allclose(np.sort(d.loc[d["group"] == 1, "x"]), np.quantile(a, ppoints(20)))


def test_quantile_dotplot_of_distributions(layer_data):
    df = pd.DataFrame({"g": ["a"], "d": [stats.norm(0, 1)]})
    default = layer_data(ggplot(df, aes(y="g", xdist="d")) + stat_dots())
    assert len(default) == 100
    np.testing.assert_allclose(np.sort(default["x"]), stats.norm.ppf(ppoints(100)))
    fifty = layer_data(ggplot(df, aes(y="g", xdist="d")) + stat_dots(quantiles=50))
    assert len(fifty) == 50


def test_dotsinterval_has_intervals_and_dot_widths(layer_data):
    d = layer_data(ggplot(DF, aes("v", "g")) + stat_dotsinterval(aes(fill=after_stat("level"))))
    assert set(d["datatype"]) == {"slab", "interval"}
    dots = d[d["datatype"] == "slab"]
    assert set(dots[".width"].dropna()) == {0.66, 0.95}
    assert dots["fill"].nunique() == 3  # two levels + outside both (NaN)


def test_continuous_position_gets_one_distribution_per_value(layer_data):
    df = pd.DataFrame({"x": np.repeat([1, 2, 3], 100), "y": rng.normal(np.repeat([1, 2, 3], 100))})
    d = layer_data(ggplot(df, aes("x", "y")) + stat_dotsinterval())
    iv = d[d["datatype"] == "interval"]
    assert sorted(iv["x"].unique()) == [1, 2, 3]
    assert len(iv) == 3 * 2


# geoms ----------------------------------------------------------------------------------


def _dots_artist(plot):
    from plotnine_ggdist.geom_dotsinterval import _DotsArtist

    import io

    fig = plot.draw()
    fig.savefig(io.BytesIO(), format="png")
    artists = [a for ax in fig.axes for a in ax.artists if isinstance(a, _DotsArtist)]
    return fig, artists


def test_auto_binwidth_fits_the_space():
    fig, (artist,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots())
    w = artist.binwidth_px
    assert w > 0
    for g in artist.groups:
        v_px = artist._to_px(g["values"], g["region"][0])
        lo, hi = artist._other_px(g["values"][0], np.array(g["region"]))
        _, off = layout_dots(v_px, w, "bin", g["side"])
        assert layout_height(off, w, g["side"]) <= abs(hi - lo) * (1 + 1e-6)


def test_fixed_binwidth_units():
    fig, (artist,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots(binwidth=unit(2, "mm"), overflow="keep"))
    assert artist.binwidth_px == pytest.approx(2 / 25.4 * fig.dpi)


def test_binwidth_in_data_units():
    fig, (artist,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots(binwidth=0.1, overflow="keep"))
    px_per_unit = abs(artist._to_px(np.array([1.0]), 0)[0] - artist._to_px(np.array([0.0]), 0)[0])
    assert artist.binwidth_px == pytest.approx(0.1 * px_per_unit)


def test_overflow_warns():
    with pytest.warns(UserWarning, match="overflow"):
        _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots(binwidth=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots(binwidth=1.0, overflow="compress"))


def test_swarm_defaults():
    fig, (artist,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_swarm())
    assert artist.params["layout"] == "swarm"
    assert {g["side"] for g in artist.groups} == {"both"}
    assert artist.binwidth_px == pytest.approx(1.5 / 25.4 * fig.dpi)


def test_orientation_and_coord_flip():
    _, (h,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots())
    _, (v,) = _dots_artist(ggplot(DF, aes("g", "v")) + geom_dots())
    _, (f,) = _dots_artist(ggplot(DF, aes("v", "g")) + geom_dots() + coord_flip())
    assert h.value_on_x and not v.value_on_x and not f.value_on_x


def test_geom_dotsinterval_with_precomputed_data():
    from plotnine_ggdist import median_qi

    iv = median_qi(DF, "v", by="g", width=[0.66, 0.95]).rename(columns={"v": "v_point"})
    p = (
        ggplot(DF, aes("v", "g"))
        + geom_dots()
        + geom_dotsinterval(aes(x="v_point", xmin=".lower", xmax=".upper", y="g"), data=iv)
    )
    p.draw()


def test_bad_parameters():
    with pytest.raises(PlotnineError, match="layout"):
        (ggplot(DF, aes("v", "g")) + geom_dots(layout="spiral")).draw()
    with pytest.raises(PlotnineError, match="overflow"):
        (ggplot(DF, aes("v", "g")) + geom_dotsinterval(overflow="explode")).draw()
