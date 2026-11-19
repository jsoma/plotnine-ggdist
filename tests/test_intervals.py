"""
Tests for point and interval summaries (no plotting)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from plotnine_ggdist import (
    Mode,
    hdci,
    hdi,
    ll,
    mean_hdi,
    mean_qi,
    median_qi,
    mode_hdi,
    point_interval,
    qi,
    ul,
)

rng = np.random.default_rng(2026)
NORMAL = rng.normal(size=20_000)
BIMODAL = np.r_[rng.normal(0, 1, 10_000), rng.normal(6, 1, 10_000)]


# interval functions -------------------------------------------------------------


def test_qi_matches_numpy_quantile():
    out = qi(NORMAL, 0.9)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], np.quantile(NORMAL, [0.05, 0.95]))


def test_qi_of_distribution_uses_ppf():
    np.testing.assert_allclose(qi(stats.norm(), 0.95)[0], [-1.959964, 1.959964], atol=1e-6)


def test_ll_and_ul():
    d = stats.norm()
    np.testing.assert_allclose(ll(d, 0.95)[0], [d.ppf(0.05), np.inf])
    np.testing.assert_allclose(ul(d, 0.95)[0], [-np.inf, d.ppf(0.95)])


def test_hdci_is_shortest_window():
    x = rng.exponential(size=5000)
    lo, hi = hdci(x, 0.8)[0]
    inside = np.mean((x >= lo) & (x <= hi))
    assert inside == pytest.approx(0.8, abs=1e-3)
    # shorter than the equal-tailed interval for a skewed sample
    q = qi(x, 0.8)[0]
    assert hi - lo < q[1] - q[0]
    # exponential: the shortest 80% interval starts at the minimum
    assert lo == pytest.approx(x.min(), abs=0.01)


def test_hdci_of_distribution():
    d = stats.gamma(2)
    lo, hi = hdci(d, 0.9)[0]
    assert d.cdf(hi) - d.cdf(lo) == pytest.approx(0.9, abs=1e-6)
    # at the shortest interval the density is equal at both ends
    assert d.pdf(lo) == pytest.approx(d.pdf(hi), rel=1e-3)


def test_hdi_unimodal_matches_hdci():
    d = stats.gamma(3)
    np.testing.assert_allclose(hdi(d, 0.9), hdci(d, 0.9), atol=2e-3)


def test_hdi_symmetric_distribution_matches_qi():
    np.testing.assert_allclose(hdi(stats.norm(), 0.95), qi(stats.norm(), 0.95), atol=1e-3)


def test_hdi_bimodal_sample_gives_two_intervals():
    out = hdi(BIMODAL, 0.8)
    assert out.shape == (2, 2)
    assert out[0, 1] < 3 < out[1, 0]
    mass = sum(np.mean((BIMODAL >= lo) & (BIMODAL <= hi)) for lo, hi in out)
    assert mass == pytest.approx(0.8, abs=0.01)
    # hdci gives one interval covering both modes
    assert hdci(BIMODAL, 0.8).shape == (1, 2)


def test_hdi_discrete_distribution():
    out = hdi(stats.poisson(3), 0.9)
    assert out.tolist() == [[1.0, 6.0]]


def test_weighted_qi():
    x = np.array([0.0, 1.0])
    w = np.array([1.0, 3.0])
    # weighted median between the two values, closer to 1
    out = qi(np.repeat(x, [1000, 3000]), 0.5)
    lo, hi = qi(x, 0.5, weights=w)[0]
    assert 0 <= lo <= hi <= 1
    assert out[0, 1] == 1


# point summaries ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dist", "mode"),
    [(stats.gamma(3), 2.0), (stats.beta(2, 5), 0.2), (stats.norm(1, 2), 1.0)],
)
def test_mode_of_distributions(dist, mode):
    assert Mode(dist) == pytest.approx(mode, abs=1e-4)


def test_mode_of_sample_and_integers():
    assert Mode(rng.normal(3, 1, 20_000)) == pytest.approx(3, abs=0.2)
    assert Mode(np.array([1, 2, 2, 3, 2, 5])) == 2


# point_interval ---------------------------------------------------------------------


def test_point_interval_vector_returns_y_ymin_ymax():
    out = point_interval(NORMAL, width=[0.66, 0.95])
    assert out.columns.tolist() == ["y", "ymin", "ymax", ".width", ".point", ".interval"]
    assert out[".width"].tolist() == [0.66, 0.95]
    assert out[".point"].unique().tolist() == ["median"]
    assert out[".interval"].unique().tolist() == ["qi"]


def test_point_interval_dataframe_simple_names():
    df = pd.DataFrame({"x": NORMAL})
    out = median_qi(df, "x", width=0.9)
    assert out.columns.tolist() == ["x", ".lower", ".upper", ".width", ".point", ".interval"]
    assert out["x"].iloc[0] == pytest.approx(np.median(NORMAL))


def test_point_interval_by_group_and_groupby_object():
    df = pd.DataFrame({"g": np.repeat(["a", "b"], 1000), "x": NORMAL[:2000]})
    by = median_qi(df, "x", by="g", width=[0.5, 0.95])
    grouped = df.groupby("g").pipe(median_qi, "x", width=[0.5, 0.95])
    pd.testing.assert_frame_equal(by, grouped)
    assert by["g"].tolist() == ["a", "a", "b", "b"]


def test_point_interval_multiple_columns():
    df = pd.DataFrame({"a": NORMAL[:1000], "b": NORMAL[1000:2000] * 2, ".draw": np.arange(1000)})
    out = mean_qi(df)  # all numeric columns except .draw
    assert out.columns.tolist() == [
        "a", "a.lower", "a.upper", "b", "b.lower", "b.upper", ".width", ".point", ".interval",
    ]


def test_point_interval_distribution_column():
    df = pd.DataFrame({"g": ["a", "b"], "d": [stats.norm(0, 1), stats.norm(5, 2)]})
    out = mean_qi(df, "d", by="g")
    np.testing.assert_allclose(out["d"], [0, 5])
    np.testing.assert_allclose(out[".upper"], [1.959964, 5 + 2 * 1.959964], atol=1e-5)


def test_point_interval_nan_handling():
    x = np.r_[NORMAL[:100], np.nan]
    df = pd.DataFrame({"x": x})
    assert np.isnan(median_qi(df, "x")["x"].iloc[0])
    assert np.isfinite(median_qi(df, "x", na_rm=True)["x"].iloc[0])


def test_multimodal_hdi_rows():
    out = mode_hdi(BIMODAL, width=0.8)
    assert len(out) == 2
    assert (out[".width"] == 0.8).all()


def test_custom_functions():
    out = point_interval(NORMAL, point=np.mean, interval=lambda x, w: [[x.min(), x.max()]])
    assert out["ymin"].iloc[0] == NORMAL.min()
    assert out[".point"].iloc[0] == "mean"


def test_bad_names_raise():
    with pytest.raises(ValueError, match="Unknown interval"):
        point_interval(NORMAL, interval="nope")
    with pytest.raises(ValueError, match="point summary"):
        point_interval(NORMAL, point="nope")


def test_mean_hdi_name():
    assert mean_hdi.__name__ == "mean_hdi"
