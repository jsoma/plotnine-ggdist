"""
Point and interval summaries

Python counterparts of ggdist's `point_interval()` family: point summaries
(`mean`, `median`, `Mode`) and interval functions (`qi`, `hdi`, `hdci`,
`ll`, `ul`) for samples or frozen scipy.stats distributions, and the
`point_interval()` data-frame interface with its shortcuts (`median_qi`,
`mean_hdi`, ...).

This module does not import plotnine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from .density import density_bounded, weighted_quantile
from .dist import dist_cdf, dist_limits, dist_pdf, dist_quantile, is_discrete, is_dist

__all__ = (
    "Mode",
    "qi",
    "hdi",
    "hdci",
    "ll",
    "ul",
    "point_interval",
    "POINTS",
    "INTERVALS",
)

EXCLUDE = (".chain", ".iteration", ".draw", ".row")


# Point summaries --------------------------------------------------------------


def _mean(x: Any, weights: Any | None = None) -> float:
    if is_dist(x):
        return float(x.mean())
    x = np.asarray(x, dtype=float)
    return float(np.average(x, weights=weights))


def _median(x: Any, weights: Any | None = None) -> float:
    if is_dist(x):
        return float(x.median())
    return float(weighted_quantile(x, 0.5, weights))


def Mode(x: Any, weights: Any | None = None) -> float:
    """
    The mode of a sample or distribution

    For samples, the maximum of a bounded kernel density estimate (for
    integer-valued samples, the most frequent value). For distributions,
    the maximum of the density (or mass) function.
    """
    if is_dist(x):
        lo, hi = dist_limits(x)
        if is_discrete(x):
            k = np.arange(np.floor(lo), np.ceil(hi) + 1)
            return float(k[np.argmax(dist_pdf(x, k))])
        grid = np.linspace(lo, hi, 2001)
        i = int(np.argmax(dist_pdf(x, grid)))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if a == b:
            return float(grid[i])
        res = optimize.minimize_scalar(
            lambda v: -float(dist_pdf(x, np.array([v]))[0]),
            bounds=(a, b),
            method="bounded",
        )
        return float(res.x)

    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    x = x[finite]
    w = None if weights is None else np.asarray(weights, dtype=float)[finite]
    if len(x) == 0:
        return float("nan")
    if len(np.unique(x)) == 1:
        return float(x[0])
    if np.all(x == np.round(x)):
        vals, inv = np.unique(x, return_inverse=True)
        counts = np.bincount(inv, weights=w)
        return float(vals[np.argmax(counts)])
    d = density_bounded(x, weights=w, n=2001)
    return float(d.x[np.argmax(d.pdf)])


POINTS: dict[str, Callable[..., float]] = {
    "mean": _mean,
    "median": _median,
    "mode": Mode,
}


# Interval functions -----------------------------------------------------------
#
# Each takes (x, width, weights) and returns an (k, 2) array of intervals:
# k = 1 except for hdi of multimodal distributions.


def qi(x: Any, width: float = 0.95, weights: Any | None = None) -> np.ndarray:
    """
    Quantile (equal-tailed) interval

    Returns
    -------
    numpy.ndarray
        Array of shape (1, 2): ``[[lower, upper]]``.
    """
    p = [(1 - width) / 2, (1 + width) / 2]
    if is_dist(x):
        return dist_quantile(x, p).reshape(1, 2)
    return np.asarray(weighted_quantile(x, p, weights)).reshape(1, 2)


def ll(x: Any, width: float = 0.95, weights: Any | None = None) -> np.ndarray:
    """
    Lower limit: ``[[q(1 - width), inf]]``
    """
    lo = qi(x, 1 - 2 * (1 - width), weights)[0, 0] if width < 1 else -np.inf
    return np.array([[lo, np.inf]])


def ul(x: Any, width: float = 0.95, weights: Any | None = None) -> np.ndarray:
    """
    Upper limit: ``[[-inf, q(width)]]``
    """
    hi = qi(x, 1 - 2 * (1 - width), weights)[0, 1] if width < 1 else np.inf
    return np.array([[-np.inf, hi]])


def hdci(x: Any, width: float = 0.95, weights: Any | None = None) -> np.ndarray:
    """
    Highest-density continuous interval (shortest probability interval)

    For multimodal distributions this may not be a highest-density
    interval; see [](`~plotnine_ggdist.hdi`).
    """
    if is_dist(x):
        if width >= 1:
            return np.array([dist_limits(x, (0, 1))])

        def length(a: float) -> float:
            q = dist_quantile(x, [a, a + width])
            return float(q[1] - q[0])

        # coarse grid over the lower-tail probability, then refine
        grid = np.linspace(0, 1 - width, 201)[1:-1] if width < 1 else np.array([0.0])
        lens = np.array([length(a) for a in grid])
        i = int(np.nanargmin(lens))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = optimize.minimize_scalar(length, bounds=(a, b), method="bounded")
        a = float(res.x) if res.fun <= lens[i] else float(grid[i])
        return dist_quantile(x, [a, a + width]).reshape(1, 2)

    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    x = x[finite]
    if weights is not None:
        w = np.asarray(weights, dtype=float)[finite]
        order = np.argsort(x, kind="stable")
        xs, ws = x[order], w[order] / w.sum()
        cw = np.concatenate([[0], np.cumsum(ws)])
        best = (np.inf, xs[0], xs[-1])
        j = 0
        for i in range(len(xs)):
            while j < len(xs) and cw[j + 1] - cw[i] < width - 1e-12:
                j += 1
            if j >= len(xs):
                break
            if xs[j] - xs[i] < best[0]:
                best = (xs[j] - xs[i], xs[i], xs[j])
        return np.array([[best[1], best[2]]])

    xs = np.sort(x)
    n = len(xs)
    if n == 0:
        return np.array([[np.nan, np.nan]])
    m = min(n, max(1, int(np.ceil(width * n))))
    spans = xs[m - 1 :] - xs[: n - m + 1]
    i = int(np.argmin(spans))
    return np.array([[xs[i], xs[i + m - 1]]])


def hdi(
    x: Any,
    width: float = 0.95,
    weights: Any | None = None,
    n: int = 4096,
) -> np.ndarray:
    """
    Highest-density interval(s)

    The region of highest density containing `width` of the probability.
    For a multimodal distribution this can be several disjoint intervals,
    returned one per row. For samples, the density is a bounded KDE.
    """
    if width >= 1:
        if is_dist(x):
            return np.array([dist_limits(x, (0, 1))])
        x = np.asarray(x, dtype=float)
        return np.array([[np.nanmin(x), np.nanmax(x)]])

    if is_dist(x):
        if is_discrete(x):
            return _hdi_discrete(x, width)
        lo, hi = dist_limits(x, (1e-6, 1 - 1e-6))
        grid = np.linspace(lo, hi, n)
        pdf = dist_pdf(x, grid)

        def cdf(v: np.ndarray) -> np.ndarray:
            return dist_cdf(x, v)

    else:
        xv = np.asarray(x, dtype=float)
        finite = np.isfinite(xv)
        w = None if weights is None else np.asarray(weights, dtype=float)[finite]
        xv = xv[finite]
        if len(np.unique(xv)) < 2:
            return np.array([[xv[0], xv[0]]])
        d = density_bounded(xv, weights=w, n=n)
        grid, pdf = d.x, d.pdf
        dcdf = d.cdf

        def cdf(v: np.ndarray) -> np.ndarray:
            return np.interp(v, grid, dcdf)

    def regions(t: float) -> np.ndarray:
        above = pdf >= t
        if not above.any():
            i = int(np.argmax(pdf))
            return np.array([[grid[i], grid[i]]])
        edges = np.diff(above.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0])
        if above[0]:
            starts.insert(0, 0)
        if above[-1]:
            ends.append(len(grid) - 1)
        out = []
        for s, e in zip(starts, ends, strict=True):
            out.append([_cross(grid, pdf, t, s, left=True), _cross(grid, pdf, t, e, left=False)])
        return np.array(out)

    def mass(t: float) -> float:
        r = regions(t)
        return float(np.sum(cdf(r[:, 1]) - cdf(r[:, 0])))

    lo_t, hi_t = 0.0, float(np.max(pdf))
    for _ in range(60):
        mid = (lo_t + hi_t) / 2
        if mass(mid) > width:
            lo_t = mid
        else:
            hi_t = mid
    return regions(lo_t)


def _cross(grid: np.ndarray, pdf: np.ndarray, t: float, i: int, left: bool) -> float:
    """
    Linearly interpolate where pdf crosses t next to grid index i
    """
    j = i - 1 if left else i + 1
    if j < 0 or j >= len(grid):
        return float(grid[i])
    p0, p1 = pdf[j], pdf[i]
    if p1 == p0:
        return float(grid[i])
    f = (t - p0) / (p1 - p0)
    return float(grid[j] + f * (grid[i] - grid[j]))


def _hdi_discrete(dist: Any, width: float) -> np.ndarray:
    lo, hi = dist_limits(dist, (1e-9, 1 - 1e-9))
    k = np.arange(np.floor(lo), np.ceil(hi) + 1)
    pmf = dist_pdf(dist, k)
    order = np.argsort(-pmf, kind="stable")
    keep = order[: int(np.searchsorted(np.cumsum(pmf[order]), width) + 1)]
    sel = np.sort(k[keep])
    # split into runs of consecutive integers
    breaks = np.where(np.diff(sel) > 1)[0]
    runs = np.split(sel, breaks + 1)
    return np.array([[r[0], r[-1]] for r in runs], dtype=float)


INTERVALS: dict[str, Callable[..., np.ndarray]] = {
    "qi": qi,
    "hdi": hdi,
    "hdci": hdci,
    "ll": ll,
    "ul": ul,
}


# point_interval ---------------------------------------------------------------


def _resolve(fn: str | Callable, table: dict[str, Callable], what: str) -> tuple[Callable, str]:
    if callable(fn):
        return fn, getattr(fn, "__name__", what)
    try:
        return table[fn.lower()], fn.lower()
    except KeyError as e:
        raise ValueError(f"Unknown {what} {fn!r}; use one of {sorted(table)} or a function") from e


def _call_point(fn: Callable, x: Any, weights: Any | None) -> float:
    if weights is None:
        return float(fn(x))
    try:
        return float(fn(x, weights=weights))
    except TypeError:
        return float(fn(x))


def _call_interval(fn: Callable, x: Any, width: float, weights: Any | None) -> np.ndarray:
    if weights is None:
        out = fn(x, width)
    else:
        try:
            out = fn(x, width, weights=weights)
        except TypeError:
            out = fn(x, width)
    return np.atleast_2d(np.asarray(out, dtype=float))


def summarise(
    x: Any,
    width: float | Sequence[float] = 0.95,
    point: str | Callable = "median",
    interval: str | Callable = "qi",
    weights: Any | None = None,
) -> pd.DataFrame:
    """
    Summarise one sample or distribution: columns y, ymin, ymax, .width

    One row per interval (several per width for a multimodal hdi).
    """
    point_fn, point_name = _resolve(point, POINTS, "point summary")
    interval_fn, interval_name = _resolve(interval, INTERVALS, "interval")
    widths = np.atleast_1d(np.asarray(width, dtype=float))
    if not is_dist(x):
        x = np.asarray(x, dtype=float)
    y = _call_point(point_fn, x, weights)
    rows = []
    for w in widths:
        for lo, hi in _call_interval(interval_fn, x, float(w), weights):
            rows.append((y, lo, hi, float(w)))
    out = pd.DataFrame(rows, columns=["y", "ymin", "ymax", ".width"])
    out[".point"] = point_name
    out[".interval"] = interval_name
    return out


def point_interval(
    data: Any,
    *columns: str,
    width: float | Sequence[float] = 0.95,
    point: str | Callable = "median",
    interval: str | Callable = "qi",
    by: str | Sequence[str] | None = None,
    simple_names: bool = True,
    na_rm: bool = False,
    exclude: Sequence[str] = EXCLUDE,
) -> pd.DataFrame:
    """
    Point and interval summaries of samples or distributions

    Parameters
    ----------
    data :
        One of

        - a DataFrame: summarise `columns` (default: every numeric or
          distribution column not in `by` or `exclude`), within groups
          given by `by`;
        - a pandas ``DataFrameGroupBy``: as above, grouped;
        - a 1-d array-like of draws, or a frozen scipy.stats distribution:
          returns columns ``y``, ``ymin``, ``ymax``.
    *columns :
        Columns of `data` to summarise.
    width :
        Probability mass of the interval(s); one output row per width.
        ggdist's ``.width``.
    point :
        ``"median"``, ``"mean"``, ``"mode"`` or a function of a sample.
    interval :
        ``"qi"``, ``"hdi"``, ``"hdci"``, ``"ll"``, ``"ul"`` or a function
        ``f(x, width)`` returning ``[[lower, upper], ...]``.
    by :
        Grouping column(s).
    simple_names :
        With a single summarised column ``x``, name the interval columns
        ``.lower``/``.upper`` (otherwise ``x.lower``/``x.upper``).
    na_rm :
        Drop missing values before summarising (otherwise they propagate
        to NaN results).
    exclude :
        Columns never summarised automatically (MCMC bookkeeping).

    Returns
    -------
    pandas.DataFrame
        The group columns, the point summary (named after the column),
        the interval ends, ``.width``, ``.point`` and ``.interval``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from plotnine_ggdist import median_qi
    >>> df = pd.DataFrame({"g": ["a"] * 500 + ["b"] * 500,
    ...                    "x": np.random.default_rng(1).normal(size=1000)})
    >>> median_qi(df, "x", by="g", width=[0.66, 0.95])  # doctest: +SKIP
    """
    point_fn, point_name = _resolve(point, POINTS, "point summary")
    interval_fn, interval_name = _resolve(interval, INTERVALS, "interval")

    if isinstance(data, pd.core.groupby.DataFrameGroupBy):
        keys = data.keys
        by_cols = [keys] if isinstance(keys, str) else list(keys)
        data = data.obj
        if by is not None:
            raise ValueError("Pass either a grouped DataFrame or `by`, not both")
        by = by_cols

    if not isinstance(data, pd.DataFrame):
        if columns:
            raise ValueError("`columns` only apply when `data` is a DataFrame")
        x = data
        if not is_dist(x):
            x = np.asarray(x, dtype=float)
            if na_rm:
                x = x[np.isfinite(x)]
        return summarise(x, width, point_fn, interval_fn).assign(
            **{".point": point_name, ".interval": interval_name}
        )

    by_list = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if columns:
        cols = list(columns)
    else:
        cols = [
            c
            for c in data.columns
            if c not in by_list
            and c not in exclude
            and (pd.api.types.is_numeric_dtype(data[c]) or _is_dist_col(data[c]))
        ]
    if not cols:
        raise ValueError("No columns to summarise")
    simple = simple_names and len(cols) == 1

    groups = data.groupby(by_list, sort=True, observed=True, dropna=False) if by_list else [((), data)]
    pieces = []
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        out = None
        for c in cols:
            part = _summarise_column(sub[c], width, point_fn, interval_fn, na_rm)
            names = (
                {"y": c, "ymin": ".lower", "ymax": ".upper"}
                if simple
                else {"y": c, "ymin": f"{c}.lower", "ymax": f"{c}.upper"}
            )
            part = part.rename(columns=names)
            if out is None:
                out = part
            else:
                if len(part) != len(out):
                    raise ValueError(
                        "Columns produced different numbers of intervals "
                        "(e.g. a multimodal hdi); summarise them separately"
                    )
                out = pd.concat([out, part.drop(columns=".width").reset_index(drop=True)], axis=1)
        assert out is not None
        for k, v in zip(by_list, key, strict=False):
            out.insert(by_list.index(k), k, v)
        pieces.append(out)

    res = pd.concat(pieces, ignore_index=True)
    res = res[[c for c in res.columns if c != ".width"] + [".width"]]
    res[".point"] = point_name
    res[".interval"] = interval_name
    return res


def _is_dist_col(s: pd.Series) -> bool:
    return s.dtype == object and len(s) > 0 and all(is_dist(v) for v in s)


def _summarise_column(
    s: pd.Series, width: Any, point_fn: Callable, interval_fn: Callable, na_rm: bool
) -> pd.DataFrame:
    if _is_dist_col(s):
        if len(s) != 1:
            # one row per distribution
            return pd.concat(
                [summarise(d, width, point_fn, interval_fn) for d in s], ignore_index=True
            )[["y", "ymin", "ymax", ".width"]]
        return summarise(s.iloc[0], width, point_fn, interval_fn)[["y", "ymin", "ymax", ".width"]]
    x = s.to_numpy(dtype=float)
    if na_rm:
        x = x[np.isfinite(x)]
    elif np.isnan(x).any():
        widths = np.atleast_1d(np.asarray(width, dtype=float))
        return pd.DataFrame({"y": np.nan, "ymin": np.nan, "ymax": np.nan, ".width": widths})
    return summarise(x, width, point_fn, interval_fn)[["y", "ymin", "ymax", ".width"]]


def _make_shortcut(point: str, interval: str) -> Callable[..., pd.DataFrame]:
    fn = partial(point_interval, point=point, interval=interval)

    def shortcut(data: Any, *columns: str, **kwargs: Any) -> pd.DataFrame:
        return fn(data, *columns, **kwargs)

    shortcut.__name__ = f"{point}_{interval}"
    shortcut.__qualname__ = shortcut.__name__
    shortcut.__doc__ = (
        f"{point.capitalize()} with {interval} interval(s)\n\n"
        f"Shortcut for ``point_interval(..., point={point!r}, interval={interval!r})``; "
        "see [](`~plotnine_ggdist.point_interval`)."
    )
    return shortcut


SHORTCUTS: dict[str, Callable[..., pd.DataFrame]] = {
    f"{p}_{i}": _make_shortcut(p, i) for p in POINTS for i in INTERVALS
}


def parse_point_interval(spec: Any) -> tuple[Callable, Callable, str, str]:
    """
    Resolve a stat's `point_interval` argument

    Accepts ``"median_qi"``-style names or a ``(point, interval)`` pair of
    names / functions.
    """
    if isinstance(spec, str):
        try:
            p, i = spec.lower().split("_", 1)
        except ValueError as e:
            raise ValueError(f"Unknown point_interval {spec!r}; e.g. 'median_qi'") from e
        spec = (p, i)
    if isinstance(spec, tuple) and len(spec) == 2:
        pf, pn = _resolve(spec[0], POINTS, "point summary")
        inf, inn = _resolve(spec[1], INTERVALS, "interval")
        return pf, inf, pn, inn
    raise ValueError(
        "point_interval must be a name like 'median_qi' or a (point, interval) tuple"
    )
