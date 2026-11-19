"""
Dot plot layout

Bins values into stacks of dots and lays them out, for dot plots and
quantile dot plots. Independent implementations of the layouts ggdist
documents:

- ``"bin"``: a Wilkinson (1999) dot-density binning. Bins are formed from
  the centre of the data outwards, so symmetric data give a symmetric plot,
  and overlapping bins can be nudged apart.
- ``"weave"``: the same rows as ``"bin"``, but each dot keeps its own
  position along the value axis.
- ``"hex"``: ``"bin"`` with alternate rows offset by a quarter bin width,
  for hexagonal packing with ``stackratio < 1``.
- ``"bar"``: for discrete values: the dots of each distinct value fill a
  rectangular bar, row by row, up to 90% of the gap between values wide.
- ``"swarm"``: a compact beeswarm; each dot goes as close to the baseline as
  it can without overlapping the dots already placed, taking next the dot
  that can sit closest (for more than 400 dots, dots are taken in order
  of value instead, which is much faster).

All positions are in one unit system (e.g. pixels, or data units with a
`heightratio` to convert): values along the value axis and offsets from the
baseline along the other axis. Nothing here imports plotnine.
"""

from __future__ import annotations

import heapq
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = (
    "ppoints",
    "dot_quantiles",
    "wilkinson_bins",
    "nudge_apart",
    "layout_dots",
    "layout_height",
    "find_binwidth",
    "bin_dots",
    "find_dotplot_binwidth",
)

Layout = Literal["bin", "weave", "hex", "swarm"]
Side = Literal["top", "bottom", "both"]
LAYOUTS = ("bin", "weave", "hex", "swarm", "bar")
BAR_FILL = 0.9  # fraction of the gap between values a "bar" may span
COMPACT_SWARM_MAX = 400  # above this many dots, "swarm" uses a one-pass order


def ppoints(n: int) -> np.ndarray:
    """
    Probability points for `n` quantiles, as R's ``ppoints()``

    ``(i - a) / (n + 1 - 2a)`` for ``i = 1..n``, with ``a = 3/8`` for
    ``n <= 10`` and ``1/2`` otherwise.
    """
    n = int(n)
    if n <= 0:
        return np.array([])
    a = 3 / 8 if n <= 10 else 1 / 2
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def dot_quantiles(x: Any, n: int, weights: Any | None = None) -> np.ndarray:
    """
    `n` quantiles of a sample or frozen scipy.stats distribution, for a
    quantile dot plot (Kay et al. 2016)
    """
    from .density import weighted_quantile
    from .dist import is_dist

    p = ppoints(n)
    if is_dist(x):
        return np.asarray(x.ppf(p), dtype=float)
    return np.asarray(weighted_quantile(x, p, weights), dtype=float)


# Binning ------------------------------------------------------------------------------


def wilkinson_bins(x_sorted: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Dot-density bins of sorted values, from the centre outwards

    A middle bin holds the values within ``width / 2`` of the centre of the
    data (the midpoint of the median pair). To its right each bin starts at
    the first value not yet binned and takes every value less than
    ``width`` above it; to its left the same is done going down. So
    symmetric data give mirror-image bins.

    Returns
    -------
    bin :
        Bin index (0, 1, ... from left to right) of each value.
    centers :
        Centre of each bin: the midpoint of its smallest and largest value.
    """
    x = np.asarray(x_sorted, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=int), np.array([])
    if not width > 0:
        raise ValueError("bin width must be positive")
    c = (x[(n - 1) // 2] + x[n // 2]) / 2
    m0 = int(np.searchsorted(x, c - width / 2, side="left"))
    m1 = int(np.searchsorted(x, c + width / 2, side="right"))
    right: list[tuple[int, int]] = []
    i = m1
    while i < n:
        j = max(int(np.searchsorted(x, x[i] + width, side="left")), i + 1)
        right.append((i, j))
        i = j
    left: list[tuple[int, int]] = []
    k = m0  # exclusive end
    while k > 0:
        s = min(int(np.searchsorted(x, x[k - 1] - width, side="right")), k - 1)
        left.append((s, k))
        k = s
    ranges = left[::-1] + ([(m0, m1)] if m1 > m0 else []) + right
    bins = np.empty(n, dtype=int)
    centers = np.empty(len(ranges))
    for b, (s, e) in enumerate(ranges):
        bins[s:e] = b
        centers[b] = (x[s] + x[e - 1]) / 2
    return bins, centers


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted isotonic (non-decreasing) regression, pool-adjacent-violators
    """
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            v1, w1, l1 = vals.pop(), wts.pop(), lens.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww)
            wts.append(ww)
            lens.append(l1 + l2)
    return np.repeat(vals, lens)


def nudge_apart(positions: np.ndarray, width: float, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Move sorted positions the least (in weighted squared distance) so that
    neighbours are at least `width` apart

    With ``d_i = p_i - i * width`` the constraint ``p_{i+1} - p_i >= width``
    becomes ``d`` non-decreasing, so this is an isotonic regression.
    """
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return p.copy()
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    idx = np.arange(len(p)) * width
    return _pava(p - idx, w) + idx


# Layout -------------------------------------------------------------------------------


def _row_offsets(
    rows: np.ndarray, counts: np.ndarray, width: float, stackratio: float, side: str, aligned: bool = False
) -> np.ndarray:
    """
    Offsets of dot centres from the baseline for rows 0, 1, ... of stacks

    With ``side="both"`` stacks are centred on the baseline; `aligned`
    instead puts rows on a common grid (0, +1, -1, +2, ...) so rows line up
    across bins, which "weave" and "hex" need.
    """
    step = width * stackratio
    if side == "both":
        if aligned:
            level = np.where(rows % 2 == 1, (rows + 1) // 2, -(rows // 2))
            return level * step
        return (rows - (counts - 1) / 2) * step
    off = width / 2 + rows * step
    return off if side == "top" else -off


def layout_dots(
    values: Any,
    width: float,
    layout: str = "bin",
    side: str = "top",
    stackratio: float = 1.0,
    overlaps: str = "nudge",
    order: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Positions of dots of diameter `width`

    Parameters
    ----------
    values :
        Positions along the value axis.
    width :
        Bin width (dot diameter, before `dotsize`).
    layout :
        ``"bin"``, ``"weave"``, ``"hex"``, ``"swarm"`` or ``"bar"``.
    side :
        ``"top"`` (stacks grow up from the baseline), ``"bottom"`` or
        ``"both"`` (centred on it).
    stackratio :
        Distance between dot centres in a stack, relative to `width`.
    overlaps :
        ``"nudge"`` moves overlapping bins (or dots in a row, for
        ``"weave"``) apart; ``"keep"`` leaves them.
    order :
        Stacking order within bins (``"bin"``, ``"hex"`` and ``"bar"``);
        default the values themselves.

    Returns
    -------
    x, offset :
        Value-axis position and offset from the baseline of each dot, in
        the order of `values`.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return np.array([]), np.array([])
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    if side not in ("top", "bottom", "both"):
        raise ValueError(f"side must be 'top', 'bottom' or 'both', not {side!r}")
    if layout == "swarm":
        return _swarm(v, width, side, stackratio)
    if layout == "bar":
        return _bar(v, width, side, stackratio, order)

    srt = np.argsort(v, kind="stable")
    xs = v[srt]
    bins, centers = wilkinson_bins(xs, width)
    counts = np.bincount(bins, minlength=len(centers))
    if overlaps == "nudge" and layout != "weave":
        centers = nudge_apart(centers, width, counts)

    # row of each dot within its bin
    rows_sorted = np.empty(n, dtype=int)
    if layout == "weave":
        # the dot nearest the bin centre at the base, then outwards
        key = np.abs(xs - centers[bins])
    else:
        key = np.zeros(n) if order is None else np.asarray(order)[srt]
    for b in range(len(centers)):
        members = np.flatnonzero(bins == b)
        if layout == "weave":
            sub = members[np.lexsort((xs[members], key[members]))]
        elif order is None:
            sub = members
        else:
            sub = members[np.argsort(key[members], kind="stable")]
        rows_sorted[sub] = np.arange(len(sub))

    if layout == "weave":
        x_sorted = xs.copy()
        if overlaps == "nudge":
            # rows on the two sides of a "both" layout are distinct rows
            for r in np.unique(rows_sorted):
                m = np.flatnonzero(rows_sorted == r)  # sorted by bin, i.e. by x
                m = m[np.argsort(x_sorted[m], kind="stable")]
                x_sorted[m] = nudge_apart(x_sorted[m], width)
    else:
        x_sorted = centers[bins].copy()
        if layout == "hex":
            x_sorted += np.where(rows_sorted % 2 == 0, width / 4, -width / 4)

    off_sorted = _row_offsets(rows_sorted, counts[bins], width, stackratio, side, aligned=layout != "bin")
    x_out = np.empty(n)
    off_out = np.empty(n)
    x_out[srt] = x_sorted
    off_out[srt] = off_sorted
    return x_out, off_out


def _bar_columns(v: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Distinct values, their counts and the number of columns per bar
    """
    vals, counts = np.unique(v, return_counts=True)
    if len(vals) > 1:
        res = float(np.min(np.diff(vals)))
        ncol = max(1, int(np.floor(BAR_FILL * res / width + 1e-9)))
    else:
        ncol = max(1, int(np.ceil(np.sqrt(counts[0]))))
    return vals, counts, ncol


def _bar(v: np.ndarray, width: float, side: str, stackratio: float, order: Any) -> tuple[np.ndarray, np.ndarray]:
    n = len(v)
    vals, counts, ncol = _bar_columns(v, width)
    key = np.zeros(n) if order is None else np.asarray(order)
    x = np.empty(n)
    off = np.empty(n)
    for val, c in zip(vals, counts, strict=True):
        members = np.flatnonzero(v == val)
        members = members[np.argsort(key[members], kind="stable")]
        k = np.arange(c)
        cols_here = min(ncol, c)
        col, row = k % ncol, k // ncol
        nrows = int(np.ceil(c / ncol))
        # bars centred on the value; a partial top row starts at the left
        x[members] = val + (col - (cols_here - 1) / 2) * width
        off[members] = _row_offsets(row, np.full(c, nrows), width, stackratio, side)
    return x, off


def _bar_height(xs_sorted: np.ndarray, width: float, stackratio: float) -> float:
    _, counts, ncol = _bar_columns(xs_sorted, width)
    rows = int(np.ceil(counts.max() / ncol))
    return width * (1 + (rows - 1) * stackratio)


def _swarm(v: np.ndarray, width: float, side: str, stackratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compact beeswarm: repeatedly place the dot that can sit closest to the
    baseline without overlapping placed dots
    """
    n = len(v)
    srt = np.argsort(v, kind="stable")
    xs = v[srt]
    w2 = (width * stackratio) ** 2
    lo_idx = np.searchsorted(xs, xs - width, side="right")
    hi_idx = np.searchsorted(xs, xs + width, side="left")
    placed = np.zeros(n, dtype=bool)
    y = np.zeros(n)
    center = (xs[(n - 1) // 2] + xs[n // 2]) / 2
    one_sided = side != "both"
    balance = [0]  # running sign balance, to alternate ties

    def best(i: int) -> float:
        nb = np.arange(lo_idx[i], hi_idx[i])
        nb = nb[placed[nb]]
        if not len(nb):
            return 0.0
        dx = xs[nb] - xs[i]
        dy = np.sqrt(np.maximum(w2 - dx**2, 0))
        cands = np.concatenate([[0.0], y[nb] + dy, y[nb] - dy])
        if one_sided:
            cands = cands[cands >= 0]
        # feasible candidates: clear of every placed neighbour
        ok = np.all(dx[None, :] ** 2 + (y[nb][None, :] - cands[:, None]) ** 2 >= w2 * (1 - 1e-9), axis=1)
        cands = cands[ok]
        a = np.abs(cands)
        m = a.min()
        tied = cands[a <= m + 1e-12]
        if len(tied) > 1:
            # alternate sides on ties, for a balanced swarm
            return float(tied.max() if balance[0] <= 0 else tied.min())
        return float(tied[0])

    if n > COMPACT_SWARM_MAX:
        # large samples: place dots in order of value (one pass), which is
        # much faster than the compact order and looks similar at this size
        for i in range(n):
            y[i] = best(i)
            placed[i] = True
            balance[0] += int(np.sign(y[i]))
        heap: list = []
    else:
        current = np.zeros(n)
        version = np.zeros(n, dtype=int)
        heap = [(0.0, abs(xs[i] - center), i, 0) for i in range(n)]
        heapq.heapify(heap)
    while heap:
        _, _, i, ver = heapq.heappop(heap)
        if placed[i] or ver != version[i]:
            continue
        y[i] = current[i]
        placed[i] = True
        balance[0] += int(np.sign(y[i]))
        # placing a dot only removes positions, so only neighbours whose
        # current best position collides with it need a new one
        js = np.arange(lo_idx[i], hi_idx[i])
        js = js[~placed[js]]
        hit = js[(xs[js] - xs[i]) ** 2 + (current[js] - y[i]) ** 2 < w2 * (1 - 1e-9)]
        for j in hit:
            current[j] = best(j)
            version[j] += 1
            heapq.heappush(heap, (abs(current[j]), abs(xs[j] - center), j, int(version[j])))

    if side == "top":
        y = y + width / 2
    elif side == "bottom":
        y = -(y + width / 2)
    x_out = np.empty(n)
    y_out = np.empty(n)
    x_out[srt] = xs
    y_out[srt] = y
    return x_out, y_out


def layout_height(offsets: np.ndarray, width: float, side: str) -> float:
    """
    Extent of a layout from the baseline (one-sided) or overall (``"both"``)
    """
    if not len(offsets):
        return 0.0
    if side == "both":
        return float(np.ptp(offsets) + width)
    return float(np.max(np.abs(offsets)) + width / 2)


def _bin_height(xs_sorted: np.ndarray, width: float, stackratio: float) -> float:
    bins, centers = wilkinson_bins(xs_sorted, width)
    c = np.bincount(bins).max()
    return width * (1 + (c - 1) * stackratio)


def find_binwidth(
    values: Any,
    maxheight: float,
    stackratio: float = 1.0,
    layout: str = "bin",
    side: str = "top",
) -> float:
    """
    The largest bin width whose tallest stack fits in `maxheight`

    Binary search over the bin width (in log space) using the
    dot-density binning; for ``"swarm"``, the result is then shrunk until
    the swarm itself fits.
    """
    xs = np.sort(np.asarray(values, dtype=float))
    xs = xs[np.isfinite(xs)]
    n = len(xs)
    if n == 0 or not maxheight > 0:
        return float(maxheight) if maxheight > 0 else 1.0
    height = _bar_height if layout == "bar" else _bin_height
    hi = float(maxheight)
    lo = maxheight / (1 + (n - 1) * stackratio)
    if layout == "bar":
        # the widest dots that still fit the bars between values
        vals = np.unique(xs)
        if len(vals) > 1:
            hi = min(hi, BAR_FILL * float(np.min(np.diff(vals))))
        lo = min(lo, hi)
    if height(xs, hi, stackratio) <= maxheight:
        best = hi
    else:
        best = lo
        a, b = np.log(lo), np.log(hi)
        for _ in range(40):
            m = (a + b) / 2
            if height(xs, float(np.exp(m)), stackratio) <= maxheight:
                best = max(best, float(np.exp(m)))
                a = m
            else:
                b = m
    if layout == "swarm":
        for _ in range(30):
            _, off = _swarm(xs, best, side, stackratio)
            h = layout_height(off, best, side)
            if h <= maxheight * (1 + 1e-9):
                break
            best *= max(0.5, min(0.98, maxheight / h))
    return best


# Public, data-unit interface -----------------------------------------------------------


def bin_dots(
    x: Any,
    y: float,
    binwidth: float,
    heightratio: float = 1.0,
    stackratio: float = 1.0,
    layout: str = "bin",
    side: str = "topright",
    orientation: str = "horizontal",
    overlaps: str = "nudge",
) -> pd.DataFrame:
    """
    Lay out a dot plot of `x` with bins of `binwidth`, in data units

    Parameters
    ----------
    x :
        Values (along the value axis).
    y :
        Baseline position on the other axis.
    binwidth :
        Bin width in units of `x`.
    heightratio :
        Ratio of dot height (in units of the other axis) to `binwidth`.
    stackratio, layout, overlaps :
        See [](`~plotnine_ggdist.geom_dotsinterval`).
    side :
        ``"topright"``/``"top"``/``"right"``, ``"bottomleft"``/``"bottom"``/
        ``"left"``, or ``"both"``.
    orientation :
        ``"horizontal"``/``"y"`` (values on x) or ``"vertical"``/``"x"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``x``, ``y`` (dot centres) and ``bin`` (bin number; for
        ``"swarm"``, the bins the dots would have had).
    """
    s = {"topright": "top", "top": "top", "right": "top", "bottomleft": "bottom",
         "bottom": "bottom", "left": "bottom", "both": "both"}.get(side, side)  # fmt: skip
    vals = np.asarray(x, dtype=float)
    pos, off = layout_dots(vals, binwidth, layout, s, stackratio, overlaps)
    srt = np.argsort(vals, kind="stable")
    bins_sorted, _ = wilkinson_bins(vals[srt], binwidth)
    bins = np.empty(len(vals), dtype=int)
    bins[srt] = bins_sorted
    other = y + off * heightratio
    if orientation in ("vertical", "x"):
        return pd.DataFrame({"x": other, "y": pos, "bin": bins})
    return pd.DataFrame({"x": pos, "y": other, "bin": bins})


def find_dotplot_binwidth(
    x: Any,
    maxheight: float,
    heightratio: float = 1.0,
    stackratio: float = 1.0,
    layout: str = "bin",
) -> float:
    """
    A bin width (in units of `x`) whose tallest stack fits in `maxheight`

    `maxheight` is in units of the other axis; `heightratio` converts a bin
    width to a dot height in those units.
    """
    return find_binwidth(np.asarray(x, dtype=float), maxheight / heightratio, stackratio, layout)
