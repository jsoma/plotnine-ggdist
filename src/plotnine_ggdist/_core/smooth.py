"""
Smoothers for dot positions

Used with the `smooth` parameter of the dots geoms. Each takes the dot
positions and returns the same number of new positions, in the same rank
order. Called without data, each returns a version of itself with the
given options, so ``smooth=smooth_discrete(kernel="epanechnikov")`` works
as in ggdist.

This module does not import plotnine.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np

from .density import get_density
from .kernels import get_kernel

__all__ = (
    "smooth_none",
    "smooth_unbounded",
    "smooth_bounded",
    "smooth_discrete",
    "smooth_bar",
    "get_smooth",
    "data_resolution",
)


def data_resolution(x: np.ndarray) -> float:
    """
    Smallest gap between distinct values (1 if there is only one value)
    """
    u = np.unique(np.asarray(x, dtype=float)[np.isfinite(x)])
    if len(u) < 2:
        return 1.0
    return float(np.min(np.diff(u)))


def _by_rank(x: np.ndarray, new_sorted: np.ndarray) -> np.ndarray:
    out = np.empty(len(x))
    out[np.argsort(x, kind="stable")] = new_sorted
    return out


def smooth_none(x: Any = None, **kwargs: Any) -> Any:
    """
    No smoothing: return the positions unchanged (the default)
    """
    if x is None:
        return partial(smooth_none, **kwargs)
    return np.asarray(x, dtype=float)


def _smooth_density(x: np.ndarray, density: Any, **kwargs: Any) -> np.ndarray:
    from .dots import ppoints

    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0:
        return x.copy()
    d = get_density(density)(x, **kwargs)
    # quantiles of the estimate: invert its cdf at evenly spaced probabilities
    cdf, grid = np.maximum.accumulate(d.cdf), d.x
    p = ppoints(len(x)) * (cdf[-1] - cdf[0]) + cdf[0]
    return _by_rank(x, np.interp(p, cdf, grid))


def smooth_unbounded(
    x: Any = None, density: Any = "unbounded", trim: bool = False, **kwargs: Any
) -> Any:
    """
    Replace the positions by evenly spaced quantiles of a kernel density
    estimate (a "density dotplot", Zvinca 2018)

    Recommended only for large samples, where individual positions matter
    little. Extra keyword arguments (``kernel``, ``bandwidth``, ``adjust``,
    ...) go to the density estimator.
    """
    if x is None:
        return partial(smooth_unbounded, density=density, trim=trim, **kwargs)
    return _smooth_density(x, density, trim=trim, **kwargs)


def smooth_bounded(
    x: Any = None,
    density: Any = "bounded",
    bounds: tuple[float | None, float | None] = (None, None),
    bounder: str = "cooke",
    trim: bool = False,
    **kwargs: Any,
) -> Any:
    """
    As [](`~plotnine_ggdist.smooth_unbounded`), with a bounded density
    estimator; give known `bounds` where you can
    """
    if x is None:
        return partial(smooth_bounded, density=density, bounds=bounds, bounder=bounder, trim=trim, **kwargs)
    return _smooth_density(x, density, bounds=bounds, bounder=bounder, trim=trim, **kwargs)


def smooth_discrete(x: Any = None, kernel: str = "rectangular", width: float = 0.7, **kwargs: Any) -> Any:
    """
    Smooth discrete values into bins around each value

    A kernel density smoother whose kernel spans about `width` times the
    resolution of the data (the smallest gap between values), so each value
    becomes a bin of dots. ``layout="bar"`` usually looks better.
    """
    if x is None:
        return partial(smooth_discrete, kernel=kernel, width=width, **kwargs)
    x = np.asarray(x, dtype=float)
    k = get_kernel(kernel)
    half = k.support if np.isfinite(k.support) else 2.0  # gaussian: +/- 2 sd
    bw = width * data_resolution(x) / (2 * half)
    return smooth_unbounded(x, kernel=k, bandwidth=bw, **kwargs)


def smooth_bar(x: Any = None, width: float = 0.7, **kwargs: Any) -> Any:
    """
    Spread duplicated values evenly over ``+/- width / 2`` resolutions
    around each value
    """
    if x is None:
        return partial(smooth_bar, width=width, **kwargs)
    x = np.asarray(x, dtype=float)
    res = data_resolution(x)
    out = x.copy()
    srt = np.argsort(x, kind="stable")
    xs = x[srt]
    vals, start, counts = np.unique(xs, return_index=True, return_counts=True)
    spread = np.empty(len(xs))
    for v, s, c in zip(vals, start, counts, strict=True):
        spread[s : s + c] = v + ((np.arange(c) + 0.5) / c - 0.5) * width * res
    out[srt] = spread
    return out


_SMOOTHS: dict[str, Callable[..., Any]] = {
    "none": smooth_none,
    "unbounded": smooth_unbounded,
    "bounded": smooth_bounded,
    "discrete": smooth_discrete,
    "bar": smooth_bar,
}


def get_smooth(smooth: str | Callable[..., Any] | None) -> Callable[..., Any]:
    """
    Resolve a smoother from a name (``"bounded"`` or ``"smooth_bounded"``)
    or a function
    """
    if smooth is None:
        return smooth_none
    if callable(smooth):
        return smooth
    try:
        return _SMOOTHS[str(smooth).removeprefix("smooth_")]
    except KeyError as e:
        raise ValueError(f"smooth must be one of {sorted(_SMOOTHS)} or a function, not {smooth!r}") from e


