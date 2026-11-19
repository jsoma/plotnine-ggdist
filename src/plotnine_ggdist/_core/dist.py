"""
Helpers for analytical distributions

plotnine-ggdist uses frozen ``scipy.stats`` distributions (e.g.
``scipy.stats.norm(0, 1)``) where ggdist uses distributional objects.
Anything with the frozen-distribution interface works: ``pdf``/``pmf``,
``cdf``, ``ppf``, ``mean``, ``median`` and ``support``.

This module does not import plotnine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = (
    "is_dist",
    "is_discrete",
    "dist_pdf",
    "dist_cdf",
    "dist_quantile",
    "dist_limits",
    "dist_name",
)

# Used when a limit of the support is infinite (ggdist's p_limits default)
DEFAULT_P_LIMITS = (0.001, 0.999)


def is_dist(obj: Any) -> bool:
    """
    Whether `obj` looks like a frozen scipy.stats distribution
    """
    return (
        obj is not None
        and not isinstance(obj, (str, bytes, float, int, np.number))
        and hasattr(obj, "cdf")
        and hasattr(obj, "ppf")
        and (hasattr(obj, "pdf") or hasattr(obj, "pmf"))
    )


def is_discrete(dist: Any) -> bool:
    """
    Whether `dist` is a discrete distribution (has a pmf, not a pdf)
    """
    inner = getattr(dist, "dist", None)
    if inner is not None:
        # scipy frozen distributions: rv_discrete vs rv_continuous
        from scipy import stats

        return isinstance(inner, stats.rv_discrete)
    return hasattr(dist, "pmf") and not hasattr(dist, "pdf")


def dist_pdf(dist: Any, x: np.ndarray) -> np.ndarray:
    """
    Density (or mass, for discrete distributions) of `dist` at `x`
    """
    x = np.asarray(x, dtype=float)
    if is_discrete(dist):
        return np.asarray(dist.pmf(x), dtype=float)
    return np.asarray(dist.pdf(x), dtype=float)


def dist_cdf(dist: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(dist.cdf(np.asarray(x, dtype=float)), dtype=float)


def dist_quantile(dist: Any, p: np.ndarray) -> np.ndarray:
    return np.asarray(dist.ppf(np.asarray(p, dtype=float)), dtype=float)


def dist_limits(
    dist: Any,
    p_limits: tuple[float | None, float | None] = (None, None),
) -> tuple[float, float]:
    """
    Value limits for drawing the slab of `dist`

    A missing (``None``/NaN) probability limit uses the end of the support
    when it is finite, otherwise the 0.001 / 0.999 quantile. This is
    ggdist's `p_limits` behaviour.
    """
    lo_p, hi_p = (None if _isna(p) else float(p) for p in p_limits)
    try:
        s_lo, s_hi = (float(v) for v in dist.support())
    except Exception:  # pragma: no cover - objects without support()
        s_lo, s_hi = -np.inf, np.inf

    if lo_p is None:
        lo = s_lo if np.isfinite(s_lo) else float(dist.ppf(DEFAULT_P_LIMITS[0]))
    else:
        lo = float(dist.ppf(lo_p))
    if hi_p is None:
        hi = s_hi if np.isfinite(s_hi) else float(dist.ppf(DEFAULT_P_LIMITS[1]))
    else:
        hi = float(dist.ppf(hi_p))
    return lo, hi


def dist_name(dist: Any) -> str:
    """
    A short name for `dist`, e.g. ``"norm(0, 1)"``
    """
    inner = getattr(dist, "dist", None)
    name = getattr(inner, "name", None) or type(dist).__name__
    args = list(getattr(dist, "args", ()))
    kwds = getattr(dist, "kwds", {}) or {}
    parts = [_fmt(a) for a in args] + [f"{k}={_fmt(v)}" for k, v in kwds.items()]
    return f"{name}({', '.join(parts)})"


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def _isna(v: Any) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))
