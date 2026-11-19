"""
Blur functions for blurry dot plots

Each takes distances `x` (>= 0) from a dot's centre along the value axis,
the dot's radius `r` and the blur's standard deviation `sd` (all in the
same units) and returns opacities in [0, 1]. Called without `x`, each
returns a version of itself with the given options, e.g.
``blur_interval(width=0.9)``.

This module does not import plotnine.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
from scipy.special import ndtr, ndtri

__all__ = ("blur_gaussian", "blur_interval", "get_blur", "blur_reach")


def blur_gaussian(x: Any = None, r: float = 0.5, sd: float = 0.0) -> Any:
    """
    A dot of radius `r` blurred by a Gaussian with standard deviation `sd`

    The opacity at distance x is ``Phi((x + r) / sd) - Phi((x - r) / sd)``:
    the share of a Gaussian centred at x that falls on the dot.
    """
    if x is None:
        return partial(blur_gaussian, r=r, sd=sd) if (r, sd) != (0.5, 0.0) else blur_gaussian
    x = np.abs(np.asarray(x, dtype=float))
    if not sd > 0:
        return (x <= r).astype(float)
    return ndtr((x + r) / sd) - ndtr((x - r) / sd)


def blur_interval(x: Any = None, r: float = 0.5, sd: float = 0.0, width: float = 0.95) -> Any:
    """
    A solid dot with a half-opaque band covering a Gaussian interval

    The band covers ``+/- z * sd`` with ``z`` the normal quantile for an
    interval of probability `width` (ggdist's ``.width``), default 0.95.
    """
    if x is None:
        return partial(_blur_interval, width=width)
    return _blur_interval(x, r, sd, width)


def _blur_interval(x: Any, r: float = 0.5, sd: float = 0.0, width: float = 0.95) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    half = ndtri((1 + width) / 2) * max(sd, 0.0)
    return np.where(x <= r, 1.0, np.where(x <= half, 0.5, 0.0))


_BLURS: dict[str, Callable[..., Any]] = {"gaussian": blur_gaussian, "interval": blur_interval}


def get_blur(blur: str | Callable[..., Any]) -> Callable[..., np.ndarray]:
    """
    Resolve a blur from a name (``"gaussian"`` or ``"blur_gaussian"``) or a
    function ``f(x, r, sd)``
    """
    if callable(blur):
        return blur if blur is not blur_interval else partial(_blur_interval, width=0.95)
    try:
        fn = _BLURS[str(blur).removeprefix("blur_")]
    except KeyError as e:
        raise ValueError(f"blur must be one of {sorted(_BLURS)} or a function, not {blur!r}") from e
    return fn if fn is blur_gaussian else partial(_blur_interval, width=0.95)


def blur_reach(blur: Callable[..., np.ndarray], r: float, sd: float) -> float:
    """
    Distance beyond which the blur is (practically) transparent
    """
    far = r + 6 * max(sd, 0.0) + 1e-12
    xs = np.linspace(0, far, 241)
    alpha = np.asarray(blur(xs, r, sd), dtype=float)
    visible = np.flatnonzero(alpha > 1e-3)
    return float(xs[visible[-1]]) if len(visible) else r
