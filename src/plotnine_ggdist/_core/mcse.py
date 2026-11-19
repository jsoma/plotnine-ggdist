"""
Monte Carlo standard errors of quantiles

Implements the method in Vehtari, Gelman, Simpson, Carpenter & Bürkner
(2021), "Rank-normalization, folding, and localization: An improved R-hat
for assessing convergence of MCMC", Bayesian Analysis 16(2): the effective
sample size of the indicator ``x <= q`` gives a Beta distribution for the
probability level of the sample quantile; its 16% and 84% quantiles
(+/- one standard deviation for a normal), mapped back through the sorted
draws, give the standard error.

The draws are treated as one chain, in the order given. This module does
not import plotnine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ("ess_basic", "mcse_quantile")


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    n = len(x)
    xc = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, size)
    acov = np.fft.irfft(f * np.conj(f), size)[:n] / n
    return acov / acov[0]


def ess_basic(x: Any) -> float:
    """
    Effective sample size of one chain of draws

    Geyer's initial monotone sequence estimator: sums of adjacent pairs of
    autocorrelations are added while positive and forced to be
    non-increasing. Capped at ``n * log10(n)`` as in Vehtari et al. (2021).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or np.ptp(x) == 0:
        return float(n)
    rho = _autocorrelation(x)
    pairs = rho[0 : n - 1 : 2][: (n - 1) // 2] + rho[1:n:2][: (n - 1) // 2]
    # initial positive sequence
    stop = np.flatnonzero(pairs <= 0)
    pairs = pairs[: stop[0]] if len(stop) else pairs
    if not len(pairs):
        return float(n)
    pairs = np.minimum.accumulate(pairs)  # initial monotone sequence
    tau = -1 + 2 * np.sum(pairs)
    tau = max(tau, 1 / np.log10(n))
    return float(n / tau)


def mcse_quantile(x: Any, probs: Any) -> np.ndarray:
    """
    Monte Carlo standard error of the `probs` quantiles of draws `x`

    Like ``posterior::mcse_quantile()`` for a single chain.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    n = len(x)
    out = np.full(len(probs), np.nan)
    if n < 2:
        return out
    srt = np.sort(x)
    for i, p in enumerate(probs):
        q = np.quantile(x, p)
        ess = ess_basic((x <= q).astype(float))
        a, b = stats.beta.ppf([0.1586553, 0.8413447], ess * p + 1, ess * (1 - p) + 1)
        th1 = srt[max(int(np.floor(a * n)), 1) - 1]
        th2 = srt[min(int(np.ceil(b * n)), n) - 1]
        out[i] = (th2 - th1) / 2
    return out
