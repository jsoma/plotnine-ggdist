"""
Smoothing kernels for the density estimators

Each kernel is scaled to unit standard deviation, so a bandwidth is the
kernel's standard deviation for every kernel, as in R's ``density()``. The
kernels and their names follow ``stats::density()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import ndtr

__all__ = ("Kernel", "KERNELS", "get_kernel")


@dataclass(frozen=True)
class Kernel:
    """
    A kernel with unit standard deviation

    Attributes
    ----------
    name :
        Kernel name.
    support :
        Half-width of the support (``inf`` for the Gaussian).
    pdf, cdf :
        Density and distribution functions of the standardised kernel.
    """

    name: str
    support: float
    pdf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]

    def reach(self, cut: float = 5.0) -> float:
        """
        How far (in standard deviations) the kernel has any mass
        """
        return min(self.support, cut)


def _clip(u: np.ndarray, a: float) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float) / a, -1, 1)


def _inside(u: np.ndarray, a: float) -> np.ndarray:
    return np.abs(u) <= a


_SQ3, _SQ5, _SQ6, _SQ7 = np.sqrt(3), np.sqrt(5), np.sqrt(6), np.sqrt(7)
_COS_A = 1 / np.sqrt(1 / 3 - 2 / np.pi**2)
_OPTCOS_A = 1 / np.sqrt(1 - 8 / np.pi**2)


def _tri_cdf(u: np.ndarray) -> np.ndarray:
    t = _clip(u, _SQ6)
    return np.where(t < 0, (1 + t) ** 2 / 2, 1 - (1 - t) ** 2 / 2)


KERNELS: dict[str, Kernel] = {
    "gaussian": Kernel(
        "gaussian",
        np.inf,
        lambda u: np.exp(-0.5 * np.asarray(u) ** 2) / np.sqrt(2 * np.pi),
        lambda u: ndtr(u),
    ),
    "rectangular": Kernel(
        "rectangular",
        _SQ3,
        lambda u: np.where(_inside(u, _SQ3), 1 / (2 * _SQ3), 0.0),
        lambda u: (_clip(u, _SQ3) + 1) / 2,
    ),
    "triangular": Kernel(
        "triangular",
        _SQ6,
        lambda u: np.maximum(1 - np.abs(u) / _SQ6, 0) / _SQ6,
        _tri_cdf,
    ),
    "epanechnikov": Kernel(
        "epanechnikov",
        _SQ5,
        lambda u: np.where(_inside(u, _SQ5), 3 / (4 * _SQ5) * (1 - (np.asarray(u) / _SQ5) ** 2), 0.0),
        lambda u: 0.5 + 0.75 * (_clip(u, _SQ5) - _clip(u, _SQ5) ** 3 / 3),
    ),
    "biweight": Kernel(
        "biweight",
        _SQ7,
        lambda u: np.where(_inside(u, _SQ7), 15 / (16 * _SQ7) * (1 - (np.asarray(u) / _SQ7) ** 2) ** 2, 0.0),
        lambda u: 0.5 + 15 / 16 * (_clip(u, _SQ7) - 2 * _clip(u, _SQ7) ** 3 / 3 + _clip(u, _SQ7) ** 5 / 5),
    ),
    "cosine": Kernel(
        "cosine",
        _COS_A,
        lambda u: np.where(_inside(u, _COS_A), (1 + np.cos(np.pi * np.asarray(u) / _COS_A)) / (2 * _COS_A), 0.0),
        lambda u: 0.5 + (_clip(u, _COS_A) + np.sin(np.pi * _clip(u, _COS_A)) / np.pi) / 2,
    ),
    "optcosine": Kernel(
        "optcosine",
        _OPTCOS_A,
        lambda u: np.where(
            _inside(u, _OPTCOS_A), np.pi / 4 * np.cos(np.pi * np.asarray(u) / (2 * _OPTCOS_A)) / _OPTCOS_A, 0.0
        ),
        lambda u: 0.5 + np.sin(np.pi * _clip(u, _OPTCOS_A) / 2) / 2,
    ),
}


def get_kernel(kernel: str | Kernel) -> Kernel:
    """
    Look up a kernel by (partial) name, as ``stats::density()`` does
    """
    if isinstance(kernel, Kernel):
        return kernel
    matches = [k for k in KERNELS if k.startswith(str(kernel).lower())]
    if len(matches) != 1:
        raise ValueError(f"kernel must (partially) match one of {list(KERNELS)}, not {kernel!r}")
    return KERNELS[matches[0]]
