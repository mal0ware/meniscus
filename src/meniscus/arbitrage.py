"""Static-arbitrage checks for SVI slices and surfaces.

Equation numbers refer to Gatheral and Jacquier (2014).

* butterfly: the function ``g`` of eq. (2.1) must be non-negative (Lemma 2.2);
* calendar: total variance must not decrease in time at fixed ``k``
  (Definition 2.2), measured by the crossedness of Definition 5.1;
* SSVI: the sufficient conditions of Corollary 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from meniscus.svi import SSVI, RawSVI, power_law_phi

FloatArray = NDArray[np.float64]

#: Grid on which slices are checked. The paper's conditions are for all real k.
DEFAULT_GRID: FloatArray = np.linspace(-3.0, 3.0, 2401)


def g_function(p: RawSVI, k: ArrayLike) -> FloatArray:
    """Eq. (2.1): the density is ``g`` times a positive factor."""
    kk = np.asarray(k, dtype=np.float64)
    w, w1, w2 = p.w(kk), p.dw(kk), p.d2w(kk)
    out: FloatArray = (
        (1.0 - kk * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    )
    return out


def density(p: RawSVI, k: ArrayLike) -> FloatArray:
    """Risk-neutral density of ``k`` implied by the slice (proof of Lemma 2.2)."""
    kk = np.asarray(k, dtype=np.float64)
    w = p.w(kk)
    d_minus = -kk / np.sqrt(w) - 0.5 * np.sqrt(w)
    out: FloatArray = g_function(p, kk) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus**2)
    return out


@dataclass(frozen=True)
class ButterflyCheck:
    """Result of the butterfly check on one slice."""

    min_g: float
    argmin_k: float
    positive_variance: bool
    wing_ok: bool

    @property
    def free(self) -> bool:
        """True when no butterfly arbitrage was found on the grid."""
        return self.min_g >= 0.0 and self.positive_variance and self.wing_ok


def check_butterfly(p: RawSVI, k: ArrayLike | None = None) -> ButterflyCheck:
    """Evaluate Lemma 2.2 on a grid.

    ``wing_ok`` covers the second condition of the lemma. The wings of raw SVI
    have slopes ``b (1 - rho)`` and ``b (1 + rho)``; both are required to be
    below 2, which is Lee's moment bound (cited in the paper after Theorem 4.2).
    """
    grid = DEFAULT_GRID if k is None else np.asarray(k, dtype=np.float64)
    positive = p.min_variance() > 0.0
    if not positive:
        w = p.w(grid)
        grid = grid[w > 1e-12]
    g = g_function(p, grid)
    i = int(np.argmin(g))
    return ButterflyCheck(
        min_g=float(g[i]),
        argmin_k=float(grid[i]),
        positive_variance=positive,
        wing_ok=p.b * (1.0 + abs(p.rho)) < 2.0,
    )


def crossedness(early: RawSVI, late: RawSVI, k: ArrayLike | None = None) -> float:
    """Largest amount by which the earlier slice lies above the later one.

    Definition 5.1 evaluates the gap between the roots of a quartic. Here the
    gap is taken as the maximum over a fine grid, which finds the same
    crossings without the root-finding. Zero means no calendar arbitrage was
    found on the grid.
    """
    grid = DEFAULT_GRID if k is None else np.asarray(k, dtype=np.float64)
    return float(np.max(np.maximum(early.w(grid) - late.w(grid), 0.0)))


@dataclass(frozen=True)
class SSVICheck:
    """The four conditions of Corollary 4.1 for an SSVI surface."""

    theta_increasing: bool
    calendar_ok: bool
    butterfly_slope_ok: bool
    butterfly_curvature_ok: bool

    @property
    def free(self) -> bool:
        """True when all four sufficient conditions hold."""
        return (
            self.theta_increasing
            and self.calendar_ok
            and self.butterfly_slope_ok
            and self.butterfly_curvature_ok
        )


def check_ssvi(s: SSVI, theta_grid: ArrayLike | None = None) -> SSVICheck:
    """Check Corollary 4.1 numerically on a grid of ``theta``."""
    th = (
        np.geomspace(1e-6, 50.0, 4000)
        if theta_grid is None
        else np.asarray(theta_grid, dtype=np.float64)
    )
    phi = power_law_phi(th, s.eta, s.gamma)
    # d(theta phi)/d theta for eq. (4.5), in closed form
    dthetaphi = phi * (1.0 - s.gamma) / (1.0 + th)
    bound = np.inf if s.rho == 0.0 else (1.0 + np.sqrt(1.0 - s.rho**2)) / s.rho**2
    a = 1.0 + abs(s.rho)
    tol = 1e-12
    return SSVICheck(
        theta_increasing=bool(np.all(np.diff(s.theta) >= 0.0) and s.theta[0] > 0.0),
        calendar_ok=bool(np.all(dthetaphi >= 0.0) and np.all(dthetaphi <= bound * phi + tol)),
        butterfly_slope_ok=bool(np.all(th * phi * a < 4.0)),
        butterfly_curvature_ok=bool(np.all(th * phi * phi * a <= 4.0 + tol)),
    )
