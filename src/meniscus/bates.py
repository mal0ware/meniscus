"""Bates (1996) prices: Heston with lognormal jumps in the price.

The model adds a compound Poisson process to the Heston dynamics of
``x = log(S / F)``. Jumps arrive at rate ``lam`` per year and each adds a
normal ``N(mu_j, delta_j^2)`` to ``x``. The drift is compensated so the price
stays a martingale, which multiplies the Heston characteristic function by

    exp(lam t (exp(i u mu_j - delta_j^2 u^2 / 2) - 1 - i u (exp(mu_j + delta_j^2 / 2) - 1))).

That factor is the characteristic function of Merton's (1976) jump part. The
Heston part is taken from :mod:`meniscus.heston` and the pricing integral is
the same one, Cui et al. (2017) eq. (9).

The model is used here only to generate surfaces that Heston cannot fit
exactly. Nothing is calibrated to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from meniscus import heston

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: Rule for generating surfaces: finer and longer than the calibration default,
#: because a jump smile at one or two weeks decays slowly in ``u``.
GENERATION_QUADRATURE = heston.Quadrature.panels(panels=32, n=32, umax=800.0)


@dataclass(frozen=True)
class BatesParams:
    """Heston parameters ``[v0, vbar, rho, kappa, sigma]`` and the jump part."""

    v0: float
    vbar: float
    rho: float
    kappa: float
    sigma: float
    lam: float
    mu_j: float
    delta_j: float

    @property
    def heston(self) -> FloatArray:
        """The diffusion part as a Heston parameter vector."""
        return np.array([self.v0, self.vbar, self.rho, self.kappa, self.sigma])

    @property
    def jump_variance_rate(self) -> float:
        """Variance of ``x`` added by jumps per year: ``lam (mu_j^2 + delta_j^2)``."""
        return self.lam * (self.mu_j**2 + self.delta_j**2)


def jump_factor(p: BatesParams, u: ArrayLike, t: float) -> ComplexArray:
    """Characteristic function of the compensated jump part at (complex) ``u``."""
    uu = np.asarray(u, dtype=np.complex128)
    mean_jump = np.exp(p.mu_j + 0.5 * p.delta_j**2) - 1.0
    one_jump = np.exp(1j * uu * p.mu_j - 0.5 * p.delta_j**2 * uu * uu)
    out: ComplexArray = np.exp(p.lam * t * (one_jump - 1.0 - 1j * uu * mean_jump))
    return out


def charfn(p: BatesParams, u: ArrayLike, t: float) -> ComplexArray:
    """Characteristic function of ``log(S_T / F)`` under Bates."""
    out: ComplexArray = heston.charfn(p.heston, u, t, "cui") * jump_factor(p, u, t)
    return out


def call_price(
    p: BatesParams,
    k: ArrayLike,
    t: ArrayLike,
    *,
    quadrature: heston.Quadrature | None = None,
) -> FloatArray:
    """Undiscounted call price per unit of forward."""
    quad_ = quadrature or GENERATION_QUADRATURE
    k_, t_ = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(t, dtype=np.float64))
    shape = k_.shape
    k_, t_ = k_.ravel(), t_.ravel()
    u = quad_.nodes.astype(np.complex128)
    out = np.empty(k_.size)
    for tt in np.unique(t_):
        idx = np.flatnonzero(t_ == tt)
        phi1 = charfn(p, u - 1j, float(tt))
        phi2 = charfn(p, u, float(tt))
        kk = k_[idx]
        osc = np.exp(-1j * np.outer(kk, quad_.nodes)) / (1j * quad_.nodes)
        strike = np.exp(kk)
        i1 = (osc * phi1).real @ quad_.weights
        i2 = (osc * phi2).real @ quad_.weights
        out[idx] = 0.5 * (1.0 - strike) + (i1 - strike * i2) / np.pi
    return out.reshape(shape)


def call_price_reference(p: BatesParams, k: float, t: float) -> float:
    """Slow adaptive-quadrature price, used to check the fixed-node rule."""
    strike = float(np.exp(k))

    def integrand(u: float) -> float:
        z = np.exp(-1j * u * k) / (1j * u)
        p1 = charfn(p, np.array([u - 1j]), t)[0]
        p2 = charfn(p, np.array([u]), t)[0]
        return float((z * (p1 - strike * p2)).real)

    umax = 50.0
    while abs(charfn(p, np.array([umax]), t)[0]) > 1e-14 and umax < 1e6:
        umax *= 2.0
    total = 0.0
    for a, b in pairwise(np.linspace(0.0, umax, 129)):
        total += quad(integrand, max(a, 1e-12), b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return 0.5 * (1.0 - strike) + total / np.pi
