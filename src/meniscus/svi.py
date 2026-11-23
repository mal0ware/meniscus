"""SVI parameterisations of total implied variance and their fits.

Equation numbers refer to Gatheral and Jacquier (2014).

* raw SVI, eq. (3.1):      ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``
* natural SVI, eq. (3.2), with the maps (3.3) and (3.4)
* SVI jump-wings, eq. (3.5)
* SSVI, eq. (4.1), with the power-law curvature function of eq. (4.5)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RawSVI:
    """Raw SVI parameters ``(a, b, rho, m, sigma)`` of eq. (3.1)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> FloatArray:
        """Parameters as ``[a, b, rho, m, sigma]``."""
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def w(self, k: ArrayLike) -> FloatArray:
        """Total variance, eq. (3.1)."""
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))
        return out

    def dw(self, k: ArrayLike) -> FloatArray:
        """First derivative of ``w`` in ``k``."""
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))
        return out

    def d2w(self, k: ArrayLike) -> FloatArray:
        """Second derivative of ``w`` in ``k``."""
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5
        return out

    def min_variance(self) -> float:
        """Minimum of ``w``: ``a + b sigma sqrt(1 - rho^2)`` (text below eq. 3.1)."""
        return self.a + self.b * self.sigma * float(np.sqrt(1.0 - self.rho**2))


@dataclass(frozen=True)
class NaturalSVI:
    """Natural SVI parameters ``(delta, mu, rho, omega, zeta)`` of eq. (3.2)."""

    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def w(self, k: ArrayLike) -> FloatArray:
        """Total variance, eq. (3.2)."""
        x = self.zeta * (np.asarray(k, dtype=np.float64) - self.mu)
        out: FloatArray = self.delta + 0.5 * self.omega * (
            1.0 + self.rho * x + np.sqrt((x + self.rho) ** 2 + 1.0 - self.rho**2)
        )
        return out


def natural_to_raw(p: NaturalSVI) -> RawSVI:
    """Eq. (3.3)."""
    root = float(np.sqrt(1.0 - p.rho**2))
    return RawSVI(
        a=p.delta + 0.5 * p.omega * (1.0 - p.rho**2),
        b=0.5 * p.omega * p.zeta,
        rho=p.rho,
        m=p.mu - p.rho / p.zeta,
        sigma=root / p.zeta,
    )


def raw_to_natural(p: RawSVI) -> NaturalSVI:
    """Eq. (3.4)."""
    root = float(np.sqrt(1.0 - p.rho**2))
    omega = 2.0 * p.b * p.sigma / root
    return NaturalSVI(
        delta=p.a - 0.5 * omega * (1.0 - p.rho**2),
        mu=p.m + p.rho * p.sigma / root,
        rho=p.rho,
        omega=omega,
        zeta=root / p.sigma,
    )


@dataclass(frozen=True)
class JumpWings:
    """SVI-JW parameters ``(v, psi, p, c, v_tilde)`` of eq. (3.5)."""

    v: float
    psi: float
    p: float
    c: float
    v_tilde: float


def raw_to_jump_wings(p: RawSVI, t: float) -> JumpWings:
    """Eq. (3.5)."""
    root = float(np.sqrt(p.m**2 + p.sigma**2))
    v = (p.a + p.b * (-p.rho * p.m + root)) / t
    sw = float(np.sqrt(v * t))
    return JumpWings(
        v=v,
        psi=p.b / sw * 0.5 * (-p.m / root + p.rho),
        p=p.b / sw * (1.0 - p.rho),
        c=p.b / sw * (1.0 + p.rho),
        v_tilde=p.min_variance() / t,
    )


@dataclass(frozen=True)
class SVIFit:
    """A fitted slice with its residual statistics."""

    params: RawSVI
    rmse_vol: float
    max_abs_vol: float


_LOWER = np.array([-2.0, 1e-6, -0.999, -3.0, 1e-4])
_UPPER = np.array([4.0, 10.0, 0.999, 3.0, 5.0])


def fit_raw(
    k: ArrayLike,
    vol: ArrayLike,
    t: float,
    *,
    weights: ArrayLike | None = None,
    penalty: Callable[[RawSVI], FloatArray] | None = None,
    start: RawSVI | None = None,
    only_start: bool = False,
) -> SVIFit:
    """Least-squares raw SVI fit to implied volatilities at one expiry.

    The residual is ``weight * (sigma_SVI(k) - vol)``. Several starting points
    are tried and the best fit kept; the starts are deterministic. ``penalty``
    maps a candidate slice to extra residuals, which is how arbitrage
    constraints are imposed (Gatheral and Jacquier, section 5.2). ``start``
    adds one more starting point, and ``only_start`` drops the others.
    """
    k_ = np.asarray(k, dtype=np.float64)
    v_ = np.asarray(vol, dtype=np.float64)
    wt = np.ones_like(k_) if weights is None else np.asarray(weights, dtype=np.float64)
    w_obs = v_ * v_ * t

    def resid(x: FloatArray) -> FloatArray:
        p = RawSVI(*x)
        out: FloatArray = wt * (np.sqrt(np.maximum(p.w(k_), 1e-12) / t) - v_)
        if penalty is not None:
            out = np.concatenate([out, penalty(p)])
        return out

    i_min = int(np.argmin(w_obs))
    span = float(k_.max() - k_.min())
    slope = float((w_obs[-1] - w_obs[0]) / max(span, 1e-6)) if k_.size > 1 else 0.0
    w_lo, w_hi = float(w_obs.min()), float(w_obs.max())
    starts = []
    for rho0 in (-0.7, -0.3, 0.0, 0.3):
        for sig0 in (0.05, 0.2, 0.6):
            b0 = max((w_hi - w_lo) / max(span, 1e-6), 0.02) + abs(slope) * 0.5
            starts.append(np.array([0.5 * w_lo, b0, rho0, float(k_[i_min]), sig0 * max(span, 0.1)]))
    if start is not None:
        starts = [start.as_array()] if only_start else [start.as_array(), *starts]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, _LOWER + 1e-9, _UPPER - 1e-9)
        sol = least_squares(
            resid, x0, bounds=(_LOWER, _UPPER), x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    params = RawSVI(*(float(x) for x in best.x))
    err = np.sqrt(np.maximum(params.w(k_), 1e-12) / t) - v_
    return SVIFit(
        params=params,
        rmse_vol=float(np.sqrt(np.mean(err**2))),
        max_abs_vol=float(np.max(np.abs(err))),
    )


# --------------------------------------------------------------------------- SSVI


def power_law_phi(theta: ArrayLike, eta: float, gamma: float) -> FloatArray:
    """Curvature function of eq. (4.5): ``eta / (theta^gamma (1 + theta)^(1 - gamma))``."""
    th = np.asarray(theta, dtype=np.float64)
    out: FloatArray = eta / (th**gamma * (1.0 + th) ** (1.0 - gamma))
    return out


@dataclass(frozen=True)
class SSVI:
    """SSVI surface, eq. (4.1), with ``phi`` from eq. (4.5).

    ``theta`` is the at-the-money total variance at each expiry in ``t``; both
    are increasing. Between expiries ``theta`` is interpolated linearly in
    time, and from zero before the first expiry.
    """

    t: FloatArray
    theta: FloatArray
    rho: float
    eta: float
    gamma: float

    def theta_at(self, t: ArrayLike) -> FloatArray:
        """At-the-money total variance at time ``t``."""
        tt = np.concatenate(([0.0], self.t))
        th = np.concatenate(([0.0], self.theta))
        out: FloatArray = np.interp(np.asarray(t, dtype=np.float64), tt, th)
        return out

    def w_theta(self, k: ArrayLike, theta: ArrayLike) -> FloatArray:
        """Total variance at log-moneyness ``k`` for ATM total variance ``theta``."""
        th = np.asarray(theta, dtype=np.float64)
        x = power_law_phi(th, self.eta, self.gamma) * np.asarray(k, dtype=np.float64)
        out: FloatArray = (
            0.5 * th * (1.0 + self.rho * x + np.sqrt((x + self.rho) ** 2 + 1.0 - self.rho**2))
        )
        return out

    def w(self, k: ArrayLike, t: ArrayLike) -> FloatArray:
        """Total variance at ``(k, t)``."""
        return self.w_theta(k, self.theta_at(t))

    def slice(self, i: int) -> RawSVI:
        """The raw SVI slice at expiry ``i``, through eq. (3.3)."""
        th = float(self.theta[i])
        phi = float(power_law_phi(th, self.eta, self.gamma))
        return natural_to_raw(NaturalSVI(delta=0.0, mu=0.0, rho=self.rho, omega=th, zeta=phi))


def monotone_theta(theta: ArrayLike) -> FloatArray:
    """Smallest non-decreasing sequence above ``theta`` (running maximum)."""
    out: FloatArray = np.maximum.accumulate(np.asarray(theta, dtype=np.float64))
    return out


def fit_ssvi(
    k: list[FloatArray],
    vol: list[FloatArray],
    t: ArrayLike,
    theta: ArrayLike,
    *,
    weights: list[FloatArray] | None = None,
) -> SSVI:
    """Fit ``(rho, eta, gamma)`` of SSVI to implied volatilities at several expiries.

    ``theta`` is taken as given (one value per expiry) and made non-decreasing.
    ``eta`` is searched as a fraction of its bound ``2 / (1 + |rho|)``, the
    condition stated after eq. (4.5), and ``gamma`` is kept in ``(0, 1/2]``.
    Together these imply all of Corollary 4.1 (see PAPER.md), so the fitted
    surface is free of static arbitrage by construction.
    """
    t_ = np.asarray(t, dtype=np.float64)
    th = monotone_theta(theta)
    wts = [np.ones_like(x) for x in k] if weights is None else weights

    def build(x: FloatArray) -> SSVI:
        rho = float(x[0])
        return SSVI(
            t=t_, theta=th, rho=rho, eta=float(x[1]) * 2.0 / (1.0 + abs(rho)), gamma=float(x[2])
        )

    def resid(x: FloatArray) -> FloatArray:
        s = build(x)
        parts = [
            wts[i] * (np.sqrt(s.w_theta(k[i], th[i]) / t_[i]) - vol[i]) for i in range(t_.size)
        ]
        out: FloatArray = np.concatenate(parts)
        return out

    best = None
    for rho0 in (-0.6, -0.2, 0.2):
        for frac0 in (0.3, 0.8):
            for g0 in (0.2, 0.45):
                sol = least_squares(
                    resid,
                    np.array([rho0, frac0, g0]),
                    bounds=([-0.999, 1e-4, 1e-3], [0.999, 1.0, 0.5]),
                    xtol=1e-13,
                    ftol=1e-13,
                    gtol=1e-13,
                )
                if best is None or sol.cost < best.cost:
                    best = sol
    assert best is not None
    return build(best.x)
