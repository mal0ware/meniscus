"""Least-squares calibration of Heston with the analytic gradient.

The problem is Cui et al. (2017) eqs. (2)-(5): minimise ``f = 0.5 * ||r||^2``
with ``r_i = weight_i * (C(theta; K_i, T_i) - C*_i)``. Cui et al. use unit
weights (price residuals). Dividing by the Black vega turns the residual into
an implied-volatility error to first order, which is the unit a bid-ask spread
is quoted in; both are available here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from meniscus import black, heston

FloatArray = NDArray[np.float64]
Weighting = Literal["price", "vega"]

#: Search box, ordered [v0, vbar, rho, kappa, sigma].
LOWER: FloatArray = np.array([1e-4, 1e-4, -0.999, 1e-3, 1e-3])
UPPER: FloatArray = np.array([5.0, 5.0, 0.999, 60.0, 15.0])


@dataclass(frozen=True)
class Problem:
    """A set of target options: log-moneyness, expiry and implied volatility."""

    k: FloatArray
    t: FloatArray
    vol: FloatArray
    weighting: Weighting = "vega"

    @property
    def price(self) -> FloatArray:
        """Target call prices per unit of forward."""
        return black.call_price(self.k, self.vol**2 * self.t)

    @property
    def weights(self) -> FloatArray:
        """Residual weights: one, or the reciprocal of the Black vega."""
        if self.weighting == "price":
            return np.ones_like(self.k)
        return 1.0 / black.vega(self.k, self.vol, self.t)

    def residuals(self, theta: ArrayLike) -> FloatArray:
        """Weighted price residuals, eq. (2)."""
        out: FloatArray = self.weights * (heston.call_price(theta, self.k, self.t) - self.price)
        return out

    def residuals_and_jacobian(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Residuals and their ``(n, 5)`` Jacobian from the analytic gradient."""
        c, g = heston.call_price_and_gradient(theta, self.k, self.t)
        w = self.weights
        return w * (c - self.price), w[:, None] * g

    def model_vol(self, theta: ArrayLike) -> FloatArray:
        """Heston implied volatility at every target option."""
        return black.implied_vol(self.k, self.t, heston.call_price(theta, self.k, self.t))

    def vol_errors(self, theta: ArrayLike) -> FloatArray:
        """Exact implied-volatility errors, model minus target."""
        out: FloatArray = self.model_vol(theta) - self.vol
        return out

    def hessian(self, theta: ArrayLike) -> FloatArray:
        """Gauss-Newton Hessian ``J^T J``, eq. (35)."""
        _, jac = self.residuals_and_jacobian(theta)
        out: FloatArray = jac.T @ jac
        return out


@dataclass(frozen=True)
class Fit:
    """A calibration result."""

    theta: FloatArray
    cost: float
    rmse_vol: float
    max_abs_vol: float
    nfev: int
    success: bool


def calibrate(
    problem: Problem,
    theta0: ArrayLike,
    *,
    free: ArrayLike | None = None,
    tol: float = 1e-14,
    max_nfev: int = 400,
) -> Fit:
    """Minimise the least-squares objective from ``theta0``.

    ``free`` is a boolean mask of the parameters that may move; the others stay
    at their value in ``theta0``. That is how the profile of the loss along one
    parameter is computed. The solver is SciPy's bounded trust-region
    least-squares method with the analytic Jacobian.
    """
    th0 = np.clip(np.asarray(theta0, dtype=np.float64), LOWER, UPPER)
    mask = np.ones(5, dtype=bool) if free is None else np.asarray(free, dtype=bool)

    def full(x: FloatArray) -> FloatArray:
        th = th0.copy()
        th[mask] = x
        return th

    cache: dict[bytes, tuple[FloatArray, FloatArray]] = {}

    def both(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            r, jac = problem.residuals_and_jacobian(full(x))
            cache[key] = (r, jac[:, mask])
        return cache[key]

    lo, hi = LOWER[mask], UPPER[mask]
    x0 = np.clip(th0[mask], lo + 1e-10, hi - 1e-10)
    sol = least_squares(
        lambda x: both(x)[0],
        x0,
        jac=lambda x: both(x)[1],
        bounds=(lo, hi),
        x_scale="jac",
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=max_nfev,
    )
    theta = full(sol.x)
    err = problem.vol_errors(theta)
    return Fit(
        theta=theta,
        cost=float(sol.cost),
        rmse_vol=float(np.sqrt(np.nanmean(err**2))),
        max_abs_vol=float(np.nanmax(np.abs(err))),
        nfev=int(sol.nfev),
        success=bool(sol.success),
    )


def random_starts(n: int, seed: int) -> FloatArray:
    """Starting points drawn uniformly from the ranges of Cui et al., Table 5."""
    rng = np.random.default_rng(seed)
    lo = np.array([0.05, 0.05, -0.90, 0.50, 0.05])
    hi = np.array([0.95, 0.95, -0.10, 5.00, 0.95])
    out: FloatArray = lo + (hi - lo) * rng.random((n, 5))
    return out


def scaled_hessian(hess: FloatArray, theta: ArrayLike) -> FloatArray:
    """Hessian with respect to ``log``-scaled parameters: ``D H D``, ``D = diag(|theta|)``.

    Cui et al. note that the raw Hessian mixes parameters of very different
    size and suggest rescaling. This is the Hessian in relative changes, so a
    step of one unit is a 100% move in that parameter.
    """
    d = np.abs(np.asarray(theta, dtype=np.float64))
    out: FloatArray = hess * np.outer(d, d)
    return out


def condition_number(hess: FloatArray) -> float:
    """Ratio of the largest to the smallest eigenvalue of a symmetric matrix."""
    ev = np.linalg.eigvalsh(hess)
    return float(ev[-1] / ev[0])
