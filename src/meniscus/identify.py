"""How well a set of option quotes pins down the Heston parameters.

The tools here map the calibration loss around its minimum. The basic object
is the profile: fix one or two parameters on a grid, re-optimise the rest, and
record how far the refitted smile is from a reference smile. A parameter set
is called *indistinguishable* from the reference when every implied volatility
is within the quote's half bid-ask spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from meniscus.calibrate import Problem, calibrate
from meniscus.heston import PARAM_NAMES

FloatArray = NDArray[np.float64]

KAPPA = PARAM_NAMES.index("kappa")
SIGMA = PARAM_NAMES.index("sigma")


def spread_ratio(
    problem: Problem, theta: ArrayLike, reference_vol: FloatArray, half_spread: ArrayLike
) -> float:
    """Largest ``|vol(theta) - reference| / half_spread`` over the options.

    A value of at most one means every option is priced inside its spread.
    """
    h = np.broadcast_to(np.asarray(half_spread, dtype=np.float64), reference_vol.shape)
    err = np.abs(problem.model_vol(theta) - reference_vol) / h
    return float(np.nanmax(err)) if np.all(np.isfinite(err)) else float("inf")


@dataclass(frozen=True)
class Profile:
    """Loss profile along one parameter.

    ``theta[i]`` is the best parameter set with the profiled parameter fixed at
    ``grid[i]``; ``rmse_vol`` is its fit to the target and ``max_dvol`` its
    largest implied-volatility distance from the reference smile.
    """

    index: int
    grid: FloatArray
    theta: FloatArray
    rmse_vol: FloatArray
    max_dvol: FloatArray
    ratio: FloatArray

    def interval(self, level: float = 1.0) -> tuple[float, float, bool, bool]:
        """Range of the grid around the best point where ``ratio <= level``.

        Returns ``(low, high, low_open, high_open)``. The ends are linearly
        interpolated in the log of the grid. An ``open`` flag says the range
        runs off that end of the grid, so the true end is further out.
        """
        ok = self.ratio <= level
        best = int(np.argmin(self.ratio))
        if not ok[best]:
            return float("nan"), float("nan"), False, False
        lo = best
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        hi = best
        while hi < self.grid.size - 1 and ok[hi + 1]:
            hi += 1

        def cross(inside: int, outside: int) -> float:
            x0, x1 = np.log(self.grid[inside]), np.log(self.grid[outside])
            y0, y1 = self.ratio[inside], self.ratio[outside]
            return float(np.exp(x0 + (level - y0) / (y1 - y0) * (x1 - x0)))

        low_open, high_open = lo == 0, hi == self.grid.size - 1
        low = float(self.grid[0]) if low_open else cross(lo, lo - 1)
        high = float(self.grid[-1]) if high_open else cross(hi, hi + 1)
        return low, high, low_open, high_open


def profile_1d(
    problem: Problem,
    theta_hat: ArrayLike,
    grid: ArrayLike,
    half_spread: ArrayLike,
    *,
    index: int = KAPPA,
    reference_vol: FloatArray | None = None,
) -> Profile:
    """Profile the loss along one parameter, warm-starting outwards from ``theta_hat``.

    The value of the parameter at ``theta_hat`` is added to the grid, so the
    profile always contains the reference point itself.
    """
    th_hat = np.asarray(theta_hat, dtype=np.float64)
    g = np.unique(np.append(np.asarray(grid, dtype=np.float64), th_hat[index]))
    ref = problem.model_vol(th_hat) if reference_vol is None else reference_vol
    free = np.ones(5, dtype=bool)
    free[index] = False
    thetas = np.empty((g.size, 5))
    rmse = np.empty(g.size)
    start = int(np.argmin(np.abs(np.log(g) - np.log(th_hat[index]))))
    order = list(range(start, g.size)) + list(range(start - 1, -1, -1))
    for i in order:
        if i == start:
            warm = th_hat.copy()
        elif i > start:
            warm = thetas[i - 1].copy()
        else:
            warm = thetas[i + 1].copy()
        warm[index] = g[i]
        fit = calibrate(problem, warm, free=free)
        thetas[i] = fit.theta
        rmse[i] = fit.rmse_vol
    h = np.broadcast_to(np.asarray(half_spread, dtype=np.float64), ref.shape)
    dvol = np.array([np.nanmax(np.abs(problem.model_vol(th) - ref)) for th in thetas])
    ratio = np.array([spread_ratio(problem, th, ref, h) for th in thetas])
    return Profile(index=index, grid=g, theta=thetas, rmse_vol=rmse, max_dvol=dvol, ratio=ratio)


@dataclass(frozen=True)
class ProfileMap:
    """Loss profile over a grid of two parameters, the other three re-optimised."""

    index_x: int
    index_y: int
    grid_x: FloatArray
    grid_y: FloatArray
    max_dvol: FloatArray
    ratio: FloatArray


def profile_2d(
    problem: Problem,
    theta_hat: ArrayLike,
    grid_x: ArrayLike,
    grid_y: ArrayLike,
    half_spread: ArrayLike,
    *,
    index_x: int = KAPPA,
    index_y: int = SIGMA,
    reference_vol: FloatArray | None = None,
) -> ProfileMap:
    """Profile over two parameters. ``max_dvol[j, i]`` is at ``(grid_x[i], grid_y[j])``."""
    th_hat = np.asarray(theta_hat, dtype=np.float64)
    gx = np.asarray(grid_x, dtype=np.float64)
    gy = np.asarray(grid_y, dtype=np.float64)
    ref = problem.model_vol(th_hat) if reference_vol is None else reference_vol
    h = np.broadcast_to(np.asarray(half_spread, dtype=np.float64), ref.shape)
    free = np.ones(5, dtype=bool)
    free[[index_x, index_y]] = False
    dvol = np.empty((gy.size, gx.size))
    ratio = np.empty((gy.size, gx.size))
    row_start = th_hat.copy()
    for j in range(gy.size):
        warm = row_start.copy()
        for i in range(gx.size):
            warm[index_x], warm[index_y] = gx[i], gy[j]
            fit = calibrate(problem, warm, free=free, tol=1e-10, max_nfev=60)
            warm = fit.theta.copy()
            if i == 0:
                row_start = fit.theta.copy()
            err = np.abs(problem.model_vol(fit.theta) - ref)
            dvol[j, i] = np.nanmax(err)
            ratio[j, i] = np.nanmax(err / h)
    return ProfileMap(
        index_x=index_x, index_y=index_y, grid_x=gx, grid_y=gy, max_dvol=dvol, ratio=ratio
    )


def equivalent_sets(
    problem: Problem,
    profile: Profile,
    half_spread: ArrayLike,
    *,
    n: int = 5,
    reference_vol: FloatArray,
    level: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Pick ``n`` parameter sets spread along the valley, all inside the spread.

    The profiled parameter is spaced geometrically between the ends of the
    interval where the profile stays within ``level`` half-spreads, pulled in
    by 2% so the end points are strictly inside. Returns ``(theta, ratio)``.
    """
    low, high, _, _ = profile.interval(level)
    if not np.isfinite(low):
        return np.empty((0, 5)), np.empty(0)
    values = np.geomspace(low * 1.02, high / 1.02, n) if high / low > 1.05 else np.full(n, low)
    free = np.ones(5, dtype=bool)
    free[profile.index] = False
    thetas, ratios = [], []
    for v in values:
        nearest = int(np.argmin(np.abs(np.log(profile.grid) - np.log(v))))
        warm = profile.theta[nearest].copy()
        warm[profile.index] = v
        fit = calibrate(problem, warm, free=free)
        thetas.append(fit.theta)
        ratios.append(spread_ratio(problem, fit.theta, reference_vol, half_spread))
    return np.array(thetas), np.array(ratios)


# --------------------------------------------------------------------------- statistical intervals

#: Standard normal quantile and chi-square(1) quantile for a two-sided 90% interval.
Z90 = 1.6448536269514722
CHI2_90 = Z90 * Z90


def uniform_noise_sd(half_spread: float) -> float:
    """Standard deviation of a uniform draw on ``[-half_spread, half_spread]``."""
    return half_spread / float(np.sqrt(3.0))


def ls_covariance(problem: Problem, theta: ArrayLike, noise_sd: ArrayLike) -> FloatArray:
    """Linearised covariance of the unweighted least-squares estimate.

    With vega weights the Jacobian ``J`` of the residuals is the Jacobian of
    the implied volatilities, to first order and exactly at a perfect fit.
    Linearising the model around ``theta`` gives the sandwich
    ``(J^T J)^-1 J^T S J (J^T J)^-1`` with ``S = diag(noise_sd^2)``. When
    ``noise_sd`` is one number this is ``noise_sd^2 (J^T J)^-1``, the usual
    standard error of nonlinear least squares under independent noise of
    equal variance.

    This is not a Cramer-Rao bound. It would be one only for Gaussian noise.
    The noise in this study is uniform, and for uniform noise least squares
    is not the efficient estimator. The formula describes least squares
    itself, so it agrees with a Monte Carlo of least squares whenever the
    model is close to linear over the range of the noise and the noise
    variances are the ones assumed.
    """
    if problem.weighting != "vega":
        raise ValueError("the covariance is defined here for vega-weighted residuals")
    _, jac = problem.residuals_and_jacobian(theta)
    bread = np.linalg.inv(jac.T @ jac)
    s = np.broadcast_to(np.asarray(noise_sd, dtype=np.float64), (jac.shape[0],))
    meat = jac.T @ (s[:, None] ** 2 * jac)
    out: FloatArray = bread @ meat @ bread
    return out


def ls_sd(problem: Problem, theta: ArrayLike, noise_sd: ArrayLike) -> FloatArray:
    """Square roots of the diagonal of :func:`ls_covariance`."""
    out: FloatArray = np.sqrt(np.diag(ls_covariance(problem, theta, noise_sd)))
    return out


def likelihood_interval(
    profile: Profile, n_options: int, noise_sd: float, *, chi2: float = CHI2_90
) -> tuple[float, float, bool, bool]:
    """Profile-likelihood interval for the profiled parameter, at noise-free data.

    For Gaussian noise the log-likelihood ratio along the profile is the rise
    in the sum of squared volatility errors over ``noise_sd^2``. On exact data
    the minimum is zero, so the interval is where
    ``n rmse^2 <= chi2 noise_sd^2``. The ends are interpolated on the square
    root of that ratio, which is close to linear in the parameter near the
    optimum. Returns the same tuple as :meth:`Profile.interval`.
    """
    stat = np.sqrt(n_options / chi2) * profile.rmse_vol / noise_sd
    scaled = Profile(
        profile.index, profile.grid, profile.theta, profile.rmse_vol, profile.max_dvol, stat
    )
    return scaled.interval()


def spread_needed(profile: Profile, true_value: float, tolerance: float) -> float:
    """Largest half-spread that keeps the profiled parameter within ``tolerance``.

    ``tolerance`` is relative: 0.25 means from ``true / 1.25`` to ``1.25 true``.
    The answer is the smaller of the profile's implied-volatility distances at
    those two points, interpolated in the log of the grid. Any half-spread
    below it excludes both. ``nan`` when a point is off the grid.
    """
    x = np.log(profile.grid)
    out = []
    for target in (true_value / (1.0 + tolerance), true_value * (1.0 + tolerance)):
        if not profile.grid[0] <= target <= profile.grid[-1]:
            return float("nan")
        out.append(float(np.interp(np.log(target), x, profile.max_dvol)))
    return min(out)
