"""Synthetic Heston surfaces with known parameters."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from meniscus import black, heston
from meniscus.calibrate import Problem, Weighting

FloatArray = NDArray[np.float64]

#: Cui et al. (2017), Table 1, ordered [v0, vbar, rho, kappa, sigma].
CUI_THETA: FloatArray = np.array([0.08, 0.10, -0.80, 3.00, 0.25])
#: Cui et al. (2017), Table 4: maturities in days, with 252 days to the year.
CUI_DAYS: tuple[int, ...] = (30, 60, 90, 120, 150, 180, 252, 360)
#: Call deltas of the five options per maturity in Table 4:
#: 10-delta put, 25-delta put, at the money, 25-delta call, 10-delta call.
CUI_CALL_DELTAS: tuple[float, ...] = (0.90, 0.75, 0.50, 0.25, 0.10)
#: Nine strikes per maturity, used by the experiment on sets of expiries.
NINE_CALL_DELTAS: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def strike_for_delta(theta: ArrayLike, t: float, call_delta: float) -> float:
    """Log-moneyness at which the Black delta of the Heston smile is ``call_delta``.

    The delta is ``N(d1)`` at the option's own Heston implied volatility. With
    no dividends this equals the spot delta used by Cui et al.
    """
    th = np.asarray(theta, dtype=np.float64)
    return _strike_for_delta(tuple(float(x) for x in th), float(t), float(call_delta))


@lru_cache(maxsize=4096)
def _strike_for_delta(theta: tuple[float, ...], t: float, call_delta: float) -> float:
    def gap(k: float) -> float:
        vol = black.implied_vol(k, t, heston.call_price(theta, k, t))
        return float(black.call_delta(k, vol, t)) - call_delta

    width = 4.0 * float(np.sqrt(max(theta[0], theta[1]) * t))
    return float(brentq(gap, -width, width, xtol=1e-12))


def delta_grid(
    theta: ArrayLike,
    days: tuple[int, ...] = CUI_DAYS,
    call_deltas: tuple[float, ...] = CUI_CALL_DELTAS,
    *,
    year: float = 252.0,
    weighting: Weighting = "vega",
) -> Problem:
    """A calibration problem with exact Heston prices at delta-spaced strikes."""
    ks, ts = [], []
    for d in days:
        t = d / year
        for delta in call_deltas:
            ks.append(strike_for_delta(theta, t, delta))
            ts.append(t)
    k = np.array(ks)
    t_arr = np.array(ts)
    vol = black.implied_vol(k, t_arr, heston.call_price(theta, k, t_arr))
    return Problem(k=k, t=t_arr, vol=vol, weighting=weighting)


def with_noise(problem: Problem, half_spread: ArrayLike, seed: int) -> Problem:
    """Shift every target volatility uniformly within its half-spread."""
    rng = np.random.default_rng(seed)
    h = np.broadcast_to(np.asarray(half_spread, dtype=np.float64), problem.vol.shape)
    shift = (2.0 * rng.random(problem.vol.shape) - 1.0) * h
    return Problem(k=problem.k, t=problem.t, vol=problem.vol + shift, weighting=problem.weighting)
