"""Black (1976) prices on a forward, and the inversion to implied volatility.

Everything is in forward units with no discounting. With log-moneyness
``k = log(K / F)`` and total variance ``w = sigma**2 * T`` the call is

    c(k, w) = N(d1) - exp(k) * N(d2),    d1 = -k / sqrt(w) + sqrt(w) / 2,
                                         d2 = d1 - sqrt(w),

which is the form used in Gatheral and Jacquier (2014), section 2.2. A price in
currency is ``D * F * c`` for a discount factor ``D``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
#: Upper end of the bracket for ``sqrt(w)`` in the inversion.
MAX_TOTAL_VOL = 12.0


def _as_float(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def call_price(k: ArrayLike, w: ArrayLike) -> FloatArray:
    """Undiscounted call price per unit of forward."""
    k_, w_ = np.broadcast_arrays(_as_float(k), _as_float(w))
    s = np.sqrt(np.maximum(w_, 0.0))
    intrinsic = np.maximum(1.0 - np.exp(k_), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k_ / s + 0.5 * s
        price = ndtr(d1) - np.exp(k_) * ndtr(d1 - s)
    out: FloatArray = np.where(s > 0.0, price, intrinsic)
    return out


def put_price(k: ArrayLike, w: ArrayLike) -> FloatArray:
    """Undiscounted put price per unit of forward."""
    k_, w_ = np.broadcast_arrays(_as_float(k), _as_float(w))
    s = np.sqrt(np.maximum(w_, 0.0))
    intrinsic = np.maximum(np.exp(k_) - 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k_ / s + 0.5 * s
        price = np.exp(k_) * ndtr(s - d1) - ndtr(-d1)
    out: FloatArray = np.where(s > 0.0, price, intrinsic)
    return out


def otm_price(k: ArrayLike, w: ArrayLike) -> FloatArray:
    """Price of the out-of-the-money option: a put for ``k < 0``, else a call."""
    k_ = _as_float(k)
    out: FloatArray = np.where(k_ < 0.0, put_price(k_, w), call_price(k_, w))
    return out


def vega(k: ArrayLike, sigma: ArrayLike, t: ArrayLike) -> FloatArray:
    """Derivative of the normalised price with respect to ``sigma``."""
    k_, sig, t_ = np.broadcast_arrays(_as_float(k), _as_float(sigma), _as_float(t))
    s = sig * np.sqrt(t_)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k_ / s + 0.5 * s
        v = np.sqrt(t_) * np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    out: FloatArray = np.where(s > 0.0, v, 0.0)
    return out


def call_delta(k: ArrayLike, sigma: ArrayLike, t: ArrayLike) -> FloatArray:
    """Forward delta ``N(d1)`` of the call."""
    k_, sig, t_ = np.broadcast_arrays(_as_float(k), _as_float(sigma), _as_float(t))
    s = sig * np.sqrt(t_)
    out: FloatArray = ndtr(-k_ / s + 0.5 * s)
    return out


def implied_total_vol(k: ArrayLike, price: ArrayLike, *, is_call: ArrayLike = True) -> FloatArray:
    """Solve ``price = c(k, s**2)`` (or the put) for ``s = sigma * sqrt(T)``.

    The price is first converted to the out-of-the-money side with put-call
    parity, which keeps the time value from being lost next to a large
    intrinsic value. The root is then bracketed in ``[0, MAX_TOTAL_VOL]`` and
    found by bisection, which cannot diverge. Prices outside the no-arbitrage
    bounds give ``nan``.
    """
    k_, p_, call_ = np.broadcast_arrays(
        _as_float(k), _as_float(price), np.asarray(is_call, dtype=bool)
    )
    ek = np.exp(k_)
    # c - p = 1 - exp(k)
    call = np.where(call_, p_, p_ + 1.0 - ek)
    target = np.where(k_ < 0.0, call - 1.0 + ek, call)
    upper = np.where(k_ < 0.0, ek, 1.0)
    valid = (target > 0.0) & (target < upper) & np.isfinite(target)

    lo = np.zeros(k_.shape)
    hi = np.full(k_.shape, MAX_TOTAL_VOL)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        too_low = otm_price(k_, mid * mid) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    s = 0.5 * (lo + hi)
    valid &= s < MAX_TOTAL_VOL * (1.0 - 1e-9)
    out: FloatArray = np.where(valid, s, np.nan)
    return out


def implied_vol(
    k: ArrayLike, t: ArrayLike, price: ArrayLike, *, is_call: ArrayLike = True
) -> FloatArray:
    """Annualised implied volatility from a normalised price."""
    s = implied_total_vol(k, price, is_call=is_call)
    out: FloatArray = s / np.sqrt(_as_float(t))
    return out
