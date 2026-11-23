"""A synthetic option chain that looks like a listed market.

No market data is used. The chain is generated from a Bates model (Heston
with jumps), on a grid of listed-style expiries and strikes, with a bid and
an ask around each mid. The spread widens in the wings and at short expiries,
and a strike is quoted only while its out-of-the-money option is worth more
than a minimum price, so short expiries have a narrow range of strikes.

Every number in :class:`MarketSpec` is a choice made for this study. The
values are meant to give an equity-index-like shape: a negative skew that is
steepest at the short end, and at-the-money volatility near 20%. They were
not estimated from any data set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from meniscus import arbitrage, bates, black
from meniscus.svi import RawSVI, fit_raw

FloatArray = NDArray[np.float64]


def _listed_strikes() -> FloatArray:
    """Strike over forward: steps of 1% near the money, 2.5% and 5% further out."""
    near = np.arange(0.90, 1.1001, 0.01)
    mid = np.concatenate([np.arange(0.70, 0.8999, 0.025), np.arange(1.125, 1.3001, 0.025)])
    far = np.concatenate([np.arange(0.40, 0.6999, 0.05), np.arange(1.35, 1.6001, 0.05)])
    out: FloatArray = np.unique(np.round(np.concatenate([near, mid, far]), 6))
    return out


#: The generating model: a diffusion with a negative skew, and about one down-jump every two years.
INDEX_LIKE = bates.BatesParams(
    v0=0.035, vbar=0.05, rho=-0.70, kappa=2.0, sigma=0.45, lam=0.5, mu_j=-0.10, delta_j=0.12
)


@dataclass(frozen=True)
class MarketSpec:
    """Everything that defines the synthetic chain."""

    #: The generating model. ``kappa`` is 2, so the Heston fit can be compared with it.
    model: bates.BatesParams = INDEX_LIKE
    #: Calendar days to expiry: weeklies, monthlies, quarterlies and one long date.
    days: tuple[int, ...] = (7, 14, 28, 42, 63, 91, 126, 182, 273, 364, 546)
    #: Listed strikes as a fraction of the forward.
    strikes: FloatArray = field(default_factory=_listed_strikes)
    #: Half bid-ask spread at the money for expiries of 30 days or more, in volatility.
    half_spread_atm: float = 0.0020
    #: The spread grows as ``1 + wing * z^2`` with ``z = k / (vol_ref sqrt(t))``.
    wing: float = 0.25
    vol_ref: float = 0.20
    #: Largest half-spread, in volatility.
    half_spread_cap: float = 0.03
    #: A strike is quoted while its out-of-the-money mid is at least this, per unit of forward.
    min_price: float = 1.0e-4
    #: Each mid sits off the true volatility by a uniform draw of this fraction of the half-spread.
    mid_noise: float = 0.5
    min_quotes: int = 8
    year: float = 365.0

    def half_spread(self, k: ArrayLike, t: float) -> FloatArray:
        """Half bid-ask spread in volatility at log-moneyness ``k`` and expiry ``t``."""
        kk = np.asarray(k, dtype=np.float64)
        z = kk / (self.vol_ref * np.sqrt(t))
        short = max(1.0, float(np.sqrt(30.0 / (t * self.year))))
        out: FloatArray = np.minimum(
            self.half_spread_atm * short * (1.0 + self.wing * z * z), self.half_spread_cap
        )
        return out


@dataclass(frozen=True)
class Quotes:
    """Two-sided quotes at one expiry, as implied volatilities, with the true smile."""

    days: int
    t: float
    k: FloatArray
    true_vol: FloatArray
    mid: FloatArray
    bid: FloatArray
    ask: FloatArray

    @property
    def half_spread(self) -> FloatArray:
        """Half the bid-ask spread in volatility."""
        out: FloatArray = 0.5 * (self.ask - self.bid)
        return out


def true_vol(spec: MarketSpec, k: ArrayLike, t: float) -> FloatArray:
    """Implied volatility of the generating model."""
    kk = np.asarray(k, dtype=np.float64)
    return black.implied_vol(kk, t, bates.call_price(spec.model, kk, t))


def generate(spec: MarketSpec, seed: int) -> list[Quotes]:
    """The quoted chain, shortest expiry first. The same seed gives the same chain."""
    rng = np.random.default_rng(seed)
    out = []
    for d in spec.days:
        t = d / spec.year
        k_all = np.log(spec.strikes)
        vol_all = true_vol(spec, k_all, t)
        keep = np.isfinite(vol_all)
        keep[keep] &= black.otm_price(k_all[keep], vol_all[keep] ** 2 * t) >= spec.min_price
        k, vol = k_all[keep], vol_all[keep]
        h = spec.half_spread(k, t)
        # one draw per listed strike, quoted or not, so the chain does not depend on the cut
        shift = (2.0 * rng.random(k_all.size) - 1.0)[keep] * spec.mid_noise * h
        mid = vol + shift
        if k.size >= spec.min_quotes:
            out.append(Quotes(days=d, t=t, k=k, true_vol=vol, mid=mid, bid=mid - h, ask=mid + h))
    return out


@dataclass(frozen=True)
class Slice:
    """One expiry after smoothing: a raw SVI fit to the quotes, with its statistics."""

    days: int
    t: float
    n_quotes: int
    k_min: float
    k_max: float
    svi: RawSVI
    rmse_vol: float
    share_inside: float
    median_half_spread: float
    rmse_vol_to_truth: float
    spec: MarketSpec

    def half_spread(self, k: ArrayLike) -> FloatArray:
        """Half-spread in volatility at log-moneyness ``k``."""
        return self.spec.half_spread(k, self.t)

    def vol(self, k: ArrayLike) -> FloatArray:
        """SVI implied volatility at log-moneyness ``k``."""
        out: FloatArray = np.sqrt(self.svi.w(k) / self.t)
        return out


def fit_slice(q: Quotes, spec: MarketSpec) -> Slice:
    """Fit raw SVI to one expiry, weighting each quote by one over its half-spread."""
    h = q.half_spread
    fit = fit_raw(q.k, q.mid, q.t, weights=1.0 / h)
    model = np.sqrt(fit.params.w(q.k) / q.t)
    inside = (model >= q.bid) & (model <= q.ask)
    return Slice(
        days=q.days,
        t=q.t,
        n_quotes=int(q.k.size),
        k_min=float(q.k.min()),
        k_max=float(q.k.max()),
        svi=fit.params,
        rmse_vol=fit.rmse_vol,
        share_inside=float(np.mean(inside)),
        median_half_spread=float(np.median(h)),
        rmse_vol_to_truth=float(np.sqrt(np.mean((model - q.true_vol) ** 2))),
        spec=spec,
    )


def butterfly_on_range(s: Slice) -> tuple[float, bool]:
    """Butterfly check restricted to the quoted strikes: ``(min of g, free)``.

    Only the range counts here: the variance must be positive and ``g``
    non-negative between the lowest and the highest quoted strike.
    """
    k = np.linspace(s.k_min, s.k_max, 801)
    w = s.svi.w(k)
    if np.any(w <= 0.0):
        return float("nan"), False
    min_g = float(np.min(arbitrage.g_function(s.svi, k)))
    return min_g, min_g >= 0.0
