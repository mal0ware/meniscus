import numpy as np
import pytest

from meniscus import black


def test_put_call_parity():
    k = np.linspace(-1.0, 1.0, 21)
    w = 0.09
    assert np.allclose(black.call_price(k, w) - black.put_price(k, w), 1.0 - np.exp(k), atol=1e-15)


def test_at_the_money_closed_form():
    # c(0, w) = 2 N(sqrt(w) / 2) - 1
    from scipy.special import ndtr

    for w in (1e-4, 0.04, 1.0):
        assert black.call_price(0.0, w) == pytest.approx(
            2.0 * ndtr(np.sqrt(w) / 2.0) - 1.0, abs=1e-15
        )


def test_zero_variance_is_intrinsic():
    k = np.array([-0.2, 0.0, 0.3])
    assert np.allclose(black.call_price(k, 0.0), np.maximum(1.0 - np.exp(k), 0.0))
    assert np.allclose(black.put_price(k, 0.0), np.maximum(np.exp(k) - 1.0, 0.0))


def test_price_increases_in_variance_and_decreases_in_strike():
    w = np.linspace(0.001, 2.0, 50)
    assert np.all(np.diff(black.call_price(0.1, w)) > 0)
    k = np.linspace(-1, 1, 50)
    assert np.all(np.diff(black.call_price(k, 0.2)) < 0)


def test_vega_matches_finite_difference():
    k, sigma, t = np.array([-0.3, 0.0, 0.2]), 0.4, 0.7
    h = 1e-6
    fd = (black.call_price(k, (sigma + h) ** 2 * t) - black.call_price(k, (sigma - h) ** 2 * t)) / (
        2 * h
    )
    assert np.allclose(black.vega(k, sigma, t), fd, rtol=1e-7)


def test_implied_vol_round_trip_calls_and_puts():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.01, 2.0, 200)
    vol = rng.uniform(0.05, 1.5, 200)
    w = vol**2 * t
    k = rng.uniform(-3.0, 3.0, 200) * np.sqrt(w)  # up to three standard deviations out
    assert np.allclose(black.implied_vol(k, t, black.call_price(k, w)), vol, atol=1e-9)
    assert np.allclose(
        black.implied_vol(k, t, black.put_price(k, w), is_call=False), vol, atol=1e-9
    )


def test_implied_vol_deep_in_the_money_uses_time_value():
    # a call at k = -1.5 is almost all intrinsic value; the inversion must still work
    k, t, vol = -1.5, 0.25, 0.8
    price = black.call_price(k, vol**2 * t)
    assert black.implied_vol(k, t, price) == pytest.approx(vol, abs=1e-6)


def test_implied_vol_rejects_prices_outside_bounds():
    assert np.isnan(black.implied_vol(0.1, 1.0, 0.0))
    assert np.isnan(black.implied_vol(0.1, 1.0, 1.01))
    assert np.isnan(black.implied_vol(-0.1, 1.0, 1.0 - np.exp(-0.1) - 1e-6))


def test_delta_is_half_at_the_forward_adjusted_strike():
    # N(d1) = 1/2 where k = w / 2
    sigma, t = 0.3, 1.0
    assert black.call_delta(sigma**2 * t / 2, sigma, t) == pytest.approx(0.5)
