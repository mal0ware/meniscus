import numpy as np
import pytest

from meniscus import arbitrage, black, svi
from meniscus.pipeline import VOGT


def test_g_is_one_for_a_flat_smile():
    flat = svi.RawSVI(a=0.04, b=1e-12, rho=0.0, m=0.0, sigma=1.0)
    assert np.allclose(arbitrage.g_function(flat, np.linspace(-2, 2, 41)), 1.0, atol=1e-9)


def test_density_matches_second_derivative_of_call_prices():
    # proof of Lemma 2.2: p(k) = d2C/dK2 at K = exp(k), in forward units
    p = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    k0 = np.array([-0.4, 0.0, 0.3])
    strike = np.exp(k0)
    h = 1e-4

    def call(x):
        return black.call_price(np.log(x), p.w(np.log(x)))

    second = (call(strike + h) - 2 * call(strike) + call(strike - h)) / h**2
    # density of K is density of k divided by K
    assert np.allclose(arbitrage.density(p, k0) / strike, second, rtol=1e-4)


def test_density_integrates_to_one_for_a_clean_slice():
    p = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    k = np.linspace(-60, 20, 800001)
    assert np.trapezoid(arbitrage.density(p, k), k) == pytest.approx(1.0, abs=1e-6)


def test_vogt_smile_is_flagged():
    # Gatheral and Jacquier (2014), Example 3.1 and Figure 1
    check = arbitrage.check_butterfly(VOGT, np.linspace(-1.5, 1.5, 3001))
    assert not check.free
    assert check.min_g < 0
    assert 0.5 < check.argmin_k < 1.5


def test_clean_slice_passes_and_broken_slices_fail():
    clean = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    assert arbitrage.check_butterfly(clean).free
    # same slice with the wings made steeper than the moment bound allows
    assert not arbitrage.check_butterfly(svi.RawSVI(0.02, 1.5, -0.5, 0.1, 0.2)).free
    # a slice broken on purpose: a sharp kink at the money on almost no variance
    assert not arbitrage.check_butterfly(svi.RawSVI(0.001, 0.4, 0.0, 0.0, 0.001)).free
    # negative total variance
    assert not arbitrage.check_butterfly(svi.RawSVI(-0.05, 0.1, 0.0, 0.0, 0.1)).free


def test_crossedness():
    early = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    late = svi.RawSVI(0.03, 0.3, -0.5, 0.1, 0.2)
    assert arbitrage.crossedness(early, late) == 0.0
    assert arbitrage.crossedness(late, early) == pytest.approx(0.01)
    tilted = svi.RawSVI(0.03, 0.3, 0.5, 0.1, 0.2)
    assert arbitrage.crossedness(early, tilted) > 0.0


def test_ssvi_conditions():
    t, theta = np.array([0.5, 1.0]), np.array([0.04, 0.09])
    good = svi.SSVI(t=t, theta=theta, rho=-0.5, eta=1.3, gamma=0.5)
    assert arbitrage.check_ssvi(good).free
    assert not arbitrage.check_ssvi(svi.SSVI(t=t, theta=theta, rho=-0.5, eta=6.0, gamma=0.5)).free
    backwards = svi.SSVI(t=t, theta=theta[::-1], rho=-0.5, eta=1.0, gamma=0.5)
    assert not arbitrage.check_ssvi(backwards).theta_increasing
    # gamma above one half breaks the second butterfly condition for small theta
    steep = svi.SSVI(t=t, theta=theta, rho=-0.5, eta=1.3, gamma=0.8)
    assert not arbitrage.check_ssvi(steep).butterfly_curvature_ok


def test_ssvi_slices_inside_the_bound_have_no_butterfly_arbitrage():
    rng = np.random.default_rng(1)
    for _ in range(25):
        rho = rng.uniform(-0.95, 0.95)
        gamma = rng.uniform(0.05, 0.5)
        eta = rng.uniform(0.1, 1.0) * 2.0 / (1.0 + abs(rho))
        theta = np.array([rng.uniform(0.001, 2.0)])
        s = svi.SSVI(t=np.array([1.0]), theta=theta, rho=rho, eta=eta, gamma=gamma)
        assert arbitrage.check_butterfly(s.slice(0)).min_g >= 0.0
