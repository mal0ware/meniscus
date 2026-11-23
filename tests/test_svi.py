import numpy as np
import pytest

from meniscus import arbitrage, svi
from meniscus.pipeline import VOGT, VOGT_JW


def test_natural_and_raw_are_the_same_curve():
    nat = svi.NaturalSVI(delta=0.01, mu=0.05, rho=-0.4, omega=0.2, zeta=1.7)
    raw = svi.natural_to_raw(nat)
    k = np.linspace(-1, 1, 31)
    assert np.allclose(nat.w(k), raw.w(k), atol=1e-15)
    back = svi.raw_to_natural(raw)
    assert np.allclose(
        [back.delta, back.mu, back.rho, back.omega, back.zeta],
        [nat.delta, nat.mu, nat.rho, nat.omega, nat.zeta],
    )


def test_derivatives_match_finite_differences():
    p = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    k = np.linspace(-1, 1, 21)
    h = 1e-5
    assert np.allclose(p.dw(k), (p.w(k + h) - p.w(k - h)) / (2 * h), atol=1e-9)
    assert np.allclose(p.d2w(k), (p.w(k + h) - 2 * p.w(k) + p.w(k - h)) / h**2, atol=1e-5)


def test_minimum_variance():
    p = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2)
    assert p.w(np.linspace(-2, 2, 400001)).min() == pytest.approx(p.min_variance(), abs=1e-10)


def test_jump_wings_of_the_vogt_smile_match_the_paper():
    # Gatheral and Jacquier (2014), Example 5.1
    jw = svi.raw_to_jump_wings(VOGT, 1.0)
    assert np.allclose([jw.v, jw.psi, jw.p, jw.c, jw.v_tilde], VOGT_JW, rtol=2e-6)


def test_jump_wings_skew_is_the_atm_vol_slope_times_root_t():
    # eq. (3.5) as printed gives the ATM volatility skew only at t = 1; in general
    # psi = sqrt(t) * d sigma / dk, which is also what Lemma 4.1 and eq. (4.2) imply
    p, t = svi.RawSVI(0.02, 0.3, -0.5, 0.1, 0.2), 0.75
    h = 1e-6
    slope = (np.sqrt(p.w(h) / t) - np.sqrt(p.w(-h) / t)) / (2 * h)
    assert svi.raw_to_jump_wings(p, t).psi == pytest.approx(slope * np.sqrt(t), rel=1e-6)


def test_fit_recovers_its_own_parameters():
    truth = svi.RawSVI(a=0.02, b=0.25, rho=-0.45, m=0.05, sigma=0.15)
    k = np.linspace(-0.6, 0.4, 25)
    fit = svi.fit_raw(k, np.sqrt(truth.w(k) / 0.5), 0.5)
    assert np.allclose(fit.params.as_array(), truth.as_array(), atol=1e-7)
    assert fit.rmse_vol < 1e-9


def test_fit_is_deterministic():
    truth = svi.RawSVI(0.01, 0.2, 0.2, -0.05, 0.3)
    k = np.linspace(-0.5, 0.5, 15)
    vol = np.sqrt(truth.w(k)) + 0.002 * np.sin(40 * k)
    a = svi.fit_raw(k, vol, 1.0).params.as_array()
    b = svi.fit_raw(k, vol, 1.0).params.as_array()
    assert np.array_equal(a, b)


def test_ssvi_slice_is_the_surface():
    s = svi.SSVI(
        t=np.array([0.1, 0.5, 1.0]),
        theta=np.array([0.01, 0.04, 0.09]),
        rho=-0.4,
        eta=1.0,
        gamma=0.4,
    )
    k = np.linspace(-1, 1, 21)
    for i in range(3):
        assert np.allclose(s.slice(i).w(k), s.w(k, s.t[i]), atol=1e-15)
    # at the money the surface returns theta (definition 4.1)
    assert np.allclose(s.w(0.0, s.t), s.theta)


def test_ssvi_fit_recovers_parameters_and_is_arbitrage_free():
    truth = svi.SSVI(
        t=np.array([0.1, 0.5, 1.0, 2.0]),
        theta=np.array([0.01, 0.04, 0.09, 0.2]),
        rho=-0.35,
        eta=0.9,
        gamma=0.3,
    )
    ks = [np.linspace(-0.5, 0.5, 11) * (1 + i) for i in range(4)]
    vols = [np.sqrt(truth.w_theta(ks[i], truth.theta[i]) / truth.t[i]) for i in range(4)]
    fit = svi.fit_ssvi(ks, vols, truth.t, truth.theta)
    assert np.allclose(
        [fit.rho, fit.eta, fit.gamma], [truth.rho, truth.eta, truth.gamma], atol=1e-6
    )
    assert arbitrage.check_ssvi(fit).free
    assert all(arbitrage.check_butterfly(fit.slice(i)).free for i in range(4))


def test_monotone_theta():
    assert np.array_equal(svi.monotone_theta([1.0, 0.5, 2.0, 1.5]), [1.0, 1.0, 2.0, 2.0])
