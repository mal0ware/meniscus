import numpy as np
import pytest

from meniscus import black, heston
from meniscus.pipeline import LEFLOCH_CASES

CUI = np.array([0.08, 0.10, -0.80, 3.00, 0.25])


@pytest.mark.parametrize("case", LEFLOCH_CASES, ids=["table1_call", "table3_put"])
def test_published_reference_prices(case):
    # reference prices from the adaptive Filon quadrature paper (2016), see PAPER.md
    _, theta, strike, t, kind, published, scale = case
    call = heston.call_price_reference(theta, float(np.log(strike)), t)
    value = (call if kind == "call" else call - 1.0 + strike) * scale
    assert value == pytest.approx(published, abs=2e-6 if kind == "put" else 6e-8)


def test_three_forms_of_the_characteristic_function_agree_at_short_maturity():
    u = np.linspace(0.1, 60.0, 200)
    trap = heston.charfn(CUI, u, 0.5, "trap")
    assert np.allclose(heston.charfn(CUI, u, 0.5, "original"), trap, atol=1e-13)
    assert np.allclose(heston.charfn(CUI, u, 0.5, "cui"), trap, atol=1e-13)
    assert np.allclose(
        heston.charfn(CUI, u - 1j, 0.5, "cui"), heston.charfn(CUI, u - 1j, 0.5, "trap"), atol=1e-13
    )


def test_original_form_breaks_at_long_maturity_and_the_stable_forms_do_not():
    # Cui et al. Fig. 2: T = 15 with the parameters of their Table 1
    u = np.linspace(0.05, 4.0, 400)
    trap = heston.charfn(CUI, u, 15.0, "trap")
    assert np.max(np.abs(heston.charfn(CUI, u, 15.0, "cui") - trap)) < 1e-13
    assert np.max(np.abs(heston.charfn(CUI, u, 15.0, "original") - trap)) > 1e-4
    # the stable forms are continuous in u
    assert np.max(np.abs(np.diff(trap))) < 0.05


def test_characteristic_function_basics():
    assert heston.charfn(CUI, np.array([0.0 + 0j]), 1.0)[0] == pytest.approx(1.0)
    # martingale: E[S_T / F] = phi(-i) = 1
    assert heston.charfn(CUI, np.array([-1j]), 1.0)[0] == pytest.approx(1.0, abs=1e-13)


def test_black_scholes_limit():
    # vol of variance to zero with v0 = vbar gives constant volatility
    theta = [0.04, 0.04, 0.0, 1.0, 1e-4]
    k = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    t = np.array([0.25, 0.5, 1.0, 1.0, 2.0])
    assert np.allclose(heston.call_price(theta, k, t), black.call_price(k, 0.04 * t), atol=1e-8)


def test_deterministic_variance_limit():
    # sigma -> 0 with v0 != vbar: total variance is the integral of the mean path
    v0, vbar, kappa, t = 0.09, 0.04, 2.0, 1.5
    w = vbar * t + (v0 - vbar) * (1.0 - np.exp(-kappa * t)) / kappa
    price = heston.call_price([v0, vbar, 0.0, kappa, 1e-4], np.array([-0.2, 0.0, 0.2]), t)
    assert np.allclose(price, black.call_price(np.array([-0.2, 0.0, 0.2]), w), atol=1e-7)


def test_fixed_nodes_match_adaptive_quadrature():
    k = np.array([-0.4, -0.1, 0.0, 0.15, 0.3])
    t = np.array([7 / 365, 30 / 365, 0.25, 1.0, 2.0])
    fast = heston.call_price(CUI, k, t)
    slow = np.array([heston.call_price_reference(CUI, a, b) for a, b in zip(k, t, strict=True)])
    assert np.allclose(fast, slow, atol=1e-11)


def test_price_bounds_and_monotonicity():
    k = np.linspace(-0.6, 0.6, 41)
    c = heston.call_price(CUI, k, 0.5)
    assert np.all(c >= np.maximum(1.0 - np.exp(k), 0.0) - 1e-12)
    assert np.all(c <= 1.0)
    assert np.all(np.diff(c) < 0)
    # convex in strike: the density is non-negative
    strikes = np.linspace(0.6, 1.6, 101)
    cs = heston.call_price(CUI, np.log(strikes), 0.5)
    assert np.all(np.diff(cs, 2) > -1e-12)


def test_negative_correlation_gives_a_downward_skew():
    vol = black.implied_vol(
        np.array([-0.2, 0.2]), 0.5, heston.call_price(CUI, np.array([-0.2, 0.2]), 0.5)
    )
    assert vol[0] > vol[1]


@pytest.mark.parametrize(
    "theta",
    [
        CUI,
        np.array([0.04, 0.04, -0.9, 0.5, 1.0]),
        np.array([0.3, 0.25, -0.38, 13.0, 5.0]),
        np.array([0.09, 0.09, 0.3, 1.0, 1.0]),
    ],
)
def test_analytic_gradient_matches_central_differences(theta):
    k = np.array([-0.3, -0.1, 0.0, 0.1, 0.25, -0.2, 0.05])
    t = np.array([0.05, 0.1, 0.5, 1.0, 1.0, 2.0, 2.0])
    price, grad = heston.call_price_and_gradient(theta, k, t)
    assert np.allclose(price, heston.call_price(theta, k, t), atol=1e-14)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-6 * max(1.0, abs(theta[j]))
        fd = (heston.call_price(theta + e, k, t) - heston.call_price(theta - e, k, t)) / (2 * e[j])
        assert np.allclose(grad[:, j], fd, atol=2e-8), heston.PARAM_NAMES[j]


def test_gradient_does_not_overflow_at_large_vol_of_variance():
    _, grad = heston.call_price_and_gradient(
        [0.3, 0.3, -0.3, 40.0, 14.0], np.array([0.0, 0.3]), np.array([2.0, 2.0])
    )
    assert np.all(np.isfinite(grad))


def test_theta_must_have_five_entries():
    with pytest.raises(ValueError):
        heston.call_price([0.04, 0.04, 0.0], 0.0, 1.0)
