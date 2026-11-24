import numpy as np
import pytest

from meniscus import bates, black, chain, heston

SPEC = chain.ChainSpec()
SEED = 7


@pytest.fixture(scope="module")
def quotes():
    return chain.generate(SPEC, SEED)


def test_jump_factor_keeps_the_forward_a_martingale():
    p = SPEC.model
    for t in (0.02, 0.5, 2.0):
        assert bates.jump_factor(p, np.array([0.0]), t)[0] == pytest.approx(1.0)
        assert bates.jump_factor(p, np.array([-1j]), t)[0] == pytest.approx(1.0, abs=1e-13)
        assert bates.charfn(p, np.array([-1j]), t)[0] == pytest.approx(1.0, abs=1e-12)


def test_bates_without_jumps_is_heston():
    p = bates.BatesParams(0.04, 0.06, -0.6, 1.5, 0.4, lam=0.0, mu_j=-0.1, delta_j=0.1)
    k = np.linspace(-0.3, 0.2, 7)
    for t in (0.1, 1.0):
        plain = heston.call_price(p.heston, k, t, quadrature=bates.GENERATION_QUADRATURE)
        assert np.allclose(bates.call_price(p, k, t), plain, atol=1e-14, rtol=0)


def test_bates_with_no_diffusion_skew_matches_the_merton_series():
    # kappa large and sigma tiny: variance is constant, so the model is Merton (1976)
    v, lam, mu, delta, t = 0.04, 0.8, -0.08, 0.15, 0.5
    p = bates.BatesParams(v, v, 0.0, 5.0, 1e-4, lam=lam, mu_j=mu, delta_j=delta)
    mean_jump = np.exp(mu + 0.5 * delta**2) - 1.0
    for k in (-0.2, 0.0, 0.15):
        total, weight = 0.0, np.exp(-lam * t)
        for n in range(40):
            # forward of the conditional lognormal, relative to the unconditional forward
            fwd = np.exp(-lam * mean_jump * t + n * (mu + 0.5 * delta**2))
            w = v * t + n * delta**2
            total += weight * fwd * float(black.call_price(k - np.log(fwd), w))
            weight *= lam * t / (n + 1)
        assert float(bates.call_price(p, k, t)) == pytest.approx(total, abs=2e-7)


def test_fixed_rule_matches_adaptive_quadrature_at_the_shortest_expiries():
    for days, k in ((7, -0.19), (7, 0.05), (14, -0.2), (91, 0.0), (546, -0.9)):
        t = days / 365.0
        fixed = float(bates.call_price(SPEC.model, k, t))
        assert fixed == pytest.approx(bates.call_price_reference(SPEC.model, k, t), abs=1e-11)


def test_chain_is_seeded_and_ordered(quotes):
    again = chain.generate(SPEC, SEED)
    other = chain.generate(SPEC, SEED + 1)
    assert [q.days for q in quotes] == sorted(SPEC.days)
    for a, b, c in zip(quotes, again, other, strict=True):
        assert np.array_equal(a.mid, b.mid) and np.array_equal(a.k, b.k)
        assert np.array_equal(a.true_vol, c.true_vol) and not np.array_equal(a.mid, c.mid)


def test_quotes_look_like_a_listed_chain(quotes):
    widths = [q.k.max() - q.k.min() for q in quotes]
    assert widths == sorted(widths)  # longer expiries quote a wider range of strikes
    for q in quotes:
        assert q.k.size >= SPEC.min_quotes and np.all(np.diff(q.k) > 0)
        assert np.all(q.bid < q.mid) and np.all(q.mid < q.ask)
        # the truth is inside every spread, and the cheapest quoted option clears the minimum
        assert np.all((q.true_vol > q.bid) & (q.true_vol < q.ask))
        assert black.otm_price(q.k, q.true_vol**2 * q.t).min() >= SPEC.min_price
        atm = int(np.argmin(np.abs(q.k)))
        # the spread is narrowest at the money and the smile is skewed to the downside
        assert q.half_spread[atm] == pytest.approx(q.half_spread.min(), rel=0.05)
        assert q.half_spread.max() <= SPEC.half_spread_cap + 1e-15
        assert q.true_vol[0] > q.true_vol[atm] > 0.15
    assert quotes[0].half_spread.min() > quotes[-1].half_spread.min()


def test_slice_fit_is_inside_the_spread_and_close_to_the_truth(quotes):
    for q in (quotes[2], quotes[-1]):
        s = chain.fit_slice(q, SPEC)
        assert s.share_inside >= 0.9
        assert s.rmse_vol_to_truth < 0.004
        assert s.n_quotes == q.k.size and s.k_min == q.k.min()
        assert np.allclose(s.half_spread(q.k), q.half_spread)
        min_g, free = chain.butterfly_on_range(s)
        assert free and min_g > 0
