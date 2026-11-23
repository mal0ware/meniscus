import numpy as np
import pytest

from meniscus import black, calibrate, identify, synthetic
from meniscus.pipeline import cui_hessian_rows

THETA = synthetic.CUI_THETA


@pytest.fixture(scope="module")
def problem():
    return synthetic.delta_grid(THETA)


def test_delta_grid_hits_its_deltas(problem):
    delta = black.call_delta(problem.k, problem.vol, problem.t).reshape(8, 5)
    assert np.allclose(delta, np.array(synthetic.CUI_CALL_DELTAS)[None, :], atol=1e-9)
    assert problem.k.size == 40


def test_residuals_vanish_at_the_truth(problem):
    r, jac = problem.residuals_and_jacobian(THETA)
    assert np.max(np.abs(r)) < 1e-12
    assert jac.shape == (40, 5)
    assert np.max(np.abs(problem.vol_errors(THETA))) < 1e-12


def test_vega_weighted_residual_is_a_vol_error_to_first_order(problem):
    theta = THETA * np.array([1.01, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(problem.residuals(theta), problem.vol_errors(theta), rtol=0.05, atol=1e-6)


def test_jacobian_matches_finite_differences(problem):
    theta = np.array([0.06, 0.12, -0.6, 2.0, 0.4])
    _, jac = problem.residuals_and_jacobian(theta)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-6
        fd = (problem.residuals(theta + e) - problem.residuals(theta - e)) / 2e-6
        assert np.allclose(jac[:, j], fd, atol=1e-6)


@pytest.mark.parametrize("start", [[0.2, 0.2, -0.6, 1.2, 0.3], [0.05, 0.3, -0.2, 4.5, 0.8]])
def test_calibration_recovers_the_truth(problem, start):
    fit = calibrate.calibrate(problem, start)
    assert np.allclose(fit.theta, THETA, rtol=1e-6)
    assert fit.rmse_vol < 1e-10


def test_fixed_parameters_stay_fixed(problem):
    free = np.array([True, True, True, False, True])
    start = THETA.copy()
    start[3] = 5.0
    fit = calibrate.calibrate(problem, start, free=free)
    assert fit.theta[3] == 5.0
    assert 1e-5 < fit.max_abs_vol < 0.01


def test_hessian_is_the_same_order_as_cui_table_8():
    rows, head = cui_hessian_rows()
    ratios = {(r[0], r[1]): r[4] for r in rows[:-1]}
    # two printed entries are ten times too large for a positive semidefinite matrix
    assert head["cui_table8_min_eigenvalue_as_printed"] < 0
    for key, ratio in ratios.items():
        if key in {("vbar", "vbar"), ("v0", "vbar")}:
            assert 0.08 < ratio < 0.12
        else:
            assert 0.7 < ratio < 1.5
    assert 1e6 < head["cui_condition_number"] < 1e7


def test_scaled_hessian_and_condition_number():
    h = np.diag([1.0, 4.0])
    assert calibrate.condition_number(h) == pytest.approx(4.0)
    assert np.allclose(calibrate.scaled_hessian(h, [2.0, -0.5]), np.diag([4.0, 1.0]))


def test_random_starts_are_seeded_and_in_range():
    a, b = calibrate.random_starts(5, 3), calibrate.random_starts(5, 3)
    assert np.array_equal(a, b)
    assert np.all(a[:, 2] < 0) and np.all(a[:, 3] >= 0.5)


def test_profile_is_zero_at_the_truth_and_grows_away_from_it(problem):
    profile = identify.profile_1d(problem, THETA, [1.0, 2.0, 6.0, 12.0], 0.0025)
    assert 3.0 in profile.grid
    i = int(np.flatnonzero(profile.grid == 3.0)[0])
    assert profile.max_dvol[i] < 1e-9
    assert np.all(np.diff(profile.max_dvol[: i + 1]) < 0)
    assert np.all(np.diff(profile.max_dvol[i:]) > 0)
    assert np.all(profile.theta[:, 3] == profile.grid)
    low, high, low_open, high_open = profile.interval()
    assert 2.0 < low < 3.0 < high < 6.0
    assert not low_open and not high_open


def test_equivalent_sets_are_inside_the_spread_and_differ(problem):
    profile = identify.profile_1d(problem, THETA, np.geomspace(1.0, 9.0, 9), 0.0025)
    sets, ratios = identify.equivalent_sets(problem, profile, 0.0025, reference_vol=problem.vol)
    assert sets.shape == (5, 5)
    assert np.all(ratios <= 1.0)
    assert sets[-1, 3] / sets[0, 3] > 1.3
    for th in sets:
        assert np.max(np.abs(problem.model_vol(th) - problem.vol)) <= 0.0025


def test_interval_reports_an_open_end():
    grid = np.array([1.0, 2.0, 4.0])
    ratio = np.array([0.5, 0.0, 2.0])
    p = identify.Profile(3, grid, np.zeros((3, 5)), np.zeros(3), np.zeros(3), ratio)
    low, high, low_open, high_open = p.interval()
    assert low == 1.0 and low_open and not high_open
    assert 2.0 < high < 4.0


def test_noise_is_seeded_and_bounded(problem):
    a = synthetic.with_noise(problem, 0.0025, 5)
    b = synthetic.with_noise(problem, 0.0025, 5)
    assert np.array_equal(a.vol, b.vol)
    assert np.max(np.abs(a.vol - problem.vol)) <= 0.0025
