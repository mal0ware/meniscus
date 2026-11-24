import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from meniscus import arbitrage, chain, pipeline, svi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def table(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = pipeline.Config.tiny()
    head_a, _ = pipeline.run_synthetic(cfg, out / "a")
    head_b, _ = pipeline.run_synthetic(cfg, out / "b")
    mkt_a, data = pipeline.run_chain(cfg, out / "a" / "chain")
    mkt_b, _ = pipeline.run_chain(cfg, out / "b" / "chain")
    pipeline.write_json(out / "a" / "headline.json", {**head_a, "chain": mkt_a})
    pipeline.write_json(out / "b" / "headline.json", {**head_b, "chain": mkt_b})
    return out, head_a, mkt_a, data


def test_two_runs_write_identical_tables(tiny):
    out, *_ = tiny
    files = sorted(p.relative_to(out / "a") for p in (out / "a").rglob("*") if p.is_file())
    assert len(files) >= 24
    for rel in files:
        assert (out / "a" / rel).read_bytes() == (out / "b" / rel).read_bytes(), rel


def test_synthetic_headline_is_sane(tiny):
    _, head, _, _ = tiny
    assert head["vogt_min_g"] < 0
    assert head["gradient_max_error"] < 1e-8
    assert head["quadrature_max_error"] < 1e-11
    assert head["charfn_gap_cui_T15"] < 1e-12 < head["charfn_gap_original_T15"]
    assert head["recovery_within_1e-4"] >= 1
    assert head["synthetic_kappa_low"] < 3.0 < head["synthetic_kappa_high"]
    assert head["synthetic_sets_max_ratio"] <= 1.0
    assert head["svi_self_recovery_max_param_error"] < 1e-6


def test_statistical_intervals_sit_inside_the_all_quotes_interval(tiny):
    out, head, _, _ = tiny
    assert head["synthetic_kappa_low"] < head["ls_kappa_q05"] < 3.0
    assert 3.0 < head["ls_kappa_q95"] < head["synthetic_kappa_high"]
    assert head["ls_kappa_q05"] < head["likelihood_kappa_high"]
    assert head["ls_correlation_rho_sigma"] > 0.9
    rows = table(out / "a" / "kappa_intervals.csv")
    assert len(rows) == 4 and all(float(r["kappa_low"]) < float(r["kappa_high"]) for r in rows)
    names = [r["parameter"] for r in table(out / "a" / "identifiability.csv")]
    assert names == ["v0", "vbar", "rho", "kappa", "sigma"]


def test_noise_assumption_table(tiny):
    out, head, _, _ = tiny
    rows = table(out / "a" / "noise_assumptions.csv")
    assert len(rows) == 15
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    assert len(by_case) == 3
    for case, rs in by_case.items():
        for r in rs:
            equal, sandwich = float(r["ls_sd_equal_variance"]), float(r["ls_sd_sandwich"])
            if case.startswith("uniform"):
                assert sandwich == pytest.approx(equal, rel=1e-6)
            assert float(r["monte_carlo_sd"]) > 0
    low, high = head["wings_half_spread_range"]
    assert low == pytest.approx(pipeline.SYNTHETIC_HALF_SPREAD, rel=0.2) and high > 1.5 * low


def test_takeaway_table_more_data_needs_less_tight_quotes(tiny):
    out, *_ = tiny
    rows = table(out / "a" / "takeaway.csv")
    assert len(rows) == len(pipeline.TAKEAWAY_EXPIRIES) * len(pipeline.TAKEAWAY_DELTAS)
    need = {
        (int(r["n_expiries"]), int(r["strikes_per_expiry"])): float(
            r["half_spread_for_90pct_within_25pct"]
        )
        for r in rows
    }
    for strikes in (5, 9, 17):
        by_expiries = [need[n, strikes] for n in (1, 2, 3, 5, 8, 10)]
        assert by_expiries == sorted(by_expiries)
        assert need[1, strikes] < 1e-5  # one expiry: no realistic spread is tight enough
    for n in (2, 3, 5, 8, 10):
        assert need[n, 5] < need[n, 9] < need[n, 17]
    # the three tolerance columns are proportional
    r = rows[-1]
    assert float(r["half_spread_for_90pct_within_50pct"]) == pytest.approx(
        5 * float(r["half_spread_for_90pct_within_10pct"]), rel=1e-6
    )
    expiry = table(out / "a" / "expiry_study.csv")
    assert expiry[0]["low_open_25bp"] == "True" and expiry[0]["high_open_25bp"] == "True"
    needed = [float(d["half_spread_to_exclude_kappa_off_by_25pct"]) for d in expiry]
    assert needed == sorted(needed)


def test_chain_run(tiny):
    out, _, mkt, data = tiny
    spec = data["spec"]
    assert mkt["n_expiries"] == len(spec.days) == 11
    assert mkt["heston_expiries"] == 10  # the 7-day slice is left out
    assert mkt["n_quotes"] == len(table(out / "a" / "chain" / "quotes.csv"))
    assert mkt["svi_share_inside"] > 0.95
    assert mkt["svi_slices_with_butterfly_arbitrage_quoted_range"] == 0
    assert mkt["svi_calendar_crossings_quoted_range"] == 0
    assert mkt["ssvi_static_arbitrage_free"]
    assert mkt["repaired_slices_butterfly_free"] == 11
    assert mkt["repaired_calendar_crossings"] == 0
    assert mkt["repaired_share_inside"] > 0.9
    low, high = mkt["heston_kappa_low"], mkt["heston_kappa_high"]
    assert low <= mkt["heston_theta"]["kappa"] <= high
    assert data["valley"].sets.shape == (5, 5)
    # Heston cannot reproduce a smile made with jumps, even without quote noise
    assert mkt["heston_rmse_vol_fitted_to_true_smile"] > 1e-3
    assert mkt["heston_rmse_vol_at_true_kappa"] > mkt["heston_rmse_vol"]
    assert mkt["generator"]["kappa"] == spec.model.kappa


def test_arbitrage_free_refit_repairs_slices_that_cross_in_the_wing(tiny):
    *_, data = tiny
    quotes, spec = data["quotes"][4:6], data["spec"]
    first, second = (chain.fit_slice(q, spec) for q in quotes)
    # a steeper earlier slice: the two cross beyond the quoted strikes
    broken = replace(first.svi, b=first.svi.b * 2.5, a=first.svi.a - 0.004)
    assert arbitrage.crossedness(broken, second.svi) > 0
    # a start with no static arbitrage: two slices of one SSVI surface
    surf = svi.SSVI(
        t=np.array([first.t, second.t]),
        theta=np.array([float(first.svi.w(0.0)), float(second.svi.w(0.0))]),
        rho=-0.6,
        eta=0.8,
        gamma=0.4,
    )
    start = [surf.slice(0), surf.slice(1)]
    assert arbitrage.crossedness(*start) == 0.0
    fixed = pipeline.arbitrage_free_slices(quotes, start)
    assert arbitrage.crossedness(fixed[0], fixed[1]) == 0.0
    assert all(arbitrage.check_butterfly(p).free for p in fixed)
    # the refit is closer to the quotes than its start was
    for q, p0, p1 in zip(quotes, start, fixed, strict=True):
        err0 = np.sqrt(p0.w(q.k) / q.t) - q.mid
        err1 = np.sqrt(p1.w(q.k) / q.t) - q.mid
        assert np.sum((err1 / q.half_spread) ** 2) < np.sum((err0 / q.half_spread) ** 2)
        assert np.mean(np.abs(err1) <= q.half_spread) >= 0.9


def test_chain_problem_uses_nine_deltas_inside_the_quoted_range(tiny):
    *_, data = tiny
    prob, hs, used = pipeline.chain_problem(data["slices"])
    assert len(used) == 10 and prob.k.size == hs.size <= 90
    for s in used:
        k = prob.k[prob.t == s.t]
        assert k.min() >= s.k_min and k.max() <= s.k_max


def test_reproduce_fast_smoke(tmp_path, monkeypatch):
    import reproduce

    monkeypatch.setattr(pipeline.Config, "fast", staticmethod(pipeline.Config.tiny))
    reproduce.run(tmp_path, fast=True)
    assert (tmp_path / "fig_valley_synthetic.png").stat().st_size > 10_000
    assert (tmp_path / "fig_intervals.png").stat().st_size > 10_000
    assert (tmp_path / reproduce.CHAIN_DIR / "fig_surface.png").exists()
    assert (tmp_path / "headline.json").exists()
    assert (tmp_path / reproduce.CHAIN_DIR / "headline.json").exists()


def test_no_code_reaches_the_network_and_raw_data_is_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in ignore and ".env" in ignore
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "urllib" not in text and "requests" not in text and "http" not in text, path
