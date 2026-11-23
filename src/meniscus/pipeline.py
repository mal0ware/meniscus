"""The experiments behind every table in ``results/``.

Two groups:

* :func:`run_synthetic` validates the code against published values and maps
  the Heston calibration loss on Heston surfaces with known parameters.
* :func:`run_market` generates a market-like chain from a Bates model, with
  bid-ask spreads and a listed strike grid, and then treats it the way a real
  chain would be treated: SVI slices, arbitrage checks, SSVI, and Heston.

Nothing here reads market data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from meniscus import arbitrage, black, heston, identify, market, svi, synthetic
from meniscus.calibrate import (
    Problem,
    calibrate,
    condition_number,
    random_starts,
    scaled_hessian,
)
from meniscus.heston import PARAM_NAMES
from meniscus.market import MarketSpec, Quotes, Slice, butterfly_on_range

FloatArray = NDArray[np.float64]

#: Half bid-ask spread, in volatility, used as the tolerance on synthetic surfaces.
SYNTHETIC_HALF_SPREAD = 0.0025
TOLERANCES = (0.0005, 0.001, 0.0025, 0.005, 0.01)
#: Expiries shorter than this are left out of the Heston calibration.
HESTON_MIN_DAYS = 10.0
#: Relative tolerances on kappa used in the "what do you need" tables.
KAPPA_TOLERANCES = (0.10, 0.25, 0.50)
#: Expiry sets (days, 252 to the year) and strikes per expiry of the Fisher design table.
TAKEAWAY_EXPIRIES: tuple[tuple[int, ...], ...] = (
    (90,),
    (30, 90),
    (30, 90, 180),
    (30, 60, 90, 180, 360),
    synthetic.CUI_DAYS,
    (*synthetic.CUI_DAYS, 504, 756),
)
TAKEAWAY_DELTAS: tuple[tuple[float, ...], ...] = (
    synthetic.CUI_CALL_DELTAS,
    synthetic.NINE_CALL_DELTAS,
    tuple(float(x) for x in np.round(np.linspace(0.9, 0.1, 17), 2)),
)

#: Gatheral and Jacquier (2014), Example 3.1, in the order (a, b, rho, m, sigma).
VOGT = svi.RawSVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
#: Example 5.1: the jump-wings parameters of the same smile at t = 1.
VOGT_JW = (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249)

#: Le Floc'h (2016), Tables 1 and 3: (theta, strike / forward, T, kind, published, scale).
LEFLOCH_CASES = (
    ("Le Floc'h Table 1, call, F=1000 K=1400 T=0.0182",
     (0.826, 0.320, -0.557, 0.254, 0.344), 1.4, 0.0182, "call", 0.1073414, 1000.0),
    ("Le Floc'h Table 3, put, K/F=0.25 T=1, per 1e6 notional",
     (0.0225, 0.01, 0.5, 0.1, 2.0), 0.25, 1.0, "put", 119.385324, 1.0e6),
)  # fmt: skip

#: Cui et al. (2017), Table 8, lower triangle, order (kappa, vbar, sigma, rho, v0).
CUI_TABLE8_ORDER = ("kappa", "vbar", "sigma", "rho", "v0")
CUI_TABLE8 = (
    (5.26e-5,),
    (9.65e-3, 2.26e1),
    (-5.49e-4, -7.66e-2, 7.46e-3),
    (1.61e-4, 2.00e-2, -2.34e-3, 7.56e-4),
    (5.28e-3, 1.18e1, -3.53e-2, 8.40e-3, 9.69e-1),
)
CUI_CONDITION_NUMBER = 3.978e6
#: A fitted rho at or below this is counted as stuck on the edge of the search box.
LOWER_RHO_EDGE = -0.9985


@dataclass(frozen=True)
class Config:
    """Sizes of the experiments."""

    n_profile: int = 41
    n_map_x: int = 31
    n_map_y: int = 25
    n_starts: int = 100
    n_noise: int = 500
    n_market_starts: int = 20
    seed: int = 20261123

    @staticmethod
    def fast() -> Config:
        """A reduced run for CI: the same experiments on coarser grids."""
        return Config(n_profile=13, n_map_x=9, n_map_y=7, n_starts=10, n_noise=30, n_market_starts=4)

    @staticmethod
    def tiny() -> Config:
        """The smallest run that still exercises every experiment (tests)."""
        return Config(n_profile=7, n_map_x=4, n_map_y=3, n_starts=3, n_noise=3, n_market_starts=2)


def fmt(x: Any) -> Any:
    """Numbers to eight significant digits; everything else unchanged."""
    if isinstance(x, bool | np.bool_):
        return str(bool(x))
    if isinstance(x, float | np.floating):
        return f"{float(x):.8g}"
    if isinstance(x, int | np.integer):
        return str(int(x))
    return x


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write a small table with deterministic number formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(x) for x in row])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with sorted keys and rounded floats."""

    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [clean(v) for v in x]
        if isinstance(x, bool | np.bool_):
            return bool(x)
        if isinstance(x, float | np.floating):
            return float(f"{float(x):.8g}") if np.isfinite(x) else None
        if isinstance(x, int | np.integer):
            return int(x)
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _kappa_grid(n: int) -> FloatArray:
    out: FloatArray = np.geomspace(0.1, 40.0, n)
    return out


def _interval_row(profile: identify.Profile, h: float) -> list[Any]:
    scaled = identify.Profile(
        profile.index, profile.grid, profile.theta, profile.rmse_vol, profile.max_dvol,
        profile.max_dvol / h,
    )  # fmt: skip
    low, high, low_open, high_open = scaled.interval()
    return [low, high, high / low, low_open, high_open]


# --------------------------------------------------------------------------- validation


def validation_rows(cfg: Config) -> tuple[list[list[Any]], dict[str, Any]]:
    """Checks of the code against published numbers and independent calculations."""
    rows: list[list[Any]] = []
    head: dict[str, Any] = {}

    for name, theta, strike, t, kind, published, scale in LEFLOCH_CASES:
        k = float(np.log(strike))
        call = heston.call_price_reference(theta, k, t)
        value = (call if kind == "call" else call - 1.0 + strike) * scale
        rows.append([name, value, published, value - published])

    jw = svi.raw_to_jump_wings(VOGT, 1.0)
    for label, mine, pub in zip(
        ("v", "psi", "p", "c", "v_tilde"), (jw.v, jw.psi, jw.p, jw.c, jw.v_tilde), VOGT_JW,
        strict=True,
    ):  # fmt: skip
        rows.append([f"Gatheral-Jacquier Example 5.1, SVI-JW {label}", mine, pub, mine - pub])
    vogt = arbitrage.check_butterfly(VOGT, np.linspace(-1.5, 1.5, 3001))
    rows.append(["Gatheral-Jacquier Example 3.1, min of g on [-1.5, 1.5]", vogt.min_g, "< 0", ""])
    head["vogt_min_g"] = vogt.min_g
    head["vogt_argmin_k"] = vogt.argmin_k

    # analytic gradient against central differences, and fixed nodes against adaptive quadrature
    truth = synthetic.CUI_THETA
    prob = synthetic.delta_grid(truth, weighting="price")
    price, grad = heston.call_price_and_gradient(truth, prob.k, prob.t)
    fd = np.empty_like(grad)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-6
        fd[:, j] = (
            heston.call_price(truth + e, prob.k, prob.t)
            - heston.call_price(truth - e, prob.k, prob.t)
        ) / 2e-6
    grad_err = float(np.max(np.abs(grad - fd)))
    ref = np.array(
        [
            heston.call_price_reference(truth, float(k), float(t))
            for k, t in zip(prob.k, prob.t, strict=True)
        ]
    )
    quad_err = float(np.max(np.abs(price - ref)))
    cui_rule = heston.Quadrature.gauss_legendre()
    cui_err = float(
        np.max(np.abs(heston.call_price(truth, prob.k, prob.t, quadrature=cui_rule) - ref))
    )
    rows.append(["analytic gradient minus central difference, max abs", grad_err, 0.0, grad_err])
    rows.append(["256-node rule minus adaptive quadrature, max abs price", quad_err, 0.0, quad_err])
    rows.append(
        ["Cui 64-node rule minus adaptive quadrature, max abs price", cui_err, 0.0, cui_err]
    )
    head.update(
        gradient_max_error=grad_err, quadrature_max_error=quad_err, cui_rule_max_error=cui_err
    )

    # the three forms of the characteristic function at T = 15 (Cui Fig. 2)
    u = np.linspace(0.05, 4.0, 400)
    trap = heston.charfn(truth, u, 15.0, "trap")
    forms: tuple[heston.Form, ...] = ("original", "cui")
    for form in forms:
        gap = float(np.max(np.abs(heston.charfn(truth, u, 15.0, form) - trap)))
        rows.append([f"char. function, {form} form minus trap form, T=15, max abs", gap, "", ""])
        head[f"charfn_gap_{form}_T15"] = gap
    return rows, head


def cui_hessian_rows() -> tuple[list[list[Any]], dict[str, Any]]:
    """Gauss-Newton Hessian at the optimum of the Cui design, against their Table 8."""
    prob = synthetic.delta_grid(synthetic.CUI_THETA, weighting="price")
    hess = prob.hessian(synthetic.CUI_THETA)
    idx = [PARAM_NAMES.index(n) for n in CUI_TABLE8_ORDER]
    rows: list[list[Any]] = []
    for i, name_i in enumerate(CUI_TABLE8_ORDER):
        for j in range(i + 1):
            mine = float(hess[idx[i], idx[j]])
            pub = CUI_TABLE8[i][j]
            rows.append([name_i, CUI_TABLE8_ORDER[j], mine, pub, mine / pub])
    cond = condition_number(hess)
    rows.append(["condition number", "", cond, CUI_CONDITION_NUMBER, cond / CUI_CONDITION_NUMBER])
    published = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1):
            published[i, j] = published[j, i] = CUI_TABLE8[i][j]
    head = {
        "cui_condition_number": cond,
        "cui_condition_number_published": CUI_CONDITION_NUMBER,
        "cui_table8_min_eigenvalue_as_printed": float(np.linalg.eigvalsh(published)[0]),
    }
    return rows, head


# --------------------------------------------------------------------------- SVI on synthetic data


def svi_synthetic_rows() -> tuple[list[list[Any]], dict[str, Any]]:
    """SVI recovers its own parameters, and fits Heston smiles."""
    rows: list[list[Any]] = []
    truth = svi.RawSVI(a=0.02, b=0.25, rho=-0.45, m=0.05, sigma=0.15)
    k = np.linspace(-0.6, 0.4, 25)
    t = 0.5
    fit = svi.fit_raw(k, np.sqrt(truth.w(k) / t), t)
    err = float(np.max(np.abs(fit.params.as_array() - truth.as_array())))
    min_g = arbitrage.check_butterfly(fit.params, k).min_g
    rows.append(["SVI to SVI data", t, fit.rmse_vol, err, min_g])
    worst = 0.0
    for days in (30, 90, 252, 504):
        tt = days / 252.0
        kk = np.array(
            [
                synthetic.strike_for_delta(synthetic.CUI_THETA, tt, d)
                for d in np.linspace(0.95, 0.05, 19)
            ]
        )
        vol = black.implied_vol(kk, tt, heston.call_price(synthetic.CUI_THETA, kk, tt))
        f = svi.fit_raw(kk, vol, tt)
        min_g = arbitrage.check_butterfly(f.params, np.linspace(kk.min(), kk.max(), 801)).min_g
        rows.append([f"SVI to Heston smile, {days} days", tt, f.rmse_vol, "", min_g])
        worst = max(worst, f.rmse_vol)
    return rows, {"svi_self_recovery_max_param_error": err, "svi_to_heston_worst_rmse_vol": worst}


# --------------------------------------------------------------------------- synthetic identifiability


@dataclass(frozen=True)
class Valley:
    """Everything the figures need about one identifiability study."""

    problem: Problem
    theta_hat: FloatArray
    half_spread: FloatArray
    profile: identify.Profile
    sets: FloatArray
    set_ratios: FloatArray
    surface: identify.ProfileMap


def _profile_rows(profile: identify.Profile) -> list[list[Any]]:
    return [
        [
            profile.grid[i],
            *profile.theta[i],
            profile.rmse_vol[i],
            profile.max_dvol[i],
            profile.ratio[i],
        ]
        for i in range(profile.grid.size)
    ]


PROFILE_HEADER = [
    "kappa_fixed",
    *PARAM_NAMES,
    "rmse_vol_to_target",
    "max_dvol_to_reference",
    "ratio_to_half_spread",
]


def _map_rows(m: identify.ProfileMap) -> list[list[Any]]:
    return [
        [m.grid_x[i], m.grid_y[j], m.max_dvol[j, i], m.ratio[j, i]]
        for j in range(m.grid_y.size)
        for i in range(m.grid_x.size)
    ]


def _set_rows(
    problem: Problem, sets: FloatArray, ratios: FloatArray, ref: FloatArray
) -> list[list[Any]]:
    rows = []
    for th, r in zip(sets, ratios, strict=True):
        dv = float(np.max(np.abs(problem.model_vol(th) - ref)))
        feller = 2.0 * th[3] * th[1] / th[4] ** 2
        rows.append([*th, dv, r, feller])
    return rows


SET_HEADER = [*PARAM_NAMES, "max_dvol_to_reference", "ratio_to_half_spread", "feller_ratio"]


def run_synthetic(cfg: Config, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every experiment that needs no data. Returns ``(headline, plot_data)``."""
    head: dict[str, Any] = {}
    rows, h = validation_rows(cfg)
    head.update(h)
    write_csv(out / "validation.csv", ["check", "value", "published", "difference"], rows)
    rows, h = cui_hessian_rows()
    head.update(h)
    write_csv(out / "cui_hessian.csv", ["row", "column", "this_repo", "cui_table_8", "ratio"], rows)
    rows, h = svi_synthetic_rows()
    head.update(h)
    write_csv(
        out / "svi_synthetic.csv",
        ["case", "t", "rmse_vol", "max_param_error", "min_g_on_fitted_range"],
        rows,
    )

    theta = synthetic.CUI_THETA
    prob = synthetic.delta_grid(theta)
    hs = SYNTHETIC_HALF_SPREAD

    # 1. does the optimiser find the truth from random starts (Cui Algorithm 5.1, one optimum)
    starts = random_starts(cfg.n_starts, cfg.seed)
    fits = [calibrate(prob, s) for s in starts]
    rel = np.array([np.max(np.abs(f.theta / theta - 1.0)) for f in fits])
    write_csv(
        out / "recovery.csv",
        [
            "start",
            *[f"start_{n}" for n in PARAM_NAMES],
            *PARAM_NAMES,
            "rmse_vol",
            "max_rel_param_error",
            "nfev",
        ],
        [[i, *starts[i], *f.theta, f.rmse_vol, rel[i], f.nfev] for i, f in enumerate(fits)],
    )
    head["recovery_starts"] = cfg.n_starts
    head["recovery_within_1e-4"] = int(np.sum(rel < 1e-4))
    head["recovery_worst_rmse_vol"] = float(max(f.rmse_vol for f in fits))

    # 2. the Hessian in relative parameter changes
    hess = scaled_hessian(prob.hessian(theta), theta)
    ev, evec = np.linalg.eigh(hess)
    write_csv(
        out / "hessian_eigen.csv",
        ["eigenvalue", *[f"d_log_{n}" for n in PARAM_NAMES]],
        [[ev[i], *evec[:, i]] for i in range(5)],
    )
    head["scaled_hessian_condition_number"] = float(ev[-1] / ev[0])

    # 3. profile along kappa, and how wide the valley is at each tolerance
    profile = identify.profile_1d(prob, theta, _kappa_grid(cfg.n_profile), hs)
    write_csv(out / "profile_kappa_synthetic.csv", PROFILE_HEADER, _profile_rows(profile))
    write_csv(
        out / "valley_width_synthetic.csv",
        ["half_spread_vol", "kappa_low", "kappa_high", "high_over_low", "low_open", "high_open"],
        [[tol, *_interval_row(profile, tol)] for tol in TOLERANCES],
    )
    low, high, _, _ = profile.interval()
    head.update(
        synthetic_true_kappa=float(theta[3]),
        synthetic_half_spread_vol=hs,
        synthetic_kappa_low=low,
        synthetic_kappa_high=high,
        synthetic_max_dvol_over_grid=float(profile.max_dvol.max()),
        synthetic_kappa_grid=[float(profile.grid[0]), float(profile.grid[-1])],
    )

    # 4. five parameter sets inside the spread
    sets, ratios = identify.equivalent_sets(prob, profile, hs, reference_vol=prob.vol)
    write_csv(
        out / "equivalent_sets_synthetic.csv", SET_HEADER, _set_rows(prob, sets, ratios, prob.vol)
    )
    head["synthetic_sets_max_ratio"] = float(ratios.max())

    # 5. the loss over (kappa, sigma)
    surface = identify.profile_2d(
        prob, theta, np.geomspace(0.3, 30.0, cfg.n_map_x), np.geomspace(0.05, 2.0, cfg.n_map_y), hs
    )
    write_csv(
        out / "profile_map_synthetic.csv",
        ["kappa", "sigma", "max_dvol", "ratio_to_half_spread"],
        _map_rows(surface),
    )

    # 6. the same question for different sets of expiries
    designs = (
        ("1 expiry: 90 days", (90,)),
        ("3 expiries: 30 to 180 days", (30, 90, 180)),
        ("8 expiries: 30 to 360 days", synthetic.CUI_DAYS),
        ("10 expiries: 30 to 756 days", (*synthetic.CUI_DAYS, 504, 756)),
    )
    design_rows: list[list[Any]] = []
    design_profiles = []
    noise_sd = identify.uniform_noise_sd(hs)
    for label, days in designs:
        p = synthetic.delta_grid(theta, days, synthetic.NINE_CALL_DELTAS)
        cond = condition_number(scaled_hessian(p.hessian(theta), theta))
        prof = identify.profile_1d(p, theta, _kappa_grid(cfg.n_profile), hs)
        design_profiles.append((label, prof))
        sd_kappa = float(identify.fisher_sd(p, theta, noise_sd)[identify.KAPPA])
        lik = identify.likelihood_interval(prof, p.k.size, noise_sd)
        design_rows.append(
            [
                label,
                p.k.size,
                cond,
                *_interval_row(prof, 0.0025),
                *_interval_row(prof, 0.005),
                float(prof.max_dvol.max()),
                *[identify.spread_needed(prof, float(theta[3]), x) for x in KAPPA_TOLERANCES],
                sd_kappa,
                *lik,
            ]
        )
    write_csv(
        out / "design_study.csv",
        ["design", "n_options", "scaled_hessian_condition_number",
         "kappa_low_25bp", "kappa_high_25bp", "high_over_low_25bp", "low_open_25bp", "high_open_25bp",
         "kappa_low_50bp", "kappa_high_50bp", "high_over_low_50bp", "low_open_50bp", "high_open_50bp",
         "max_dvol_over_grid",
         *[f"half_spread_to_exclude_kappa_off_by_{round(100 * x)}pct" for x in KAPPA_TOLERANCES],
         "fisher_sd_kappa_25bp", "likelihood90_low_25bp", "likelihood90_high_25bp",
         "likelihood90_low_open", "likelihood90_high_open"],
        design_rows,
    )  # fmt: skip

    # 6b. the same from the Fisher information alone, for more designs
    take_rows: list[list[Any]] = []
    for days in TAKEAWAY_EXPIRIES:
        for deltas in TAKEAWAY_DELTAS:
            p = synthetic.delta_grid(theta, days, deltas)
            # standard deviation of kappa per unit of noise standard deviation
            unit = float(identify.fisher_sd(p, theta, 1.0)[identify.KAPPA])
            rel90 = identify.Z90 * unit * noise_sd / float(theta[3])
            take_rows.append(
                [
                    len(days), days[0], days[-1], len(deltas), p.k.size, unit * noise_sd, rel90,
                    *[x * float(theta[3]) / (identify.Z90 * unit) * np.sqrt(3.0)
                      for x in KAPPA_TOLERANCES],
                ]
            )  # fmt: skip
    write_csv(
        out / "takeaway.csv",
        ["n_expiries", "first_day", "last_day", "strikes_per_expiry", "n_options",
         "fisher_sd_kappa_25bp", "fisher_90pct_half_width_relative_25bp",
         *[f"half_spread_for_90pct_within_{round(100 * x)}pct" for x in KAPPA_TOLERANCES]],
        take_rows,
    )  # fmt: skip

    # 7. calibrate to quotes moved inside their spread
    est = np.array(
        [
            calibrate(synthetic.with_noise(prob, hs, cfg.seed + 1 + i), theta).theta
            for i in range(cfg.n_noise)
        ]
    )
    q = np.quantile(est, [0.0, 0.05, 0.5, 0.95, 1.0], axis=0)
    write_csv(
        out / "noise_study.csv",
        ["parameter", "true", "min", "q05", "median", "q95", "max"],
        [[n, theta[j], *q[:, j]] for j, n in enumerate(PARAM_NAMES)],
    )
    head.update(
        noise_replications=cfg.n_noise,
        noise_kappa_q05=float(q[1, 3]),
        noise_kappa_q95=float(q[3, 3]),
        noise_kappa_min=float(q[0, 3]),
        noise_kappa_max=float(q[4, 3]),
    )


    # 8. three ways to put an interval on each parameter, on the 40-option design
    cov = identify.fisher_covariance(prob, theta, noise_sd)
    f_sd = np.sqrt(np.diag(cov))
    mc_sd = est.std(axis=0, ddof=1)
    at_rho_bound = float(np.mean(est[:, 2] <= LOWER_RHO_EDGE))
    write_csv(
        out / "identifiability.csv",
        ["parameter", "true", "fisher_sd", "monte_carlo_sd", "monte_carlo_over_fisher",
         "fisher_q05", "fisher_q95", "monte_carlo_q05", "monte_carlo_q95", "monte_carlo_mean"],
        [
            [n, theta[j], f_sd[j], mc_sd[j], mc_sd[j] / f_sd[j],
             theta[j] - identify.Z90 * f_sd[j], theta[j] + identify.Z90 * f_sd[j],
             q[1, j], q[3, j], float(est[:, j].mean())]
            for j, n in enumerate(PARAM_NAMES)
        ],
    )  # fmt: skip
    lik_low, lik_high, _, _ = identify.likelihood_interval(profile, prob.k.size, noise_sd)
    k_true, k_sd = float(theta[3]), float(f_sd[3])
    write_csv(
        out / "kappa_intervals.csv",
        ["method", "kappa_low", "kappa_high", "high_over_low"],
        [
            [name, a, b, b / a]
            for name, a, b in (
                ("every quote inside the half-spread (profile, largest error)", low, high),
                ("profile likelihood, 90%", lik_low, lik_high),
                ("Fisher information, 90%", k_true - identify.Z90 * k_sd, k_true + identify.Z90 * k_sd),
                ("recalibration to noisy quotes, 5% to 95%", float(q[1, 3]), float(q[3, 3])),
            )
        ],
    )  # fmt: skip
    head.update(
        noise_sd_vol=noise_sd,
        fisher_sd_kappa=k_sd,
        monte_carlo_sd_kappa=float(mc_sd[3]),
        fisher_kappa_q05=k_true - identify.Z90 * k_sd,
        fisher_kappa_q95=k_true + identify.Z90 * k_sd,
        likelihood_kappa_low=lik_low,
        likelihood_kappa_high=lik_high,
        fisher_correlation_kappa_vbar=float(cov[3, 1] / (f_sd[3] * f_sd[1])),
        fisher_correlation_rho_sigma=float(cov[2, 4] / (f_sd[2] * f_sd[4])),
        noise_share_at_rho_bound=at_rho_bound,
    )

    valley = Valley(prob, theta, np.full(prob.k.shape, hs), profile, sets, ratios, surface)
    return head, {
        "valley": valley,
        "design_profiles": design_profiles,
        "noise_estimates": est,
        "fisher_sd": f_sd,
    }


# --------------------------------------------------------------------------- a market-like chain


def _delta_strikes(s: Slice) -> FloatArray:
    """Log-moneyness at nine Black deltas on the SVI slice, kept inside the quoted range."""
    out = []
    for delta in synthetic.NINE_CALL_DELTAS:

        def gap(k: float, delta: float = delta) -> float:
            vol = float(s.vol(np.array([k]))[0])
            return float(black.call_delta(k, vol, s.t)) - delta

        lo, hi = s.k_min, s.k_max
        if gap(lo) * gap(hi) < 0.0:
            out.append(float(brentq(gap, lo, hi, xtol=1e-12)))
    return np.array(out)


def market_problem(slices: list[Slice]) -> tuple[Problem, FloatArray, list[Slice]]:
    """Heston calibration target built from the fitted slices: ``(problem, half_spread, used)``."""
    used = [s for s in slices if s.days >= HESTON_MIN_DAYS]
    ks, ts, vols, hs = [], [], [], []
    for s in used:
        k = _delta_strikes(s)
        ks.append(k)
        ts.append(np.full(k.shape, s.t))
        vols.append(s.vol(k))
        hs.append(s.half_spread(k))
    prob = Problem(k=np.concatenate(ks), t=np.concatenate(ts), vol=np.concatenate(vols))
    return prob, np.concatenate(hs), used


#: Penalty weight and number of passes over the expiries of the arbitrage-free refit.
REFIT_PENALTY = 1.0e4
REFIT_SWEEPS = 3
#: Grid on which the refit penalises negative density and crossed slices.
REFIT_GRID: FloatArray = np.linspace(-3.0, 3.0, 601)
#: Margins the refit keeps: ``g`` above the first, the gap between slices above the second.
REFIT_G_MARGIN = 1.0e-3
REFIT_GAP_MARGIN = 1.0e-6
#: How far past the quoted strikes the extrapolation is scored, as a multiple of the range end.
EXTRAPOLATION = 1.25


def arbitrage_free_slices(
    quotes: list[Quotes], start: list[svi.RawSVI], *, sweeps: int = REFIT_SWEEPS
) -> list[svi.RawSVI]:
    """Refit every slice to its quotes with penalties for butterfly and calendar arbitrage.

    This follows the recipe of Gatheral and Jacquier, section 5.2. The start
    is a surface that is already free of static arbitrage (the SSVI fit), and
    each slice is then refitted to its own quotes with penalties for a
    negative ``g`` and for crossing either neighbour. The slices are visited
    in order of expiry, ``sweeps`` times, so a slice can move again after its
    neighbours have. The penalties are soft, so the result is checked
    afterwards and the check is what is reported.
    """
    out = list(start)
    for _ in range(sweeps):
        for i, q in enumerate(quotes):
            before = out[i - 1] if i else None
            after = out[i + 1] if i + 1 < len(out) else None

            def penalty(
                p: svi.RawSVI,
                before: svi.RawSVI | None = before,
                after: svi.RawSVI | None = after,
            ) -> FloatArray:
                w = p.w(REFIT_GRID)
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = np.where(w > 1e-10, arbitrage.g_function(p, REFIT_GRID), -1.0)
                parts = [
                    REFIT_PENALTY * np.minimum(g - REFIT_G_MARGIN, 0.0),
                    REFIT_PENALTY * np.minimum(w, 0.0),
                ]
                if before is not None:
                    gap = w - before.w(REFIT_GRID)
                    parts.append(REFIT_PENALTY * np.minimum(gap - REFIT_GAP_MARGIN, 0.0))
                if after is not None:
                    gap = after.w(REFIT_GRID) - w
                    parts.append(REFIT_PENALTY * np.minimum(gap - REFIT_GAP_MARGIN, 0.0))
                return np.concatenate(parts)

            fit = svi.fit_raw(
                q.k, q.mid, q.t, weights=1.0 / q.half_spread, penalty=penalty, start=out[i],
                only_start=True,
            )  # fmt: skip
            out[i] = fit.params
    return out


def market_starts(n: int, seed: int) -> FloatArray:
    """Starting points for the market calibration, uniform in a wide box."""
    rng = np.random.default_rng(seed)
    lo = np.array([0.01, 0.01, -0.90, 0.30, 0.10])
    hi = np.array([0.20, 0.20, 0.30, 10.0, 2.00])
    out: FloatArray = lo + (hi - lo) * rng.random((n, 5))
    return out


def _extrapolation_errors(spec: MarketSpec, s: Slice, p: svi.RawSVI) -> tuple[float, float]:
    """Fitted minus true volatility just past the lowest and the highest quoted strike."""
    k = np.array([EXTRAPOLATION * s.k_min, EXTRAPOLATION * s.k_max])
    with np.errstate(invalid="ignore"):
        fitted = np.sqrt(p.w(k) / s.t)
    err = fitted - market.true_vol(spec, k, s.t)
    return float(err[0]), float(err[1])


def run_market(
    cfg: Config, out: Path, spec: MarketSpec | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a market-like chain and run the surface fits and the Heston study on it."""
    spec = spec or MarketSpec()
    head: dict[str, Any] = {}
    quotes = market.generate(spec, cfg.seed + 11)
    slices = [market.fit_slice(q, spec) for q in quotes]
    model = spec.model
    write_csv(
        out / "quotes.csv",
        ["days", "log_moneyness", "true_vol", "mid_vol", "bid_vol", "ask_vol"],
        [
            [q.days, q.k[i], q.true_vol[i], q.mid[i], q.bid[i], q.ask[i]]
            for q in quotes
            for i in range(q.k.size)
        ],
    )
    head.update(
        generator={
            "model": "Bates",
            **{n: float(v) for n, v in zip(PARAM_NAMES, model.heston, strict=True)},
            "jump_rate": model.lam,
            "jump_mean": model.mu_j,
            "jump_sd": model.delta_j,
        },
        generator_feller_ratio=float(2.0 * model.kappa * model.vbar / model.sigma**2),
        generator_jump_share_of_long_run_variance=float(
            model.jump_variance_rate / (model.jump_variance_rate + model.vbar)
        ),
        half_spread_atm_vol=spec.half_spread_atm,
        half_spread_cap_vol=spec.half_spread_cap,
    )

    # SVI slices: fit quality and static arbitrage
    rows: list[list[Any]] = []
    n_bfly_range = n_bfly_all = 0
    for i, s in enumerate(slices):
        min_g_range, free_range = butterfly_on_range(s)
        everywhere = arbitrage.check_butterfly(s.svi)
        if i + 1 < len(slices):
            nxt = slices[i + 1]
            lo, hi = max(s.k_min, nxt.k_min), min(s.k_max, nxt.k_max)
            cross_range = arbitrage.crossedness(s.svi, nxt.svi, np.linspace(lo, hi, 801))
            cross_all = arbitrage.crossedness(s.svi, nxt.svi)
        else:
            cross_range = cross_all = 0.0
        n_bfly_range += not free_range
        n_bfly_all += not everywhere.free
        rows.append([
            s.days, s.n_quotes, s.k_min, s.k_max, float(s.vol(np.array([0.0]))[0]),
            s.rmse_vol, s.share_inside, s.median_half_spread, s.rmse_vol_to_truth,
            *_extrapolation_errors(spec, s, s.svi),
            min_g_range, everywhere.min_g, everywhere.argmin_k, everywhere.free, cross_range, cross_all,
        ])  # fmt: skip
    write_csv(
        out / "svi_slices.csv",
        ["days", "n_quotes", "k_min", "k_max", "atm_vol", "rmse_vol_to_mid", "share_inside_bid_ask",
         "median_half_spread_vol", "rmse_vol_to_truth", "low_strike_extrapolation_error",
         "high_strike_extrapolation_error", "min_g_quoted_range", "min_g_all_k", "argmin_g_all_k",
         "butterfly_free_all_k", "crossedness_next_quoted_range", "crossedness_next_all_k"],
        rows,
    )  # fmt: skip
    n_quotes = sum(s.n_quotes for s in slices)
    extrap = np.array([[r[9], r[10]] for r in rows])
    head.update(
        n_expiries=len(slices),
        n_quotes=n_quotes,
        svi_share_inside=float(sum(s.share_inside * s.n_quotes for s in slices) / n_quotes),
        svi_rmse_vol_to_truth=float(
            np.sqrt(sum(s.rmse_vol_to_truth**2 * s.n_quotes for s in slices) / n_quotes)
        ),
        svi_max_abs_extrapolation_error=float(np.nanmax(np.abs(extrap))),
        svi_slices_with_butterfly_arbitrage_quoted_range=int(n_bfly_range),
        svi_slices_with_butterfly_arbitrage_all_k=int(n_bfly_all),
        svi_calendar_crossings_quoted_range=int(sum(r[15] > 0.0 for r in rows)),
        svi_calendar_crossings_all_k=int(sum(r[16] > 1e-12 for r in rows)),
    )

    # SSVI: one surface, free of static arbitrage by construction
    t = np.array([s.t for s in slices])
    theta_atm = np.array([float(s.svi.w(0.0)) for s in slices])
    grids = [_delta_strikes(s) for s in slices]
    vols = [s.vol(g) for s, g in zip(slices, grids, strict=True)]
    wts = [1.0 / s.half_spread(g) for s, g in zip(slices, grids, strict=True)]
    surf = svi.fit_ssvi(grids, vols, t, theta_atm, weights=wts)
    check_surface = arbitrage.check_ssvi(surf)
    ssvi_rows: list[list[Any]] = []
    inside_total = 0
    for i, s in enumerate(slices):
        fitted = np.sqrt(surf.w_theta(grids[i], surf.theta[i]) / s.t)
        err = fitted - vols[i]
        inside = int(np.sum(np.abs(err) <= s.half_spread(grids[i])))
        inside_total += inside
        ssvi_rows.append([
            s.days, grids[i].size, surf.theta[i], theta_atm[i],
            float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))), inside / grids[i].size,
        ])  # fmt: skip
    write_csv(
        out / "ssvi_fit.csv",
        ["days", "n_points", "theta_used", "theta_from_slice", "rmse_vol_vs_svi",
         "max_abs_vol_vs_svi", "share_inside_half_spread"],
        ssvi_rows,
    )  # fmt: skip
    n_points = sum(r[1] for r in ssvi_rows)
    head.update(
        ssvi_rho=surf.rho,
        ssvi_eta=surf.eta,
        ssvi_gamma=surf.gamma,
        ssvi_static_arbitrage_free=check_surface.free,
        ssvi_theta_monotone_in_data=bool(np.all(np.diff(theta_atm) >= 0.0)),
        ssvi_rmse_vol_vs_svi=float(np.sqrt(sum(r[4] ** 2 * r[1] for r in ssvi_rows) / n_points)),
        ssvi_share_inside=inside_total / n_points,
    )

    # the same slices refitted with penalties for butterfly and calendar arbitrage
    start = [surf.slice(i) for i in range(len(slices))]
    repaired = arbitrage_free_slices(quotes, start)
    grid_all = arbitrage.DEFAULT_GRID
    rep_rows: list[list[Any]] = []
    quotes_inside = 0
    for i, (s, q, p) in enumerate(zip(slices, quotes, repaired, strict=True)):
        at_quotes = np.sqrt(p.w(q.k) / q.t)
        err = at_quotes - q.mid
        in_quotes = int(np.sum((at_quotes >= q.bid) & (at_quotes <= q.ask)))
        quotes_inside += in_quotes
        cross = arbitrage.crossedness(repaired[i - 1], p, grid_all) if i else 0.0
        check = arbitrage.check_butterfly(p)
        rep_rows.append([
            s.days, p.a, p.b, p.rho, p.m, p.sigma,
            float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))),
            in_quotes / q.k.size, float(np.sqrt(np.mean((at_quotes - q.true_vol) ** 2))),
            *_extrapolation_errors(spec, s, p), check.min_g, check.free, cross,
        ])  # fmt: skip
    write_csv(
        out / "svi_slices_arbitrage_free.csv",
        ["days", "a", "b", "rho", "m", "sigma", "rmse_vol_to_mid", "max_abs_vol_to_mid",
         "share_inside_bid_ask", "rmse_vol_to_truth",
         "low_strike_extrapolation_error", "high_strike_extrapolation_error",
         "min_g_all_k", "butterfly_free_all_k", "crossedness_previous_all_k"],
        rep_rows,
    )  # fmt: skip
    rep_extrap = np.array([[r[10], r[11]] for r in rep_rows])
    start_inside = sum(
        int(np.sum((np.sqrt(p.w(q.k) / q.t) >= q.bid) & (np.sqrt(p.w(q.k) / q.t) <= q.ask)))
        for p, q in zip(start, quotes, strict=True)
    )
    head.update(
        repaired_slices_butterfly_free=int(sum(bool(r[13]) for r in rep_rows)),
        repaired_calendar_crossings=int(sum(r[14] > 0.0 for r in rep_rows)),
        repaired_max_crossedness=float(max(r[14] for r in rep_rows)),
        repaired_share_inside=quotes_inside / n_quotes,
        repaired_rmse_vol_to_truth=float(
            np.sqrt(sum(r[9] ** 2 * s.n_quotes for r, s in zip(rep_rows, slices, strict=True)) / n_quotes)
        ),
        repaired_max_abs_extrapolation_error=float(np.nanmax(np.abs(rep_extrap))),
        ssvi_share_of_quotes_inside_bid_ask=start_inside / n_quotes,
    )

    # Heston on the SVI-smoothed quotes
    prob, hs, used = market_problem(slices)
    starts = market_starts(cfg.n_market_starts, cfg.seed + 7)
    fits = [calibrate(prob, s) for s in starts]
    best = min(fits, key=lambda f: f.cost)
    write_csv(
        out / "heston_multistart.csv",
        ["start", *PARAM_NAMES, "rmse_vol", "max_abs_vol", "max_abs_dvol_vs_best", "nfev"],
        [
            [i, *f.theta, f.rmse_vol, f.max_abs_vol,
             float(np.max(np.abs(prob.model_vol(f.theta) - prob.model_vol(best.theta)))), f.nfev]
            for i, f in enumerate(fits)
        ],
    )  # fmt: skip
    ref = prob.model_vol(best.theta)
    err = ref - prob.vol
    same = [f for f in fits if f.rmse_vol <= best.rmse_vol * 1.001 + 1e-9]
    # the same calibration against the true smile, to separate the model's misfit from quote noise
    truth_vol = np.concatenate([market.true_vol(spec, prob.k[prob.t == s.t], s.t) for s in used])
    clean = calibrate(Problem(k=prob.k, t=prob.t, vol=truth_vol), best.theta)
    head.update(
        heston_expiries=len(used),
        heston_options=int(prob.k.size),
        heston_theta={n: float(best.theta[j]) for j, n in enumerate(PARAM_NAMES)},
        heston_rmse_vol=best.rmse_vol,
        heston_max_abs_vol=best.max_abs_vol,
        heston_share_inside_half_spread=float(np.mean(np.abs(err) <= hs)),
        heston_median_half_spread_vol=float(np.median(hs)),
        heston_starts=cfg.n_market_starts,
        heston_starts_reaching_best_rmse=len(same),
        heston_kappa_range_among_best=[
            float(min(f.theta[3] for f in same)),
            float(max(f.theta[3] for f in same)),
        ],
        heston_feller_ratio=float(2.0 * best.theta[3] * best.theta[1] / best.theta[4] ** 2),
        heston_theta_fitted_to_true_smile={
            n: float(clean.theta[j]) for j, n in enumerate(PARAM_NAMES)
        },
        heston_rmse_vol_fitted_to_true_smile=clean.rmse_vol,
        target_rmse_vol_to_truth=float(np.sqrt(np.mean((prob.vol - truth_vol) ** 2))),
    )
    write_csv(
        out / "heston_fit_by_expiry.csv",
        ["days", "n_options", "rmse_vol", "max_abs_vol", "median_half_spread_vol", "share_inside_half_spread"],
        [
            [s.days, int(np.sum(prob.t == s.t)), float(np.sqrt(np.mean(err[prob.t == s.t] ** 2))),
             float(np.max(np.abs(err[prob.t == s.t]))), float(np.median(hs[prob.t == s.t])),
             float(np.mean(np.abs(err[prob.t == s.t]) <= hs[prob.t == s.t]))]
            for s in used
        ],
    )  # fmt: skip

    grid = np.geomspace(0.05, 50.0, cfg.n_profile)
    profile = identify.profile_1d(prob, best.theta, grid, hs)
    write_csv(out / "profile_kappa.csv", PROFILE_HEADER, _profile_rows(profile))
    low, high, low_open, high_open = profile.interval()
    sets, ratios = identify.equivalent_sets(prob, profile, hs, reference_vol=ref)
    write_csv(out / "equivalent_sets.csv", [*SET_HEADER, "rmse_vol_to_target"],
              [[*row, float(np.sqrt(np.mean(prob.vol_errors(th) ** 2)))] for row, th in zip(_set_rows(prob, sets, ratios, ref), sets, strict=True)])  # fmt: skip
    surface = identify.profile_2d(
        prob,
        best.theta,
        np.geomspace(max(best.theta[3] / 30.0, 0.02), min(best.theta[3] * 30.0, 55.0), cfg.n_map_x),
        np.geomspace(best.theta[4] / 8.0, min(best.theta[4] * 8.0, 14.0), cfg.n_map_y),
        hs,
    )
    write_csv(
        out / "profile_map.csv",
        ["kappa", "sigma", "max_dvol", "ratio_to_half_spread"],
        _map_rows(surface),
    )
    head.update(
        heston_kappa_low=low,
        heston_kappa_high=high,
        heston_kappa_low_open=low_open,
        heston_kappa_high_open=high_open,
        heston_kappa_grid=[float(grid[0]), float(grid[-1])],
        heston_profile_rmse_range=[float(profile.rmse_vol.min()), float(profile.rmse_vol.max())],
        heston_rmse_vol_at_true_kappa=float(
            np.interp(np.log(model.kappa), np.log(profile.grid), profile.rmse_vol)
        ),
        heston_sets_max_ratio=float(ratios.max()) if ratios.size else None,
        heston_sets_rmse_range=[float(min(np.sqrt(np.mean(prob.vol_errors(th) ** 2)) for th in sets)),
                                float(max(np.sqrt(np.mean(prob.vol_errors(th) ** 2)) for th in sets))] if len(sets) else None,
    )  # fmt: skip
    valley = Valley(prob, best.theta, hs, profile, sets, ratios, surface)
    return head, {
        "valley": valley, "slices": slices, "used": used, "ssvi": surf, "spec": spec,
        "quotes": quotes, "repaired": repaired,
    }  # fmt: skip
