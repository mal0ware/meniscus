# Sources, formulas and comparison

Every formula in `src/meniscus/` is listed here with where it comes from. Every number that could be compared with a source is compared. All numbers are from `results/`, written by `scripts/reproduce.py`, unless a line says otherwise. Every surface is synthetic. No market data was used.

## Sources read

| Short name | Citation | How it was read |
|---|---|---|
| GJ | J. Gatheral and A. Jacquier, "Arbitrage-free SVI volatility surfaces", Quantitative Finance 14(1), 2014. arXiv:1204.0646 | arXiv PDF |
| Cui | Y. Cui, S. del Baño Rollin and G. Germano, "Full and fast calibration of the Heston stochastic volatility model", European Journal of Operational Research 263(2), 2017. arXiv:1511.08718 | arXiv PDF |
| Trap | H. Albrecher, P. Mayer, W. Schoutens and J. Tistaert, "The Little Heston Trap", Wilmott Magazine, January 2007 | author's version, revised 11 September 2006 |
| LF | F. Le Floc'h, "An adaptive Filon quadrature for stochastic volatility models", v1.1, April 2016 | author's PDF |
| Heston | S. Heston, "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options", Review of Financial Studies 6(2), 1993 | not read |
| Bates | D. Bates, "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options", Review of Financial Studies 9(1), 1996 | not read |
| Merton | R. Merton, "Option pricing when underlying stock returns are discontinuous", Journal of Financial Economics 3, 1976 | not read |

Heston (1993) was not fetched. Its characteristic function is taken from two restatements that agree with each other: Cui eq. (10) and Trap eq. (1). Anything marked "Heston" below is therefore from a secondary source. Bates (1996) and Merton (1976) were not fetched either. The jump factor used here is the characteristic function of a compensated compound Poisson process with normal jump sizes, which is textbook material, and it is checked numerically (see "The generator" below).

## Formulas

Notation: `k = log(K/F)` is log-moneyness, `w = sigma^2 T` is total implied variance, prices are undiscounted and per unit of forward.

### Black price and implied volatility (`black.py`)

| Formula | Source |
|---|---|
| `c(k, w) = N(d1) - exp(k) N(d2)`, `d1 = -k/sqrt(w) + sqrt(w)/2`, `d2 = d1 - sqrt(w)` | GJ section 2.2, the line before eq. (2.1) |
| Implied volatility by bisection on the out-of-the-money price | own; no source needed |

### SVI (`svi.py`)

| Formula | Source |
|---|---|
| Raw SVI `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))` | GJ eq. (3.1) |
| Minimum `a + b sigma sqrt(1 - rho^2)` | GJ, text under eq. (3.1) |
| Natural SVI | GJ eq. (3.2) |
| Natural to raw and back | GJ eqs. (3.3), (3.4) |
| SVI jump-wings from raw | GJ eq. (3.5) |
| SSVI `w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))` | GJ eq. (4.1) |
| `phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))` | GJ eq. (4.5) |

### Static arbitrage (`arbitrage.py`)

| Formula | Source |
|---|---|
| `g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2`, no butterfly arbitrage iff `g >= 0` and `d+ -> -inf` | GJ eq. (2.1), Lemma 2.2 |
| Density `p(k) = g(k) / sqrt(2 pi w) exp(-d-^2 / 2)` | GJ, proof of Lemma 2.2 |
| No calendar arbitrage iff `w` is non-decreasing in `t` at fixed `k` | GJ Definition 2.2 |
| Crossedness of two slices | GJ Definition 5.1 |
| SSVI conditions | GJ Theorem 4.1, Theorem 4.2, Corollary 4.1 |
| Wing slopes below 2 | Lee's moment bound, as cited in GJ after Theorem 4.2 |

### Heston (`heston.py`)

The parameter vector is `[v0, vbar, rho, kappa, sigma]`, as in Cui.

| Formula | Source |
|---|---|
| Call price as two integrals of the characteristic function | Cui eq. (9) |
| `xi = kappa - sigma rho i u`, `d = sqrt(xi^2 + sigma^2 (u^2 + i u))` | Cui eqs. (11a), (11b) |
| Original form of the characteristic function | Cui eq. (10); Trap eq. (1). Heston (1993) eq. (17) not read |
| Stable form | Cui eq. (12); Trap eq. (2) |
| Form used for the gradient | Cui eq. (18), with `A` from (15) and `D` from (17b) |
| Gradient of the price | Cui eq. (22) |
| `h(u) = grad(phi) / phi` | Cui eq. (23a) to (23e) |
| Gauss-Legendre quadrature on `[0, 200]` | Cui section 5.2 |

### Bates (`bates.py`)

| Formula | Source |
|---|---|
| `phi_Bates(u) = phi_Heston(u) exp(lam t (exp(i u mu_j - delta_j^2 u^2 / 2) - 1 - i u (exp(mu_j + delta_j^2 / 2) - 1)))` | standard; attributed to Bates (1996), not read |
| Same pricing integral as Heston | Cui eq. (9) |

### Calibration (`calibrate.py`, `identify.py`)

| Formula | Source |
|---|---|
| Residuals and objective `f = 1/2 ||r||^2` | Cui eqs. (2) to (5) |
| Gauss-Newton Hessian `J J^T` | Cui eq. (35) |
| Random starting ranges | Cui Table 5 |
| Profile of the loss, spread ratio, equivalent sets | own definitions, described in the README |
| Fisher information `J^T J / s^2` for independent noise of standard deviation `s` on each implied volatility, and its inverse as the covariance bound | standard (Cramer-Rao); `J` is the vega-weighted Jacobian of Cui eq. (22) |
| Profile-likelihood interval: `n rmse(kappa)^2 <= chi2_1(0.90) s^2` at noise-free data | standard (Gaussian likelihood ratio) |
| `s = h / sqrt(3)` for a uniform draw on `[-h, h]` | variance of the uniform distribution |

## Where the code differs from the sources

1. **Forward units.** Cui prices with spot and a rate. Here everything is per unit of forward with no discounting. With `S0 = 1` the two give the same price and the same gradient, because the discount factor times the forward is the spot.
2. **Intermediate derivatives.** Cui gives `h(u)` through the rearranged identities (27) to (30). The code differentiates `A1`, `A2` and `d` directly and carries the hyperbolic functions with `exp(-d t / 2)` factored out, so nothing overflows. The result is checked against central differences: the largest difference over the 40 options of the Cui design is 6.6e-10.
3. **Quadrature.** Cui uses 64 Gauss-Legendre nodes on `[0, 200]`. The default here is 8 panels of 32 nodes on the same interval. On the Cui design the 64-node rule is off by at most 2.2e-11 against adaptive quadrature and the 256-node rule by 1.5e-13. The larger rule is kept because the second experiment calibrates to an expiry of 14 days, shorter than anything in Cui's design.
4. **Optimiser.** Cui uses Levenberg-Marquardt from the LEVMAR package with no bounds. The code uses SciPy's bounded trust-region least squares with the same analytic Jacobian. The box is `v0, vbar` in [1e-4, 5], `rho` in [-0.999, 0.999], `kappa` in [1e-3, 60], `sigma` in [1e-3, 15].
5. **Residual weights.** Cui uses price residuals. The studies here divide by the Black vega, which makes the residual an implied-volatility error to first order. The Hessian comparison below uses price residuals, as Cui does.
6. **Calendar check.** GJ Lemma 3.3 finds crossings from the roots of a quartic. The code takes the largest gap on a grid of 2,401 points on `k` in [-3, 3]. This finds the same crossings inside that range and none outside it.
7. **SVI fit.** GJ section 5.2 fits prices, starts from a square-root SVI fit, and penalises crossing. The code fits implied volatility weighted by one over the half bid-ask spread, from a fixed set of 12 starting points. The arbitrage-free refit starts from the power-law SSVI fit instead of the square-root one. It visits the slices in order of expiry three times and refits each to its own quotes, with penalties for `g` below 0.001 and for a slice less than 1e-6 above the one before or below the one after, on 601 points on [-3, 3].
8. **Interpolation between expiries.** GJ Lemma 5.1 is not implemented. The SSVI surface interpolates `theta` linearly in time, which is monotone and so satisfies Theorem 4.1.
9. **`gamma` at most one half.** GJ says after eq. (4.5) that the surface is free of static arbitrage if `eta (1 + |rho|) <= 2`. With `gamma > 1/2` the second condition of Theorem 4.2, `theta phi^2 (1 + |rho|) <= 4`, fails as `theta` goes to zero, because `theta phi^2` behaves like `theta^(1 - 2 gamma)`. The sentence in GJ follows a discussion of `gamma = 1/2`. The fit here keeps `gamma` in (0, 1/2], where both conditions hold for every `theta`.

## Comparison with published numbers

### Heston prices (LF)

| Case | This repo | Published | Difference |
|---|---|---|---|
| Table 1: call, `F = 1000`, `K = 1400`, `T = 0.0182`, `v0 = 0.826`, `kappa = 0.254`, `theta = 0.320`, `sigma = 0.344`, `rho = -0.557` | 0.10734145 | 0.1073414 | 5e-8 |
| Table 3: put, `K/F = 0.25`, `T = 1`, `v0 = 0.0225`, `kappa = 0.1`, `theta = 0.01`, `sigma = 2.0`, `rho = 0.5` | 119.38532 | 119.385324 | 4e-8 |

Two notes on LF. Table 4 reprices the Table 1 option with five methods at a tolerance of 1e-8 and prints 0.107341552 four times and 0.107341448 once. This repo gets 0.107341448. Both round to the 0.1073414 of Table 1, so the test uses that. Table 3's caption says the notional is 100,000; the printed price matches this repo to nine digits with a notional of 1,000,000 and is ten times too large for 100,000.

### The characteristic function at long maturity (Cui Fig. 2)

With the parameters of Cui Table 1 and `T = 15`, the original form differs from the stable form by up to 0.86 for `u` in [0.05, 4]. The form of eq. (18) differs from it by 3.6e-15. This matches the figure: eq. (10) jumps, eqs. (12) and (18) do not.

### The Hessian at the optimum (Cui Table 8)

The design is rebuilt from the description in Cui section 5.1: maturities of 30 to 360 days with 252 days to the year, and five strikes per maturity at Black deltas of 10 and 25 for puts, 50, and 25 and 10 for calls.

| Entry | This repo | Cui Table 8 | Ratio |
|---|---|---|---|
| kappa, kappa | 5.75e-5 | 5.26e-5 | 1.09 |
| vbar, kappa | 9.91e-3 | 9.65e-3 | 1.03 |
| vbar, vbar | 2.21 | 22.6 | 0.098 |
| sigma, kappa | -6.40e-4 | -5.49e-4 | 1.17 |
| sigma, vbar | -8.42e-2 | -7.66e-2 | 1.10 |
| sigma, sigma | 9.14e-3 | 7.46e-3 | 1.23 |
| rho, kappa | 1.90e-4 | 1.61e-4 | 1.18 |
| rho, vbar | 2.29e-2 | 2.00e-2 | 1.15 |
| rho, sigma | -2.85e-3 | -2.34e-3 | 1.22 |
| rho, rho | 9.07e-4 | 7.56e-4 | 1.20 |
| v0, kappa | 5.86e-3 | 5.28e-3 | 1.11 |
| v0, vbar | 1.23 | 11.8 | 0.104 |
| v0, sigma | -4.69e-2 | -3.53e-2 | 1.33 |
| v0, rho | 1.23e-2 | 8.40e-3 | 1.47 |
| v0, v0 | 1.06 | 0.969 | 1.09 |
| condition number | 2.73e6 | 3.978e6 | 0.69 |

Thirteen of fifteen entries agree to within 3% to 47%. The other two are smaller than the printed values by a factor of ten. The matrix as printed is not positive semidefinite: its smallest eigenvalue is -4.2, and `11.8^2 > 22.6 x 0.969`. A Gauss-Newton Hessian cannot have a negative eigenvalue, so I think the exponents of those two entries are misprinted (`+1` for `+0`).

The remaining gap is probably the strikes. Cui Table 4 gives the surface as implied volatilities by delta, and this repo does not reproduce it: at 30 days Cui prints 0.2808, 0.2540 and 0.2369 for the at-the-money, 25-delta and 10-delta calls, and this repo gets 0.2862, 0.2761 and 0.2674. At 360 days the two are closer (0.3007, 0.2897, 0.2803 against 0.2996, 0.2866, 0.2758). The first two columns of Table 4 as extracted from the PDF (2.5096, 1.4359 at 30 days) cannot be volatilities, so the table may hold something else there. Counting 365 days to the year was tried and did not match either. The qualitative claim does reproduce: the Hessian is ill-conditioned by a factor of a few million, and the `kappa` and `rho` diagonal entries are the smallest.

### Recovery from random starts (Cui section 5.2)

Cui reports 9,843 successes in 10,000 runs over 100 random true parameter sets. This repo runs 100 random starts, drawn from the ranges of Cui Table 5, against the one true set of Table 1. All 100 reach the true parameters to a relative error below 1e-4. The worst implied-volatility RMSE is 4.4e-15. The experiment is smaller than Cui's and covers one true set, so it does not test their 98.4% rate.

### SVI jump-wings (GJ Example 5.1)

| Parameter | This repo | GJ |
|---|---|---|
| `v` | 0.017426253 | 0.01742625 |
| `psi` | -0.17521114 | -0.1752111 |
| `p` | 0.6997381 | 0.6997381 |
| `c` | 1.3167982 | 1.316798 |
| `v_tilde` | 0.011624903 | 0.0116249 |

One difference from the text of GJ. The paper says after eq. (3.5) that `psi` is the at-the-money volatility skew. As printed, eq. (3.5) gives the skew times `sqrt(t)`, which is also what Lemma 4.1 and eq. (4.2) imply when read together. The two agree at `t = 1`, which is the case in Example 5.1. The code implements eq. (3.5) as printed and a test checks the factor.

### The Vogt smile (GJ Example 3.1, Figure 1)

With `(a, b, m, rho, sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)` the function `g` has a minimum of -0.0329 at `k = 0.879`. GJ Figure 1 shows the same dip below zero near `k = 0.9`. See `results/fig_vogt.png`.

## Claims tested

| Claim | Result |
|---|---|
| The stable characteristic function is continuous where the original form jumps (Trap, Cui Fig. 2) | Reproduced |
| The analytic gradient equals the numerical one (Cui section 3.3.4) | Reproduced to 6.6e-10 |
| The calibration finds the true parameters from random starts (Cui section 5.2) | 100 of 100 on one true set |
| The Hessian at the optimum is ill-conditioned, flattest along `kappa` and `rho` (Cui Table 8) | Reproduced in kind; entries differ by up to 47%, two look misprinted |
| "Local minima" in earlier work come from the flat directions (Cui section 5.2) | Consistent with what was found: one optimum, and a range of `kappa` inside any given tolerance |
| A raw SVI slice can have butterfly arbitrage (GJ Example 3.1) | Reproduced |
| SSVI inside the bound has no static arbitrage (GJ Corollary 4.1) | Holds for 25 random surfaces and for the fitted one, with `gamma <= 1/2` |
| A raw SVI fit can be free of arbitrage on the quoted strikes and not beyond them | Seen on the synthetic chain: 2 of 11 slices and 7 of 10 pairs fail outside the quoted range, none inside |
| A penalised slice-by-slice fit removes calendar and butterfly arbitrage (GJ section 5.2) | Worked on the synthetic chain: no arbitrage left on the grid, 96.5% of quotes inside the spread |
| Least squares attains the Cramer-Rao bound when the noise is small | Holds for all five parameters at a half-spread of 0.25 points: recalibration over Fisher standard deviations are 0.95 to 1.00 |

## The design claim that did not hold

The design README promised "a long flat valley, with five different parameter sets that price the same options to within the bid-ask spread". That is what one expiry gives. With eight expiries the five sets exist but they are close together: `kappa` from 2.30 to 3.88 around a true value of 3, for a half-spread of 0.25 volatility points. On the chain generated with jumps the five sets run from `kappa` = 4.17 to 5.90 around a best fit of 4.94.

## Variants tried

Everything that was run and then changed, in order.

1. Year length for the Cui design: 252 days (as the paper says) and 365 days. Neither reproduces Cui Table 4. 252 was kept.
2. Quadrature default: 64 nodes, then 8 panels of 32. Changed before any calibration was run.
3. Recalibrations to noisy quotes: 200, then 500, to get the standard deviations to about 3%. With 200 the 5% to 95% range of `kappa` was 2.68 to 3.37; with 500 it is 2.66 to 3.36.
4. Arbitrage-free refit, first version: shortest expiry to longest, starting from the raw fit, target the raw fit on 41 points, penalty against the slice before only. On the synthetic chain this left 9 crossings of up to 2.1e-5 and put 52.5% of points inside the spread, because the first repaired slice took a steep right wing and every later slice had to stay above it. These numbers are from a run that was not kept. Changed to the version described above (start from SSVI, both neighbours, three passes, fitted to the quotes): no crossings, 96.5% of quotes inside. The second is reported.
5. Butterfly check "between the quoted strikes": first used the whole-line flags for positive variance and wing slope, which counted the 42-day slice as failing inside its range when the failure is outside it. Changed to test the variance and `g` on the quoted range only.
6. Ends of the likelihood interval: first interpolated on `n rmse^2`, which is quadratic in `kappa`, between grid points 16% apart. That gave 2.67 to 3.29, narrower than the Fisher interval for no real reason. Changed to interpolate on its square root, which is close to linear: 2.65 to 3.37.
7. A test compared Bates with no jumps against Heston on the two different quadrature rules and failed at 1e-11. The test now uses one rule for both.

Not varied after seeing results: the generator's parameters, the strike grid, the spread model, the 10-day cut-off for Heston, the nine deltas, the vega weighting, the 0.25-point half-spread used on Heston surfaces, the three passes of the refit, and the seeds. The table of valley widths in the README gives five half-spreads so the reader is not tied to that one choice.

## The three intervals

The 40-option design, half-spread `h` = 0.25 volatility points, so `s` = 0.144 points.

| Method | Definition | `kappa` |
|---|---|---|
| Every quote inside its spread | largest `kappa` range where the refitted smile is within `h` of the truth at all 40 options | 2.25 to 3.96 |
| Profile likelihood, 90% | `40 rmse(kappa)^2 <= 2.706 s^2` along the same profile | 2.65 to 3.37 |
| Fisher information, 90% | `3 +- 1.645 sqrt((J^T J)^-1_kk) s` | 2.64 to 3.36 |
| Recalibration | 5% and 95% points of 500 fits, each to quotes shifted uniformly within `+-h` | 2.66 to 3.36 |

The noise is uniform, not Gaussian. The Fisher and likelihood rows use only its variance, which is what the large-sample distribution of a least-squares estimate depends on. The three statistical rows agree to within 0.03 at each end, so at this spread the loss is close to quadratic in `kappa` over the range the noise reaches.

The Fisher correlation between `rho` and `sigma` is 0.96 and between `kappa` and `vbar` it is -0.84.

The design table in the README (`results/takeaway.csv`) uses the Fisher row only. Its entries are `x kappa sqrt(3) / (1.645 u)`, where `u` is the standard deviation of `kappa` per unit of noise standard deviation and `x` is the tolerance. The three tolerance columns are therefore proportional. For one expiry `u` is about 1.6e6, so the formula returns a spread near 1e-7. That number says the parameter is not identified and should not be read as a spread.

## The generator

The second experiment needs a surface Heston cannot fit exactly. Bates was picked because it adds one well-understood thing, jumps in the price, and has a closed-form characteristic function.

| Setting | Value |
|---|---|
| Diffusion `[v0, vbar, rho, kappa, sigma]` | 0.035, 0.05, -0.70, 2.0, 0.45 (Feller ratio 0.99) |
| Jumps | rate 0.5 per year, mean log jump -0.10, standard deviation 0.12 |
| Share of long-run variance from jumps | 20% |
| Expiries | 7, 14, 28, 42, 63, 91, 126, 182, 273, 364, 546 days, 365 days to the year |
| Strikes over forward | 0.90 to 1.10 by 0.01, 0.70 to 1.30 by 0.025, 0.40 to 1.60 by 0.05 |
| Quoted when | the out-of-the-money option is worth at least 1e-4 of the forward |
| Half-spread in volatility | `0.002 max(1, sqrt(30 / days)) (1 + 0.25 z^2)`, `z = k / (0.2 sqrt(t))`, capped at 0.03 |
| Mid | true volatility plus a uniform draw within half of the half-spread |

None of these values was estimated. They are a guess at what an index chain looks like, written down before any fit was run and not changed afterwards.

Checks on the generator, all in `tests/test_market.py`: the characteristic function is 1 at `u = -i`; with a jump rate of zero the price equals the Heston price to 1e-14; with constant variance it equals a 40-term Merton series to 2e-7; and the fixed rule (32 panels of 32 nodes on [0, 800]) agrees with adaptive quadrature to 1e-11 at 7, 14, 91 and 546 days.

## What the chain shows

| Quantity | Value |
|---|---|
| Quotes, expiries | 399, 11 |
| Raw SVI: quotes inside bid-ask | 99.5% |
| Raw SVI: RMSE to the true smile at the quotes | 0.17 points |
| Raw SVI: slices with butterfly arbitrage, quoted range / all `k` | 0 / 2 |
| Raw SVI: neighbouring pairs that cross, quoted range / all `k` | 0 / 7 |
| Raw SVI: largest error at 1.25 times the last quoted strike | 4.3 points |
| SSVI: `rho`, `eta`, `gamma` | -0.676, 0.663, 0.444 |
| SSVI: quotes inside bid-ask | 56.1% |
| Refit: slices free of butterfly arbitrage, crossings | 11 of 11, 0 |
| Refit: quotes inside bid-ask | 96.5% |
| Refit: RMSE to the true smile at the quotes | 0.37 points |
| Refit: largest error at 1.25 times the last quoted strike | 3.8 points |
| Heston: starts reaching the best fit | 20 of 20 |
| Heston: `[v0, vbar, rho, kappa, sigma]` | 0.0422, 0.0588, -0.704, 4.94, 0.764 |
| Heston: RMSE, median half-spread, share inside | 0.27, 0.23, 68% |
| Heston: `kappa` within one half-spread of the best fit | 4.09 to 6.02 |
| Heston: RMSE with `kappa` held at 2 | 0.48 points |
| Heston fitted to the true smile, no noise: `kappa`, RMSE | 4.89, 0.24 points |

The two slices with butterfly arbitrage are at 28 and 42 days, and the failure is on the call side beyond the last quoted strike. The 42-day raw fit has negative total variance there. A fit weighted by the spread has little reason to care about a region with no quotes, which is the practical point of GJ's penalties.

The Heston result is the one that would not show up on Heston's own prices. The profile, the Hessian and the Fisher information all describe the fitted model. On this chain they report a well-identified `kappa` near 5 when the process that made the data reverts at 2.
