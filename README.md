# Meniscus

**Fit a volatility surface that cannot be arbitraged, then show why calibrating Heston has more than one answer.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

Option prices imply a different volatility at every strike and expiry. Can you fit that surface so that it admits no free lunch? And when you calibrate a stochastic-volatility model to it, is the answer unique?

## What this will contain

- Implied volatility inversion
- Raw and natural SVI fits per expiry, and a full surface
- Butterfly and calendar arbitrage checks
- Heston pricing by characteristic function, in the numerically stable form
- Heston calibration with an analytic gradient
- An identifiability study that maps the loss around the optimum

## The result it is built to show

A plot of the Heston calibration loss as a long flat valley, with five different parameter sets that price the same options to within the bid-ask spread.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Gatheral and Jacquier, "Arbitrage-free SVI volatility surfaces", Quantitative Finance 14(1), 2014
- Heston, "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options", Review of Financial Studies 6(2), 1993
- Albrecher, Mayer, Schoutens and Tistaert, "The Little Heston Trap", Wilmott Magazine, 2007 (the stable characteristic function)
- Cui, del Baño Rollin and Germano, "Full and fast calibration of the Heston stochastic volatility model", European Journal of Operational Research 263(2), 2017

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Synthetic surfaces are committed. There is no free option-chain history that may be redistributed, so real chains are fetched by a script for personal use and never stored in the repo.

## How it is tested

The maths is tested before any result is trusted.

- Heston prices match published reference values
- SVI recovers its own parameters from generated data
- The arbitrage checks flag a slice that was broken on purpose

## Layout

```
src/meniscus/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Implied volatility inversion
- [ ] Raw and natural SVI fits per expiry, and a full surface
- [ ] Butterfly and calendar arbitrage checks
- [ ] Heston pricing by characteristic function, in the numerically stable form
- [ ] Heston calibration with an analytic gradient
- [ ] An identifiability study that maps the loss around the optimum
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Most results are on synthetic data, because real option history is not free to share.
- A surface with no static arbitrage can still be wrong.
- The flat valley is a property of the model and the data together. More expiries narrow it; they do not remove it.
- This is research code. It is not investment advice and it does not trade.

## The name

A meniscus is the curved surface of a liquid in a glass. Implied volatility bends the same way across strikes.

## Licence

MIT. See `LICENSE`.
