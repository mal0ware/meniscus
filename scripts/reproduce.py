"""Rebuild every table and figure in ``results/``.

    uv run python scripts/reproduce.py            # full run
    uv run python scripts/reproduce.py --fast     # coarse grids, for CI
    uv run python scripts/reproduce.py --out DIR  # write somewhere else

Everything is generated from models with known parameters and fixed seeds, so
the same command writes the same files every time. No market data is read and
no network is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniscus import pipeline, plots, synthetic  # noqa: E402

MARKET_DIR = "market"


def run(out: Path, *, fast: bool, figures: bool = True) -> None:
    """Run the experiments and write tables, figures and the two ``headline.json`` files."""
    cfg = pipeline.Config.fast() if fast else pipeline.Config()
    mode = "fast" if fast else "full"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    head, data = pipeline.run_synthetic(cfg, out)
    if figures:
        plots.valley_figure(
            data["valley"], out / "fig_valley_synthetic.png",
            title="Heston calibrated to its own prices: 40 options, true κ = 3",
            true_label="true parameters",
        )  # fmt: skip
        plots.designs_figure(
            data["design_profiles"], pipeline.SYNTHETIC_HALF_SPREAD, out / "fig_designs.png"
        )
        plots.smiles_figure(
            data["valley"], out / "fig_five_sets_synthetic.png",
            title="Five parameter sets, one surface: differences from the true smile",
        )  # fmt: skip
        plots.intervals_figure(
            data["noise_estimates"], data["fisher_sd"], synthetic.CUI_THETA, data["valley"],
            out / "fig_intervals.png",
        )  # fmt: skip
        plots.vogt_figure(out / "fig_vogt.png")
    pipeline.write_json(out / "headline.json", {"mode": mode, **head})
    print(f"Heston-on-Heston experiments done in {time.time() - started:.0f}s")

    sub = out / MARKET_DIR
    started = time.time()
    head, data = pipeline.run_market(cfg, sub)
    if figures:
        plots.valley_figure(
            data["valley"], sub / "fig_valley.png",
            title="Heston calibrated to a chain generated with jumps (true κ = 2)",
            true_label="best fit",
        )  # fmt: skip
        plots.surface_figure(
            data["slices"], data["used"], data["ssvi"], data["valley"], sub / "fig_surface.png",
            title="A synthetic chain from a Bates model: three fits and the truth",
        )  # fmt: skip
        plots.smiles_figure(
            data["valley"], sub / "fig_five_sets.png",
            title="Five parameter sets inside the spread of the best Heston fit",
        )  # fmt: skip
    pipeline.write_json(sub / "headline.json", {"mode": mode, **head})
    print(f"market-like experiments done in {time.time() - started:.0f}s")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="coarse grids, about a minute")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.out, fast=args.fast)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
