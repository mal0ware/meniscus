"""Figures for ``results/``. One chart per claim, no decoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from meniscus import arbitrage, identify, market, svi
from meniscus.market import Slice
from meniscus.pipeline import VOGT, Valley

FloatArray = NDArray[np.float64]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#0d366b", "#256abf", "#86b6ef", "#e9f1fc"])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _log_ticks(ax: Any, values: list[float], axis: str = "x") -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(matplotlib.ticker.FixedLocator(values))
    target.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_locator(matplotlib.ticker.NullLocator())


def valley_figure(v: Valley, path: Path, *, title: str, true_label: str) -> None:
    """The loss around the optimum: a map over (kappa, sigma) and the profile along kappa."""
    _style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"wspace": 0.28})
    m = v.surface
    z = np.clip(m.max_dvol * 100.0, 1e-3, None)
    levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, max(16.0, float(z.max()) * 1.01)]
    cf = ax0.contourf(
        m.grid_x, m.grid_y, z, levels=levels,
        colors=[BLUES(i / (len(levels) - 2)) for i in range(len(levels) - 1)],
    )  # fmt: skip
    ax0.contour(m.grid_x, m.grid_y, m.ratio, levels=[1.0], colors=[INK], linewidths=1.6)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    kx = [x for x in (0.03, 0.1, 0.3, 1, 3, 10, 30) if m.grid_x[0] <= x <= m.grid_x[-1]]
    ky = [y for y in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if m.grid_y[0] <= y <= m.grid_y[-1]]
    _log_ticks(ax0, kx, "x")
    _log_ticks(ax0, ky, "y")
    ax0.grid(False)
    if len(v.sets):
        ax0.plot(v.sets[:, 3], v.sets[:, 4], "o", ms=6, color=SERIES[1], mec=SURFACE, mew=1.2,
                 label="five sets inside the spread")  # fmt: skip
    ax0.plot(v.theta_hat[3], v.theta_hat[4], "+", ms=12, mew=2.2, color=INK, label=true_label)
    ax0.plot([], [], color=INK, lw=1.6, label="one half-spread")
    ax0.set_xlabel("mean reversion κ (per year)")
    ax0.set_ylabel("vol of variance σ")
    ax0.set_title("κ and σ fixed, other three refitted")
    ax0.legend(loc="upper left", fontsize=8, labelcolor=INK,
               facecolor=SURFACE, frameon=True, framealpha=0.85, edgecolor="none")  # fmt: skip
    cb = fig.colorbar(cf, ax=ax0, pad=0.02, ticks=levels[:-1])
    cb.set_label("largest implied-vol error (vol points)", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)

    p = v.profile
    ax1.plot(p.grid, p.max_dvol * 100.0, color=SERIES[0])
    median_h = float(np.median(v.half_spread)) * 100.0
    ax1.axhline(median_h, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax1.text(p.grid[0] * 1.05, median_h, f"half-spread {median_h:.2f}", va="bottom", fontsize=8.5,
             color=INK2)  # fmt: skip
    if len(v.sets):
        d = [float(np.max(np.abs(v.problem.model_vol(th) - v.problem.model_vol(v.theta_hat)))) * 100
             for th in v.sets]  # fmt: skip
        ax1.plot(v.sets[:, 3], d, "o", ms=6, color=SERIES[1], mec=SURFACE, mew=1.2)
    ax1.set_xscale("log")
    _log_ticks(ax1, [x for x in (0.05, 0.1, 0.3, 1, 3, 10, 30) if p.grid[0] <= x <= p.grid[-1]])
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlabel("mean reversion κ held fixed (per year)")
    ax1.set_ylabel("largest implied-vol error (vol points)")
    ax1.set_title("κ fixed, other four refitted")
    fig.suptitle(title, x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.02)
    _save(fig, path)


def designs_figure(
    profiles: list[tuple[str, identify.Profile]], half_spread: float, path: Path
) -> None:
    """The kappa profile for different sets of expiries."""
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (label, p) in enumerate(profiles):
        ax.plot(p.grid, p.max_dvol * 100.0, color=SERIES[i], label=label)
    ax.axhline(half_spread * 100.0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.105, half_spread * 100.0, f"half-spread {half_spread * 100:.2f}", va="bottom",
            fontsize=8.5, color=INK2)  # fmt: skip
    ax.set_xscale("log")
    _log_ticks(ax, [0.1, 0.3, 1, 3, 10, 30])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("mean reversion κ held fixed (per year); true value 3")
    ax.set_ylabel("largest implied-vol error (vol points)")
    ax.set_title("More expiries narrow the valley")
    ax.legend(loc="upper center", labelcolor=INK)
    _save(fig, path)


def smiles_figure(v: Valley, path: Path, *, title: str, max_panels: int = 4) -> None:
    """The five parameter sets as differences from the reference smile, with the spread band."""
    _style()
    ts = np.unique(v.problem.t)
    pick = ts[np.unique(np.linspace(0, ts.size - 1, min(max_panels, ts.size)).round().astype(int))]
    fig, axes = plt.subplots(1, pick.size, figsize=(2.9 * pick.size, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})  # fmt: skip
    ref = v.problem.model_vol(v.theta_hat)
    vols = [v.problem.model_vol(th) for th in v.sets]
    for ax, t in zip(np.atleast_1d(axes), pick, strict=True):
        sel = v.problem.t == t
        k = v.problem.k[sel]
        ax.fill_between(k, -v.half_spread[sel] * 100, v.half_spread[sel] * 100, color="#e9f1fc",
                        lw=0, label="bid to ask")  # fmt: skip
        for i, (th, vol) in enumerate(zip(v.sets, vols, strict=True)):
            ax.plot(k, (vol[sel] - ref[sel]) * 100, color=SERIES[i], lw=1.6, marker="o", ms=3,
                    label=f"κ = {th[3]:.2f}")  # fmt: skip
        ax.axhline(0.0, color=AXIS, lw=0.8)
        ax.set_title(f"t = {t:.2f} years", fontsize=9.5)
        ax.set_xlabel("log-moneyness k")
    np.atleast_1d(axes)[0].set_ylabel("implied vol minus reference (vol points)")
    np.atleast_1d(axes)[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=INK)
    fig.suptitle(title, x=0.125, ha="left", fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, path)


def vogt_figure(path: Path) -> None:
    """Gatheral and Jacquier's Figure 1: a raw SVI smile with a negative density."""
    _style()
    k = np.linspace(-1.5, 1.5, 601)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"wspace": 0.25})
    ax0.plot(k, VOGT.w(k), color=SERIES[0])
    ax0.set_xlabel("log-moneyness k")
    ax0.set_ylabel("total variance w")
    ax0.set_title("Raw SVI smile, Example 3.1")
    g = arbitrage.g_function(VOGT, k)
    ax1.plot(k, g, color=SERIES[0])
    ax1.fill_between(k, g, 0.0, where=g < 0.0, color=SERIES[7], lw=0, alpha=0.45)
    ax1.axhline(0.0, color=INK, lw=0.8)
    ax1.annotate("negative density", xy=(float(k[np.argmin(g)]), float(g.min())),
                 xytext=(0.1, 0.35), color=INK2, fontsize=8.5,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})  # fmt: skip
    ax1.set_xlabel("log-moneyness k")
    ax1.set_ylabel("g(k), eq. (2.1)")
    ax1.set_title("Its g function goes below zero")
    _save(fig, path)


def surface_figure(
    slices: list[Slice],
    used: list[Slice],
    ssvi: svi.SSVI,
    v: Valley,
    path: Path,
    *,
    title: str,
    panels: int = 7,
) -> None:
    """Fitted smiles on the market-like chain: SVI with its spread band, SSVI, Heston, truth."""
    _style()
    idx_all = {s.days: i for i, s in enumerate(slices)}
    pick = np.unique(np.linspace(0, len(used) - 1, min(panels, len(used))).round().astype(int))
    shown = [used[i] for i in pick]
    fig, axes = plt.subplots(
        2, 4, figsize=(12.0, 5.6), gridspec_kw={"wspace": 0.22, "hspace": 0.42}
    )
    flat = axes.ravel()
    for ax, s in zip(flat, shown, strict=False):
        sel = v.problem.t == s.t
        k = np.linspace(v.problem.k[sel].min(), v.problem.k[sel].max(), 121)
        vol = s.vol(k) * 100
        h = s.half_spread(k) * 100
        ax.fill_between(k, vol - h, vol + h, color="#cde2fb", lw=0, label="SVI fit ± half-spread")
        ax.plot(k, market.true_vol(s.spec, k, s.t) * 100, color=INK, lw=1.0, ls=(0, (2, 2)),
                label="true smile (Bates)")  # fmt: skip
        i = idx_all[s.days]
        ax.plot(k, np.sqrt(ssvi.w_theta(k, ssvi.theta[i]) / s.t) * 100, color=SERIES[2], lw=1.6,
                label="SSVI (3 parameters and the ATM curve)")  # fmt: skip
        ax.plot(v.problem.k[sel], v.problem.model_vol(v.theta_hat)[sel] * 100, color=SERIES[1],
                lw=1.6, marker="o", ms=3, label="Heston (5 parameters)")  # fmt: skip
        ax.set_title(f"{s.days} days", fontsize=9.5)
        ax.set_xlabel("log-moneyness k")
    for ax in flat[len(shown) :]:
        ax.axis("off")
    flat[0].set_ylabel("implied vol (%)")
    flat[4].set_ylabel("implied vol (%)")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[min(len(shown), 7)].legend(handles, labels, loc="center", labelcolor=INK)
    fig.suptitle(title, x=0.125, ha="left", fontsize=12, fontweight="bold", y=0.97)
    _save(fig, path)


def intervals_figure(
    estimates: FloatArray, fisher_sd: FloatArray, theta: FloatArray, v: Valley, path: Path
) -> None:
    """Recalibrated kappa against the Fisher prediction, with the three intervals."""
    _style()
    kappa = estimates[:, identify.KAPPA]
    true, sd = float(theta[identify.KAPPA]), float(fisher_sd[identify.KAPPA])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    lo, hi = v.profile.interval()[:2]
    span = max(hi - true, true - lo) * 1.25
    edges = np.linspace(true - span, true + span, 31)
    ax.hist(np.clip(kappa, edges[0], edges[-1]), bins=edges, density=True, color="#cde2fb",
            lw=0, label=f"{kappa.size} recalibrations to noisy quotes")  # fmt: skip
    x = np.linspace(edges[0], edges[-1], 301)
    ax.plot(x, np.exp(-0.5 * ((x - true) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi)),
            color=SERIES[0], label="Fisher information (normal)")  # fmt: skip
    top = ax.get_ylim()[1]
    rows = (
        ("every quote inside the spread", lo, hi, INK),
        ("Fisher, 90%", true - identify.Z90 * sd, true + identify.Z90 * sd, SERIES[0]),
        ("recalibrations, 5% to 95%", float(np.quantile(kappa, 0.05)),
         float(np.quantile(kappa, 0.95)), SERIES[1]),
    )  # fmt: skip
    left = edges[0] - 0.42 * (edges[-1] - edges[0])
    for j, (label, a, b, color) in enumerate(rows):
        y = top * (1.30 - 0.09 * j)
        ax.plot([a, b], [y, y], color=color, lw=2.4, solid_capstyle="butt")
        ax.text(left, y, "  " + label, va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0.0, top * 1.38)
    ax.set_xlim(left, edges[-1])
    ax.set_xticks([t for t in ax.get_xticks() if edges[0] <= t <= edges[-1]])
    ax.axvline(true, color=AXIS, lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("calibrated mean reversion κ (per year); true value 3")
    ax.set_title("Three intervals for κ at a half-spread of 0.25 volatility points")
    ax.legend(loc="center left", labelcolor=INK, bbox_to_anchor=(0.0, 0.45))
    ax.grid(False)
    _save(fig, path)
