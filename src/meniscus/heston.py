"""Heston (1993) prices and their parameter gradient by characteristic function.

The parameter vector is ordered as in Cui, del Bano Rollin and Germano (2017):
``theta = [v0, vbar, rho, kappa, sigma]``.

All prices are undiscounted and per unit of forward, for log-moneyness
``k = log(K / F)``. Under the forward measure ``x_T = log(S_T / F)`` has

    dx = -v/2 dt + sqrt(v) dW,   dv = kappa (vbar - v) dt + sigma sqrt(v) dZ,
    d<W, Z> = rho dt,

and ``phi(u) = E[exp(i u x_T)]``. Three algebraically equal forms of ``phi`` are
implemented, numbered as in Cui et al.:

* ``original``  eq. (10), the form in Heston (1993), unstable at long maturity;
* ``trap``      eq. (12), the form of Albrecher et al. (2007), stable;
* ``cui``       eq. (18), stable and easy to differentiate.

The pricing integral is Cui eq. (9) and the gradient is Cui eq. (22).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Form = Literal["original", "trap", "cui"]

PARAM_NAMES = ("v0", "vbar", "rho", "kappa", "sigma")
#: Gauss-Legendre settings used by Cui et al. (section 5.2).
CUI_NODES = 64
CUI_UMAX = 200.0


@dataclass(frozen=True)
class Quadrature:
    """Gauss-Legendre nodes and weights on ``[0, umax]``."""

    nodes: FloatArray
    weights: FloatArray

    @staticmethod
    def gauss_legendre(n: int = CUI_NODES, umax: float = CUI_UMAX) -> Quadrature:
        """``n``-point rule mapped from ``[-1, 1]`` to ``[0, umax]``."""
        x, w = np.polynomial.legendre.leggauss(n)
        return Quadrature(nodes=0.5 * umax * (x + 1.0), weights=0.5 * umax * w)

    @staticmethod
    def panels(panels: int = 8, n: int = 32, umax: float = CUI_UMAX) -> Quadrature:
        """Composite rule: ``n`` points on each of ``panels`` equal panels."""
        x, w = np.polynomial.legendre.leggauss(n)
        edges = np.linspace(0.0, umax, panels + 1)
        half = 0.5 * np.diff(edges)
        nodes = (edges[:-1, None] + half[:, None] * (x[None, :] + 1.0)).ravel()
        weights = (half[:, None] * w[None, :]).ravel()
        return Quadrature(nodes=nodes, weights=weights)


DEFAULT_QUADRATURE = Quadrature.panels()


def _unpack(theta: ArrayLike) -> tuple[float, float, float, float, float]:
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (5,):
        raise ValueError("theta must be [v0, vbar, rho, kappa, sigma]")
    return float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4])


def charfn(theta: ArrayLike, u: ArrayLike, t: float, form: Form = "cui") -> ComplexArray:
    """Characteristic function of ``log(S_T / F)`` at (complex) ``u``."""
    v0, vbar, rho, kappa, sigma = _unpack(theta)
    uu = np.asarray(u, dtype=np.complex128)
    xi = kappa - sigma * rho * 1j * uu  # (11a)
    d = np.sqrt(xi * xi + sigma * sigma * (uu * uu + 1j * uu))  # (11b)
    s2 = sigma * sigma
    if form == "original":
        with np.errstate(over="ignore", invalid="ignore"):
            g1 = (xi + d) / (xi - d)  # (11c)
            edt = np.exp(d * t)
            expo = kappa * vbar / s2 * (
                (xi + d) * t - 2.0 * np.log((1.0 - g1 * edt) / (1.0 - g1))
            ) + v0 / s2 * (xi + d) * (1.0 - edt) / (1.0 - g1 * edt)
    elif form == "trap":
        g2 = (xi - d) / (xi + d)  # (13)
        emdt = np.exp(-d * t)
        expo = kappa * vbar / s2 * (
            (xi - d) * t - 2.0 * np.log((1.0 - g2 * emdt) / (1.0 - g2))
        ) + v0 / s2 * (xi - d) * (1.0 - emdt) / (1.0 - g2 * emdt)
    elif form == "cui":
        emdt = np.exp(-d * t)
        a2 = 0.5 * ((d + xi) + (d - xi) * emdt)  # A2 * exp(-d t / 2), from (16c)
        big_a = (uu * uu + 1j * uu) * 0.5 * (1.0 - emdt) / a2  # (15a)
        big_d = np.log(d) + 0.5 * (kappa - d) * t - np.log(a2)  # (17b)
        expo = (
            -t * kappa * vbar * rho * 1j * uu / sigma - v0 * big_a + 2.0 * kappa * vbar / s2 * big_d
        )  # (18)
    else:
        raise ValueError(f"unknown form {form!r}")
    out: ComplexArray = np.exp(expo)
    return out


def _charfn_and_h(
    theta: tuple[float, float, float, float, float], uu: ComplexArray, t: float
) -> tuple[ComplexArray, ComplexArray]:
    """``phi(u)`` in the form (18) and ``h(u) = grad(phi) / phi``, shape (5, n).

    ``h`` is Cui eq. (23). The intermediate derivatives are obtained by
    differentiating ``A1``, ``A2`` and ``d`` directly instead of through the
    rearranged identities (27)-(30); the two are algebraically equal. The
    hyperbolic functions are carried with a factor ``exp(-d t / 2)`` taken
    out, so nothing overflows for large ``u``.
    """
    v0, vbar, rho, kappa, sigma = theta
    s2 = sigma * sigma
    iu = 1j * uu
    q = uu * uu + iu
    xi = kappa - sigma * rho * iu
    d = np.sqrt(xi * xi + s2 * q)
    emdt = np.exp(-d * t)
    ch = 0.5 * (1.0 + emdt)  # cosh(d t / 2) * exp(-d t / 2)
    sh = 0.5 * (1.0 - emdt)  # sinh(d t / 2) * exp(-d t / 2)
    a2 = d * ch + xi * sh
    big_a = q * sh / a2
    big_d = np.log(d) + 0.5 * (kappa - d) * t - np.log(a2)
    phi = np.exp(
        -t * kappa * vbar * rho * iu / sigma - v0 * big_a + 2.0 * kappa * vbar / s2 * big_d
    )

    def partials(
        xi_p: ComplexArray | complex, d_p: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]:
        a1_p = q * 0.5 * t * d_p * ch
        a2_p = d_p * ch + d * 0.5 * t * d_p * sh + xi_p * sh + xi * 0.5 * t * d_p * ch
        big_a_p = a1_p / a2 - big_a * a2_p / a2
        log_b_p = d_p / d - a2_p / a2
        return big_a_p, log_b_p

    c = 2.0 * kappa * vbar / s2
    a_rho, lb_rho = partials(-sigma * iu, -xi * sigma * iu / d)
    a_kap, lb_kap = partials(1.0, xi / d)
    a_sig, lb_sig = partials(-rho * iu, (-xi * rho * iu + sigma * q) / d)

    h = np.empty((5, uu.size), dtype=np.complex128)
    h[0] = -big_a  # (23a)
    h[1] = 2.0 * kappa / s2 * big_d - t * kappa * rho * iu / sigma  # (23b)
    h[2] = -v0 * a_rho + c * lb_rho - t * kappa * vbar * iu / sigma  # (23c)
    h[3] = (
        -v0 * a_kap + 2.0 * vbar / s2 * big_d + c * (lb_kap + 0.5 * t) - t * vbar * rho * iu / sigma
    )  # (23d)
    h[4] = (
        -v0 * a_sig
        - 4.0 * kappa * vbar / (s2 * sigma) * big_d
        + c * lb_sig
        + t * kappa * vbar * rho * iu / s2
    )  # (23e)
    return phi, h


def _by_maturity(t: FloatArray) -> list[tuple[float, NDArray[np.intp]]]:
    return [(float(tt), np.flatnonzero(t == tt)) for tt in np.unique(t)]


def call_price(
    theta: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    *,
    quadrature: Quadrature | None = None,
    form: Form = "cui",
) -> FloatArray:
    """Undiscounted call price per unit of forward, Cui eq. (9)."""
    quad_ = quadrature or DEFAULT_QUADRATURE
    k_, t_ = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(t, dtype=np.float64))
    shape = k_.shape
    k_, t_ = k_.ravel(), t_.ravel()
    u = quad_.nodes.astype(np.complex128)
    out = np.empty(k_.size)
    for tt, idx in _by_maturity(t_):
        phi1 = charfn(theta, u - 1j, tt, form)
        phi2 = charfn(theta, u, tt, form)
        kk = k_[idx]
        osc = np.exp(-1j * np.outer(kk, quad_.nodes)) / (1j * quad_.nodes)
        i1 = (osc * phi1).real @ quad_.weights
        i2 = (osc * phi2).real @ quad_.weights
        strike = np.exp(kk)
        out[idx] = 0.5 * (1.0 - strike) + (i1 - strike * i2) / np.pi
    return out.reshape(shape)


def call_price_and_gradient(
    theta: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    *,
    quadrature: Quadrature | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Price and its gradient with respect to ``theta``, Cui eqs. (9) and (22).

    Returns ``(price, grad)`` with ``grad`` of shape ``(n, 5)``.
    """
    quad_ = quadrature or DEFAULT_QUADRATURE
    th = _unpack(theta)
    k_ = np.atleast_1d(np.asarray(k, dtype=np.float64))
    t_ = np.broadcast_to(np.asarray(t, dtype=np.float64), k_.shape)
    u = quad_.nodes.astype(np.complex128)
    price = np.empty(k_.size)
    grad = np.empty((k_.size, 5))
    for tt, idx in _by_maturity(t_):
        phi1, h1 = _charfn_and_h(th, u - 1j, tt)
        phi2, h2 = _charfn_and_h(th, u, tt)
        kk = k_[idx]
        strike = np.exp(kk)
        osc = np.exp(-1j * np.outer(kk, quad_.nodes)) / (1j * quad_.nodes)
        w = quad_.weights
        i1 = (osc * phi1).real @ w
        i2 = (osc * phi2).real @ w
        price[idx] = 0.5 * (1.0 - strike) + (i1 - strike * i2) / np.pi
        for j in range(5):
            g1 = (osc * (phi1 * h1[j])).real @ w
            g2 = (osc * (phi2 * h2[j])).real @ w
            grad[idx, j] = (g1 - strike * g2) / np.pi
    return price, grad


def call_price_reference(theta: ArrayLike, k: float, t: float, *, form: Form = "trap") -> float:
    """Slow adaptive-quadrature price, used to check the fixed-node rule.

    Integrates the same formula (9) with ``scipy.integrate.quad`` out to the
    point where the characteristic function has decayed below ``1e-14``.
    """
    strike = float(np.exp(k))

    def integrand(u: float) -> float:
        z = np.exp(-1j * u * k) / (1j * u)
        p1 = charfn(theta, np.array([u - 1j]), t, form)[0]
        p2 = charfn(theta, np.array([u]), t, form)[0]
        return float((z * (p1 - strike * p2)).real)

    umax = 50.0
    while abs(charfn(theta, np.array([umax]), t, "trap")[0]) > 1e-14 and umax < 1e6:
        umax *= 2.0
    edges = np.linspace(0.0, umax, 65)
    total = 0.0
    for a, b in pairwise(edges):
        total += quad(integrand, max(a, 1e-12), b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return 0.5 * (1.0 - strike) + total / np.pi
