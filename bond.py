"""
bond.py — fixed income and rates derivatives in one file.

Usage
-----
    import bond

    # Bond pricing
    p = bond.price(face=1000, coupon=0.05, maturity=10, ytm=0.04)
    y = bond.ytm(face=1000, coupon=0.05, maturity=10, price=1081.76)
    d = bond.duration(1000, 0.05, 10, 0.05)
    v = bond.dv01(1000, 0.05, 10, 0.05)

    # Yield curves
    params = bond.fit_nelson_siegel([1, 2, 5, 10, 30], [0.05, 0.049, 0.047, 0.045, 0.043])
    r5 = bond.spot_rate(5.0, *params)

    # Swaps
    curve = bond.bootstrap({0.25: 0.053, 0.5: 0.052}, {1: 0.051, 5: 0.046, 10: 0.044})
    pv    = bond.price_irs(curve, notional=10_000_000, fixed_rate=0.045, maturity=5)

    # Swaption vol surface
    alpha, rho, nu = bond.fit_sabr({0.03: 0.28, 0.04: 0.24, 0.05: 0.22}, F=0.04, T=5.0)
    sigma = bond.sabr_vol(F=0.04, K=0.045, T=5.0, alpha=alpha, beta=0.5, rho=rho, nu=nu)
    pv    = bond.black_swaption(F=0.04, K=0.045, sigma=sigma, T=5.0, annuity_factor=4.3)

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import calendar
import math
import warnings
from datetime import date
from typing import Literal, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq, minimize
from scipy.stats import norm

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _MPL = True
except ImportError:
    _MPL = False


# =============================================================================
# UTILS — day counts, year fractions, discount factors
# =============================================================================

DayCount = Literal["ACT_ACT", "ACT_360", "ACT_365", "THIRTY_360", "THIRTY_E_360"]


def _to_date(d):
    return date.fromisoformat(d) if isinstance(d, str) else d


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def year_fraction(start, end, convention: DayCount = "ACT_ACT") -> float:
    """Year fraction between two dates under a given day count convention.

    Conventions: ACT_ACT, ACT_360, ACT_365, THIRTY_360, THIRTY_E_360
    """
    s, e = _to_date(start), _to_date(end)
    if s == e:
        return 0.0
    if e < s:
        return -year_fraction(e, s, convention)
    actual = (e - s).days
    if convention == "ACT_ACT":
        total, cur = 0.0, s
        while cur.year < e.year:
            nxt = date(cur.year + 1, 1, 1)
            total += (nxt - cur).days / (366 if _is_leap(cur.year) else 365)
            cur = nxt
        return total + (e - cur).days / (366 if _is_leap(e.year) else 365)
    if convention == "ACT_360":
        return actual / 360
    if convention == "ACT_365":
        return actual / 365
    if convention in ("THIRTY_360", "THIRTY_E_360"):
        d1, m1, y1 = s.day, s.month, s.year
        d2, m2, y2 = e.day, e.month, e.year
        if convention == "THIRTY_360":
            if d1 == 31: d1 = 30
            if d2 == 31 and d1 == 30: d2 = 30
        else:
            if d1 == 31: d1 = 30
            if d2 == 31: d2 = 30
        return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360
    raise ValueError(f"Unknown day count: {convention!r}")


def discount_factor(zero_rate: float, t: float, compounding: str = "continuous") -> float:
    """Discount factor P(0,t) from a zero rate."""
    if compounding == "continuous":
        return float(np.exp(-zero_rate * t))
    if compounding == "annual":
        return 1.0 / (1 + zero_rate) ** t
    if compounding in ("semi-annual", "semiannual"):
        return 1.0 / (1 + zero_rate / 2) ** (2 * t)
    raise ValueError(f"Unknown compounding: {compounding!r}")


# =============================================================================
# CORE — bond pricing, YTM, duration, convexity, DV01
# =============================================================================

def price(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
) -> float:
    """Dirty (full) price of a fixed-rate bond.

    Parameters
    ----------
    face      : par value
    coupon    : annual coupon rate (e.g. 0.05 = 5%)
    maturity  : years to maturity
    ytm       : annual yield to maturity
    frequency : coupon payments per year (1=annual, 2=semi-annual)

    Examples
    --------
    >>> price(1000, 0.05, 10, 0.05)   # par bond
    1000.0
    >>> price(1000, 0.05, 10, 0.04)   # premium bond
    1081.76
    """
    c = face * coupon / frequency
    r = ytm / frequency
    n = int(round(maturity * frequency))
    if abs(r) < 1e-12:
        return c * n + face
    return c * (1 - (1 + r) ** -n) / r + face / (1 + r) ** n


def accrued_interest(
    face: float = 1000.0,
    coupon: float = 0.05,
    frequency: int = 2,
    t_since_last: float = 0.0,
) -> float:
    """Accrued interest since last coupon date.

    Parameters
    ----------
    t_since_last : fraction of coupon period elapsed (0=just paid, 1=about to pay)
    """
    return face * coupon / frequency * t_since_last


def clean_price(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
    t_since_last: float = 0.0,
) -> float:
    """Clean price = dirty price − accrued interest."""
    return price(face, coupon, maturity, ytm, frequency) - accrued_interest(face, coupon, frequency, t_since_last)


def ytm(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    price: float = 1000.0,
    frequency: int = 2,
) -> float:
    """Yield to maturity (solved numerically via Brent's method).

    Examples
    --------
    >>> ytm(1000, 0.05, 10, 1081.76)
    0.04001...
    """
    def obj(y):
        c = face * coupon / frequency
        r = y / frequency
        n = int(round(maturity * frequency))
        if abs(r) < 1e-12:
            pv = c * n + face
        else:
            pv = c * (1 - (1 + r) ** -n) / r + face / (1 + r) ** n
        return pv - price

    try:
        return brentq(obj, 1e-6, 9.99, xtol=1e-10, maxiter=500)
    except ValueError:
        raise ValueError(f"YTM not found for price={price:.4f} — check inputs.")


def duration(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
) -> float:
    """Macaulay duration in years.

    For a zero-coupon bond, duration equals maturity.

    Examples
    --------
    >>> duration(1000, 0.0, 10, 0.05)   # zero-coupon
    10.0
    """
    c = face * coupon / frequency
    r = ytm / frequency
    n = int(round(maturity * frequency))
    p = weighted = 0.0
    for i in range(1, n + 1):
        cf = c if i < n else c + face
        pv = cf / (1 + r) ** i
        p += pv
        weighted += (i / frequency) * pv
    return weighted / p if p else 0.0


def modified_duration(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
) -> float:
    """Modified duration = Macaulay duration / (1 + ytm/frequency)."""
    mac = duration(face, coupon, maturity, ytm, frequency)
    return mac / (1 + ytm / frequency)


def convexity(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
) -> float:
    """Convexity (second-order price sensitivity to yield).

    ΔP/P ≈ -MD·Δy + ½·C·Δy²
    """
    c = face * coupon / frequency
    r = ytm / frequency
    n = int(round(maturity * frequency))
    p = cx = 0.0
    for i in range(1, n + 1):
        cf = c if i < n else c + face
        pv = cf / (1 + r) ** i
        t = i / frequency
        p += pv
        cx += pv * t * (t + 1 / frequency)
    return cx / (p * (1 + r) ** 2) if p else 0.0


def dv01(
    face: float = 1000.0,
    coupon: float = 0.05,
    maturity: float = 10.0,
    ytm: float = 0.05,
    frequency: int = 2,
) -> float:
    """Dollar value of 1bp (DV01 / PVBP) — computed via central difference.

    Returns the price change for a +1bp parallel shift in yield.

    Examples
    --------
    >>> dv01(1000, 0.05, 10, 0.05)   # ≈ $0.779 per $1000 face
    0.779
    """
    bp = 0.0001
    p_up = price(face, coupon, maturity, ytm + bp, frequency)
    p_dn = price(face, coupon, maturity, ytm - bp, frequency)
    return (p_dn - p_up) / 2


def yield_to_call(
    face: float = 1000.0,
    coupon: float = 0.05,
    call_price: float = 1050.0,
    years_to_call: float = 5.0,
    market_price: float = 1000.0,
    frequency: int = 2,
) -> float:
    """Yield to call — treats call_price as face and years_to_call as maturity."""
    return ytm(
        face=call_price,
        coupon=coupon * face / call_price,
        maturity=years_to_call,
        price=market_price,
        frequency=frequency,
    )


def price_from_spot_curve(
    face: float,
    coupon: float,
    maturity: float,
    spot_curve_fn,
    frequency: int = 2,
) -> float:
    """Price a bond by discounting cash flows at continuously-compounded spot rates.

    Parameters
    ----------
    spot_curve_fn : callable(t) -> spot rate for maturity t
    """
    c = face * coupon / frequency
    n = int(round(maturity * frequency))
    pv = 0.0
    for i in range(1, n + 1):
        t = i / frequency
        r = spot_curve_fn(t)
        cf = c if i < n else c + face
        pv += cf * math.exp(-r * t)
    return pv


# =============================================================================
# CURVES — Nelson-Siegel, Svensson, cubic spline, FRED data
# =============================================================================

def _ns_rate(t, b0, b1, b2, tau):
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = t / tau
        factor = np.where(x < 1e-8, 1.0, (1 - np.exp(-x)) / x)
        hump = np.where(x < 1e-8, 0.0, factor - np.exp(-x))
    return b0 + b1 * factor + b2 * hump


def fit_nelson_siegel(
    maturities: Sequence[float],
    yields: Sequence[float],
) -> tuple[float, float, float, float]:
    """Fit Nelson-Siegel model to market yields.

    r(t) = β₀ + β₁·L(t) + β₂·H(t)

    Parameters
    ----------
    maturities : list of tenors in years
    yields     : corresponding market yields (decimal)

    Returns
    -------
    (b0, b1, b2, tau)

    Examples
    --------
    >>> params = fit_nelson_siegel([1,2,5,10,30], [0.05,0.049,0.047,0.045,0.043])
    >>> spot_rate(5.0, *params)
    0.047...
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)

    def obj(params):
        b0, b1, b2, tau = params
        if tau <= 0 or b0 <= 0:
            return 1e10
        return float(np.sum((_ns_rate(t, b0, b1, b2, tau) - y) ** 2))

    best, best_val = None, np.inf
    for tau0 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj, [y[-1], y[0] - y[-1], 0.0, tau0],
                           method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000})
        if res.fun < best_val:
            best_val, best = res.fun, res
    b0, b1, b2, tau = best.x
    return float(b0), float(b1), float(b2), float(abs(tau))


def spot_rate(t, b0: float, b1: float, b2: float, tau: float):
    """Nelson-Siegel spot rate at maturity t (scalar or array).

    Examples
    --------
    >>> params = fit_nelson_siegel([1,5,10,30], [0.05,0.047,0.045,0.043])
    >>> spot_rate(5.0, *params)
    0.047...
    """
    return _ns_rate(t, b0, b1, b2, tau)


def _sv_rate(t, b0, b1, b2, b3, tau1, tau2):
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x1, x2 = t / tau1, t / tau2
        f1 = np.where(x1 < 1e-8, 1.0, (1 - np.exp(-x1)) / x1)
        h1 = np.where(x1 < 1e-8, 0.0, f1 - np.exp(-x1))
        f2 = np.where(x2 < 1e-8, 1.0, (1 - np.exp(-x2)) / x2)
        h2 = np.where(x2 < 1e-8, 0.0, f2 - np.exp(-x2))
    return b0 + b1 * f1 + b2 * h1 + b3 * h2


def fit_svensson(
    maturities: Sequence[float],
    yields: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    """Fit Svensson (6-parameter) model to market yields.

    Returns (b0, b1, b2, b3, tau1, tau2)
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)

    def obj(params):
        b0, b1, b2, b3, tau1, tau2 = params
        if tau1 <= 0 or tau2 <= 0 or b0 <= 0:
            return 1e10
        return float(np.sum((_sv_rate(t, b0, b1, b2, b3, tau1, tau2) - y) ** 2))

    best, best_val = None, np.inf
    for t1, t2 in [(1.0, 5.0), (0.5, 3.0), (2.0, 8.0)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj, [y[-1], y[0] - y[-1], 0.0, 0.0, t1, t2],
                           method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
        if res.fun < best_val:
            best_val, best = res.fun, res
    b0, b1, b2, b3, tau1, tau2 = best.x
    return float(b0), float(b1), float(b2), float(b3), float(abs(tau1)), float(abs(tau2))


def spot_rate_svensson(t, b0, b1, b2, b3, tau1, tau2):
    """Svensson spot rate at maturity t."""
    return _sv_rate(t, b0, b1, b2, b3, tau1, tau2)


def fit_cubic_spline(maturities: Sequence[float], yields: Sequence[float]):
    """Natural cubic spline through market data points.

    Returns a CubicSpline callable: cs(t) → yield at t.
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    idx = np.argsort(t)
    return CubicSpline(t[idx], y[idx], bc_type="natural")


def curve_forward_rate(t1: float, t2: float, spot_fn) -> float:
    """Continuously-compounded forward rate f(t1,t2) from a spot rate function.

    f(t1,t2) = (r(t2)·t2 − r(t1)·t1) / (t2 − t1)
    """
    if t2 <= t1:
        raise ValueError("t2 must be > t1")
    return (spot_fn(t2) * t2 - spot_fn(t1) * t1) / (t2 - t1)


_FRED_SERIES = {
    0.083: "DGS1MO", 0.25: "DGS3MO", 0.5: "DGS6MO",
    1.0: "DGS1", 2.0: "DGS2", 3.0: "DGS3", 5.0: "DGS5",
    7.0: "DGS7", 10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30",
}


def fetch_treasury_yields(api_key: str | None = None) -> tuple[list, list]:
    """Fetch current US Treasury par yields from FRED.

    Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html

    Returns
    -------
    (maturities, yields) — two lists of floats

    Examples
    --------
    >>> mats, ylds = fetch_treasury_yields(api_key="your_key_here")
    >>> params = fit_nelson_siegel(mats, ylds)
    """
    import requests
    base = "https://api.stlouisfed.org/fred/series/observations"
    mats, ylds = [], []
    for mat, sid in sorted(_FRED_SERIES.items()):
        params = {"series_id": sid, "sort_order": "desc", "limit": 5,
                  "file_type": "json", "api_key": api_key or "abcdefghijklmnopqrstuvwxyz123456"}
        try:
            r = requests.get(base, params=params, timeout=10)
            r.raise_for_status()
            for o in r.json().get("observations", []):
                v = o.get("value", ".")
                if v != ".":
                    mats.append(mat)
                    ylds.append(float(v) / 100)
                    break
        except Exception as e:
            warnings.warn(f"Could not fetch {sid}: {e}")
    return mats, ylds


def plot_curve(params, model: str = "ns", t_min: float = 0.1, t_max: float = 30.0,
               market_mats=None, market_yields=None, label=None, show: bool = True):
    """Plot a fitted yield curve.

    Parameters
    ----------
    params : tuple from fit_nelson_siegel() / fit_svensson() / fit_cubic_spline()
    model  : 'ns', 'sv', or 'spline'
    """
    if not _MPL:
        raise ImportError("matplotlib required")
    t = np.linspace(t_min, t_max, 300)
    rates = (_ns_rate(t, *params) if model == "ns" else
             _sv_rate(t, *params) if model == "sv" else
             params(t))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, rates * 100, lw=2, label=label or f"{model.upper()} curve")
    if market_mats is not None and market_yields is not None:
        ax.scatter(market_mats, np.asarray(market_yields) * 100,
                   color="black", zorder=5, s=40, label="Market")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
    ax.set_title("Yield Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


# =============================================================================
# SWAPS — curve bootstrapping, IRS pricing, DV01, par rates
# =============================================================================

class DiscountCurve:
    """Bootstrapped discount curve with log-linear interpolation.

    Attributes
    ----------
    tenors : sorted maturity times (years)
    dfs    : discount factors P(0,t)
    """

    def __init__(self, tenors: list, dfs: list):
        self.tenors = np.asarray(tenors, dtype=float)
        self.dfs = np.asarray(dfs, dtype=float)
        self._interp = interp1d(
            self.tenors, np.log(self.dfs), kind="linear",
            bounds_error=False,
            fill_value=(np.log(self.dfs[0]), np.log(self.dfs[-1])),
        )

    def df(self, t):
        """Discount factor P(0,t)."""
        return np.exp(self._interp(t))

    def zero_rate(self, t, compounding: str = "continuous"):
        """Zero rate for maturity t."""
        t = np.asarray(t, dtype=float)
        d = self.df(t)
        with np.errstate(divide="ignore"):
            r = -np.log(d) / t
        if compounding == "annual":
            r = np.exp(r) - 1
        elif compounding in ("semi-annual", "semiannual"):
            r = 2 * (np.exp(r / 2) - 1)
        return float(r) if r.ndim == 0 else r

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate f(t1,t2)."""
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        return -math.log(float(self.df(t2)) / float(self.df(t1))) / (t2 - t1)

    def __repr__(self):
        return (f"DiscountCurve(tenors=[{self.tenors[0]:.2f}..{self.tenors[-1]:.2f}], "
                f"n={len(self.tenors)})")


def bootstrap(
    deposits: dict,
    swaps: dict,
    frequency: int = 2,
) -> DiscountCurve:
    """Bootstrap a discount curve from deposit and swap par rates.

    Algorithm
    ---------
    1. Deposits: P(0,t) = 1 / (1 + r·t)  [simple interest]
    2. Swaps:    solve for P(0,T) such that fixed leg PV = float leg PV

    Parameters
    ----------
    deposits : {maturity_yr: rate}  — short-end deposit rates
    swaps    : {maturity_yr: rate}  — swap par rates (semi-annual fixed vs float)
    frequency : fixed leg frequency (default 2 = semi-annual)

    Returns
    -------
    DiscountCurve

    Examples
    --------
    >>> curve = bootstrap({0.25: 0.053, 0.5: 0.052},
    ...                   {1: 0.051, 2: 0.049, 5: 0.046, 10: 0.044})
    >>> curve.df(5.0)
    0.796...
    >>> curve.zero_rate(5.0)
    0.046...
    """
    tenors, dfs = [0.0], [1.0]

    for t in sorted(deposits):
        tenors.append(t)
        dfs.append(1.0 / (1.0 + deposits[t] * t))

    curve = DiscountCurve(tenors, dfs)

    for T in sorted(swaps):
        K = swaps[T]
        dt = 1.0 / frequency
        times = np.arange(dt, T + 1e-9, dt)

        def residual(df_T, T=T, K=K, times=times, tenors=tenors, dfs=dfs):
            all_t, all_df = tenors + [T], dfs + [df_T]
            trial = DiscountCurve(all_t, all_df)
            fixed_pv = K * dt * sum(trial.df(ti) for ti in times)
            return fixed_pv + trial.df(T) - 1.0

        known_sum = K * dt * sum(curve.df(ti) for ti in times[:-1])
        guess = max(0.01, min((1.0 - known_sum) / (1.0 + K * dt), 0.9999))
        try:
            df_T = brentq(residual, 0.001, 1.5, xtol=1e-12, maxiter=500)
        except ValueError:
            df_T = guess

        tenors.append(T)
        dfs.append(df_T)
        curve = DiscountCurve(tenors, dfs)

    return curve


def annuity(curve: DiscountCurve, maturity: float, frequency: int = 2) -> float:
    """Annuity factor A(0,T) = (1/freq) · Σ P(0,tᵢ)."""
    dt = 1.0 / frequency
    times = np.arange(dt, maturity + 1e-9, dt)
    return dt * float(np.sum(curve.df(times)))


def par_rate(curve: DiscountCurve, maturity: float, frequency: int = 2) -> float:
    """Par swap rate K* = (1 − P(0,T)) / A(0,T).

    Examples
    --------
    >>> curve = bootstrap({0.5: 0.05}, {1: 0.049, 5: 0.046, 10: 0.044})
    >>> par_rate(curve, 5.0)
    0.046...
    """
    A = annuity(curve, maturity, frequency)
    return (1.0 - float(curve.df(maturity))) / A if A else 0.0


def forward_rate(curve: DiscountCurve, t1: float, t2: float) -> float:
    """Continuously-compounded forward rate from the discount curve."""
    return curve.forward_rate(t1, t2)


def price_irs(
    curve: DiscountCurve,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.05,
    maturity: float = 5.0,
    frequency: int = 2,
    pay_fixed: bool = True,
) -> float:
    """Price a vanilla fixed-for-floating interest rate swap.

    NPV (payer) = N · [(1 − P(0,T)) − K · A(0,T)]

    Parameters
    ----------
    curve      : bootstrapped DiscountCurve
    notional   : notional principal
    fixed_rate : fixed coupon rate
    maturity   : swap maturity in years
    frequency  : fixed leg payment frequency
    pay_fixed  : True = payer (pay fixed, receive float)

    Examples
    --------
    >>> curve = bootstrap({0.25: 0.053}, {1: 0.051, 5: 0.046})
    >>> price_irs(curve, 10_000_000, 0.045, 5, pay_fixed=True)
    48_521...  # positive = in the money
    """
    A = annuity(curve, maturity, frequency)
    float_leg = 1.0 - float(curve.df(maturity))
    fixed_leg = fixed_rate * A
    sign = 1.0 if pay_fixed else -1.0
    return sign * notional * (float_leg - fixed_leg)


def dv01_irs(
    curve: DiscountCurve,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.05,
    maturity: float = 5.0,
    frequency: int = 2,
) -> float:
    """DV01 of an IRS via parallel curve bump (+/− 1bp).

    Examples
    --------
    >>> curve = bootstrap({0.5: 0.05}, {5: 0.046, 10: 0.044})
    >>> dv01_irs(curve, 10_000_000, 0.046, 5)
    4_450...
    """
    bp = 0.0001
    dfs_up = np.exp(np.log(curve.dfs) - bp * curve.tenors)
    dfs_dn = np.exp(np.log(curve.dfs) + bp * curve.tenors)
    c_up = DiscountCurve(list(curve.tenors), list(dfs_up))
    c_dn = DiscountCurve(list(curve.tenors), list(dfs_dn))
    pv_up = price_irs(c_up, notional, fixed_rate, maturity, frequency)
    pv_dn = price_irs(c_dn, notional, fixed_rate, maturity, frequency)
    return abs(pv_up - pv_dn) / 2


def plot_swap_curve(curve: DiscountCurve, t_max: float = 30.0, show: bool = True):
    """Plot zero rates and forward rates from a bootstrapped discount curve."""
    if not _MPL:
        raise ImportError("matplotlib required")
    t = np.linspace(0.1, t_max, 300)
    zeros = curve.zero_rate(t) * 100
    fwds = np.array([curve.forward_rate(max(0.01, ti - 0.25), ti + 0.25) * 100
                     for ti in t])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t, zeros, lw=2, color="#1565C0")
    ax1.scatter(curve.tenors[1:], curve.zero_rate(curve.tenors[1:]) * 100,
                color="red", zorder=5, s=30, label="Knots")
    ax1.set_xlabel("Maturity (years)"); ax1.set_ylabel("Zero Rate (%)")
    ax1.set_title("Zero Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(t, fwds, lw=2, color="#2E7D32")
    ax2.set_xlabel("Maturity (years)"); ax2.set_ylabel("Forward Rate (%)")
    ax2.set_title("Forward Curve"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return ax1, ax2


# =============================================================================
# VOL — SABR model, swaption pricing, vol surface
# =============================================================================

def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """SABR implied Black volatility (Hagan et al. 2002).

    Parameters
    ----------
    F     : forward rate
    K     : strike rate
    T     : option expiry (years)
    alpha : initial vol level (σ₀)
    beta  : CEV exponent 0≤β≤1 (0.5 is industry default for rates)
    rho   : correlation between forward and vol processes (−1 < ρ < 1)
    nu    : vol-of-vol (ν ≥ 0)

    Examples
    --------
    >>> sabr_vol(0.04, 0.04, 5.0, alpha=0.20, beta=0.5, rho=-0.30, nu=0.40)
    0.199...
    """
    if abs(F - K) < 1e-7:
        term1 = alpha / F ** (1 - beta)
        term2 = 1 + (
            (1 - beta) ** 2 / 24 * alpha ** 2 / F ** (2 - 2 * beta)
            + rho * beta * nu * alpha / (4 * F ** (1 - beta))
            + (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T
        return term1 * term2

    log_FK = math.log(F / K)
    FK_mid = math.sqrt(F * K)
    FK_beta = FK_mid ** (1 - beta)
    z = (nu / alpha) * FK_mid ** (1 - beta) * log_FK
    denom = 1 - 2 * rho * z + z ** 2
    x_z = math.log((math.sqrt(denom) + z - rho) / (1 - rho))
    z_over_xz = z / x_z if abs(x_z) > 1e-12 else 1.0
    prefactor = alpha / (
        FK_beta * (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
                   + (1 - beta) ** 4 / 1920 * log_FK ** 4)
    )
    corr = 1 + (
        (1 - beta) ** 2 / 24 * alpha ** 2 / FK_mid ** (2 - 2 * beta)
        + rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta))
        + (2 - 3 * rho ** 2) / 24 * nu ** 2
    ) * T
    return max(prefactor * z_over_xz * corr, 1e-6)


def fit_sabr(
    market_vols: dict,
    F: float,
    T: float,
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """Calibrate SABR α, ρ, ν to market implied vols (β fixed).

    Parameters
    ----------
    market_vols : {strike: implied_black_vol}
    F           : forward rate (ATM)
    T           : option expiry (years)
    beta        : CEV exponent (fixed, default 0.5)

    Returns
    -------
    (alpha, rho, nu)

    Examples
    --------
    >>> vols = {0.03: 0.28, 0.04: 0.24, 0.05: 0.22, 0.06: 0.23}
    >>> alpha, rho, nu = fit_sabr(vols, F=0.04, T=5.0)
    """
    strikes = np.array(list(market_vols.keys()), dtype=float)
    target = np.array(list(market_vols.values()), dtype=float)
    atm_vol = market_vols.get(F, float(np.mean(target)))
    alpha0 = atm_vol * F ** (1 - beta)

    def obj(params):
        a, r, n = params
        if a <= 0 or n < 0 or abs(r) >= 1:
            return 1e10
        return float(np.sum((np.array([sabr_vol(F, K, T, a, beta, r, n)
                                       for K in strikes]) - target) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(obj, [alpha0, -0.20, 0.40], method="L-BFGS-B",
                       bounds=[(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)],
                       options={"ftol": 1e-12, "maxiter": 2000})
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def vol_surface(
    expiries: Sequence[float],
    strikes: Sequence[float],
    forwards: dict,
    sabr_params: dict,
    beta: float = 0.5,
) -> np.ndarray:
    """Build an implied vol surface matrix from calibrated SABR parameters.

    Parameters
    ----------
    expiries    : option expiries (years)
    strikes     : strikes
    forwards    : {expiry: forward_rate}
    sabr_params : {expiry: (alpha, rho, nu)}  from fit_sabr()
    beta        : CEV exponent

    Returns
    -------
    np.ndarray of shape (len(expiries), len(strikes))

    Examples
    --------
    >>> surface = vol_surface([1,5,10], [0.02,0.04,0.06],
    ...                       {1:0.04, 5:0.044, 10:0.045},
    ...                       {1:(0.20,-0.3,0.4), 5:(0.18,-0.25,0.38), 10:(0.16,-0.2,0.35)})
    """
    result = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        a, r, n = sabr_params[T]
        F = forwards[T]
        for j, K in enumerate(strikes):
            result[i, j] = sabr_vol(F, K, T, a, beta, r, n)
    return result


def black_swaption(
    F: float,
    K: float,
    sigma: float,
    T: float,
    annuity_factor: float,
    notional: float = 1_000_000.0,
    payer: bool = True,
) -> float:
    """Black-76 swaption price.

    PV (payer) = N · A · [F·N(d₁) − K·N(d₂)]

    Parameters
    ----------
    F              : forward swap rate
    K              : strike (fixed rate)
    sigma          : Black implied vol
    T              : option expiry (years)
    annuity_factor : A(0,T) — use bond.annuity(curve, maturity)
    notional       : swap notional
    payer          : True=payer swaption, False=receiver swaption

    Examples
    --------
    >>> black_swaption(0.04, 0.04, 0.20, 5.0, annuity_factor=4.3)
    28_580...
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, (F - K) if payer else (K - F))
        return notional * annuity_factor * intrinsic
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if payer:
        pv = annuity_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        pv = annuity_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return notional * pv


def black_swaption_vol(
    F: float,
    K: float,
    T: float,
    annuity_factor: float,
    market_price: float,
    notional: float = 1_000_000.0,
    payer: bool = True,
) -> float:
    """Implied Black vol from a market swaption price (inverse Black-76)."""
    def obj(s):
        return black_swaption(F, K, s, T, annuity_factor, notional, payer) - market_price
    try:
        return brentq(obj, 1e-6, 10.0, xtol=1e-10)
    except ValueError:
        raise ValueError(f"Implied vol not found for price={market_price:.2f}")


def plot_smile(strikes, F, T, alpha, beta, rho, nu,
               market_vols=None, label=None, show=True):
    """Plot SABR vol smile for a given expiry."""
    if not _MPL:
        raise ImportError("matplotlib required")
    K = np.asarray(strikes, dtype=float)
    vols = np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in K])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K * 100, vols * 100, lw=2, label=label or f"SABR T={T}y")
    if market_vols:
        mk = np.array(list(market_vols.keys()))
        mv = np.array(list(market_vols.values()))
        ax.scatter(mk * 100, mv * 100, color="black", zorder=5, s=40, label="Market")
    ax.axvline(F * 100, ls="--", color="gray", lw=1, label=f"ATM={F:.2%}")
    ax.set_xlabel("Strike (%)"); ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"Vol Smile — {T}y expiry"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_surface(expiries, strikes, surface, title="Implied Vol Surface", show=True):
    """3D plot of an implied volatility surface."""
    if not _MPL:
        raise ImportError("matplotlib required")
    T = np.asarray(expiries)
    K = np.asarray(strikes)
    T_grid, K_grid = np.meshgrid(T, K, indexing="ij")
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T_grid, K_grid * 100, surface * 100, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Expiry (years)"); ax.set_ylabel("Strike (%)")
    ax.set_zlabel("Implied Vol (%)"); ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return ax
