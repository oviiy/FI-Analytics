# bond

[![CI](https://github.com/yourusername/bond/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/bond/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Minimal, zero-bloat Python library for fixed income and rates derivatives.

```python
from bond.core import price, ytm, duration, dv01
from bond.curves import fit_nelson_siegel, spot_rate, fetch_treasury_yields
from bond.swaps import bootstrap, price_irs, par_rate
from bond.vol import fit_sabr, sabr_vol, black_swaption
```

Four independent modules. Each works standalone. No QuantLib. No Bloomberg.
Just `numpy`, `scipy`, and `matplotlib`.

---

## Install

```bash
pip install bond-math
```

Or from source:

```bash
git clone https://github.com/yourusername/bond.git
cd bond
pip install -e ".[dev]"
```

---

## Modules

### `bond.core` — Bond Pricing

Price, yield, duration, convexity, DV01, accrued interest.

```python
from bond.core import price, ytm, duration, modified_duration, convexity, dv01

# 5% semi-annual 10-year bond at 4% yield
p = price(face=1000, coupon=0.05, maturity=10, ytm=0.04, frequency=2)
# → 1081.11

# Solve for yield given price
y = ytm(face=1000, coupon=0.05, maturity=10, price=1081.109, frequency=2)
# → 0.0400

# Risk measures
mac = duration(1000, 0.05, 10, 0.05, 2)       # → 7.99 years (Macaulay)
mod = modified_duration(1000, 0.05, 10, 0.05, 2)  # → 7.79 years
cx  = convexity(1000, 0.05, 10, 0.05, 2)       # → 76.3
d   = dv01(1000, 0.05, 10, 0.05, 2)            # → $7.79 per $1000 face
```

Full API: `price`, `full_price`, `clean_price`, `ytm`, `accrued_interest`,
`duration`, `modified_duration`, `convexity`, `dv01`, `yield_to_call`,
`price_change_approx`, `price_from_spot_curve`.

---

### `bond.curves` — Yield Curve Fitting

Nelson-Siegel, Svensson, cubic spline. Pull live Treasury data from FRED.

```python
from bond.curves import fit_nelson_siegel, spot_rate, fetch_treasury_yields, plot_curve

# Fit Nelson-Siegel to market data
maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
yields     = [0.053, 0.052, 0.051, 0.049, 0.047, 0.046, 0.044, 0.044, 0.043, 0.043]

params = fit_nelson_siegel(maturities, yields)   # → (b0, b1, b2, tau)

# Query any maturity
r_5y  = spot_rate(5.0, *params)    # → 0.0456
r_10y = spot_rate(10.0, *params)   # → 0.0440

# Live Treasury data (requires internet)
mats, ylds = fetch_treasury_yields()
params = fit_nelson_siegel(mats, ylds)

# Plot
plot_curve(params, model='ns', market_mats=maturities, market_yields=yields)
```

Full API: `fit_nelson_siegel`, `spot_rate`, `fit_svensson`, `spot_rate_svensson`,
`fit_cubic_spline`, `forward_rate`, `fetch_treasury_yields`, `plot_curve`.

---

### `bond.swaps` — Swap Pricing

Bootstrap a discount curve, price vanilla IRS, compute DV01 and par rates.

```python
from bond.swaps import bootstrap, price_irs, dv01_irs, par_rate, annuity

# Market data
deposits = {0.25: 0.053, 0.5: 0.052}          # deposit rates
swaps    = {1: 0.051, 2: 0.049, 5: 0.046,     # swap par rates
            7: 0.045, 10: 0.044, 30: 0.043}

# Bootstrap discount curve
curve = bootstrap(deposits, swaps)

# Discount factors and zero rates
curve.df(5.0)           # → 0.796
curve.zero_rate(5.0)    # → 0.0459
curve.forward_rate(5, 7)  # → 0.0422

# Par swap rate
par_rate(curve, 5.0)    # → 0.0460

# Price a 5Y pay-fixed swap ($10M notional)
pv = price_irs(curve, notional=10_000_000, fixed_rate=0.045, maturity=5, pay_fixed=True)
# → $48,521 (in the money — paying below par)

# DV01
d = dv01_irs(curve, 10_000_000, 0.045, 5)
# → $4,450 per basis point
```

Full API: `bootstrap`, `DiscountCurve`, `price_irs`, `dv01_irs`, `par_rate`,
`annuity`, `forward_rate`, `plot_curve`.

---

### `bond.vol` — SABR Vol Surface

Calibrate SABR to market smiles, build a full vol surface, price swaptions.

```python
from bond.vol import fit_sabr, sabr_vol, vol_surface, black_swaption, plot_surface

# Market vol quotes: {strike: implied_vol}
market_vols = {
    0.030: 0.310, 0.035: 0.280, 0.040: 0.250,   # ITM
    0.045: 0.240, 0.050: 0.245, 0.060: 0.255,   # OTM
}
F = 0.042   # forward rate
T = 5.0     # option expiry

# Calibrate SABR
alpha, rho, nu = fit_sabr(market_vols, F=F, T=T, beta=0.5)

# Query any strike
sigma = sabr_vol(F, K=0.045, T=5.0, alpha=alpha, beta=0.5, rho=rho, nu=nu)

# Black-76 swaption price
pv = black_swaption(
    F=F, K=0.045, sigma=sigma, T=T,
    annuity_factor=4.3, notional=10_000_000,
    payer=True,
)

# Build and plot a full surface
surface = vol_surface(expiries, strikes, forwards, sabr_params)
plot_surface(expiries, strikes, surface)
```

Full API: `sabr_vol`, `fit_sabr`, `vol_surface`, `black_swaption`,
`black_swaption_vol`, `plot_smile`, `plot_surface`.

---

## Notebooks

Interactive Jupyter notebooks (Colab-ready, one-click):

| Notebook | Topics |
|---|---|
| [01 — Bond Math](notebooks/01_bond_math.ipynb) | Price, YTM, duration, convexity, DV01, accrued interest |
| [02 — Yield Curves](notebooks/02_yield_curves.ipynb) | Nelson-Siegel, Svensson, cubic spline, FRED data |
| [03 — Swaps](notebooks/03_swaps.ipynb) | Bootstrapping, IRS pricing, par rates, DV01 |
| [04 — Vol Surface](notebooks/04_vol_surface.ipynb) | SABR, swaption pricing, 3D surface |

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

95+ tests covering known textbook values, round-trip invariants,
put-call parity, and calibration accuracy.

---

## Design principles

- **Independent modules** — import only what you need. `bond.core` has no dependency
  on `bond.curves` or `bond.swaps`.
- **No magic** — every function is documented with the formula it implements.
- **Readable over clever** — code is meant to be read alongside a textbook.
- **No QuantLib** — the only dependencies are `numpy`, `scipy`, `matplotlib`, and `requests`.

---

## References

- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious modeling of yield curves.*
- Svensson, L.E. (1994). *Estimating and interpreting forward interest rates.* IMF.
- Hagan, P.S. et al. (2002). *Managing smile risk.* Wilmott Magazine.
- Hull, J. (2022). *Options, Futures, and Other Derivatives.* 11th ed.

---

## License

MIT
