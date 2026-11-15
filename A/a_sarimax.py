"""A: SARIMA (Seoul only) and SARIMAX (lagged regional means) on hourly log1p PM2.5.

- Fit once on 2021-2022; parameters fixed; the Kalman filter is run through 2023-2024 and the filtered state
  a_{t|t} is pushed h steps with the transition matrix (method from the earlier project's tested sarimax_fc).
- Seasonality: deterministic daily and annual Fourier terms (2 harmonics each) -- known in advance, no leakage.
- SARIMA order: lowest AIC on 2021-2022 among (1,1,1),(2,1,1),(1,1,2),(2,1,2).
- SARIMAX, one model per horizon h: exog = log1p regional means at t+h-L with L in {h, h+3, h+6} (so every exog
  value is known at the origin t); L chosen by 2023 RMSE on the common grid. Exog gaps: forward fill up to 24 h
  (past only), then the 2021-2022 median.
- Outputs predictions A_SARIMA, A_SARIMAX and results/a_sarimax_coefficients.csv (coefficients with 95% CI).

Usage: python -I a_sarimax.py [--quick]   (--quick: fixed order (1,1,1), L = h only)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import predictions_io
from config import HORIZONS, RESULTS, WORK
from data_access import seoul_pm25
from grid import load_grid

EXOG_GROUPS = ("KR_west_coast", "CN_near", "CN_far")
ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
INDEX = pd.date_range("2021-01-01 00:00", "2024-12-31 23:00", freq="h", tz="Asia/Seoul")
TRAIN = INDEX.year <= 2022


def fourier(index):
    t = np.asarray((index - pd.Timestamp("2021-01-01", tz="Asia/Seoul")) / pd.Timedelta(hours=1))
    cols = {}
    for name, period in (("day", 24.0), ("year", 24 * 365.2425)):
        for k in (1, 2):
            cols[f"{name}_sin{k}"] = np.sin(2 * np.pi * k * t / period)
            cols[f"{name}_cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def lagged_exog(region, L):
    out = {}
    for g in EXOG_GROUPS:
        x = np.log1p(region[f"{g}_mean"]).ffill(limit=24).shift(L).reindex(INDEX)
        out[f"{g}_L{L}"] = x.fillna(x[TRAIN].median())
    return pd.DataFrame(out, index=INDEX)


def kalman_forecasts(res, endog, exog, horizons):
    """h-step forecasts at every hour from fixed parameters (ug/m3)."""
    fr = res.model.clone(endog, exog=exog).filter(res.params).filter_results
    S = fr.filtered_state
    T, Z = fr.transition[:, :, 0], fr.design[0, :, 0]
    n = S.shape[1]
    d = np.broadcast_to(fr.obs_intercept[0], (n,))
    c = np.broadcast_to(fr.state_intercept, (len(T), n))
    out = {}
    for h in range(1, max(horizons) + 1):
        S = T @ S + np.concatenate([c[:, h - 1:], np.repeat(c[:, -1:], h - 1, axis=1)], axis=1)
        if h in horizons:
            f = np.full(n, np.nan)
            f[:n - h] = (Z @ S)[:n - h] + d[h:]
            out[h] = np.expm1(f)
    return out


def fit(z, X, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(z[TRAIN], exog=X[TRAIN], order=order).fit(disp=False, maxiter=1000)


def at_grid(fc, grid, h):
    return pd.Series(fc[h], index=INDEX).reindex(grid.origin_time_kst).to_numpy()


def val_rmse(pred, grid, h):
    m = (grid.origin_split == "validation") & grid[f"eligible_h{h}"]
    e = pred[m.to_numpy()] - grid.loc[m, f"y_pm25_h{h}"].to_numpy()
    return float(np.sqrt(np.nanmean(e ** 2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    grid = load_grid()
    z = np.log1p(seoul_pm25().reindex(INDEX))
    F = fourier(INDEX)
    region = pd.read_parquet(WORK / "region_hourly.parquet").set_index("time_kst")
    orders = ORDERS[:1] if a.quick else ORDERS
    aic = {o: fit(z, F, o).aic for o in orders}
    order = min(aic, key=aic.get)
    base = fit(z, F, order)
    fc = kalman_forecasts(base, z, F, HORIZONS)
    predictions_io.wide_to_long(grid.origin_time_kst, grid.origin_split, {h: at_grid(fc, grid, h) for h in HORIZONS}, "A_SARIMA",
                                {"order": order, "quick": a.quick, "exog": "daily+annual Fourier (2 harmonics)"})
    chosen, coefs, preds = {}, [], {}
    for h in HORIZONS:
        cands = {}
        for L in ([h] if a.quick else [h, h + 3, h + 6]):
            X = pd.concat([F, lagged_exog(region, L)], axis=1)
            res = fit(z, X, order)
            p = at_grid(kalman_forecasts(res, z, X, (h,)), grid, h)
            cands[L] = (val_rmse(p, grid, h), p, res)
        L = min(cands, key=lambda k: cands[k][0])
        chosen[h] = {"L": L, "validation_rmse_by_L": {k: v[0] for k, v in cands.items()}}
        preds[h] = cands[L][1]
        res = cands[L][2]
        ci = res.conf_int()
        for name in res.params.index:
            coefs.append({"horizon": h, "L": L, "term": name, "coef": res.params[name],
                          "ci95_low": ci.loc[name, 0], "ci95_high": ci.loc[name, 1], "p": res.pvalues[name]})
    predictions_io.wide_to_long(grid.origin_time_kst, grid.origin_split, preds, "A_SARIMAX",
                                {"order": order, "quick": a.quick, "exog_groups": EXOG_GROUPS,
                                 "lag_by_horizon": {h: c["L"] for h, c in chosen.items()}})
    pd.DataFrame(coefs).to_csv(RESULTS / "a_sarimax_coefficients.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "a_sarimax_selection.json").write_text(json.dumps(
        {"order": order, "aic_by_order": {str(k): v for k, v in aic.items()}, "exog_groups": EXOG_GROUPS,
         "lag_by_horizon": chosen}, indent=1), encoding="utf-8")
    print(json.dumps({"order": order, "lags": {h: c["L"] for h, c in chosen.items()}}))
