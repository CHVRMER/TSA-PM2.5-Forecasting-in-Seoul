"""A step 2: SARIMAX region/country comparisons on top of the SARIMA order chosen in a_sarima.py.

Conditions (each is a separate SARIMAX per horizon h, exog = log1p regional mean at t+h-L, L in {h, h+3, h+6},
L chosen by 2023 RMSE -- the team rule from a_sarimax.py, unchanged):
  country, one at a time : X_KR, X_CN, X_JP, X_MN
  3 regions (W = KR_west_coast, N = CN_near < 600 km, F = CN_far >= 600 km):
    one at a time  : X_W, X_N, X_F
    stacking       : X_W -> X_WN -> X_WNF
    leave one out  : X_WNF minus one = X_NF, X_WF, X_WN
Unique conditions: 11. Fit on 2021-22 only, parameters fixed, Kalman filter through 2023-24.

Outputs (one file per condition so the run can resume):
  predictions/A_SARIMAX__<cond>.parquet (+ .meta.json)
  results/sarimax_ablation/<cond>.json  (chosen L, 2023 RMSE for every L, coefficients with 95% CI)
Usage: python a_sarimax_ablation.py [--only X_CN X_W ...]
"""
import argparse
import json

import numpy as np
import pandas as pd

import predictions_io
from a_sarimax import INDEX, TRAIN, at_grid, fit, fourier, kalman_forecasts, val_rmse
from config import HORIZONS, RESULTS, WORK
from data_access import seoul_pm25
from grid import load_grid

CONDITIONS = {
    "X_KR": ["KR_all"], "X_CN": ["CN_all"], "X_JP": ["JP_balanced"], "X_MN": ["MN"],
    "X_W": ["KR_west_coast"], "X_N": ["CN_near"], "X_F": ["CN_far"],
    "X_WN": ["KR_west_coast", "CN_near"], "X_WF": ["KR_west_coast", "CN_far"], "X_NF": ["CN_near", "CN_far"],
    "X_WNF": ["KR_west_coast", "CN_near", "CN_far"],
}
OUT = RESULTS / "sarimax_ablation"


def fit_from(z, X, order, start):
    """Same model and MLE as team fit(); the optimizer only starts from the SARIMA estimates (exog coefs 0)."""
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    mod = SARIMAX(z[TRAIN], exog=X[TRAIN], order=order)
    sp = np.array([start.get(n, 0.0) for n in mod.param_names])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(start_params=sp, disp=False, maxiter=1000)


def lagged(region, groups, L):
    """Same rule as team lagged_exog: log1p mean, past-only ffill <= 24 h, shift L, then 2021-22 median."""
    out = {}
    for g in groups:
        x = np.log1p(region[f"{g}_mean"]).ffill(limit=24).shift(L).reindex(INDEX)
        out[f"{g}_L{L}"] = x.fillna(x[TRAIN].median())
    return pd.DataFrame(out, index=INDEX)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sel = json.loads((RESULTS / "a_sarima_selection.json").read_text(encoding="utf-8"))
    order = tuple(sel["order"])
    grid = load_grid()
    z = np.log1p(seoul_pm25().reindex(INDEX))
    F = fourier(INDEX)
    region = pd.read_parquet(WORK / "region_hourly.parquet").set_index("time_kst")
    for cond, groups in CONDITIONS.items():
        if a.only and cond not in a.only:
            continue
        if (OUT / f"{cond}.json").exists():
            print("skip (done)", cond, flush=True)
            continue
        preds, info, coefs = {}, {}, []
        for h in HORIZONS:
            cands = {}
            for L in (h, h + 3, h + 6):
                X = pd.concat([F, lagged(region, groups, L)], axis=1)
                res = fit_from(z, X, order, sel["params"])
                p = at_grid(kalman_forecasts(res, z, X, (h,)), grid, h)
                cands[L] = (val_rmse(p, grid, h), p, res)
            L = min(cands, key=lambda k: cands[k][0])
            res = cands[L][2]
            preds[h] = cands[L][1]
            info[h] = {"L": L, "validation_rmse_by_L": {k: v[0] for k, v in cands.items()},
                       "converged": bool(res.mle_retvals.get("converged")), "aic": float(res.aic)}
            ci = res.conf_int()
            for name in res.params.index:
                coefs.append({"condition": cond, "horizon": h, "L": L, "term": name, "coef": float(res.params[name]),
                              "ci95_low": float(ci.loc[name, 0]), "ci95_high": float(ci.loc[name, 1]),
                              "p": float(res.pvalues[name])})
            print(cond, h, info[h], flush=True)
        predictions_io.wide_to_long(grid.origin_time_kst, grid.origin_split, preds, f"A_SARIMAX__{cond}",
                                    {"order": order, "condition": cond, "exog_groups": groups, "seed": None,
                                     "lag_by_horizon": {h: v["L"] for h, v in info.items()}})
        (OUT / f"{cond}.json").write_text(json.dumps({"condition": cond, "groups": groups, "order": order,
                                                      "by_horizon": info, "coefficients": coefs}, indent=1),
                                          encoding="utf-8")
        print("done", cond, flush=True)
