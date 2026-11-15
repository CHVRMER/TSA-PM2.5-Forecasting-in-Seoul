"""Plan A on the Regional12 conditions (A-adapted) for 3/6/12/24/36/48 h, plus SARIMA at 36/48 h.

Same SARIMA(2,1,2) + daily/annual Fourier and the same team SARIMAX rules as before
(one model per horizon, exog = log1p regional mean at t+h-L, L in {h, h+3, h+6} chosen by 2023 RMSE,
past-only ffill <= 24 h then 2021-22 median; fit on 2021-22 only, Kalman filter through 2023-24).

A-adapted Regional12 conditions (the shared files define station lists; SARIMAX gets regional MEANS):
  DN      : DOM (mean of the 514 domestic series)                       -> 1 exog
  F       : DOM + FOREIGN (mean of all 120 foreign series)              -> 2 exog
  ADD_<R> : DOM + mean of foreign region R (10 regions)                  -> 2 exog
Outputs: predictions/A_SARIMAX__R12_<cond>.parquet (3-24 h, team format, team scorer)
         predictions_ext/<model>.parquet (36/48 h, same columns, supplementary scorer only)
         results/r12/<cond>.json
Usage: python a_r12.py sarima | python a_r12.py fit [--only DN F ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import predictions_io
from a_sarimax import INDEX, kalman_forecasts, fourier
from a_sarimax_ablation import fit_from, lagged
from config import PRED, RESULTS, WORK
from data_access import seoul_pm25
from grid_ext import build

H6 = (3, 6, 12, 24, 36, 48)
FOREIGN = ["CN_SHANDONG", "CN_BTH", "CN_LIAONING", "CN_JILIN", "CN_HEILONGJIANG", "CN_INNER_MONGOLIA", "CN_OTHER",
           "JP_WEST", "JP_EAST", "MN"]
CONDITIONS = {"DN": ["DOM"], "F": ["DOM", "FOREIGN"], **{f"ADD_{r}": ["DOM", r] for r in FOREIGN}}
OUT = RESULTS / "r12"
EXT = PRED.parent / "predictions_ext"


def save_ext(origins, splits, preds, model, meta):
    """36/48 h predictions: same columns as the team format, kept apart from the team scorer inputs."""
    EXT.mkdir(exist_ok=True)
    df = pd.concat([pd.DataFrame({"origin_time_kst": origins, "horizon": h, "model": model, "split": splits,
                                  "y_hat": v}) for h, v in preds.items()], ignore_index=True)
    df.to_parquet(EXT / f"{model}.parquet", index=False)
    (EXT / f"{model}.meta.json").write_text(json.dumps({"model": model, **meta}, indent=1, default=str), encoding="utf-8")


def setup():
    grid = build(H6)
    z = np.log1p(seoul_pm25().reindex(INDEX))
    F = fourier(INDEX)
    sel = json.loads((RESULTS / "a_sarima_selection.json").read_text(encoding="utf-8"))
    return grid, z, F, sel


def at(fc_h, origins):
    return pd.Series(fc_h, index=INDEX).reindex(origins).to_numpy()


def val_rmse(p, g):
    m = (g.split == "validation") & g.eligible
    return float(np.sqrt(np.mean((p[m.to_numpy()] - g.loc[m, "y_true"].to_numpy()) ** 2)))


def run_sarima(grid, z, F, sel):
    order = tuple(sel["order"])
    res = fit_from(z, F, order, sel["params"])
    fc = kalman_forecasts(res, z, F, H6)
    g0 = grid[grid.horizon == 3]
    origins, splits = g0.origin_time_kst.to_numpy(), g0.split.to_numpy()
    old = pd.read_parquet(PRED / "A_SARIMA.parquet")
    diffs = {}
    for h in (3, 6, 12, 24):
        o = old[old.horizon == h].set_index("origin_time_kst").y_hat.reindex(pd.DatetimeIndex(origins))
        diffs[h] = float(np.nanmax(np.abs(o.to_numpy() - at(fc[h], origins))))
    save_ext(origins, splits, {h: at(fc[h], origins) for h in (36, 48)}, "A_SARIMA",
             {"order": order, "note": "same SARIMA as predictions/A_SARIMA.parquet, horizons 36/48"})
    vr = {h: val_rmse(at(fc[h], origins), grid[grid.horizon == h].reset_index(drop=True)) for h in (36, 48)}
    print(json.dumps({"max_abs_diff_vs_existing_3_24h": diffs, "validation_rmse_36_48": vr,
                      "loglik_refit": float(res.llf)}))


def run_fit(grid, z, F, sel, only):
    OUT.mkdir(parents=True, exist_ok=True)
    order = tuple(sel["order"])
    region = pd.read_parquet(WORK / "r12_hourly.parquet").set_index("time_kst")
    g0 = grid[grid.horizon == 3]
    origins, splits = g0.origin_time_kst.to_numpy(), g0.split.to_numpy()
    for cond, groups in CONDITIONS.items():
        if only and cond not in only:
            continue
        if (OUT / f"{cond}.json").exists():
            print("skip (done)", cond, flush=True)
            continue
        preds, info, coefs = {}, {}, []
        for h in H6:
            gh = grid[grid.horizon == h].reset_index(drop=True)
            cands = {}
            for L in (h, h + 3, h + 6):
                X = pd.concat([F, lagged(region, groups, L)], axis=1)
                res = fit_from(z, X, order, sel["params"])
                p = at(kalman_forecasts(res, z, X, (h,))[h], origins)
                cands[L] = (val_rmse(p, gh), p, res)
            L = min(cands, key=lambda k: cands[k][0])
            res = cands[L][2]
            preds[h] = cands[L][1]
            info[h] = {"L": L, "validation_rmse_by_L": {k: v[0] for k, v in cands.items()},
                       "converged": bool(res.mle_retvals.get("converged")), "aic": float(res.aic)}
            ci = res.conf_int()
            for name in res.params.index:
                if name.startswith(tuple(groups)):
                    coefs.append({"condition": cond, "horizon": h, "L": L, "term": name, "coef": float(res.params[name]),
                                  "ci95_low": float(ci.loc[name, 0]), "ci95_high": float(ci.loc[name, 1]),
                                  "p": float(res.pvalues[name])})
            print(cond, h, {"L": L, "converged": info[h]["converged"]}, flush=True)
        model = f"A_SARIMAX__R12_{cond}"
        meta = {"order": order, "condition": cond, "exog_groups": groups, "seed": None,
                "regional_config": "Regional12_Seed11_20261109 (A-adapted: regional means)",
                "lag_by_horizon": {h: v["L"] for h, v in info.items()}}
        predictions_io.wide_to_long(origins, splits, {h: preds[h] for h in (3, 6, 12, 24)}, model, meta)
        save_ext(origins, splits, {h: preds[h] for h in (36, 48)}, model, meta)
        (OUT / f"{cond}.json").write_text(json.dumps({"condition": cond, "groups": groups, "order": order,
                                                     "by_horizon": info, "coefficients": coefs}, indent=1),
                                         encoding="utf-8")
        print("done", cond, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sarima", "fit"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    grid, z, F, sel = setup()
    run_sarima(grid, z, F, sel) if a.step == "sarima" else run_fit(grid, z, F, sel, a.only)
