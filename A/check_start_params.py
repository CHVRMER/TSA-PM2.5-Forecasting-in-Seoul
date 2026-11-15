"""Check: SARIMAX fitted from the default start (team fit) vs from SARIMA estimates (fit_from) reach the same optimum.
One condition (X_WNF, h = 6, L = 6). Compares log-likelihood, AIC, every coefficient and the 2023 RMSE.
Usage: python check_start_params.py
"""
import json
import time

import numpy as np
import pandas as pd

from a_sarimax import INDEX, at_grid, fit, fourier, kalman_forecasts, val_rmse
from a_sarimax_ablation import CONDITIONS, fit_from, lagged
from config import RESULTS, WORK
from data_access import seoul_pm25
from grid import load_grid

sel = json.loads((RESULTS / "a_sarima_selection.json").read_text(encoding="utf-8"))
order = tuple(sel["order"])
grid = load_grid()
z = np.log1p(seoul_pm25().reindex(INDEX))
region = pd.read_parquet(WORK / "region_hourly.parquet").set_index("time_kst")
X = pd.concat([fourier(INDEX), lagged(region, CONDITIONS["X_WNF"], 6)], axis=1)
out = {}
for name, f in (("team_default_start", lambda: fit(z, X, order)), ("sarima_start", lambda: fit_from(z, X, order, sel["params"]))):
    t0 = time.time()
    r = f()
    p = at_grid(kalman_forecasts(r, z, X, (6,)), grid, 6)
    out[name] = {"seconds": round(time.time() - t0, 1), "llf": r.llf, "aic": r.aic, "iterations": r.mle_retvals.get("iterations"),
                 "converged": r.mle_retvals.get("converged"), "val_rmse_h6": val_rmse(p, grid, 6), "params": r.params}
a, b = out["team_default_start"], out["sarima_start"]
print(pd.DataFrame({"team_default_start": a["params"], "sarima_start": b["params"]}).assign(diff=lambda d: d.iloc[:, 1] - d.iloc[:, 0]).to_string())
for k in ("seconds", "iterations", "converged", "llf", "aic", "val_rmse_h6"):
    print(f"{k:12s} {a[k]!s:>22} {b[k]!s:>22}")
res = {k: {kk: (vv if kk != "params" else {n: float(x) for n, x in vv.items()}) for kk, vv in v.items()} for k, v in out.items()}
(RESULTS / "check_start_params.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
