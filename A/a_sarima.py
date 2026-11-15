"""A step 1: SARIMA (Seoul only). Order by 2021-22 AIC, then Kalman h-step forecasts at every origin.

Reuses the team functions in a_sarimax.py unchanged (fourier, fit, kalman_forecasts, at_grid).
Writes predictions/A_SARIMA.parquet and results/a_sarima_selection.json.
Usage: python -I a_sarima.py
"""
import json

import numpy as np

import predictions_io
from a_sarimax import INDEX, ORDERS, at_grid, fit, fourier, kalman_forecasts, val_rmse
from config import HORIZONS, RESULTS
from data_access import seoul_pm25
from grid import load_grid

if __name__ == "__main__":
    import sys
    done = RESULTS / "a_sarima_selection.json"
    if done.exists() and "--force" not in sys.argv:
        print("skip (done):", json.loads(done.read_text(encoding="utf-8"))["by_order"])
        sys.exit(0)
    grid = load_grid()
    z = np.log1p(seoul_pm25().reindex(INDEX))
    F = fourier(INDEX)
    fits, log = {}, {}
    for o in ORDERS:
        fits[o] = fit(z, F, o)
        log[str(o)] = {"aic": float(fits[o].aic), "converged": bool(fits[o].mle_retvals.get("converged"))}
        print(o, log[str(o)], flush=True)
    order = min(fits, key=lambda o: fits[o].aic)
    res = fits[order]
    fc = kalman_forecasts(res, z, F, HORIZONS)
    preds = {h: at_grid(fc, grid, h) for h in HORIZONS}
    predictions_io.wide_to_long(grid.origin_time_kst, grid.origin_split, preds, "A_SARIMA",
                                {"order": order, "exog": "daily+annual Fourier (2 harmonics)", "seed": None})
    out = {"order": order, "by_order": log, "validation_rmse": {h: val_rmse(preds[h], grid, h) for h in HORIZONS},
           "params": {k: float(v) for k, v in res.params.items()}}
    (RESULTS / "a_sarima_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    res.save(str(RESULTS / "a_sarima_fit.pkl"), remove_data=True)
    print(json.dumps({"order": order, "validation_rmse": out["validation_rmse"]}))
