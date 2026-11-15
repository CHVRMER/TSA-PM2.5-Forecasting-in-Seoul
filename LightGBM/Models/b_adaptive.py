"""B Adaptive: mix Domestic (B_M1) and Full (default B_M3) forecasts by condition.

y_hat = (1 - w) * y_M1 + w * y_Full, with one w per condition cell and horizon, fitted on 2023 only:
  w = clip( sum((y - a)(b - a)) / sum((b - a)^2), 0, 1 )   (least squares in ug/m3)
Cells (each x season cold Nov-Mar / warm): route sector of the trajectory 12 h back (west/north/east/local);
ASOS-108 surface wind sector at t (west/north/east/south/calm); trajectory-level wind sector over the last 6 h of
the arriving path (~500 m AGL; no 850 hPa field exists in the data). Cells with < 100 validation rows fall back to the
season-only weight. Validation scores of the adaptive models are in-sample for w -- compare them on test.

Usage: python -I b_adaptive.py [--full B_M3]
"""
import argparse

import numpy as np
import pandas as pd

import predictions_io
from config import RESULTS
from grid import load_grid

MIN_ROWS = 100


def weight(y, a, b):
    den = np.sum((b - a) ** 2)
    return float(np.clip(np.sum((y - a) * (b - a)) / den, 0, 1)) if den > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domestic", default="B_M1")
    ap.add_argument("--full", default="B_M3")
    a = ap.parse_args()
    g = load_grid()
    p = predictions_io.load([a.domestic, a.full]).pivot_table(index=["origin_time_kst", "horizon"], columns="model", values="y_hat")
    tables = []
    for kind, col in (("route", "route_group"), ("wind", "wind_sector"), ("trajwind", "traj_wind_sector")):
        rows = []
        for h in sorted(p.index.get_level_values("horizon").unique()):  # horizons the inputs actually have
            d = g.set_index("origin_time_kst")[["origin_split", "season", col, f"y_pm25_h{h}", f"eligible_h{h}"]]
            d = d.join(p.xs(h, level="horizon"), how="inner").dropna(subset=[a.domestic, a.full])
            val = d[(d.origin_split == "validation") & d[f"eligible_h{h}"]]
            season_w = {s: weight(v[f"y_pm25_h{h}"], v[a.domestic], v[a.full]) for s, v in val.groupby("season")}
            cell_w = {}
            for (s, c), v in val.groupby(["season", col]):
                cell_w[(s, c)] = weight(v[f"y_pm25_h{h}"], v[a.domestic], v[a.full]) if len(v) >= MIN_ROWS else season_w[s]
                rows.append({"horizon": h, "season": s, "cell": c, "n_validation": len(v), "w_full": cell_w[(s, c)],
                             "fallback_to_season": len(v) < MIN_ROWS})
            w = np.array([cell_w.get((s, c), season_w.get(s, 0.0)) for s, c in zip(d.season, d[col])])
            yhat = (1 - w) * d[a.domestic].to_numpy() + w * d[a.full].to_numpy()
            tables.append(pd.DataFrame({"origin_time_kst": d.index, "horizon": h, "split": d.origin_split.to_numpy(),
                                        "y_hat": yhat, "kind": kind}))
        pd.DataFrame(rows).to_csv(RESULTS / f"b_adaptive_weights_{kind}_{a.full}.csv", index=False, encoding="utf-8-sig")
    allp = pd.concat(tables)
    for kind, df in allp.groupby("kind"):
        print(predictions_io.save(df.drop(columns="kind"), f"B_ADAPT_{kind}_{a.full}",
                                  {"domestic": a.domestic, "full": a.full, "cells": kind, "min_rows": MIN_ROWS,
                                   "weights_file": f"results/b_adaptive_weights_{kind}_{a.full}.csv"}))
