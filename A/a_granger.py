"""A step 3: Granger causality tests, region/country mean -> Seoul Jung-gu PM2.5 (2021-22 training only).

Series: hourly log1p PM2.5, first-differenced (d = 1, same as SARIMA). No filling: an hour is used only when
the series and every lag it needs are observed, so lags keep the real hourly spacing.
Test: OLS  dy_t = c + sum_{k=1..p} a_k dy_{t-k} + sum_{k=1..p} b_k dx_{t-k} + e,  H0: all b_k = 0 (F test).
p in {6, 12, 24} hours. The reverse direction (Seoul -> region) is reported as a check.
statsmodels grangercausalitytests is not used directly because it would drop missing hours and shift lags.

Output: results/a_granger.csv
Usage: python a_granger.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from a_sarimax import INDEX, TRAIN
from config import RESULTS, WORK
from data_access import seoul_pm25

GROUPS = {"KR_all": "Korea (67)", "CN_all": "China (61)", "JP_balanced": "Japan (24)", "MN": "Mongolia (1)",
          "KR_west_coast": "W: KR west coast (24)", "CN_near": "N: China < 600 km (21)",
          "CN_far": "F: China >= 600 km (27)"}
LAGS = (6, 12, 24)


def granger(y, x, p):
    """F test that lags 1..p of x add to lags 1..p of y. y, x: differenced hourly series on one index."""
    cols = {"y": y}
    for k in range(1, p + 1):
        cols[f"y{k}"], cols[f"x{k}"] = y.shift(k), x.shift(k)
    d = pd.DataFrame(cols).dropna()
    ylags = [f"y{k}" for k in range(1, p + 1)]
    xlags = [f"x{k}" for k in range(1, p + 1)]
    full = sm.OLS(d.y, sm.add_constant(d[ylags + xlags])).fit()
    rest = sm.OLS(d.y, sm.add_constant(d[ylags])).fit()
    f, pval, _ = full.compare_f_test(rest)
    return {"n": len(d), "F": f, "p": pval, "sum_b": float(full.params[xlags].sum()),
            "r2_gain": full.rsquared - rest.rsquared}


if __name__ == "__main__":
    region = pd.read_parquet(WORK / "region_hourly.parquet").set_index("time_kst").reindex(INDEX)
    y = np.log1p(seoul_pm25().reindex(INDEX)).diff()[TRAIN]
    rows = []
    for g, label in GROUPS.items():
        x = np.log1p(region[f"{g}_mean"]).diff()[TRAIN]
        for p in LAGS:
            rows.append({"group": g, "label": label, "direction": "region -> Seoul", "lags_h": p, **granger(y, x, p)})
            rows.append({"group": g, "label": label, "direction": "Seoul -> region", "lags_h": p, **granger(x, y, p)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "a_granger.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.round({"F": 2, "p": 4, "sum_b": 3, "r2_gain": 4}).to_string(index=False))
