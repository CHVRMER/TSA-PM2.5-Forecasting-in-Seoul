"""Common scorer for every model.

Per horizon it keeps only rows where ALL compared models (and persistence) have a prediction, then reports
RMSE, MAE, R2, MBE (bias, pred - obs), skill vs persistence, Diebold-Mariano tests (squared error, Bartlett HAC with lag h/3 on the
3-hourly grid), moving-block bootstrap 95% CI of the RMSE difference (7-day = 56-row blocks), Holm-adjusted
p-values, and condition breakdowns (season, route sector, surface wind sector, trajectory-level wind sector,
high PM) with counts.

The 2024 test split is refused unless --final is given, so it is opened once, after settings are frozen.

Usage:
  python -I score.py --split validation --models persistence B_M1 B_M3 --pairs B_M3:B_M1
  python -I score.py --split test --final --models ...
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import predictions_io
from config import HIGH_PM, RESULTS
from grid import load_grid, long_targets


def dm_test(e1, e2, h):
    """DM on squared errors; stat < 0 means model 1 is more accurate. Lag = h/3 on the 3-hourly grid."""
    d = e1 ** 2 - e2 ** 2
    n, lag = len(d), max(1, h // 3)
    dc = d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    if lrv <= 0:  # identical forecasts (e.g. seasonal naive = persistence at 24 h): no difference to test
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * norm.sf(abs(stat)))


def block_bootstrap_rmse_diff(e1, e2, block=56, reps=1000, seed=0):
    """95% CI of RMSE(model1) - RMSE(model2) with a moving-block bootstrap in time order."""
    rng, n = np.random.default_rng(seed), len(e1)
    starts = np.arange(max(1, n - block + 1))
    k = int(np.ceil(n / block))
    diffs = np.empty(reps)
    for r in range(reps):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[r] = np.sqrt(np.mean(e1[idx] ** 2)) - np.sqrt(np.mean(e2[idx] ** 2))
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def metrics(y, yhat, ref):
    """RMSE, MAE (required); R2 = 1 - SSE/SST on the same rows, not squared correlation; MBE = mean(pred - obs),
    negative = under-prediction; skill = 1 - RMSE/RMSE_persistence (read R2 together with skill: persistence
    alone already gives a high R2 at short horizons)."""
    e = yhat - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(e)), "RMSE": rmse, "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan"), "MBE": float(np.mean(e)),
            "skill_vs_persistence": float(1 - rmse / np.sqrt(np.mean((ref - y) ** 2)))}


def evaluate(split, models, pairs=(), final=False):
    if split == "test" and not final:
        raise SystemExit("Test split is locked: freeze settings on validation, then rerun with --final once.")
    models = list(dict.fromkeys(["persistence", *models]))
    t = long_targets(load_grid(), split)
    p = predictions_io.load(models)
    p = p[p.split == split]
    clipped = p[p.y_hat < 0].groupby("model").size().to_dict()   # one rule for all models: negative -> 0
    p = p.assign(y_hat=p.y_hat.clip(lower=0))
    p = p.pivot_table(index=["origin_time_kst", "horizon"], columns="model", values="y_hat")
    df = t.set_index(["origin_time_kst", "horizon"]).join(p, how="inner").dropna(subset=models)
    df = df.sort_index()
    rows_out = df.reset_index()
    rows_out["target_time_kst"] = rows_out.origin_time_kst + pd.to_timedelta(rows_out.horizon, unit="h")
    rows, tests, cond = [], [], []
    for h, g in df.groupby(level="horizon"):
        y, ref = g.y.to_numpy(), g.persistence.to_numpy()
        for m in models:
            rows.append({"horizon": h, "model": m, **metrics(y, g[m].to_numpy(), ref)})
        for a, b in [(m, "persistence") for m in models if m != "persistence"] + [tuple(x.split(":")) for x in pairs]:
            e1, e2 = g[a].to_numpy() - y, g[b].to_numpy() - y
            stat, pv = dm_test(e1, e2, h)
            lo, hi = block_bootstrap_rmse_diff(e1, e2)
            tests.append({"horizon": h, "model": a, "versus": b, "rmse_diff": float(np.sqrt(np.mean(e1 ** 2)) - np.sqrt(np.mean(e2 ** 2))),
                          "ci95_low": lo, "ci95_high": hi, "dm_stat": stat, "p": pv})
        for col in ("season", "route_group", "wind_sector", "traj_wind_sector"):
            for level, gg in g.groupby(col):
                for m in models:
                    cond.append({"horizon": h, "condition": col, "level": level, "model": m,
                                 **metrics(gg.y.to_numpy(), gg[m].to_numpy(), gg.persistence.to_numpy())})
        hi_mask = g.y >= HIGH_PM
        for m in models:
            if hi_mask.sum():
                cond.append({"horizon": h, "condition": "y_ge_35", "level": "high", "model": m,
                             **metrics(g.y[hi_mask].to_numpy(), g[m][hi_mask].to_numpy(), g.persistence[hi_mask].to_numpy())})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = holm(tests.p)
        tests["improved"] = (tests.ci95_high < 0) & (tests.p_holm < 0.05)
    out = {"metrics": pd.DataFrame(rows), "tests": tests, "conditions": pd.DataFrame(cond)}
    out["metrics"]["negative_predictions_clipped"] = out["metrics"].model.map(clipped).fillna(0).astype(int)
    tag = f"{split}_{'_'.join(models[1:]) or 'persistence'}"[:120]
    for k, v in out.items():
        v.to_csv(RESULTS / f"score_{k}_{tag}.csv", index=False, encoding="utf-8-sig")
    # per-row record: origin, target time, horizon, truth, every model's prediction, condition labels
    rows_out.to_parquet(RESULTS / f"score_rows_{tag}.parquet", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--pairs", nargs="*", default=[], help="extra comparisons model:reference, e.g. B_M3:B_M1")
    ap.add_argument("--final", action="store_true", help="required to open the 2024 test split")
    a = ap.parse_args()
    res = evaluate(a.split, a.models, a.pairs, a.final)
    print(res["metrics"].to_string(index=False))
    if len(res["tests"]):
        print(res["tests"].round(4).to_string(index=False))
