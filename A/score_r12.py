"""Scoring for the Regional12 A-adapted runs (2023 validation only).

team : 3/6/12/24 h with the TEAM scorer (TSA/Model/Common/score.py, unchanged, run as a program).
ext  : any horizons with a SUPPLEMENTARY scorer that calls the team scorer's own functions
       (metric_values, calendar_errors, dm_calendar, bootstrap_calendar, holm) on the grid from grid_ext.py.
       It is NOT the team scorer: the team grid.py rejects horizons other than 3/6/12/24.
check: runs ext on 3/6/12/24 h and compares every metric and test with the team scorer output.

Declared pairs (fixed before results, = Regional12 comparisons.json): F vs DN, ADD_<R> vs DN (11),
plus every run vs persistence (the team scorer adds these automatically).
Usage: python score_r12.py team | ext | check
"""
import subprocess
import sys

import numpy as np
import pandas as pd

from a_r12 import CONDITIONS, EXT
from config import PRED, RESULTS, ROOT
from grid_ext import build, team_modules

MODELS = ["A_SARIMA"] + [f"A_SARIMAX__R12_{c}" for c in CONDITIONS]
PAIRS = [(f"A_SARIMAX__R12_{c}", "A_SARIMAX__R12_DN") for c in CONDITIONS if c != "DN"]
OUT_TEAM = RESULTS / "score_r12_team"
OUT_EXT = RESULTS / "score_r12_ext"


def run_team():
    cmd = [sys.executable, str(ROOT / "Model" / "Common" / "score.py"),
           "--grid", str(ROOT / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet"),
           "--predictions", *[str(PRED / f"{m}.parquet") for m in MODELS],
           "--split", "validation", "--pairs", *[f"{a}:{b}" for a, b in PAIRS], "--output-dir", str(OUT_TEAM)]
    subprocess.run(cmd, check=True, cwd=str(ROOT / "Model" / "Common"), stdout=subprocess.DEVNULL)
    print("team scorer done:", OUT_TEAM)


def run_ext(horizons, out):
    _, ts = team_modules()
    g = build(horizons)
    g = g[g.split == "validation"].reset_index(drop=True)
    preds = {}
    for m in MODELS:
        parts = []
        if any(h <= 24 for h in horizons):
            parts.append(pd.read_parquet(PRED / f"{m}.parquet"))
        if any(h > 24 for h in horizons):
            parts.append(pd.read_parquet(EXT / f"{m}.parquet"))
        p = pd.concat(parts)
        p["origin_time_kst"] = pd.to_datetime(p.origin_time_kst).dt.tz_convert("Asia/Seoul")
        preds[m] = p.set_index(["origin_time_kst", "horizon"]).y_hat
    metric_rows, test_rows = [], []
    for h in horizons:
        gh = g[g.horizon == h]
        all_times = pd.DatetimeIndex(gh.origin_time_kst).sort_values()
        act = gh[gh.eligible]
        y = act.y_true.to_numpy(float)
        ref = act.persistence_y_hat.to_numpy(float)
        key = pd.MultiIndex.from_arrays([act.origin_time_kst, act.horizon])
        runs = {"persistence": ref, "seasonal_naive": act.seasonal_y_hat.to_numpy(float)}
        raw = {}
        for m in MODELS:
            v = preds[m].reindex(key).to_numpy(float)
            assert np.isfinite(v).all(), (m, h, "missing prediction on an eligible row")
            raw[m] = v
            runs[m] = np.maximum(v, 0)
        for name, v in runs.items():
            metric_rows.append({"horizon": h, "model": name, **ts.metric_values(y, v, ref),
                                "negative_predictions_clipped": int((raw.get(name, v) < 0).sum()),
                                "seasonal_rule": "y(t+h-24) (team)" if h <= 24 else "latest same-hour <= t (y(t+h-48))"})
        pairs = [(n, "persistence") for n in runs if n != "persistence"] + PAIRS
        for a, b in pairs:
            e1, e2 = runs[a] - y, runs[b] - y
            ca, cb = ts.calendar_errors(act.origin_time_kst, e1, e2, all_times)
            stat = ts.dm_calendar(ca, cb, int(h))
            boot = ts.bootstrap_calendar(ca, cb, reps=1000, seed=0)
            test_rows.append({"horizon": h, "model": a, "versus": b, "n": len(y),
                              "rmse_diff": float(np.sqrt(np.mean(e1 ** 2)) - np.sqrt(np.mean(e2 ** 2))),
                              "dm_stat": stat["dm_stat"], "p": stat["p"], "hac_lag_slots": stat["hac_lag_slots"],
                              "ci95_low": boot["ci95_low"], "ci95_high": boot["ci95_high"]})
    tests = pd.DataFrame(test_rows)
    tests["p_holm"] = ts.holm(tests.p)
    tests["improved"] = (tests.ci95_high < 0) & (tests.p_holm < 0.05)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out / "tests.csv", index=False, encoding="utf-8-sig")
    return metrics, tests


def check():
    m_ext, t_ext = run_ext((3, 6, 12, 24), RESULTS / "score_r12_check")
    m_t = pd.read_csv(OUT_TEAM / "metrics.csv")
    m_t = m_t[m_t.prediction_kind == "single"]
    mm = m_t.merge(m_ext, on=["horizon", "model"], suffixes=("_t", "_e"))
    t_t = pd.read_csv(OUT_TEAM / "tests.csv")
    t_t["model"] = t_t.run_id.str.split("|").str[0]
    t_t["versus"] = t_t.versus_run_id.str.split("|").str[0]
    tt = t_t.merge(t_ext, on=["horizon", "model", "versus"], suffixes=("_t", "_e"))
    res = {"metric_rows": (len(m_t), len(m_ext), len(mm)), "test_rows": (len(t_t), len(t_ext), len(tt))}
    for c in ("RMSE", "MAE", "MBE", "R2", "RMSE_improvement_pct"):
        res[f"max|{c} diff|"] = float((mm[f"{c}_t"] - mm[f"{c}_e"]).abs().max())
    for c in ("rmse_diff", "p", "ci95_low", "ci95_high", "p_holm"):
        res[f"max|{c} diff|"] = float((tt[f"{c}_t"] - tt[f"{c}_e"]).abs().max())
    res["improved_identical"] = bool((tt.improved_t == tt.improved_e).all())
    from grid_ext import check_grid_ext
    import json
    out = {"scorer": res, "grid": check_grid_ext()}
    (RESULTS / "score_r12_check" / "check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "team":
        run_team()
    elif step == "ext":
        m, t = run_ext((36, 48), OUT_EXT)
        print(m[["horizon", "model", "RMSE", "RMSE_improvement_pct"]].round(3).to_string(index=False))
    else:
        check()
