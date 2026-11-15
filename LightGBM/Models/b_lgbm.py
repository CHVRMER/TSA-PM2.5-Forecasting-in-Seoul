"""B: LightGBM ladder on Origin_Features_v3, one model per horizon, target log1p PM2.5.

Models (inputs are feature-set names from feature_sets.json; missing values left to LightGBM):
  B_M1   local_plus_station_history_kr                          Domestic
  B_M2   B_M1 + team China stations within 600 km of Seoul      + near China
  B_M3   local_plus_station_history_kr_cn                       + all China incl. border proxies
  B_M4   local_plus_station_history_all_balanced                + 24 balanced Japan stations + Ulaanbaatar
  B_M3T  B_M3 + trajectory features (kr/cn/all path scopes)     + air-mass path
  B_M4T  local_plus_trajectory_and_station_history_balanced...  everything (balanced)
Settings: a 4-point grid (num_leaves x feature_fraction) chosen by 2023 RMSE with early stopping on 2023;
the chosen setting is refit with 3 seeds at the seed-0 best iteration and averaged. Grouped permutation
importance (validation) is written for every model. Test predictions are written but scored only by
score.py --final.

Usage: python -I b_lgbm.py [--models B_M1 B_M3] [--horizons 6] [--quick]
"""
import argparse
import csv
import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import predictions_io
from config import FEATURES, HORIZONS, RESULTS, SEEDS
from data_access import registry
from region_hourly import macro_groups

BASE = dict(objective="l2", learning_rate=0.03, min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbose=-1, num_threads=0)
GRID = [dict(num_leaves=nl, feature_fraction=ff) for nl in (15, 31) for ff in (0.3, 0.5)]
OBS = re.compile(r"^obs_([a-z]{2})_(.+)_pm25_")   # station ids may contain '_' (e.g. city_yanbian)


def station_key(col):
    m = OBS.match(col)
    return f"{m.group(1).upper()}:{m.group(2)}" if m else None


def feature_lists():
    sets = json.loads((FEATURES / "feature_sets.json").read_text(encoding="utf-8"))["sets"]
    near = set(macro_groups(registry())["CN_near"])
    near_cols = [c for c in sets["station_history_cn"] if station_key(c) in near]
    union = lambda *parts: list(dict.fromkeys(c for p in parts for c in p))
    return {"B_M1": sets["local_plus_station_history_kr"],
            "B_M2": union(sets["local_plus_station_history_kr"], near_cols),
            "B_M3": sets["local_plus_station_history_kr_cn"],
            "B_M4": sets["local_plus_station_history_all_balanced"],
            "B_M3T": union(sets["local_plus_trajectory_kr_cn_scopes_without_optional_gases"],
                           sets["station_history_kr"], sets["station_history_cn"]),
            "B_M4T": sets["local_plus_trajectory_and_station_history_balanced_without_optional_gases"]}


def check_features(names):
    with open(FEATURES / "feature_dictionary.csv", encoding="utf-8-sig", newline="") as f:
        role = {r["name"]: r["role"] for r in csv.DictReader(f)}
    bad = [n for n in names if role.get(n) != "feature"]
    assert not bad and len(names) == len(set(names)), f"unsafe input columns: {bad[:5]}"


def group_of(col, region_of):
    key = station_key(col)
    if key:
        return f"{key[:2]}:{region_of.get(key, 'unknown')}"
    for scope in ("all", "kr", "cn", "jp", "mn"):
        if col.startswith((f"pm25_{scope}_", f"pm10_{scope}_", f"geo_{scope}_")):
            return f"path_{scope}"
    if col.startswith(("met_", "route_")):
        return "path_weather_route"
    return "local"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def grouped_importance(booster, X, y, groups, repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rmse(np.expm1(booster.predict(X)), y)
    out = {}
    for g in sorted(set(groups)):
        cols = [i for i, x in enumerate(groups) if x == g]
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            deltas.append(rmse(np.expm1(booster.predict(Xp)), y) - base)
        out[g] = (float(np.mean(deltas)), len(cols))
    return out


if __name__ == "__main__":
    lists = feature_lists()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(lists))
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--quick", action="store_true", help="one setting, one seed, 300 rounds (code check only)")
    a = ap.parse_args()
    reg = registry()
    region_of = dict(zip(reg.station_key, reg.region_group))
    need = sorted(set().union(*(lists[m] for m in a.models)))
    meta = ["origin_time_kst", "origin_split"] + [f"y_pm25_h{h}" for h in a.horizons] + [f"eligible_h{h}" for h in a.horizons]
    table = pq.read_table(FEATURES / "origin_features_2021_2024_kst.parquet", columns=need + meta).to_pandas()
    split = table.origin_split.to_numpy()
    imp_rows, sel = [], {}
    for m in a.models:
        cols = lists[m]
        check_features(cols)
        X = table[cols].to_numpy(np.float64)
        groups = [group_of(c, region_of) for c in cols]
        preds = {}
        for h in a.horizons:
            y = table[f"y_pm25_h{h}"].to_numpy()
            ok = table[f"eligible_h{h}"].to_numpy()
            tr, va = ok & (split == "train"), ok & (split == "validation")
            dtr = lgb.Dataset(X[tr], np.log1p(y[tr]), free_raw_data=False)
            dva = lgb.Dataset(X[va], np.log1p(y[va]), reference=dtr)
            grid = GRID[:1] if a.quick else GRID
            rounds = 300 if a.quick else 5000
            trials = []
            for g in grid:
                b = lgb.train({**BASE, **g, "seed": 0}, dtr, rounds, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                trials.append((rmse(np.expm1(b.predict(X[va], num_iteration=b.best_iteration)), y[va]), g, b.best_iteration, b))
            score, g, best_iter, b0 = min(trials, key=lambda t: t[0])
            sel[f"{m}_h{h}"] = {"params": g, "best_iteration": best_iter, "validation_rmse_by_setting":
                                [{"params": t[1], "rmse": t[0], "iter": t[2]} for t in trials]}
            seeds = SEEDS[:1] if a.quick else SEEDS
            boosters = [b0] + [lgb.train({**BASE, **g, "seed": s}, dtr, best_iter) for s in seeds[1:]]
            preds[h] = np.mean([np.expm1(bb.predict(X, num_iteration=best_iter if bb is b0 else None)) for bb in boosters], axis=0)
            for grp, (delta, ncol) in grouped_importance(b0, X[va], y[va], groups).items():
                imp_rows.append({"model": m, "horizon": h, "group": grp, "n_columns": ncol, "delta_rmse_validation": delta})
            print(f"{m} h{h}: setting={g} iter={best_iter}", flush=True)
        keep = np.isin(split, ["validation", "test"])
        predictions_io.wide_to_long(table.origin_time_kst[keep], table.origin_split[keep],
                                    {h: v[keep] for h, v in preds.items()}, m,
                                    {"quick": a.quick, "seeds": list(SEEDS[:1] if a.quick else SEEDS), "n_features": len(cols),
                                     "by_horizon": {h: {k: sel[f"{m}_h{h}"][k] for k in ("params", "best_iteration")} for h in preds}})
    suffix = "_".join(a.models) if len(a.models) < len(lists) else "all"
    pd.DataFrame(imp_rows).to_csv(RESULTS / f"b_group_importance_{suffix}.csv", index=False, encoding="utf-8-sig")
    (RESULTS / f"b_lgbm_selection_{suffix}.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
