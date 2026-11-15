"""Evaluation grid for any horizon, built with the TEAM rules and the team's own baseline function.

Team rules reproduced (TSA/Model/Common/grid.py, unchanged; loaded as a module, not edited):
  origins   = the 11,688 three-hourly KST origins of the team evaluation grid
  split     = 2021-22 train / 2023 validation / 2024 test (by origin year)
  eligible  = target observed (finite, >= 0, flag 0) AND target in the same split AND inside 2021-2024
  persistence = team baseline_at_times(target_hourly, origin)          (last valid observation at or before t)
  seasonal    = team baseline_at_times(target_hourly, request), request = target - 24*ceil(h/24) hours
                -> for h <= 24 this is exactly the team rule (target - 24 h);
                -> for h = 36 / 48 the team rule (target - 24 h = t+12 / t+24) would be a FUTURE value,
                   so the latest same-clock-hour value at or before t is used (t-12 h / t). At h = 48 this equals t.
Checked against the team grid for 3/6/12/24 h by check_grid_ext() (must be identical).
"""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

from config import DATASET, ROOT


def team_modules():
    """Load the team grid.py and score.py without touching them (Plan A has its own grid.py)."""
    common = ROOT / "Model" / "Common"
    mods = {}
    for name in ("grid", "score"):
        spec = importlib.util.spec_from_file_location(f"team_{name}", common / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        if name == "score":
            sys.modules["grid"] = mods["grid"]          # score.py does `from grid import ...`
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["grid"], mods["score"]


def target_hourly():
    """Junggu 111121 hourly series exactly as the team scorer receives it (Korea_Air main + buffer, flag kept)."""
    parts = [pd.read_parquet(DATASET / "Korea_Air" / f, columns=["time_kst", "station_id", "pm25", "pm25_flag"],
                             filters=[("station_id", "==", "111121")])
             for f in ("korea_34stations_hourly_2021_2024_kst.parquet", "korea_context_buffer.parquet")]
    t = pd.concat(parts).drop(columns="station_id").drop_duplicates("time_kst").sort_values("time_kst")
    return t.reset_index(drop=True)


def build(horizons):
    tg, _ = team_modules()
    team = pd.read_parquet(DATASET / "Evaluation_Contract_20261107" / "evaluation_grid.parquet",
                           columns=["origin_time_kst", "split"]).drop_duplicates("origin_time_kst")
    origins = tg.kst(team.origin_time_kst.sort_values())
    th = target_hourly()
    valid = th[(th.pm25 >= 0) & th.pm25_flag.eq(0)].set_index(tg.kst(th[(th.pm25 >= 0) & th.pm25_flag.eq(0)].time_kst))
    rows = []
    for h in horizons:
        g = pd.DataFrame({"origin_time_kst": origins, "horizon": h})
        g["split"] = tg.split_for_time(g.origin_time_kst)
        g["target_time_kst"] = g.origin_time_kst + pd.Timedelta(hours=h)
        y = valid.pm25.reindex(g.target_time_kst).to_numpy(float)
        g["y_true"] = y
        in_study = (g.target_time_kst >= pd.Timestamp("2021-01-01", tz="Asia/Seoul")) & \
                   (g.target_time_kst < pd.Timestamp("2025-01-01", tz="Asia/Seoul"))
        same = tg.split_for_time(g.target_time_kst) == g.split.to_numpy()
        g["eligible"] = np.isfinite(y) & in_study.to_numpy() & same
        p = tg.baseline_at_times(th, g.origin_time_kst)
        s_req = g.target_time_kst - pd.Timedelta(hours=24 * math.ceil(h / 24))
        s = tg.baseline_at_times(th, s_req)
        for name, b, req in (("persistence", p, g.origin_time_kst), ("seasonal", s, s_req)):
            for col in b:
                g[f"{name}_{col}"] = b[col].to_numpy()
            g[f"{name}_request_time_kst"] = req.to_numpy()
        assert (g.seasonal_request_time_kst <= g.origin_time_kst).all(), "seasonal baseline would use the future"
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def check_grid_ext():
    """Our grid for 3/6/12/24 h must equal the team evaluation grid row for row."""
    team = pd.read_parquet(DATASET / "Evaluation_Contract_20261107" / "evaluation_grid.parquet")
    ours = build((3, 6, 12, 24))
    k = ["origin_time_kst", "horizon"]
    m = team.merge(ours, on=k, suffixes=("_t", "_o"))
    out = {"rows_team": len(team), "rows_ours": len(ours), "rows_matched": len(m),
           "eligible_equal": bool((m.eligible_t == m.eligible_o).all()),
           "split_equal": bool((m.split_t == m.split_o).all())}
    e = m.eligible_t
    out["y_true_equal_on_eligible"] = bool(np.allclose(m.loc[e, "y_true_t"], m.loc[e, "y_true_o"]))
    for b in ("persistence_y_hat", "seasonal_y_hat"):
        out[b + "_equal"] = bool(np.allclose(m[b + "_t"], m[b + "_o"]))
    return out


if __name__ == "__main__":
    print(check_grid_ext())
    g = build((36, 48))
    print(g.groupby(["split", "horizon"]).eligible.sum().unstack())
