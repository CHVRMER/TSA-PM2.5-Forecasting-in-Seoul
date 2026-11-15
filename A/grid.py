"""Common evaluation grid: 3-hourly origins t, targets y(t+h), eligibility and condition labels.

Everything comes from Origin_Features_v3 metadata, so all models are scored on the same rows.
"""
import numpy as np
import pandas as pd

from config import FEATURES, HORIZONS

COLD_MONTHS = (11, 12, 1, 2, 3)
SECTOR_GROUP = {"NW": "west", "W": "west", "SW": "west", "N": "north", "NE": "north",
                "E": "east", "SE": "east", "S": "east", "LOCAL": "local"}


def load_grid():
    """One row per origin: time, split, y_pm25_h*, eligible_h*, persistence input and condition labels."""
    cols = (["origin_time_kst", "origin_split", "route_history12h_sector", "seoul_pm25_now",
             "asos108_wind_u_now", "asos108_wind_v_now", "asos108_wind_calm_now",
             "met_b01_06_u_mean", "met_b01_06_v_mean"]
            + [f"y_pm25_h{h}" for h in HORIZONS] + [f"eligible_h{h}" for h in HORIZONS])
    g = pd.read_parquet(FEATURES / "origin_features_2021_2024_kst.parquet", columns=cols)
    g["season"] = np.where(g.origin_time_kst.dt.month.isin(COLD_MONTHS), "cold", "warm")
    g["route_group"] = g.route_history12h_sector.map(SECTOR_GROUP)
    g["wind_sector"] = wind_sector(g.asos108_wind_u_now, g.asos108_wind_v_now, g.asos108_wind_calm_now)
    # wind at trajectory level (~500 m AGL, about 950 hPa) over the last 6 h of the arriving path; the data have no
    # 850 hPa field, so this is the closest available upper-air wind. Below 0.5 m/s counts as calm.
    u, v = g.met_b01_06_u_mean, g.met_b01_06_v_mean
    g["traj_wind_sector"] = wind_sector(u, v, (np.hypot(u, v) < 0.5).astype(int))
    return g


def wind_sector(u, v, calm):
    """Direction the wind blows FROM (4 sectors) from ASOS u/v at t; 'calm' or 'unknown' otherwise."""
    frm = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    sec = np.select([(frm >= 225) & (frm < 315), (frm >= 315) | (frm < 45), (frm >= 45) & (frm < 135)],
                    ["west", "north", "east"], "south")
    out = pd.Series(sec, index=u.index).where(u.notna() & v.notna(), "unknown")
    return out.where(calm.ne(1), "calm")


def long_targets(g, split=None):
    """(origin, horizon) rows with y and eligibility; optionally one split."""
    parts = []
    for h in HORIZONS:
        p = g[["origin_time_kst", "origin_split", "season", "route_group", "wind_sector", "traj_wind_sector"]].copy()
        p["horizon"], p["y"], p["eligible"] = h, g[f"y_pm25_h{h}"], g[f"eligible_h{h}"]
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out = out[out.eligible]
    return out if split is None else out[out.origin_split == split]
