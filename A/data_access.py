"""Hourly source readers shared by the region table, node matrix and SARIMA (no imputation)."""
import numpy as np
import pandas as pd

from config import DATASET, MATCHING, TARGET_KEY

SOURCES = {  # dataset -> (country, main file, buffer file)
    "Korea_Air": ("KR", "korea_34stations_hourly_2021_2024_kst.parquet", "korea_context_buffer.parquet"),
    "Korea_East_Candidates": ("KR", "korea_east_36stations_hourly_2021_2024_kst.parquet", "korea_east_context_buffer.parquet"),
    "China": ("CN", "china_48stations_hourly_2021_2024_kst.parquet", "china_48stations_context_buffer.parquet"),
    "Japan": ("JP", "japan_west_hourly_2021_2024_kst.parquet", "japan_west_context_buffer.parquet"),
    "Mongolia": ("MN", "mongolia_ub_embassy_hourly_2021_2024_kst.parquet", "mongolia_ub_embassy_context_buffer.parquet"),
    "China_Border_Proxy": ("CN", "china_border_cities_hourly_2021_2024_kst.parquet", "china_border_cities_context_buffer.parquet"),
}
HOUR_INDEX = pd.date_range("2020-12-29 00:00", "2025-01-01 00:00", freq="h", tz="Asia/Seoul")


def registry():
    """v3 matching registry (317 external candidates) plus the Seoul target, as a DataFrame."""
    r = pd.read_csv(MATCHING / "station_registry.csv", dtype=str, encoding="utf-8-sig")
    r[["longitude", "latitude"]] = r[["longitude", "latitude"]].astype(float)
    return r


def hourly_pm25(keys):
    """Wide hourly PM2.5 table (HOUR_INDEX x station_key); flag != 0 stays NaN."""
    want = set(keys)
    cols = {}
    for ds, (cc, main, buf) in SOURCES.items():
        ids = sorted(k.split(":", 1)[1] for k in want if k.startswith(cc + ":"))
        if not ids:
            continue
        for f in (main, buf):
            t = pd.read_parquet(DATASET / ds / f, columns=["time_kst", "station_id", "pm25"],
                                filters=[("station_id", "in", ids)])
            for sid, g in t.groupby("station_id"):
                key = f"{cc}:{sid}"
                if key in want:
                    s = g.set_index("time_kst").pm25
                    cols.setdefault(key, []).append(s)
    out = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in cols.items()}).reindex(HOUR_INDEX)
    missing = want - set(out.columns)
    assert not missing, f"no source rows for {sorted(missing)[:5]}"
    return out[list(keys)]


def seoul_pm25():
    return hourly_pm25([TARGET_KEY])[TARGET_KEY]


def asos_hourly():
    """ASOS 108/102 hourly temperature, humidity, pressure, wind u/v (team rule: calm -> 0, dir 0 -> NaN)."""
    parts = []
    for f in ("asos_102_108_hourly_2021_2024_kst.parquet", "asos_context_after.parquet"):
        parts.append(pd.read_parquet(DATASET / "ASOS" / f))
    a = pd.concat(parts)
    out = {}
    for st in ("108", "102"):
        g = a[a.station_id == st].set_index("time_kst").sort_index()
        spd, dr = g.wind_speed, g.wind_direction
        good = spd.gt(0) & dr.gt(0) & dr.le(360)
        u = (-spd * np.sin(np.deg2rad(dr))).where(good)
        v = (-spd * np.cos(np.deg2rad(dr))).where(good)
        u, v = u.mask(spd.eq(0), 0.0), v.mask(spd.eq(0), 0.0)
        for name, s in (("temperature", g.temperature), ("rh", g.relative_humidity),
                        ("pressure", g.surface_pressure), ("wind_u", u), ("wind_v", v)):
            out[f"asos{st}_{name}"] = s
    return pd.DataFrame(out).reindex(HOUR_INDEX)
