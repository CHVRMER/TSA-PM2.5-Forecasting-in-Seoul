"""Hourly regional PM2.5 means (simple mean of valid stations) and valid-station counts.

Groups: every v3 region_group, plus macro groups used by SARIMAX:
  KR_west_coast  Incheon/Chungnam/Jeonbuk/Jeonnam/Jeju coastal and island stations
  CN_near / CN_far   team China stations within / beyond 600 km of Seoul (border proxies excluded)
  CN_border_RU / CN_border_MN   city-average proxies for the Russian / Mongolian border
  JP_balanced    the 24 balanced Japan stations;  MN   Ulaanbaatar
No imputation: an hour with no valid station is NaN (count 0).

Usage: python -I region_hourly.py  ->  work/region_hourly.parquet
"""
import numpy as np
import pandas as pd

from config import FEATURES, SEOUL, WORK
from data_access import hourly_pm25, registry


def haversine(lon, lat):
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, *SEOUL))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def macro_groups(reg):
    reg = reg.assign(dist_km=haversine(reg.longitude, reg.latitude))
    jp_bal = set(pd.read_csv(FEATURES / "japan_balanced_stations.csv", dtype=str, encoding="utf-8-sig").station_key)
    team_cn = (reg.country == "CN") & (reg.source_dataset == "China")
    return {
        "KR_west_coast": reg.station_key[(reg.country == "KR") & reg.region_group.isin(["인천", "충남", "전북", "전남", "제주"])],
        "CN_near": reg.station_key[team_cn & (reg.dist_km < 600)],
        "CN_far": reg.station_key[team_cn & (reg.dist_km >= 600)],
        "CN_border_RU": reg.station_key[reg.region_group == "Russia_border_proxy"],
        "CN_border_MN": reg.station_key[reg.region_group == "Mongolia_border_proxy"],
        "JP_balanced": reg.station_key[reg.station_key.isin(jp_bal)],
        "MN": reg.station_key[reg.country == "MN"],
        # Plan A country-level groups (one country at a time in SARIMAX + Granger)
        "KR_all": reg.station_key[reg.country == "KR"],          # 67 domestic stations (Jung-gu excluded)
        "CN_all": reg.station_key[reg.country == "CN"],          # 48 stations + 13 border-city averages
    }


if __name__ == "__main__":
    reg = registry()
    # Plan A uses only the macro groups (team code also built per-region_group means "rg_*" for analysis,
    # which need all 188 Japan stations; the Plan A data bundle keeps only the 24 balanced ones).
    groups = macro_groups(reg)
    pm = hourly_pm25(sorted(set().union(*map(set, groups.values()))))
    out = {}
    for name, keys in groups.items():
        block = pm[list(keys)]
        out[f"{name}_mean"] = block.mean(axis=1)
        out[f"{name}_n"] = block.notna().sum(axis=1)
    df = pd.DataFrame(out)
    df.index.name = "time_kst"
    df.reset_index().to_parquet(WORK / "region_hourly.parquet", index=False)
    pd.Series({k: len(v) for k, v in groups.items()}, name="stations").to_csv(WORK / "region_groups.csv", encoding="utf-8-sig")
    print({k: len(v) for k, v in groups.items() if not k.startswith("rg_")})
