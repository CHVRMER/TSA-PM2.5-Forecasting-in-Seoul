"""Regional12 (2026-11-09) hourly regional means for Plan A.

Station lists come only from the shared Regional12 files (regions.json, experiments.json); nothing is regrouped.
Each station is read from its own source_dataset in the shared registry. No imputation, no outlier edits
(team rule): an hour with no valid station is NaN.

Series written to work/r12_hourly.parquet:
  DOM_mean      mean of the 514 domestic external series (the DN set)
  FOREIGN_mean  mean of the 120 foreign series (F minus DN)
  <REGION>_mean for each of the 17 regions, plus *_n (valid-station counts)

Usage: python r12_data.py <Regional12 folder>
"""
import json
import sys
from pathlib import Path

import pandas as pd

from config import DATASET, WORK

FILES = {  # source_dataset -> (folder, main file, buffer file)
    "Korea_Air": ("Korea_Air", "korea_34stations_hourly_2021_2024_kst.parquet", "korea_context_buffer.parquet"),
    "Korea_East_Candidates": ("Korea_East_Candidates", "korea_east_36stations_hourly_2021_2024_kst.parquet",
                              "korea_east_context_buffer.parquet"),
    "Korea_National": ("Korea_National_2021_2024", "korea_national_pm25_hourly_2021_2024_kst.parquet",
                       "korea_national_pm25_context_buffer.parquet"),
    "China": ("China", "china_48stations_hourly_2021_2024_kst.parquet", "china_48stations_context_buffer.parquet"),
    "China_Border_Proxy": ("China_Border_Proxy", "china_border_cities_hourly_2021_2024_kst.parquet",
                           "china_border_cities_context_buffer.parquet"),
    # source name used in the Regional12 registry; read from Dataset/China_East
    "China_East_QC": ("China_East", "china_east_cities_hourly_2021_2024_kst.parquet",
                      "china_east_cities_context_buffer.parquet"),
    "Japan": ("Japan", "japan_west_hourly_2021_2024_kst.parquet", "japan_west_context_buffer.parquet"),
    "Japan_East": ("Japan_East", "japan_east_hourly_2021_2024_kst.parquet", "japan_east_context_buffer.parquet"),
    "Mongolia": ("Mongolia", "mongolia_ub_embassy_hourly_2021_2024_kst.parquet",
                 "mongolia_ub_embassy_context_buffer.parquet"),
}
HOURS = pd.date_range("2020-12-29 00:00", "2025-01-01 00:00", freq="h", tz="Asia/Seoul")


def read_all(reg):
    cols = {}
    for src, part in reg.groupby("source_dataset"):
        folder, main, buf = FILES[src]
        ids = sorted(part.station_id)
        frames = [pd.read_parquet(DATASET / folder / f, columns=["time_kst", "station_id", "pm25"],
                                  filters=[("station_id", "in", ids)]) for f in (main, buf)]
        t = pd.concat(frames)
        assert not t.duplicated(["time_kst", "station_id"]).any(), src
        wide = t.pivot(index="time_kst", columns="station_id", values="pm25")
        missing = set(ids) - set(wide.columns)
        assert not missing, (src, sorted(missing)[:5])
        for sid in ids:
            cols[part.loc[part.station_id == sid, "station_key"].iloc[0]] = wide[sid]
    return pd.DataFrame(cols).reindex(HOURS)


if __name__ == "__main__":
    cfg = Path(sys.argv[1])
    regions = json.loads((cfg / "regions.json").read_text(encoding="utf-8-sig"))["regions"]
    exp = json.loads((cfg / "experiments.json").read_text(encoding="utf-8-sig"))["conditions"]
    reg = pd.read_csv(cfg / "station_registry.csv", dtype=str, encoding="utf-8-sig")
    assert len(reg) == 634 and reg.station_key.is_unique
    pm = read_all(reg)
    groups = {"DOM": exp["DN"]["station_keys"],
              "FOREIGN": sorted(set(exp["F"]["station_keys"]) - set(exp["DN"]["station_keys"]))}
    groups.update({k: v["station_keys"] for k, v in regions.items()})
    assert len(groups["DOM"]) == 514 and len(groups["FOREIGN"]) == 120
    out = {}
    for name, keys in groups.items():
        out[f"{name}_mean"] = pm[keys].mean(axis=1)
        out[f"{name}_n"] = pm[keys].notna().sum(axis=1)
    df = pd.DataFrame(out)
    df.index.name = "time_kst"
    df.reset_index().to_parquet(WORK / "r12_hourly.parquet", index=False)
    tr = df[(df.index.year >= 2021) & (df.index.year <= 2022)]
    va = df[df.index.year == 2023]
    summary = pd.DataFrame({"stations": {k: len(v) for k, v in groups.items()},
                            "train_hours_missing_pct": {k: round(100 * tr[f"{k}_mean"].isna().mean(), 1) for k in groups},
                            "val_hours_missing_pct": {k: round(100 * va[f"{k}_mean"].isna().mean(), 1) for k in groups}})
    summary.to_csv(WORK / "r12_groups.csv", encoding="utf-8-sig")
    print(summary.to_string())
