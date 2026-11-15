"""Profile the exact current external634 hourly PM2.5 source selections.

Read-only source audit. Red values (>500 ug/m3) are display highlights, not an
error classification or an automatic QC policy. Time period is 2021--2024 KST.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"C:\Users\kimjy\Desktop\Time Series Analysis\Project\Revised_Project")
DATASET = ROOT / "Dataset"
CONFIG = ROOT / "Model/Common/Regional_Experiments_20261109"
OUTPUT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2021-01-01", tz="Asia/Seoul")
END = pd.Timestamp("2025-01-01", tz="Asia/Seoul")
HOURS = int((END - START).total_seconds() / 3600)
THRESHOLD = 500.0
SOURCES = {
    "Korea_Air": "Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet",
    "Korea_East_Candidates": "Korea_East_Candidates/korea_east_36stations_hourly_2021_2024_kst.parquet",
    "Korea_National": "Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet",
    "China": "China/china_48stations_hourly_2021_2024_kst.parquet",
    "China_Border_Proxy": "China_Border_Proxy/china_border_cities_hourly_2021_2024_kst.parquet",
    "China_East_QC": "Neighbors_Expansion_QC_20261108/china_east_hourly_qc_2021_2024_kst.parquet",
    "Japan": "Japan/japan_west_hourly_2021_2024_kst.parquet",
    "Japan_East": "Neighbors_Expansion_QC_20261108/japan_east18_hourly_2021_2024_kst.parquet",
    "Mongolia": "Mongolia/mongolia_ub_embassy_hourly_2021_2024_kst.parquet",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values, station_n, maximum_case, *, observed_rows=None):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all() and (values >= 0).all()
    n = len(values)
    expected = HOURS * station_n
    assert n <= expected
    result = {"n": n, "station_n": int(station_n), "expected_station_hours": expected,
              "observed_rows": int(observed_rows) if observed_rows is not None else None,
              "missing": expected - n, "missing_pct": 100 * (expected - n) / expected,
              "zero": int((values == 0).sum()), "n_gt500": int((values > THRESHOLD).sum()),
              "maximum_case": maximum_case}
    if not n:
        result.update({key: None for key in ["min", "p01", "q1", "median", "q3", "mean", "p99", "p999", "max", "whislo", "whishi"]})
        result.update(flier_values=[], red_values=[], n_tukey_outliers=0)
        return result
    p01, q1, med, q3, p99, p999 = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99, 0.999], method="linear")
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= lower) & (values <= upper)]
    outside = values[(values < lower) | (values > upper)]
    result.update({"min": float(values.min()), "p01": float(p01), "q1": float(q1), "median": float(med), "q3": float(q3),
                   "mean": float(values.mean()), "p99": float(p99), "p999": float(p999),
                   "max": float(values.max()), "whislo": float(inside.min()), "whishi": float(inside.max()),
                   "tukey_lower_fence": float(lower), "tukey_upper_fence": float(upper),
                   "n_tukey_outliers": len(outside), "flier_values": np.unique(outside).tolist(),
                   "red_values": np.unique(values[values > THRESHOLD]).tolist()})
    assert result["min"] <= result["whislo"] <= result["q1"] <= result["median"] <= result["q3"] <= result["whishi"] <= result["max"]
    assert result["n_gt500"] <= n and result["zero"] <= n
    return result


def case_record(row, metadata):
    return {"station_key": metadata["station_key"], "station_id": metadata["station_id"],
            "station_name": metadata["station_name"], "country": metadata["country"],
            "region": metadata["experiment_region_id"], "time_kst": row["time_kst"].isoformat(),
            "pm25": float(row["pm25"]), "pm25_flag": int(row["pm25_flag"])}


def main():
    for folder in [OUTPUT / "Results", OUTPUT / "Audit"]:
        folder.mkdir(parents=True, exist_ok=True)
    registry_path = CONFIG / "station_registry.csv"
    region_path = CONFIG / "regions.json"
    files = [registry_path, region_path, *[DATASET / rel for rel in SOURCES.values()]]
    before = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files}
    registry = pd.read_csv(registry_path, dtype={"station_key": str, "station_id": str}, keep_default_na=False)
    regions = json.loads(region_path.read_text(encoding="utf-8-sig"))["region_order"]
    assert len(registry) == 634 and registry.station_key.is_unique and len(regions) == 17
    assert set(registry.experiment_region_id) == set(regions)
    assert "KR:111121" not in set(registry.station_key)
    assert registry.source_dataset.isin(SOURCES).all()
    metadata = registry.set_index("station_key").to_dict("index")
    for key, item in metadata.items():
        item["station_key"] = key
    region_values = {region: [] for region in regions}
    region_max = {region: None for region in regions}
    region_rows = {region: 0 for region in regions}
    station_stats = {}
    extremes = []
    source_audit = {}
    qc_checks = []
    for source, relative in SOURCES.items():
        selected = registry.loc[registry.source_dataset.eq(source)].copy()
        assert not selected.empty and selected.station_id.is_unique
        station_to_key = dict(zip(selected.station_id, selected.station_key))
        station_to_region = dict(zip(selected.station_id, selected.experiment_region_id))
        path = DATASET / relative
        print(f"Reading {source}: {len(selected)} selected series", flush=True)
        frame = pq.read_table(path, columns=["time_kst", "station_id", "pm25", "pm25_flag"],
                              filters=[("station_id", "in", selected.station_id.tolist())]).to_pandas()
        frame["time_kst"] = pd.to_datetime(frame.time_kst, utc=True).dt.tz_convert("Asia/Seoul")
        frame["station_id"] = frame.station_id.astype(str)
        frame = frame.loc[(frame.time_kst >= START) & (frame.time_kst < END)].copy()
        duplicates = int(frame.duplicated(["time_kst", "station_id"]).sum())
        assert duplicates == 0, f"Duplicate source station hours: {relative}"
        assert set(frame.station_id).issubset(station_to_key)
        values = pd.to_numeric(frame.pm25, errors="coerce").to_numpy(float)
        flags = pd.to_numeric(frame.pm25_flag, errors="coerce").to_numpy(float)
        valid = (flags == 0) & np.isfinite(values) & (values >= 0)
        invalid_flag_zero = int(((flags == 0) & ~valid).sum())
        assert invalid_flag_zero == 0, f"Flag-zero invalid value: {relative}"
        frame["valid"] = valid
        source_audit[source] = {"path": relative, "selected_series": len(selected), "period_rows": len(frame),
                               "valid_rows": int(valid.sum()), "duplicate_station_hours": duplicates,
                               "invalid_flag_zero": invalid_flag_zero}
        for station_id, group in frame.groupby("station_id", sort=False, observed=True):
            key = station_to_key[station_id]
            meta = metadata[key]
            region = meta["experiment_region_id"]
            good = group.loc[group.valid]
            a = good.pm25.to_numpy(dtype=np.float64)
            maximum = case_record(good.loc[good.pm25.idxmax()], meta) if len(good) else None
            record = stats(a, 1, maximum, observed_rows=len(group))
            record.update({"station_key": key, "station_name": meta["station_name"], "country": meta["country"],
                           "region": region, "source_dataset": source})
            station_stats[key] = record
            region_values[region].append(a)
            region_rows[region] += len(group)
            if maximum and (region_max[region] is None or maximum["pm25"] > region_max[region]["pm25"]):
                region_max[region] = maximum
            for _, row in good.loc[good.pm25 > THRESHOLD].iterrows():
                extremes.append(case_record(row, meta))
        # Stations with no rows are still part of the fixed input universe.
        for _, item in selected.iterrows():
            key = item.station_key
            if key not in station_stats:
                meta = metadata[key]
                record = stats(np.array([]), 1, None, observed_rows=0)
                record.update({"station_key": key, "station_name": meta["station_name"], "country": meta["country"],
                               "region": meta["experiment_region_id"], "source_dataset": source})
                station_stats[key] = record
        if source == "China_East_QC":
            for station_id, instant, raw_value in [
                ("city_lianyungang", "2021-03-15 20:00:00", 1500),
                ("city_linyi", "2021-03-17 17:00:00", 1451),
                ("city_yancheng", "2021-03-23 17:00:00", 1689)]:
                instant = pd.Timestamp(instant, tz="Asia/Seoul")
                row = frame.loc[frame.station_id.eq(station_id) & frame.time_kst.eq(instant)]
                assert len(row) == 1 and row.pm25.isna().all() and row.pm25_flag.eq(20).all()
                qc_checks.append({"station_key": "CN:" + station_id, "time_kst": instant.isoformat(),
                                  "raw_value": raw_value, "analysis_value": None, "analysis_flag": 20,
                                  "quarantined_in_current_input": True})
        del frame, values, flags, valid
        gc.collect()
    assert set(station_stats) == set(registry.station_key)
    regional = {}
    for region in regions:
        arrays = region_values.pop(region)
        a = np.concatenate(arrays) if arrays else np.array([])
        regional[region] = stats(a, int(registry.experiment_region_id.eq(region).sum()),
                                 region_max[region], observed_rows=region_rows[region])
        regional[region]["region"] = region
        del a, arrays
    total_valid = sum(item["n"] for item in station_stats.values())
    assert total_valid == sum(item["n"] for item in regional.values()) == sum(item["valid_rows"] for item in source_audit.values())
    assert len(extremes) == sum(item["n_gt500"] for item in regional.values())
    for key, expected in [("CN:3578A", 9985), ("CN:1751A", 1976), ("KR:437421", 803)]:
        assert station_stats[key]["max"] == expected, (key, station_stats[key]["max"])
    mongolia_keys = registry.loc[registry.country.eq("MN"), "station_key"].tolist()
    assert len(mongolia_keys) == 1
    representative_keys = ["CN:3578A", "CN:1751A", "KR:437421", mongolia_keys[0]]
    for name, data in [("region_box_stats.json", regional), ("station_box_stats.json", station_stats)]:
        (OUTPUT / "Results" / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(extremes).sort_values(["pm25", "station_key", "time_kst"], ascending=[False, True, True]).to_csv(
        OUTPUT / "Results/extreme_observations_gt500.csv", index=False, encoding="utf-8-sig")
    inventory = registry.copy()
    for field in ["n", "expected_station_hours", "missing", "missing_pct", "zero", "min", "p01", "median", "p99", "p999", "max", "n_gt500"]:
        inventory[field] = inventory.station_key.map({k: v[field] for k, v in station_stats.items()})
    inventory.to_csv(OUTPUT / "Results/series_inventory.csv", index=False, encoding="utf-8-sig")
    for path in files:
        assert sha256(path) == before[str(path)]["sha256"], f"Source changed during profiling: {path}"
    summary = {"period_start_inclusive_kst": START.isoformat(), "period_end_exclusive_kst": END.isoformat(),
               "hourly_grid_hours": HOURS, "external_series": 634, "regions": len(regional),
               "country_counts": registry.country.value_counts().to_dict(),
               "expected_station_hours": HOURS * 634, "valid_station_hours": total_valid,
               "missing_station_hours": HOURS * 634 - total_valid,
               "source_files": before, "all_source_sha256_unchanged_after_profiling": True,
               "source_audit": source_audit, "red_highlight_threshold_ug_m3_exclusive": THRESHOLD,
               "red_highlight_is_error_classification": False,
               "red_highlight_note": ">500 is a visualization threshold, not an automatic data-QC rule.",
               "quartile_method": "numpy.quantile linear", "whiskers": "Most extreme observed values inside Q1-1.5IQR to Q3+1.5IQR",
               "observation_weighting": "Each valid station-hour receives equal weight within its region.",
               "representative_station_keys": representative_keys,
               "regions_gt500": [k for k, item in regional.items() if item["n_gt500"]],
               "n_observations_gt500": len(extremes), "quarantined_china_east_checks": qc_checks,
               "target_seoul_junggu_excluded": True,
               "inputs_transformed_or_overwritten": False}
    (OUTPUT / "Audit/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "valid_station_hours": total_valid,
                      "observations_gt500": len(extremes), "regions_gt500": summary["regions_gt500"],
                      "representative_station_keys": representative_keys, "source_hashes_unchanged": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
