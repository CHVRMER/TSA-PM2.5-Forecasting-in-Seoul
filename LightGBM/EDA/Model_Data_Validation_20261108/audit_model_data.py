"""Read-only audit of existing PM2.5 inputs; never imports or runs model code.

Run from any directory: python3 /absolute/path/to/this/file.py
Only new audit artifacts in this directory are written. Existing outputs are refused.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import platform
import re
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path, PureWindowsPath
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEATURES = ROOT / "Dataset/Origin_Features_2021_2024_v3"
MATCH = ROOT / "Dataset/Trajectory_Matching_2021_2024_v3"
FEATURE_FILE = FEATURES / "origin_features_2021_2024_kst.parquet"
HORIZONS = (3, 6, 12, 24)
HOUR_US = 3_600_000_000
START = pd.Timestamp("2021-01-01", tz="Asia/Seoul")
END = pd.Timestamp("2025-01-01", tz="Asia/Seoul")
SUFFIXES = ["now", "lag01h", "lag03h", "lag06h", "lag12h", "lag24h",
            "mean6h", "std6h", "coverage6h", "mean24h", "std24h", "coverage24h",
            "change3h", "change6h", "age_since_valid_h", "missing_now"]
SOURCES = {
    "Korea_Air": ("korea_34stations_hourly_2021_2024_kst.parquet", "korea_context_buffer.parquet"),
    "Korea_East_Candidates": ("korea_east_36stations_hourly_2021_2024_kst.parquet", "korea_east_context_buffer.parquet"),
    "China": ("china_48stations_hourly_2021_2024_kst.parquet", "china_48stations_context_buffer.parquet"),
    "China_Border_Proxy": ("china_border_cities_hourly_2021_2024_kst.parquet", "china_border_cities_context_buffer.parquet"),
    "Japan": ("japan_west_hourly_2021_2024_kst.parquet", "japan_west_context_buffer.parquet"),
    "Mongolia": ("mongolia_ub_embassy_hourly_2021_2024_kst.parquet", "mongolia_ub_embassy_context_buffer.parquet"),
}
REPORT = {"status": "running", "started_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
          "python": sys.version, "platform": platform.platform(), "checks": {},
          "scope": "Data validation only; no model training, prediction, tuning, imputation, or source edits.",
          "accepted_conventions": {
              "japan": "README hour-ending convention, 24h -> next-day 00:00; JST=KST; not externally revalidated",
              "M3T": "Existing 2424-column allowlist union, including all-path summaries over KR/CN/JP/MN",
              "availability": "Retrospective research; publication latency is not recreated",
          }}


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def passed(name, info):
    REPORT["checks"][name] = info
    print(f"PASS {name}: {json.dumps(info, ensure_ascii=False, default=str)[:700]}", flush=True)


def read_features(columns):
    return pq.read_table(FEATURE_FILE, columns=columns, use_threads=False).to_pandas()


def equal_numeric(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    same = np.isclose(actual, expected, rtol=1e-10, atol=1e-8, equal_nan=True)
    check(np.all(same), f"{name}: {int((~same).sum())} numeric/null mismatches")


def hour_ids(times):
    return pd.DatetimeIndex(times).as_unit("us").asi8 // HOUR_US


def split_of(times):
    year = pd.DatetimeIndex(times).year
    return np.select([(year >= 2021) & (year <= 2022), year == 2023, year == 2024],
                     ["train", "validation", "test"], default="outside")


def source_lookup(frame, field):
    frame = frame.sort_values("time_kst")
    hours = hour_ids(frame.time_kst)
    check(np.all(np.diff(hours) > 0), "Duplicate/non-increasing source hours")
    values = frame[field].to_numpy(float)
    check(not np.isinf(values).any(), f"Infinite source value: {field}")

    def get(query):
        query = np.asarray(query, dtype=np.int64)
        idx = np.searchsorted(hours, query)
        clipped = np.minimum(idx, len(hours) - 1)
        found = (idx < len(hours)) & (hours[clipped] == query)
        return np.where(found, values[clipped], np.nan)

    return hours, values, get


def moments(matrix):
    valid = np.isfinite(matrix)
    n = valid.sum(axis=1)
    avg = np.divide(np.nansum(matrix, axis=1), n, out=np.full(len(matrix), np.nan), where=n > 0)
    variance = np.divide(np.where(valid, matrix - avg[:, None], 0).__pow__(2).sum(axis=1),
                         n, out=np.full(len(matrix), np.nan), where=n > 0)
    return avg, np.sqrt(np.maximum(variance, 0)), n / matrix.shape[1]


def history_expected(frame, origin_hours, prefix, extra_lags=()):
    hours, values, get = source_lookup(frame, "pm25")
    now = get(origin_hours)
    expected = {prefix + "now": now}
    for lag in (1, 3, 6, 12, 24, *extra_lags):
        expected[prefix + f"lag{lag:02d}h"] = get(origin_hours - lag)
    for n in (6, 24):
        avg, std, coverage = moments(get(origin_hours[:, None] - np.arange(n)))
        expected.update({prefix + f"mean{n}h": avg, prefix + f"std{n}h": std,
                         prefix + f"coverage{n}h": coverage})
    for lag in (3, 6):
        expected[prefix + f"change{lag}h"] = now - get(origin_hours - lag)
    valid_hours = hours[np.isfinite(values)]
    ix = np.searchsorted(valid_hours, origin_hours, side="right") - 1
    expected[prefix + "age_since_valid_h"] = (np.where(ix >= 0, origin_hours - valid_hours[np.maximum(ix, 0)], np.nan)
                                               if len(valid_hours) else np.full(len(origin_hours), np.nan))
    expected[prefix + "missing_now"] = (~np.isfinite(now)).astype(float)
    return expected


def distance_bearing(lon1, lat1, lon2, lat2):
    x1, y1, x2, y2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((y2-y1)/2)**2 + np.cos(y1)*np.cos(y2)*np.sin((x2-x1)/2)**2
    distance = 6371 * 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1-a, 0, 1)))
    bearing = np.arctan2(np.sin(x2-x1)*np.cos(y2),
                        np.cos(y1)*np.sin(y2)-np.sin(y1)*np.cos(y2)*np.cos(x2-x1))
    return distance, np.where(distance > 1e-9, bearing, np.nan)


def main():
    manifest = json.loads((FEATURES / "feature_manifest.json").read_text())
    definitions = json.loads((FEATURES / "feature_sets.json").read_text())
    sets = definitions["sets"]
    dictionary_rows = rows(FEATURES / "feature_dictionary.csv")
    dictionary = {r["name"]: r for r in dictionary_rows}
    registry = rows(MATCH / "station_registry.csv")
    balanced = rows(FEATURES / "japan_balanced_stations.csv")

    # Stream hashes, resolving the recorded Windows location by its Dataset suffix.
    snapshots = {}
    for item in manifest["inputs"]:
        parts = PureWindowsPath(item["path"]).parts
        start = parts.index("Dataset")
        local = ROOT.joinpath(*parts[start:])
        check(local.is_file(), f"Manifest input missing: {local}")
        digest = sha(local)
        check(digest == item["sha256"], f"Manifest input SHA256 mismatch: {local}")
        snapshots[local] = digest
    for path in [FEATURE_FILE, *FEATURES.glob("*.json"), *FEATURES.glob("*.csv"),
                 *ROOT.joinpath("Models").glob("*.py")]:
        snapshots[path] = sha(path)
    for name, expected in manifest["code_sha256"].items():
        code = ROOT / "Dataset/Codes/Origin_Features_v3" / name
        check(sha(code) == expected, f"Feature producer code hash mismatch: {name}")
        snapshots[code] = expected
    passed("input_integrity", {"manifest_source_hashes_matched": len(manifest["inputs"]),
                               "producer_code_hashes_matched": len(manifest["code_sha256"])})

    pf = pq.ParquetFile(FEATURE_FILE)
    schema = pf.schema_arrow
    field_types = {f.name: f.type for f in schema}
    check(pf.metadata.num_rows == manifest["rows"] == 11688, "Feature row count mismatch")
    check(len(schema) == manifest["columns"] == 5678, "Feature column count mismatch")
    check(len(dictionary) == len(dictionary_rows) == 5678, "Duplicate/missing dictionary entries")
    check(schema.names == list(dictionary), "Dictionary/schema names or order differ")
    roles = collections.Counter(r["role"] for r in dictionary_rows)
    check(roles == {"feature": 5640, "metadata": 6, "label": 32}, "Unexpected role counts")
    for name, cols in sets.items():
        check(len(cols) == len(set(cols)) > 0, f"Duplicate/empty allowlist: {name}")
        check(all(c in dictionary and dictionary[c]["role"] == "feature" for c in cols), f"Unsafe allowlist: {name}")
        check(all(pa.types.is_floating(field_types[c]) for c in cols), f"Non-numeric input: {name}")
    for name, n in manifest["feature_set_counts"].items():
        check(len(sets[name]) == n, f"Manifest feature count mismatch: {name}")
    check(set(sets["all_candidate_features"]) == {c for c, r in dictionary.items() if r["role"] == "feature"},
          "Candidate allowlist differs from dictionary")

    check(len(registry) == len({r['station_key'] for r in registry}) == 317, "Registry identity issue")
    check(collections.Counter(r["country"] for r in registry) == {"KR": 67, "CN": 61, "JP": 188, "MN": 1}, "Country counts differ")
    check(not any(r["station_key"] == "KR:111121" for r in registry), "Target included in external registry")
    for r in registry:
        check(r["station_key"] == f"{r['country']}:{r['station_id']}", "Station identity/country mismatch")
        check(r["pm25_unit"] == "ug/m3" and r["region_group"] and r["source_dataset"] in SOURCES,
              f"Unclear registry metadata: {r['station_key']}")
        check(-180 <= float(r["longitude"]) <= 180 and -90 <= float(r["latitude"]) <= 90, "Invalid coordinates")
    target_meta = [r for r in rows(ROOT / "Dataset/Korea_Air/station_metadata.csv") if r["station_id"] == "111121"]
    check(len(target_meta) == 1 and target_meta[0]["station_name"] == "중구" and target_meta[0]["is_target"] == "1", "Target identity mismatch")
    near = []
    for r in registry:
        if r["country"] == "CN" and r["source_dataset"] == "China":
            dist, _ = distance_bearing(126.98, 37.57, float(r["longitude"]), float(r["latitude"]))
            if dist < 600:
                near.append(r["station_key"])
    check(len(near) == 21, "Near-China group is not 21 stations")
    check(len(balanced) == len({r["station_key"] for r in balanced}) == 24, "Balanced Japan count issue")
    check(set(collections.Counter(r["pref_code"] for r in balanced).values()) == {2}, "Japan prefecture balance differs")
    balanced_keys = {r["station_key"] for r in balanced}
    check(balanced_keys <= {r["station_key"] for r in registry if r["country"] == "JP"}, "Unknown balanced Japan station")
    cn_keys = {r["station_key"] for r in registry if r["country"] == "CN"}
    check(len({r["station_key"] for r in registry if r["source_dataset"] == "China_Border_Proxy"}) == 13, "Proxy group count issue")
    union = lambda *parts: list(dict.fromkeys(c for part in parts for c in part))
    obs_key = lambda c: re.sub(r"^obs_([a-z]{2})_(.+)_pm25_.*$", lambda m: f"{m[1].upper()}:{m[2]}", c)
    model_cols = {
        "M1": sets["local_plus_station_history_kr"],
        "M2": union(sets["local_plus_station_history_kr"], [c for c in sets["station_history_cn"] if obs_key(c) in near]),
        "M3": sets["local_plus_station_history_kr_cn"],
        "M4": sets["local_plus_station_history_all_balanced"],
        "M3T": union(sets["local_plus_trajectory_kr_cn_scopes_without_optional_gases"], sets["station_history_kr"], sets["station_history_cn"]),
        "M4T": sets["local_plus_trajectory_and_station_history_balanced_without_optional_gases"],
    }
    check([len(v) for v in model_cols.values()] == [1142, 1478, 2118, 2518, 2424, 2952], "Model feature counts differ")
    check(set(model_cols["M1"]) <= set(model_cols["M2"]) <= set(model_cols["M3"]) <= set(model_cols["M4"]), "Model ladder is not nested")
    check({obs_key(c) for c in sets["station_history_cn"]} == cn_keys, "CN feature identity mismatch")
    passed("schema_and_allowlists", {"rows": 11688, "columns": 5678, "roles": dict(roles),
                                    "allowlists_checked": len(sets), "model_feature_counts": {k: len(v) for k, v in model_cols.items()},
                                    "country_counts": dict(collections.Counter(r['country'] for r in registry)),
                                    "near_china_count": len(near), "near_china_keys": near,
                                    "balanced_japan_count": len(balanced), "target": target_meta[0]["station_name"]})

    meta_cols = [c for c, r in dictionary.items() if r["role"] != "feature"]
    meta = read_features(meta_cols)
    origins = pd.DatetimeIndex(meta.origin_time_kst)
    check(origins.equals(pd.date_range(START, END, freq="3h", inclusive="left")), "Origin grid differs")
    check((pd.DatetimeIndex(meta.origin_time_utc) == origins).all(), "UTC/KST instant mismatch")
    check((meta.feature_cutoff_time_kst == meta.origin_time_kst).all(), "Feature cutoff differs from t")
    check((meta.arrival_height_agl_m == 500).all(), "Wrong trajectory arrival height")
    expected_split = split_of(origins)
    check(np.array_equal(meta.origin_split, expected_split), "Origin split mismatch")
    oh = hour_ids(origins)

    target = pd.concat([pq.read_table(ROOT / "Dataset/Korea_Air" / f,
                       columns=["time_kst", "station_id", "pm25", "pm25_flag", "source_id"],
                       filters=[("station_id", "=", "111121")], use_threads=False).to_pandas()
                       for f in SOURCES["Korea_Air"]], ignore_index=True).sort_values("time_kst")
    check(not target.time_kst.duplicated().any(), "Duplicate target source hours")
    check((target.pm25.notna() == target.pm25_flag.eq(0)).all(), "Target source flag mismatch")
    check((target.pm25.dropna() >= 0).all(), "Negative valid target")
    ti = target.set_index("time_kst")
    eligibility = {}
    for h in HORIZONS:
        tt = origins + pd.Timedelta(hours=h)
        check((pd.DatetimeIndex(meta[f"target_time_h{h}"]) == tt).all(), f"h{h}: wrong target time")
        tsplit = split_of(tt)
        check(np.array_equal(meta[f"target_split_h{h}"], tsplit), f"h{h}: wrong target split")
        inside = (tt >= START) & (tt < END)
        raw = ti.reindex(tt)
        expected_y = np.where(inside, raw.pm25.to_numpy(float), np.nan)
        equal_numeric(meta[f"y_pm25_h{h}"], expected_y, f"h{h}: target")
        for field, source in [("target_flag", "pm25_flag"), ("target_source_id", "source_id")]:
            expected = np.where(inside, raw[source].to_numpy(float), np.nan)
            equal_numeric(meta[f"{field}_h{h}"], expected, f"h{h}: {field}")
        available = inside & np.isfinite(expected_y)
        same = expected_split == tsplit
        eligible = available & same
        check(np.array_equal(meta[f"label_available_h{h}"], available), f"h{h}: label availability differs")
        check(np.array_equal(meta[f"eligible_h{h}"], eligible), f"h{h}: eligibility differs")
        status = np.select([~inside, ~same & available, ~same, available],
                           ["outside_study_period", "split_boundary", "split_boundary_and_missing", "observed"], default="target_missing")
        check(np.array_equal(meta[f"target_status_h{h}"], status), f"h{h}: target status differs")
        eligibility[str(h)] = {s: int(np.sum(eligible & (expected_split == s))) for s in ("train", "validation", "test")}
        for s, count in eligibility[str(h)].items():
            check(count == manifest["target_eligible_counts"][f"h{h}_{s}"], f"h{h}: manifest eligibility count differs")
        check(not np.any(eligible & (~same | ~inside | ~np.isfinite(expected_y))), f"h{h}: invalid evaluation row")
    passed("targets_and_splits", {"origins": len(origins), "start": str(origins.min()), "last": str(origins.max()),
                                  "rows_by_year": {str(y): int(sum(origins.year == y)) for y in sorted(set(origins.year))},
                                  "target_cells_recomputed": len(origins)*4, "eligible_rows": eligibility,
                                  "eligible_cross_boundary_rows": 0, "eligible_missing_targets": 0})

    # Scan monthly row groups, not the full 5640-column file in memory.
    feature_cols = sets["all_candidate_features"]
    null_counts = collections.Counter()
    valid_train = collections.Counter()
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=feature_cols+["origin_split"], use_threads=False)
        train_mask = np.array(table["origin_split"].to_pylist()) == "train"
        for c in feature_cols:
            values = table[c].to_numpy(zero_copy_only=False)
            check(not np.isinf(values).any(), f"Infinite input feature: {c}")
            null_counts[c] += int(np.isnan(values).sum())
            valid_train[c] += int(np.isfinite(values[train_mask]).sum())
        if (rg+1) % 12 == 0:
            print(f"Scanned feature row groups {rg+1}/{pf.metadata.num_row_groups}", flush=True)
    check(all(null_counts[c] == manifest["feature_null_counts"][c] for c in feature_cols), "Feature missing counts differ from manifest")
    model_quality = {}
    for model, cols in model_cols.items():
        model_quality[model] = {"features": len(cols), "null_cell_pct": 100*sum(null_counts[c] for c in cols)/(len(origins)*len(cols)),
                                "all_missing_training_columns": [c for c in cols if valid_train[c] == 0]}
    passed("feature_value_scan", {"feature_columns": len(feature_cols), "row_groups": pf.metadata.num_row_groups,
                                  "cells_checked": len(origins)*len(feature_cols), "infinite_cells": 0,
                                  "missing_counts_match_manifest": True, "model_quality": model_quality})

    seoul_expected = history_expected(target, oh, "seoul_pm25_", extra_lags=(48,))
    actual = read_features(list(seoul_expected))
    for c, expected in seoul_expected.items():
        equal_numeric(actual[c], expected, c)
    passed("seoul_history", {"columns": len(seoul_expected), "rows_each": len(origins), "source_station": "111121", "mismatches": 0})

    # Recompute all 16 histories for every station actually used by M1-M4/T.
    selected = [r for r in registry if r["country"] != "JP" or r["station_key"] in balanced_keys]
    history_summary = {}
    for ds, files in SOURCES.items():
        stations = [r for r in selected if r["source_dataset"] == ds]
        ids = [r["station_id"] for r in stations]
        source_frames = []
        for f in files:
            path = ROOT / "Dataset" / ds / f
            schema = pq.read_schema(path)
            check(schema.field("time_kst").type == pa.timestamp("us", tz="Asia/Seoul"), f"Timezone/type mismatch: {path}")
            frame = pq.read_table(path, columns=["time_kst", "station_id", "pm25", "pm25_flag"],
                                  filters=[("station_id", "in", ids)], use_threads=False).to_pandas()
            check((frame.pm25.notna() == frame.pm25_flag.eq(0)).all(), f"Value/flag mismatch: {path}")
            check((frame.pm25.dropna() >= 0).all(), f"Negative PM2.5 in {path}")
            source_frames.append(frame)
        source = pd.concat(source_frames, ignore_index=True)
        check(not source.duplicated(["station_id", "time_kst"]).any(), f"Duplicate source station/hour: {ds}")
        prefixes = {r["station_id"]: f"obs_{r['country'].lower()}_{r['station_id']}_pm25_" for r in stations}
        columns = [prefix+suffix for prefix in prefixes.values() for suffix in SUFFIXES]
        actual = read_features(columns)
        for sid, frame in source.groupby("station_id", sort=False):
            expected = history_expected(frame, oh, prefixes[sid])
            for c, values in expected.items():
                equal_numeric(actual[c], values, c)
        history_summary[ds] = {"stations": len(stations), "columns": len(columns), "rows_each": len(origins), "mismatches": 0}
        print(f"Recomputed {ds}: {len(stations)} stations x 16 histories", flush=True)
        del actual, source, source_frames
    check(sum(s["stations"] for s in history_summary.values()) == 153, "Selected station count differs")
    passed("station_histories", history_summary)

    # ASOS: reconstruct both stations, wind convention, windows, and availability.
    weather = pd.concat([pq.read_table(ROOT / "Dataset/ASOS" / f, use_threads=False).to_pandas()
                         for f in ["asos_102_108_hourly_2021_2024_kst.parquet", "asos_context_after.parquet"]], ignore_index=True)
    fields = {"temperature": "temperature", "rh": "relative_humidity", "pressure": "surface_pressure",
              "wind_speed": "wind_speed", "precip_reported": "precipitation"}
    asos_expected = {}
    for sid in ("108", "102"):
        frame = weather[weather.station_id.eq(sid)].sort_values("time_kst").copy()
        check(not frame.time_kst.duplicated().any(), "Duplicate ASOS hour")
        for f in [*fields.values(), "wind_direction"]:
            check((frame[f].notna() == frame[f+"_flag"].eq(0)).all(), f"ASOS flag mismatch: {sid}/{f}")
        speed, direction = frame.wind_speed, frame.wind_direction
        good = speed.gt(0) & direction.gt(0) & direction.le(360)
        frame["wind_u"] = (-speed*np.sin(np.deg2rad(direction))).where(good).mask(speed.eq(0), 0)
        frame["wind_v"] = (-speed*np.cos(np.deg2rad(direction))).where(good).mask(speed.eq(0), 0)
        lookups = {k: source_lookup(frame, f)[2] for k, f in {**fields, "wind_u": "wind_u", "wind_v": "wind_v", "direction": "wind_direction"}.items()}
        prefix = f"asos{sid}_"
        for short in [*fields, "wind_u", "wind_v"]:
            asos_expected[prefix+short+"_now"] = lookups[short](oh)
        q = oh[:, None]-np.arange(6)
        for short in ("temperature", "rh", "pressure", "wind_speed", "wind_u", "wind_v"):
            asos_expected[prefix+short+"_mean6h"] = moments(lookups[short](q))[0]
        for short in ("temperature", "rh", "pressure", "wind_speed", "wind_vector"):
            valid = (np.isfinite(lookups["wind_u"](q)) & np.isfinite(lookups["wind_v"](q))) if short == "wind_vector" else np.isfinite(lookups[short](q))
            asos_expected[prefix+short+"_coverage6h"] = valid.mean(axis=1)
        rain = lookups["precip_reported"](q)
        asos_expected[prefix+"precip_reported_count6h"] = np.isfinite(rain).sum(axis=1)
        asos_expected[prefix+"precip_reported_max6h"] = np.where(np.isfinite(rain).any(axis=1), np.max(np.where(np.isfinite(rain), rain, -np.inf), axis=1), np.nan)
        spd, dr = lookups["wind_speed"](oh), lookups["direction"](oh)
        asos_expected[prefix+"wind_calm_now"] = np.where(np.isfinite(spd), (spd >= 0) & (spd <= .4), np.nan)
        asos_expected[prefix+"wind_direction_unknown_now"] = np.where(np.isfinite(spd), (spd > 0) & (~np.isfinite(dr) | (dr <= 0) | (dr > 360)), np.nan)
        asos_expected[prefix+"history_rows_fraction6h"] = np.isin(q, hour_ids(frame.time_kst)).mean(axis=1)
    actual = read_features(list(asos_expected))
    for c, expected in asos_expected.items():
        equal_numeric(actual[c], expected, c)
    epoch = pd.Timestamp("2000-01-01", tz="Asia/Seoul")
    phase = (origins-epoch).total_seconds().to_numpy()/86400/365.2425*2*np.pi
    calendar_expected = {"cal_hour_sin": np.sin(2*np.pi*origins.hour/24), "cal_hour_cos": np.cos(2*np.pi*origins.hour/24), "cal_weekday": origins.dayofweek}
    for k in (1, 2):
        calendar_expected[f"cal_annual_sin{k}"] = np.sin(k*phase)
        calendar_expected[f"cal_annual_cos{k}"] = np.cos(k*phase)
    actual = read_features(list(calendar_expected))
    for c, expected in calendar_expected.items():
        equal_numeric(actual[c], expected, c)
    passed("asos_and_calendar", {"asos_columns": len(asos_expected), "calendar_columns": len(calendar_expected), "rows_each": len(origins), "mismatches": 0,
                                  "ASOS_2020_buffer_absent": "preserved as reduced early-window availability, not filled"})

    # Complete path time/geometry checks; complete matched-edge time checks in batches.
    point_cols = ["point_id", "arrival_time_kst", "pass_time_kst", "lookback_hours", "arrival_height_agl_m", "longitude", "latitude"]
    points = pq.read_table(MATCH / "trajectory_point_coverage.parquet", columns=point_cols, use_threads=False).to_pandas().sort_values(["arrival_time_kst", "lookback_hours"])
    check(len(points) == 11688*48 and points.point_id.is_unique, "Point count/identity issue")
    ph = hour_ids(points.arrival_time_kst)
    age = points.lookback_hours.to_numpy()
    check(np.array_equal(ph.reshape(-1, 48)[:, 0], oh) and np.array_equal(age, np.tile(np.arange(1, 49), len(origins))), "Wrong 48-hour path grid")
    check(np.array_equal(hour_ids(points.pass_time_kst), ph-age), "Path pass time differs from t-age")
    check((points.arrival_height_agl_m == 500).all(), "Path arrival height mismatch")
    p12 = points[points.lookback_hours.eq(12)]
    route_actual = read_features(["route_latitude_at12h", "route_longitude_at12h"])
    equal_numeric(route_actual.route_latitude_at12h, p12.latitude, "12h latitude")
    equal_numeric(route_actual.route_longitude_at12h, p12.longitude, "12h longitude")
    dist, bearing = distance_bearing(126.98, 37.57, p12.longitude.to_numpy(), p12.latitude.to_numpy())
    compass = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    directions = compass[(((np.rad2deg(bearing) % 360)+22.5) % 360 // 45).astype(int)]
    directions = np.where(dist < 25, "LOCAL", directions)
    check(np.array_equal(directions, meta.route_history12h_sector), "Route sector differs from 12h geometry")
    edge_count = 0
    edge_pf = pq.ParquetFile(MATCH / "trajectory_station_neighbors.parquet")
    for batch in edge_pf.iter_batches(batch_size=65536, columns=["arrival_time_kst", "pass_time_kst", "observation_time_kst", "lookback_hours", "station_key", "country", "distance_km"]):
        a, p, o = (batch.column(batch.schema.get_field_index(c)).cast(pa.int64()).to_numpy() for c in ["arrival_time_kst", "pass_time_kst", "observation_time_kst"])
        age_b = batch.column(batch.schema.get_field_index("lookback_hours")).to_numpy()
        check(np.array_equal(p, a-age_b*HOUR_US) and np.array_equal(p, o) and np.all(p < a), "Future/misaligned matched observation")
        distances = batch.column(batch.schema.get_field_index("distance_km")).to_numpy()
        check(np.all(np.isfinite(distances) & (distances >= 0) & (distances <= 75+1e-9)), "Invalid matching radius")
        keys = batch.column(batch.schema.get_field_index("station_key")).to_pylist()
        countries = batch.column(batch.schema.get_field_index("country")).to_pylist()
        reg_country = {r["station_key"]: r["country"] for r in registry}
        check(all(reg_country.get(k) == country for k, country in zip(keys, countries)), "Matched station/country mismatch")
        edge_count += batch.num_rows
    check(edge_count == 3203668, "Matched edge count differs")
    passed("trajectory_times_and_direction", {"points_checked": len(points), "matched_edges_checked": edge_count,
                                              "future_or_misaligned_observation_rows": 0, "route_sector_rows_recomputed": len(origins),
                                              "direction_counts": dict(collections.Counter(directions))})

    # A deterministic 12-origin path-concentration spot check, 3 per year.
    indices = [int(i) for year in (2021, 2022, 2023, 2024) for i in np.flatnonzero(origins.year == year)[[0, 1000, -1]]]
    sample_times = origins[indices]
    edges = pq.read_table(MATCH / "trajectory_station_neighbors.parquet",
                         columns=["arrival_time_kst", "lookback_hours", "country", "distance_km", "pm25", "pm25_flag", "pm10", "pm10_flag"],
                         filters=[("arrival_time_kst", "in", sample_times.to_pydatetime().tolist())], use_threads=False).to_pandas()
    path_cols = [c for c in model_cols["M4T"] if c.startswith(("pm25_", "pm10_", "geo_")) and dictionary[c]["definition_key"] in ("traj_pm", "traj_geo")]
    path_actual = read_features(path_cols)
    checked = 0
    for row_index, origin in zip(indices, sample_times):
        e = edges[edges.arrival_time_kst.eq(origin)].copy()
        e["w"] = np.exp(-.5*(e.distance_km/50)**2)
        for scope in ("all", "kr", "cn", "jp", "mn"):
            scoped = e if scope == "all" else e[e.country.eq(scope.upper())]
            for pol in ("pm25", "pm10"):
                valid = scoped[scoped[pol+"_flag"].eq(0)].copy()
                valid["weighted"] = valid.w*valid[pol]
                numerator = valid.groupby("lookback_hours").weighted.sum()
                denominator = valid.groupby("lookback_hours").w.sum()
                hourly = (numerator/denominator).reindex(range(1, 49))
                for band, start, end in [("b01_06", 1, 6), ("b07_12", 7, 12), ("b13_24", 13, 24), ("b25_48", 25, 48)]:
                    segment = hourly.loc[start:end]
                    for stat, value in [("mean", segment.mean()), ("coverage", segment.notna().mean())]:
                        c = f"{pol}_{scope}_{band}_{stat}"
                        equal_numeric([path_actual[c].iloc[row_index]], [value], f"sample {origin}/{c}")
                        checked += 1
            for band, start, end in [("b01_06", 1, 6), ("b07_12", 7, 12), ("b13_24", 13, 24), ("b25_48", 25, 48)]:
                value = scoped.loc[scoped.lookback_hours.between(start, end), "lookback_hours"].nunique()/(end-start+1)
                c = f"geo_{scope}_{band}_fraction"
                equal_numeric([path_actual[c].iloc[row_index]], [value], f"sample {origin}/{c}")
                checked += 1
    passed("trajectory_concentration_samples", {"origins": [str(t) for t in sample_times], "values_recomputed": checked,
                                                "scope": "PM2.5/PM10 mean and coverage; geographic fractions; all 5 country scopes", "mismatches": 0})

    # Adaptive conditions are inspected only; no weights or predictions are fit.
    group_map = {"NW": "west", "W": "west", "SW": "west", "N": "north", "NE": "north", "E": "east", "SE": "east", "S": "east", "LOCAL": "local"}
    groups = pd.Series(directions).map(group_map).to_numpy()
    check(not pd.isna(groups).any(), "Unknown adaptive route category")
    seasons = np.where(origins.month.isin([11, 12, 1, 2, 3]), "cold", "warm")
    support = {}
    for h in HORIZONS:
        va = meta[f"eligible_h{h}"].to_numpy() & (expected_split == "validation")
        support[str(h)] = [{"season": s, "route": g, "n": int(np.sum(va & (seasons == s) & (groups == g))),
                             "requires_season_fallback": int(np.sum(va & (seasons == s) & (groups == g))) < 100}
                            for s in ("cold", "warm") for g in ("west", "north", "east", "local")]
    passed("adaptive_condition_support", {"validation_counts": support, "weight_estimation_performed": False,
                                          "later_decision": "Near-zero denominator rule requires approval at Adaptive stage; existing code silently chooses 0."})

    changed = [str(p.relative_to(ROOT)) for p, before in snapshots.items() if sha(p) != before]
    check(not changed, f"Source/model files changed during audit: {changed}")
    passed("source_preservation", {"files_rehashed_unchanged": len(snapshots), "model_code_imported_or_executed": False})
    REPORT["input_sha256"] = {str(p.relative_to(ROOT)): value for p, value in snapshots.items()}
    REPORT["limitations"] = [
        "No external validation of Japanese hour-ending interpretation; user accepted README convention.",
        "GDAS/finalized observations publication latency is not recreated; retrospective experiment only.",
        "Full all-candidate feature/null scan; station histories independently recomputed for 153 model-selected stations, not all 188 Japan candidates.",
        "All trajectory point/edge timestamps checked; concentration aggregation independently spot-checked at 12 origins, not fully recomputed.",
        "2024 data inspected only for schema, alignment, source consistency and eligibility; no feature selection/tuning/performance evaluation.",
        "Existing Models/config.py creates directories at import; not imported during audit.",
    ]
    REPORT["status"] = "passed"


if __name__ == "__main__":
    destination = OUT / "validation_results_final.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    began = time.monotonic()
    try:
        main()
    except Exception as exc:
        REPORT["status"] = "stopped_on_error"
        REPORT["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        REPORT["elapsed_seconds"] = round(time.monotonic()-began, 3)
        REPORT["finished_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        with destination.open("x", encoding="utf-8") as f:
            json.dump(REPORT, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        print(f"Audit status: {REPORT['status']}; report: {destination}", flush=True)
