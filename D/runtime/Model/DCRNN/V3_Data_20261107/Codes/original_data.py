"""Exact-hour DCRNN inputs for the frozen Seoul PM2.5 dataset contract.

This file is embedded directly in the Colab notebook. It deliberately does not
import the project's feature-generation code, impute a target, forward-fill an
observation, or construct a window by shifting the three-hour origin table.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


DCRNN_HORIZONS = (3, 6, 12, 24)
DCRNN_TIMEZONE = "Asia/Seoul"
DCRNN_SOURCE_FILES = {
    "Korea_Air": (
        "Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet",
        "Korea_Air/korea_context_buffer.parquet",
    ),
    "Korea_East_Candidates": (
        "Korea_East_Candidates/korea_east_36stations_hourly_2021_2024_kst.parquet",
        "Korea_East_Candidates/korea_east_context_buffer.parquet",
    ),
    "China": (
        "China/china_48stations_hourly_2021_2024_kst.parquet",
        "China/china_48stations_context_buffer.parquet",
    ),
}


def _dcrnn_kst(values):
    """Convert timezone-aware Arrow/Pandas times to one nanosecond time scale."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert(
        DCRNN_TIMEZONE
    ).as_unit("ns")


def resolve_dataset_root(path):
    """Accept Dataset/, its project parent, or an extracted ZIP wrapper folder.

    A ZIP must first be extracted by the notebook's safe extraction cell. Raw
    hourly files are required: the summary-lag feature table cannot recreate a
    complete 48-hour sequence.
    """
    base = Path(path).expanduser()
    if base.is_file():
        raise ValueError("Pass an extracted directory, not a ZIP or Parquet file.")
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    marker = Path("Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet")
    likely = [base, base / "Dataset", base / "Revised_Project" / "Dataset",
              base / "Data" / "Dataset"]
    for candidate in likely:
        if (candidate / marker).is_file():
            return candidate.resolve()
    roots = set()
    for match in base.rglob(marker.name):
        if any(part.lower() in {"trash", ".git"} for part in match.parts):
            continue
        candidate = match.parent.parent
        if (candidate / marker).is_file():
            roots.add(candidate.resolve())
    if len(roots) != 1:
        raise ValueError(
            "Select one extracted Dataset directory. Found "
            f"{len(roots)} compatible datasets under {base}."
        )
    return roots.pop()


def _dcrnn_moments(values):
    """Finite-only population moments; undefined/constant scales become one."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = np.divide(np.where(finite, values, 0.0).sum(axis=0), counts,
                      out=np.zeros(values.shape[1], dtype=np.float64), where=counts > 0)
    variance = np.divide(
        np.where(finite, values - means[None, :], 0.0).__pow__(2).sum(axis=0),
        counts, out=np.zeros_like(means), where=counts > 0,
    )
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    return means, scales, counts


def load_dcrnn_bundle(dataset_root, lookback_hours=48):
    """Read the authoritative origins and the exact-hour 116-node PM2.5 panel.

    Only ASOS/calendar columns from the positive local_baseline allowlist become
    global context. Future label flags, split labels, source IDs, country-wide
    concentration summaries, and future trajectories never enter model inputs.
    """
    if not isinstance(lookback_hours, int) or lookback_hours < 1:
        raise ValueError("lookback_hours must be a positive integer.")
    root = resolve_dataset_root(dataset_root)
    origin_dir = root / "Origin_Features_2021_2024"
    registry_path = root / "Trajectory_Matching_2021_2024/station_registry.csv"
    required = [registry_path, origin_dir / "feature_sets.json",
                origin_dir / "feature_dictionary.csv",
                origin_dir / "origin_features_2021_2024_kst.parquet"]
    required += [root / rel for rels in DCRNN_SOURCE_FILES.values() for rel in rels]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "The shared archive is missing required files:\n- " + "\n- ".join(missing)
            + "\nInclude the six processed hourly/buffer Parquets; the origin "
              "summary table alone is insufficient. Raw source CSV/XLSX files "
              "and the original large meteorology archive are not needed."
        )

    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    expected = {"station_key", "station_id", "country", "source_dataset",
                "longitude", "latitude", "pm25_unit"}
    if not expected.issubset(registry.columns):
        raise ValueError("Station registry is missing required identity columns.")
    if len(registry) != 115 or registry["station_key"].duplicated().any():
        raise ValueError("Expected exactly 115 unique external station identities.")
    if (registry["station_key"] == "KR:111121").any():
        raise ValueError("Target Junggu must be added once, outside external registry.")
    if not (registry["station_key"] == registry["country"] + ":" + registry["station_id"]).all():
        raise ValueError("Registry station_key/country/station_id disagree.")
    if not registry["source_dataset"].isin(DCRNN_SOURCE_FILES).all():
        raise ValueError("Registry contains an unsupported hourly source.")
    if not registry["pm25_unit"].eq("ug/m3").all():
        raise ValueError("All PM2.5 node observations must use ug/m3.")
    if registry.groupby("country").size().to_dict() != {"CN": 48, "KR": 67}:
        raise ValueError("Expected external registry counts KR=67 and CN=48.")
    target = {column: "" for column in registry.columns}
    target.update(station_key="KR:111121", station_id="111121", country="KR",
                  station_name="서울 중구", city="서울", region_group="Seoul_target",
                  longitude=126.98, latitude=37.57, source_dataset="Korea_Air",
                  pm25_unit="ug/m3")
    nodes = pd.concat([pd.DataFrame([target]), registry], ignore_index=True)
    nodes["longitude"] = pd.to_numeric(nodes["longitude"], errors="raise")
    nodes["latitude"] = pd.to_numeric(nodes["latitude"], errors="raise")
    coords = nodes[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or np.any(np.abs(coords[:, 0]) > 90) or np.any(np.abs(coords[:, 1]) > 180):
        raise ValueError("Invalid station latitude/longitude.")
    nodes["node_index"] = np.arange(len(nodes), dtype=np.int64)

    definitions = json.loads((origin_dir / "feature_sets.json").read_text(encoding="utf-8-sig"))
    dictionary = pd.read_csv(origin_dir / "feature_dictionary.csv", dtype=str).fillna("")
    role = dictionary.set_index("name")["role"].to_dict()
    group = dictionary.set_index("name")["group"].to_dict()
    context_names = [name for name in definitions["sets"]["local_baseline"]
                     if not name.startswith("seoul_pm25_")]
    if not context_names or len(set(context_names)) != len(context_names):
        raise ValueError("Invalid ASOS/calendar context allowlist.")
    if any(role.get(name) != "feature" or group.get(name) not in {"asos_local", "calendar"}
           for name in context_names):
        raise ValueError("A target/audit/unsupported column entered global context.")
    base_columns = ["origin_time_kst", "origin_split", "feature_cutoff_time_kst"]
    label_columns = [f"{stem}_h{h}" for h in DCRNN_HORIZONS
                     for stem in ("y_pm25", "eligible", "target_time")]
    origins = pq.read_table(origin_dir / "origin_features_2021_2024_kst.parquet",
                            columns=base_columns + label_columns + context_names).to_pandas()
    origins["origin_time_kst"] = _dcrnn_kst(origins["origin_time_kst"])
    origins["feature_cutoff_time_kst"] = _dcrnn_kst(origins["feature_cutoff_time_kst"])
    origins = origins.sort_values("origin_time_kst").reset_index(drop=True)
    expected_origins = pd.date_range("2021-01-01", "2025-01-01", freq="3h",
                                     inclusive="left", tz=DCRNN_TIMEZONE).as_unit("ns")
    if not pd.DatetimeIndex(origins["origin_time_kst"]).equals(expected_origins):
        raise ValueError("Origin table is not the frozen complete 2021–2024 3-hour grid.")
    if not origins["origin_time_kst"].equals(origins["feature_cutoff_time_kst"]):
        raise ValueError("Feature cutoff differs from forecast origin.")
    expected_splits = np.where(expected_origins.year <= 2022, "train",
                               np.where(expected_origins.year == 2023, "validation", "test"))
    if not np.array_equal(origins["origin_split"].to_numpy(), expected_splits):
        raise ValueError("Origin split differs from 2021–22 / 2023 / 2024 contract.")

    hourly_index = pd.date_range(pd.Timestamp("2021-01-01", tz=DCRNN_TIMEZONE)
                                 - pd.Timedelta(hours=lookback_hours),
                                 pd.Timestamp("2025-01-01", tz=DCRNN_TIMEZONE),
                                 freq="h", inclusive="left").as_unit("ns")
    pm_values = np.full((len(hourly_index), len(nodes)), np.nan, dtype=np.float32)
    seen = np.zeros(pm_values.shape, dtype=bool)
    source_audit = []
    for source, relative_paths in DCRNN_SOURCE_FILES.items():
        node_subset = nodes.loc[nodes["source_dataset"] == source]
        id_to_node = dict(zip(node_subset["station_id"], node_subset["node_index"]))
        for relative in relative_paths:
            path = root / relative
            frame = pq.read_table(path, columns=["time_kst", "station_id", "pm25", "pm25_flag"],
                                  filters=[("station_id", "in", list(id_to_node))]).to_pandas()
            frame["time_kst"] = _dcrnn_kst(frame["time_kst"])
            frame["station_id"] = frame["station_id"].astype(str)
            time_positions = hourly_index.get_indexer(pd.DatetimeIndex(frame["time_kst"]))
            in_range = time_positions >= 0
            frame = frame.loc[in_range].reset_index(drop=True)
            time_positions = time_positions[in_range]
            if frame.duplicated(["time_kst", "station_id"]).any():
                raise ValueError(f"Duplicate station/hour within {relative}.")
            node_positions = frame["station_id"].map(id_to_node).to_numpy(dtype=np.int64)
            if seen[time_positions, node_positions].any():
                raise ValueError(f"Overlapping station/hour across source files: {relative}.")
            values = pd.to_numeric(frame["pm25"], errors="coerce").to_numpy(dtype=np.float64)
            flags = pd.to_numeric(frame["pm25_flag"], errors="coerce").to_numpy(dtype=np.float64)
            valid = (flags == 0) & np.isfinite(values) & (values >= 0)
            if np.any((flags == 0) & ~valid):
                raise ValueError(f"Flag-0 PM2.5 is nonfinite or negative: {relative}.")
            pm_values[time_positions, node_positions] = np.where(valid, values, np.nan)
            seen[time_positions, node_positions] = True
            source_audit.append({"path": relative, "rows_in_hourly_grid": int(len(frame)),
                                 "valid_pm25_rows": int(valid.sum()),
                                 "bytes": int(path.stat().st_size),
                                 "mtime_ns": int(path.stat().st_mtime_ns)})

    origin_positions = hourly_index.get_indexer(expected_origins)
    if np.any(origin_positions < lookback_hours - 1):
        raise ValueError("Insufficient padded hourly grid for requested input history.")
    y_raw = origins[[f"y_pm25_h{h}" for h in DCRNN_HORIZONS]].to_numpy(dtype=np.float64)
    eligible = origins[[f"eligible_h{h}" for h in DCRNN_HORIZONS]].to_numpy(dtype=bool)
    for column, horizon in enumerate(DCRNN_HORIZONS):
        target_times = expected_origins + pd.Timedelta(hours=horizon)
        saved_times = _dcrnn_kst(origins[f"target_time_h{horizon}"])
        if not target_times.equals(saved_times):
            raise ValueError(f"Target timestamp misalignment at h={horizon}.")
        target_positions = hourly_index.get_indexer(target_times)
        in_study = target_positions >= 0
        source_targets = np.full(len(origins), np.nan, dtype=np.float64)
        source_targets[in_study] = pm_values[target_positions[in_study], 0]
        if not np.allclose(source_targets, y_raw[:, column], equal_nan=True, rtol=0, atol=1e-6):
            raise ValueError(f"Saved labels disagree with exact raw target hours at h={horizon}.")
        target_splits = np.where(target_times.year <= 2022, "train",
                                 np.where(target_times.year == 2023, "validation", "test"))
        computed_eligible = in_study & np.isfinite(source_targets) & (target_splits == expected_splits)
        if not np.array_equal(computed_eligible, eligible[:, column]):
            raise ValueError(f"Saved eligibility disagrees with split/target checks at h={horizon}.")

    context_values = origins[context_names].to_numpy(dtype=np.float32)
    if np.isinf(context_values).any():
        raise ValueError("Infinite ASOS/calendar context value.")
    fingerprints = [{"path": str(p.relative_to(root)), "bytes": int(p.stat().st_size),
                     "mtime_ns": int(p.stat().st_mtime_ns)} for p in required]
    return {"dataset_root": root, "nodes": nodes, "origins": origins,
            "hourly_index": hourly_index, "pm_values": pm_values,
            "context_values": context_values, "context_names": context_names,
            "y_raw": y_raw, "eligible": eligible, "origin_positions": origin_positions,
            "horizons": DCRNN_HORIZONS, "lookback_hours": lookback_hours,
            "source_fingerprints": fingerprints, "source_audit": source_audit}


def fit_dcrnn_preprocessors(bundle):
    """Fit log-PM and context moments strictly on 2021–2022 observations."""
    hours = bundle["hourly_index"]
    train_hour_mask = (hours >= pd.Timestamp("2021-01-01", tz=DCRNN_TIMEZONE)) & (
        hours < pd.Timestamp("2023-01-01", tz=DCRNN_TIMEZONE))
    train_raw = np.asarray(bundle["pm_values"][train_hour_mask], dtype=np.float64)
    train_log = np.log1p(train_raw)
    pm_mean, pm_std, pm_count = _dcrnn_moments(train_log)
    train_origins = bundle["origins"]["origin_split"].eq("train").to_numpy()
    context_mean, context_std, context_count = _dcrnn_moments(
        bundle["context_values"][train_origins])
    if pm_count[0] == 0:
        raise ValueError("No training observations for the target Junggu station.")
    return {"fit_start_inclusive": "2021-01-01T00:00:00+09:00",
            "fit_end_exclusive": "2023-01-01T00:00:00+09:00",
            "pm_mean": pm_mean.tolist(), "pm_std": pm_std.tolist(),
            "pm_valid_count": pm_count.tolist(),
            "context_mean": context_mean.tolist(), "context_std": context_std.tolist(),
            "context_valid_count": context_count.tolist(),
            "node_order": bundle["nodes"]["station_key"].tolist(),
            "context_names": list(bundle["context_names"]),
            "transform": "log1p(PM2.5); per-node train-only mean/std; missing normalized value=0, mask=1",
            "std_ddof": 0, "minimum_std": 1e-8}


def transform_dcrnn_bundle(bundle, preprocessors):
    """Apply frozen preprocessing; keep raw labels and eligibility separately."""
    if preprocessors["node_order"] != bundle["nodes"]["station_key"].tolist():
        raise ValueError("Normalizer node order differs from dataset node order.")
    if preprocessors["context_names"] != bundle["context_names"]:
        raise ValueError("Normalizer context order differs from allowlist.")
    pm_mean = np.asarray(preprocessors["pm_mean"], dtype=np.float64)
    pm_std = np.asarray(preprocessors["pm_std"], dtype=np.float64)
    context_mean = np.asarray(preprocessors["context_mean"], dtype=np.float64)
    context_std = np.asarray(preprocessors["context_std"], dtype=np.float64)
    if not (np.isfinite(pm_mean).all() and np.isfinite(pm_std).all()
            and np.isfinite(context_mean).all() and np.isfinite(context_std).all()
            and (pm_std > 0).all() and (context_std > 0).all()):
        raise ValueError("Invalid normalizer moments/scales.")
    raw_pm = bundle["pm_values"]
    pm_missing = ~np.isfinite(raw_pm)
    pm_z = (np.log1p(raw_pm.astype(np.float64)) - pm_mean[None, :]) / pm_std[None, :]
    pm_z = np.where(pm_missing, 0.0, pm_z).astype(np.float32)
    hourly_x = np.stack([pm_z, pm_missing.astype(np.float32)], axis=-1)
    context = bundle["context_values"].astype(np.float64)
    context_missing = ~np.isfinite(context)
    context_z = (context - context_mean[None, :]) / context_std[None, :]
    context_z = np.where(context_missing, 0.0, context_z).astype(np.float32)
    context_x = np.concatenate([context_z, context_missing.astype(np.float32)], axis=-1)
    y_z = (np.log1p(bundle["y_raw"]) - pm_mean[0]) / pm_std[0]
    y_z = np.where(np.isfinite(y_z), y_z, 0.0).astype(np.float32)
    if not (np.isfinite(hourly_x).all() and np.isfinite(context_x).all()
            and np.isfinite(y_z).all()):
        raise ValueError("Transformed tensors contain nonfinite values.")
    return {"hourly_x": hourly_x, "context_x": context_x, "y_z": y_z,
            "y_raw": bundle["y_raw"], "eligible": bundle["eligible"]}


def split_origin_indices(bundle, split, require_any_label=False):
    """Return fixed origin IDs; evaluation never drops input-missing cases."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    # Arrow-backed Pandas may expose a read-only NumPy view.
    mask = bundle["origins"]["origin_split"].eq(split).to_numpy().copy()
    if require_any_label:
        mask &= bundle["eligible"].any(axis=1)
    return np.flatnonzero(mask)


class DCRNNOriginDataset(Dataset):
    """Lazy exact-hour windows ending at t, with four horizon-specific masks."""
    def __init__(self, bundle, arrays, indices, lookback_hours=48):
        self.bundle = bundle
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lookback_hours = int(lookback_hours)
        if self.lookback_hours < 1 or self.lookback_hours > bundle["lookback_hours"]:
            raise ValueError("Requested window exceeds the loaded padded history.")
        if np.any(self.indices < 0) or np.any(self.indices >= len(bundle["origins"])):
            raise IndexError("Origin index outside frozen grid.")
        positions = bundle["origin_positions"][self.indices]
        if np.any(positions < self.lookback_hours - 1):
            raise ValueError("An origin lacks a complete regular-hour index window.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        origin_index = int(self.indices[item])
        end = int(self.bundle["origin_positions"][origin_index])
        start = end - self.lookback_hours + 1
        x = self.arrays["hourly_x"][start:end + 1]
        if x.shape[0] != self.lookback_hours:
            raise RuntimeError("Input window was truncated.")
        return (torch.from_numpy(x),
                torch.from_numpy(self.arrays["context_x"][origin_index]),
                torch.from_numpy(self.arrays["y_z"][origin_index]),
                torch.from_numpy(self.arrays["eligible"][origin_index].copy()),
                torch.tensor(origin_index, dtype=torch.long))


def dcrnn_bundle_summary(bundle):
    """Small audit summary suitable for notebook display and JSON output."""
    counts = {}
    for split in ("train", "validation", "test"):
        ids = split_origin_indices(bundle, split)
        counts[split] = {"origins": int(len(ids)), "eligible_by_horizon": {
            str(h): int(bundle["eligible"][ids, j].sum())
            for j, h in enumerate(DCRNN_HORIZONS)}}
    return {"nodes": int(len(bundle["nodes"])),
            "target_node": bundle["nodes"].iloc[0]["station_key"],
            "origin_rows": int(len(bundle["origins"])),
            "hourly_rows_including_padding": int(len(bundle["hourly_index"])),
            "context_features_before_mask": int(len(bundle["context_names"])),
            "splits": counts, "source_audit": bundle["source_audit"],
            "input_window": f"{bundle['lookback_hours']} hourly values ending at t",
            "future_observations_in_input": False}


def test_dcrnn_data_contract(bundle, arrays, preprocessors):
    """Check exact windows, train-only fitting, missing masks, and causality.

    Future observations/context are changed in a separate copy. Preprocessing
    must remain unchanged, and the chosen validation-origin input must remain
    identical. Targets are intentionally not involved in this input check.
    """
    valid_pm = np.isfinite(bundle["pm_values"])
    if not np.array_equal(arrays["hourly_x"][..., 1].astype(bool), ~valid_pm):
        raise AssertionError("PM missing channel disagrees with raw observations.")
    if not np.all(arrays["hourly_x"][..., 0][~valid_pm] == 0):
        raise AssertionError("Missing normalized PM must be represented by zero.")
    zero_pm = valid_pm & (bundle["pm_values"] == 0)
    if not np.all(arrays["hourly_x"][..., 1][zero_pm] == 0):
        raise AssertionError("A valid zero observation was marked missing.")
    if not np.isfinite(bundle["y_raw"][bundle["eligible"]]).all():
        raise AssertionError("Eligible label contains nonfinite values.")
    validation_ids = split_origin_indices(bundle, "validation")
    chosen = int(validation_ids[min(32, len(validation_ids) - 1)])
    end = int(bundle["origin_positions"][chosen])
    start = end - bundle["lookback_hours"] + 1
    origin = bundle["origins"].iloc[chosen]["origin_time_kst"]
    if bundle["hourly_index"][end] != origin:
        raise AssertionError("Input window does not end exactly at origin t.")
    if bundle["hourly_index"][start] != origin - pd.Timedelta(hours=bundle["lookback_hours"] - 1):
        raise AssertionError("Input history is not an exact hourly window.")
    changed = dict(bundle)
    changed["pm_values"] = bundle["pm_values"].copy()
    changed["context_values"] = bundle["context_values"].copy()
    changed["pm_values"][end + 1:] = 9999.0
    changed["context_values"][chosen + 1:] = -9999.0
    refit = fit_dcrnn_preprocessors(changed)
    for name in ("pm_mean", "pm_std", "context_mean", "context_std"):
        if not np.array_equal(np.asarray(preprocessors[name]), np.asarray(refit[name])):
            raise AssertionError(f"Future values changed train-only moments: {name}")
    changed_arrays = transform_dcrnn_bundle(changed, preprocessors)
    if not np.array_equal(arrays["hourly_x"][start:end + 1],
                          changed_arrays["hourly_x"][start:end + 1]):
        raise AssertionError("Future observation changed an input history window.")
    if not np.array_equal(arrays["context_x"][chosen], changed_arrays["context_x"][chosen]):
        raise AssertionError("Future context changed the chosen current context.")
    return {"exact_hourly_window": True, "missing_mask": True,
            "valid_zero_preserved": True, "eligible_labels_finite": True,
            "train_only_normalization": True, "post_origin_mutation_invariant": True,
            "checked_origin_kst": origin.isoformat(),
            "valid_zero_observations": int(zero_pm.sum())}