"""Fixed graphs for the Seoul-target DCRNN experiment.

Convention: A[i, j] is an edge FROM source i TO destination j.  The model
constructs its own forward/reverse diffusion operators and self loops.

The trajectory graph is a training-period, Seoul-arrival-conditional graph of
ordered geographical nearest-station assignments.  It is NOT a measured PM
transport probability, emission contribution, or causal network.
"""
from collections import deque
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


GRAPH_KST = "Asia/Seoul"
GRAPH_TARGET_KEY = "KR:111121"


def _graph_timestamp(value):
    result = pd.Timestamp(value)
    return result.tz_localize(GRAPH_KST) if result.tzinfo is None else result.tz_convert(GRAPH_KST)


def _graph_nodes(nodes):
    required = {"station_key", "latitude", "longitude"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"Node table is missing: {sorted(required - set(nodes.columns))}")
    result = nodes.reset_index(drop=True).copy()
    if result.empty or result.station_key.astype(str).duplicated().any():
        raise ValueError("Node identities must be nonempty and unique.")
    if str(result.iloc[0].station_key) != GRAPH_TARGET_KEY:
        raise ValueError("Node zero must be Seoul Junggu KR:111121.")
    result["station_key"] = result.station_key.astype(str)
    for field, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
        result[field] = pd.to_numeric(result[field], errors="raise")
        if not np.isfinite(result[field]).all() or not result[field].between(lower, upper).all():
            raise ValueError(f"Invalid {field} in node table.")
    return result


def _graph_training_rows(frame, train_start, train_end):
    """Half-open training-arrival filter, repeated after Parquet pushdown."""
    start, end = _graph_timestamp(train_start), _graph_timestamp(train_end)
    if start >= end:
        raise ValueError("train_start must precede train_end.")
    result = frame.copy()
    result["arrival_time_kst"] = pd.to_datetime(result["arrival_time_kst"], utc=True).dt.tz_convert(GRAPH_KST)
    return result.loc[(result.arrival_time_kst >= start) & (result.arrival_time_kst < end)].copy()


def _graph_haversine_matrix(nodes):
    lat = np.deg2rad(nodes.latitude.to_numpy(dtype=np.float64))
    lon = np.deg2rad(nodes.longitude.to_numpy(dtype=np.float64))
    a = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
    a += np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(distance, 0.0)
    return distance


def graph_diagnostics(adjacency, nodes):
    """Report directed connectivity before the model adds any self loops."""
    nodes = _graph_nodes(nodes)
    a = np.asarray(adjacency, dtype=np.float64)
    n = len(nodes)
    if a.shape != (n, n) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Adjacency must be a finite, nonnegative N by N matrix.")
    links = a > 0
    np.fill_diagonal(links, False)
    hops = np.full(n, -1, dtype=int)
    hops[0] = 0
    queue = deque([0])
    while queue:
        destination = queue.popleft()
        for source in np.flatnonzero(links[:, destination]):
            if hops[source] < 0:
                hops[source] = hops[destination] + 1
                queue.append(int(source))
    return {
        "nodes": n,
        "edges": int(links.sum()),
        "orientation": "A[source, destination]",
        "self_loops_in_saved_graph": int(np.count_nonzero(np.diag(a))),
        "seoul_in_degree": int(links[:, 0].sum()),
        "seoul_out_degree": int(links[0].sum()),
        "isolated_nodes": nodes.loc[~(links.any(axis=0) | links.any(axis=1)), "station_key"].tolist(),
        "cannot_reach_seoul_count": int((hops < 0).sum()),
        "cannot_reach_seoul": nodes.loc[hops < 0, "station_key"].tolist(),
        "cn_nodes": int(nodes.station_key.str.startswith("CN:").sum()),
        "cn_nodes_cannot_reach_seoul": int(((hops < 0) & nodes.station_key.str.startswith("CN:").to_numpy()).sum()),
        "within_two_hops_to_seoul": int(((hops > 0) & (hops <= 2)).sum()),
        "hops_to_seoul": {key: (None if h < 0 else int(h)) for key, h in zip(nodes.station_key, hops)},
        "adjacency_sha256": hashlib.sha256(np.asarray(adjacency, dtype="<f4").tobytes()).hexdigest(),
    }


def distance_adjacency(nodes, radius_km=500.0, sigma_km=None):
    """Symmetric Gaussian distance graph; only metadata coordinates are used."""
    nodes = _graph_nodes(nodes)
    radius_km = float(radius_km)
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("radius_km must be finite and positive.")
    distance = _graph_haversine_matrix(nodes)
    near = (distance > 0) & (distance <= radius_km)
    if sigma_km is None:
        if not near.any():
            raise ValueError("No positive-distance node pair falls within the radius.")
        sigma_km = float(np.median(distance[near]))
    sigma_km = float(sigma_km)
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        raise ValueError("sigma_km must be finite and positive.")
    a = np.where(near, np.exp(-(distance / sigma_km) ** 2), 0).astype(np.float32)
    np.fill_diagonal(a, 0)
    info = {
        "kind": "distance", "radius_km": radius_km, "sigma_km": sigma_km,
        "sigma_rule": "median of positive coordinate distances within radius, unless supplied",
        "used_pollution_or_test_labels": False,
        "diagnostics": graph_diagnostics(a, nodes),
    }
    return a, info


def nearest_trajectory_assignments(neighbors, nodes, radius_km=75.0):
    """One geographical nearest node per timestamp; simultaneous candidates never become sequential visits."""
    nodes = _graph_nodes(nodes)
    required = {"point_id", "arrival_time_kst", "lookback_hours", "station_key", "distance_km"}
    if not required.issubset(neighbors.columns):
        raise ValueError(f"Neighbor columns missing: {sorted(required - set(neighbors.columns))}")
    frame = neighbors.loc[:, sorted(required)].copy()
    lookup = {key: i for i, key in enumerate(nodes.station_key)}
    frame["station_key"] = frame.station_key.astype(str)
    if not set(frame.station_key).issubset(lookup):
        raise ValueError("The matching table contains a station absent from the node registry.")
    frame["distance_km"] = pd.to_numeric(frame.distance_km, errors="raise")
    if not np.isfinite(frame.distance_km).all() or (frame.distance_km < 0).any():
        raise ValueError("Invalid geographical matching distance.")
    frame = frame.loc[frame.distance_km <= float(radius_km) + 1e-9]
    frame = frame.sort_values(["distance_km", "station_key"], kind="mergesort").drop_duplicates("point_id", keep="first")
    frame["station_index"] = frame.station_key.map(lookup).astype(np.int64)
    if frame.duplicated(["arrival_time_kst", "lookback_hours"]).any():
        raise ValueError("More than one selected station for a trajectory timestamp.")
    return frame[["point_id", "arrival_time_kst", "lookback_hours", "station_index"]].reset_index(drop=True)


def trajectory_adjacency_from_assignments(assignments, nodes, all_arrivals=None,
                                         max_gap_hours=6.0, min_age=1, max_age=48,
                                         bridge_unmatched=True):
    """Count ordered geographical visits once per arrival with an explicit observation-gap policy.

    `assignments` has at most one station_index per arrival/age, independent of
    pollution validity.  all_arrivals must also include entirely unmatched paths.
    A[source, destination] = paths with that ordered transition / all paths.
    Age 0 is explicitly appended as Seoul node 0. With bridge_unmatched=True,
    unmatched geographical points (for example, open sea) are skipped, but an
    edge can span only max_gap_hours of actual elapsed time. This is a declared
    co-passage-order rule across a monitoring gap, NOT concentration imputation
    or confirmation of physical pollutant movement. With False, every unmatched
    hourly point breaks continuity; the latter is retained as a diagnostic.
    """
    nodes = _graph_nodes(nodes)
    if min_age != 1 or int(max_age) < min_age:
        raise ValueError("This graph contract requires ages 1..max_age and explicit Seoul age 0.")
    if not np.isfinite(float(max_gap_hours)) or float(max_gap_hours) <= 0:
        raise ValueError("max_gap_hours must be positive.")
    required = {"arrival_time_kst", "lookback_hours", "station_index"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Assignment columns missing: {sorted(required - set(assignments.columns))}")
    frame = assignments.loc[:, sorted(required)].copy()
    frame["arrival_time_kst"] = pd.to_datetime(frame.arrival_time_kst, utc=True).dt.tz_convert(GRAPH_KST)
    age_values = pd.to_numeric(frame.lookback_hours, errors="raise").to_numpy(dtype=float)
    station_values = pd.to_numeric(frame.station_index, errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(age_values).all() or not np.equal(age_values, np.floor(age_values)).all()
            or np.any((age_values < min_age) | (age_values > max_age))):
        raise ValueError("Assignment ages must be exact integer hours in 1..max_age.")
    if (not np.isfinite(station_values).all() or not np.equal(station_values, np.floor(station_values)).all()
            or np.any((station_values < 1) | (station_values >= len(nodes)))):
        raise ValueError("Historical assignments must be external nodes; Seoul is appended at age 0 only.")
    frame["lookback_hours"] = age_values.astype(np.int64)
    frame["station_index"] = station_values.astype(np.int64)
    if frame.duplicated(["arrival_time_kst", "lookback_hours"]).any():
        raise ValueError("Duplicate assignments would invent within-time transitions.")
    if all_arrivals is None:
        raise ValueError("Pass all_arrivals explicitly so entirely unmatched paths remain in the denominator.")
    arrivals = pd.DatetimeIndex(pd.to_datetime(list(all_arrivals), utc=True)).tz_convert(GRAPH_KST)
    if arrivals.has_duplicates:
        raise ValueError("all_arrivals must contain unique timestamps.")
    arrivals = arrivals.sort_values()
    if len(arrivals) == 0:
        raise ValueError("No training arrivals available to form a graph.")
    row = arrivals.get_indexer(pd.DatetimeIndex(frame.arrival_time_kst))
    if np.any(row < 0):
        raise ValueError("An assignment arrival is not present in all_arrivals.")
    visits = np.full((len(arrivals), int(max_age)), -1, dtype=np.int32)
    visits[row, int(max_age) - frame.lookback_hours.to_numpy()] = frame.station_index.to_numpy()
    counts = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    paths_with_edges = 0
    ordered_pair_occurrences = 0
    cross_gap_occurrences = 0
    over_max_gap_occurrences = 0
    accepted_gaps = []
    for stations in visits:
        pairs = set()
        previous = None
        previous_age = None
        # Columns represent ages max_age ... 1, followed by known receptor age 0.
        for column in range(int(max_age) + 1):
            age = int(max_age) - column
            station = 0 if age == 0 else int(stations[column])
            if station < 0:
                if not bridge_unmatched:
                    previous = previous_age = None
                continue
            if previous is not None and station != previous:
                gap = previous_age - age
                if 0 < gap <= float(max_gap_hours):
                    pairs.add((int(previous), station))
                    ordered_pair_occurrences += 1
                    accepted_gaps.append(float(gap))
                    cross_gap_occurrences += int(gap > 1)
                elif gap > float(max_gap_hours):
                    over_max_gap_occurrences += 1
            previous, previous_age = station, age
        if pairs:
            paths_with_edges += 1
            for source, destination in pairs:
                counts[source, destination] += 1
    np.fill_diagonal(counts, 0)
    a = (counts.astype(np.float64) / len(arrivals)).astype(np.float32)
    top = sorted(zip(*np.nonzero(counts)), key=lambda pair: (-counts[pair], pair))[:20]
    info = {
        "kind": "trajectory_directed", "n_train_arrivals": len(arrivals),
        "arrival_min": arrivals.min().isoformat(), "arrival_max": arrivals.max().isoformat(),
        "lookback_hours": [int(min_age), int(max_age)],
        "max_gap_hours": float(max_gap_hours),
        "bridge_unmatched": bool(bridge_unmatched),
        "unmatched_hour_rule": ("skip unmatched geographical points; connect ordered nearest visits only within max_gap_hours"
                                if bridge_unmatched else "break; do not connect across an unmatched point"),
        "assignment_rule": "nearest horizontal candidate per hour, ties by station_key; PM values not consulted",
        "terminal_rule": "append receptor KR:111121 as node 0 at age 0",
        "repeat_rule": "each ordered pair counts at most once per trajectory",
        "paths_with_edges": paths_with_edges,
        "ordered_transition_occurrences_before_path_dedup": ordered_pair_occurrences,
        "cross_gap_transition_occurrences_before_path_dedup": cross_gap_occurrences,
        "over_max_gap_transition_occurrences": over_max_gap_occurrences,
        "accepted_transition_gap_hours": ({"min": float(np.min(accepted_gaps)),
                                           "median": float(np.median(accepted_gaps)),
                                           "p95": float(np.percentile(accepted_gaps, 95)),
                                           "max": float(np.max(accepted_gaps))} if accepted_gaps else None),
        "sum_of_path_pair_counts": int(counts.sum()),
        "weight_definition": "number of training arrivals with ordered transition / all training arrivals",
        "interpretation": "Seoul-arrival-conditional geographical co-passage order, not physical PM transport probability",
        "top_edges": [{"source": nodes.station_key.iloc[i], "destination": nodes.station_key.iloc[j],
                       "path_count": int(counts[i, j]), "weight": float(a[i, j])} for i, j in top],
        "diagnostics": graph_diagnostics(a, nodes),
    }
    return a, info


def trajectory_adjacency(dataset_root, nodes, train_start="2021-01-01", train_end="2023-01-01",
                         radius_km=75.0, max_gap_hours=6.0, bridge_unmatched=True):
    """Build the fixed graph using training arrivals only; no validation/test trajectories or PM labels."""
    import pyarrow.parquet as pq
    root = Path(dataset_root)
    if not np.isfinite(float(radius_km)) or float(radius_km) <= 0:
        raise ValueError("radius_km must be finite and positive.")
    if (root / "Dataset").is_dir():
        root = root / "Dataset"
    folder = root / "Trajectory_Matching_2021_2024"
    point_path = folder / "trajectory_point_coverage.parquet"
    edge_path = folder / "trajectory_station_neighbors.parquet"
    if not point_path.is_file() or not edge_path.is_file():
        raise FileNotFoundError("Trajectory_Matching_2021_2024 point coverage and station neighbors are required for trajectory graphs.")
    start, end = _graph_timestamp(train_start), _graph_timestamp(train_end)
    manifest_path = folder / "matching_manifest.json"
    declared_radius = 75.0
    if manifest_path.is_file():
        declared_radius = float(json.loads(manifest_path.read_text(encoding="utf-8-sig"))["parameters"]["radius_km"])
    if float(radius_km) > declared_radius + 1e-9:
        raise ValueError(f"Saved matching supports at most {declared_radius:g} km; do not relabel it as {radius_km:g} km.")
    filters = [("arrival_time_kst", ">=", start.to_pydatetime()),
               ("arrival_time_kst", "<", end.to_pydatetime())]
    points = pq.read_table(point_path, columns=["point_id", "arrival_time_kst", "lookback_hours"], filters=filters).to_pandas()
    edges = pq.read_table(edge_path, columns=["point_id", "arrival_time_kst", "lookback_hours", "station_key", "distance_km"], filters=filters).to_pandas()
    points = _graph_training_rows(points, start, end)
    edges = _graph_training_rows(edges, start, end)
    if points.empty or points.point_id.duplicated().any():
        raise ValueError("Training trajectory points are empty or have duplicate identities.")
    ages = pd.to_numeric(points.lookback_hours, errors="raise").to_numpy()
    if not np.isin(ages, np.arange(1, 49)).all():
        raise ValueError("Expected 48 hourly past points, ages 1..48.")
    per_arrival = points.groupby("arrival_time_kst").lookback_hours.agg(["count", "nunique"])
    if not ((per_arrival["count"] == 48) & (per_arrival["nunique"] == 48)).all():
        raise ValueError("Each training arrival must have exactly ages 1..48.")
    reference = points[["point_id", "arrival_time_kst", "lookback_hours"]].rename(
        columns={"arrival_time_kst": "point_arrival", "lookback_hours": "point_age"})
    edge_check = edges.merge(reference, on="point_id", how="left", validate="many_to_one")
    if edge_check.point_arrival.isna().any() or not (edge_check.arrival_time_kst == edge_check.point_arrival).all() or not (edge_check.lookback_hours == edge_check.point_age).all():
        raise ValueError("Neighbor point identity/time disagrees with trajectory coverage.")
    if not np.isfinite(edges.distance_km).all() or (edges.distance_km > declared_radius + 1e-9).any():
        raise ValueError("Saved neighbor distances exceed the declared matching radius.")
    assignments = nearest_trajectory_assignments(edges, nodes, radius_km=radius_km)
    a, info = trajectory_adjacency_from_assignments(assignments, nodes,
        all_arrivals=per_arrival.index, max_gap_hours=max_gap_hours,
        bridge_unmatched=bridge_unmatched)
    if info["diagnostics"]["seoul_in_degree"] == 0:
        raise ValueError("No directed edge enters Seoul. Check age-0 receptor and chronological ordering.")
    info.update({"train_start_inclusive": start.isoformat(), "train_end_exclusive": end.isoformat(),
                 "matching_radius_km": float(radius_km), "saved_matching_radius_km": declared_radius,
                 "matching_radius_manifest_verified": manifest_path.is_file(),
                 "source_point_rows": int(len(points)), "source_neighbor_rows": int(len(edges)),
                 "selected_geographical_points": int(len(assignments)),
                 "source_files": [str(point_path), str(edge_path)], "used_validation_or_test_arrivals": False,
                 "used_pollution_values_or_future_targets": False})
    return a, info


def symmetric_trajectory_adjacency(adjacency, nodes):
    """Matched graph control: same learned edge weights, with orientation removed."""
    a = np.asarray(adjacency, dtype=np.float32)
    symmetric = ((a + a.T) / 2).astype(np.float32)
    np.fill_diagonal(symmetric, 0)
    return symmetric, {"kind": "trajectory_symmetric", "definition": "(A + A.T)/2 before model normalization",
                       "diagnostics": graph_diagnostics(symmetric, nodes)}


def graph_contract_tests():
    """Fast synthetic checks; no local data or PyArrow installation needed."""
    nodes = pd.DataFrame({"station_key": [GRAPH_TARGET_KEY, "KR:A", "CN:B", "KR:unused"],
                          "latitude": [37.57, 37.58, 37.59, 40.0],
                          "longitude": [126.98, 126.99, 127.0, 130.0]})
    t = _graph_timestamp("2021-01-01")
    def assignment(ages, stations):
        return pd.DataFrame({"arrival_time_kst": [t] * len(ages), "lookback_hours": ages,
                             "station_index": stations})
    # Oldest node A -> recent node B -> receptor, with residence deduplicated.
    a, info = trajectory_adjacency_from_assignments(assignment([4, 3, 2, 1], [1, 1, 2, 2]), nodes, [t], max_age=4)
    assert a[1, 2] == 1 and a[2, 0] == 1 and a[2, 1] == 0 and a[0, 2] == 0
    assert np.count_nonzero(a) == 2 and np.all(np.diag(a) == 0)
    # Strict diagnostic: missing age 2 breaks an A-to-B jump.
    gap, _ = trajectory_adjacency_from_assignments(assignment([3, 1], [1, 2]), nodes, [t], max_age=4, bridge_unmatched=False)
    assert gap[1, 2] == 0 and gap[2, 0] == 1
    # Strict diagnostic: missing age 1 does not connect a receptor edge from age 2.
    terminal, _ = trajectory_adjacency_from_assignments(assignment([2], [1]), nodes, [t], max_age=4, bridge_unmatched=False)
    assert terminal[:, 0].sum() == 0
    # Main rule: a two-hour geographical gap can be bridged, with its elapsed time reported.
    bridged, bridge_info = trajectory_adjacency_from_assignments(assignment([3, 1], [1, 2]), nodes, [t], max_age=4, max_gap_hours=2, bridge_unmatched=True)
    assert bridged[1, 2] == 1 and bridged[2, 0] == 1
    assert bridge_info["cross_gap_transition_occurrences_before_path_dedup"] == 1
    assert bridge_info["accepted_transition_gap_hours"]["max"] == 2
    blocked, _ = trajectory_adjacency_from_assignments(assignment([3, 1], [1, 2]), nodes, [t], max_age=4, max_gap_hours=1, bridge_unmatched=True)
    assert blocked[1, 2] == 0 and blocked[2, 0] == 1
    # Entirely unmatched paths stay in the denominator.
    half, _ = trajectory_adjacency_from_assignments(assignment([2, 1], [1, 2]), nodes,
                                                   [t, t + pd.Timedelta(hours=3)], max_age=4)
    assert half[1, 2] == 0.5 and half[2, 0] == 0.5
    repeats, _ = trajectory_adjacency_from_assignments(assignment([4, 3, 2, 1], [1, 2, 1, 2]), nodes, [t], max_age=4)
    assert repeats[1, 2] == 1 and repeats[2, 1] == 1
    # A point with two geographical neighbors creates only one assignment.
    neighbors = pd.DataFrame({"point_id": ["p", "p"], "arrival_time_kst": [t, t],
        "lookback_hours": [1, 1], "station_key": ["CN:B", "KR:A"], "distance_km": [10., 10.]})
    nearest = nearest_trajectory_assignments(neighbors, nodes)
    assert len(nearest) == 1 and nearest.station_index.iloc[0] == 2  # alphabetical tie-break CN:B
    sym, _ = symmetric_trajectory_adjacency(a, nodes)
    assert np.array_equal(sym, sym.T) and sym[1, 2] == 0.5
    distances, _ = distance_adjacency(nodes, radius_km=10)
    assert np.allclose(distances, distances.T) and np.all(np.diag(distances) == 0)
    boundary = pd.DataFrame({"arrival_time_kst": [t, _graph_timestamp("2022-12-31 21:00"), _graph_timestamp("2023-01-01"), _graph_timestamp("2024-01-01")]})
    selected = _graph_training_rows(boundary, "2021-01-01", "2023-01-01")
    assert len(selected) == 2
    perturbed = pd.concat([boundary, pd.DataFrame({"arrival_time_kst": [_graph_timestamp("2024-12-31")]})], ignore_index=True)
    assert selected.reset_index(drop=True).equals(_graph_training_rows(perturbed, "2021-01-01", "2023-01-01").reset_index(drop=True))
    diagnostics = graph_diagnostics(a, nodes)
    assert diagnostics["hops_to_seoul"]["KR:A"] == 2 and diagnostics["cannot_reach_seoul_count"] == 1
    return {"chronological_direction": True, "explicit_receptor": True, "no_same_time_candidate_edges": True,
            "strict_missing_point_breaks": True, "declared_gap_bridge_and_limit": True,
            "path_pair_deduplication": True,
            "unmatched_paths_in_denominator": True, "training_boundary_and_future_invariance": True,
            "symmetry_control": True, "distance_graph": True, "directed_reachability": True}