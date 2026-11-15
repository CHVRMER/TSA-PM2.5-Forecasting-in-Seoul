"""Country-safe trajectory features rebuilt from selected raw hourly observations.

The saved matching file supplies geometry only. Its cached pollutant values are
never used as model inputs: PM2.5 is rejoined from the current 635-node raw panel
at the exact historical pass time. No targets, future arrivals, gases, PM10, or
unselected Japanese candidates enter these features.
"""
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = 'gwn-country-safe-trajectory-v1.0.0'
COUNTRIES = ('KR', 'CN', 'JP', 'MN')
BANDS = ((1, 6, 'b01_06'), (7, 12, 'b07_12'), (13, 24, 'b13_24'), (25, 48, 'b25_48'))
PM_STATS = ('mean', 'std', 'max', 'inside_mix_mean', 'above_mix_mean',
            'coverage', 'valid_points', 'mean_station_count', 'unique_stations',
            'inside_mix_coverage', 'inside_mix_valid_points',
            'above_mix_coverage', 'above_mix_valid_points')
NEUTRAL_KEYS = {'route_band', 'route_endpoint', 'route_total', 'traj_met'}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_hash(values):
    a = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def origin_index(bundle):
    return pd.DatetimeIndex(bundle['origins'].origin_time_kst).tz_convert('Asia/Seoul').as_unit('ns')


def neutral_feature_names(project_root):
    folder = Path(project_root) / 'Dataset/Origin_Features_2021_2024_v3'
    d = pd.read_csv(folder / 'feature_dictionary.csv', dtype=str, keep_default_na=False)
    selected = d.loc[d.role.eq('feature') & d.definition_key.isin(NEUTRAL_KEYS)]
    if len(selected) != 114 or selected.name.duplicated().any():
        raise ValueError('Expected the unchanged 30 route + 84 meteorological neutral features.')
    if set(selected['group']) != {'trajectory_route', 'trajectory_weather'}:
        raise ValueError('Observation-dependent features entered the neutral trajectory context.')
    return selected.name.tolist()


def pm_feature_layout():
    return [(f'traj_pm25_{country.lower()}_{band}_{stat}', country, lo, hi, band, stat)
            for country in COUNTRIES for lo, hi, band in BANDS for stat in PM_STATS]


def feature_layout(project_root):
    neutral = neutral_feature_names(project_root)
    pm = pm_feature_layout()
    names = neutral + [x[0] for x in pm]
    countries = ['NEUTRAL'] * len(neutral) + [x[1] for x in pm]
    if len(names) != len(set(names)) or len(names) != 322:
        raise ValueError('The fixed trajectory context must have 322 unique columns.')
    return names, countries


def read_points(matching_folder, origins):
    """Read all 48 geometric points even when no monitoring station is nearby."""
    origins = pd.DatetimeIndex(origins).as_unit('ns')
    columns = ['point_id', 'arrival_time_kst', 'pass_time_kst', 'lookback_hours',
               'height_agl_m', 'diagnostic_mixdepth']
    frame = pd.read_parquet(Path(matching_folder) / 'trajectory_point_coverage.parquet',
                            columns=columns,
                            filters=[('arrival_time_kst', '>=', origins.min().to_pydatetime()),
                                     ('arrival_time_kst', '<=', origins.max().to_pydatetime())])
    frame['arrival_time_kst'] = pd.DatetimeIndex(frame.arrival_time_kst).as_unit('ns')
    frame['pass_time_kst'] = pd.DatetimeIndex(frame.pass_time_kst).as_unit('ns')
    frame = frame.loc[frame.arrival_time_kst.isin(origins)].copy()
    frame = frame.sort_values(['arrival_time_kst', 'lookback_hours']).reset_index(drop=True)
    per = frame.groupby('arrival_time_kst').lookback_hours.agg(['size', 'nunique'])
    if len(per) != len(origins) or not per.eq(48).all().all() or frame.point_id.duplicated().any():
        raise ValueError('Every selected arrival must have 48 unique trajectory points.')
    if not np.isin(frame.lookback_hours, np.arange(1, 49)).all():
        raise ValueError('Trajectory ages outside the fixed 1-48h range.')
    expected = frame.arrival_time_kst - pd.to_timedelta(frame.lookback_hours, unit='h')
    if not frame.pass_time_kst.eq(expected).all():
        raise ValueError('Trajectory pass time differs from arrival minus age.')
    return frame


def neighbor_batches(matching_folder, origins, batch_size=250_000):
    columns = ['point_id', 'arrival_time_kst', 'lookback_hours', 'pass_time_kst',
               'station_key', 'distance_km', 'pm25']
    # Arrow dataset filtering avoids materialising the full 16.8m-neighbor file for a pilot.
    import pyarrow.dataset as ds
    origins = pd.DatetimeIndex(origins)
    filt = (ds.field('arrival_time_kst') >= origins.min().to_pydatetime()) & (
            ds.field('arrival_time_kst') <= origins.max().to_pydatetime())
    dataset = ds.dataset(Path(matching_folder) / 'trajectory_station_neighbors.parquet', format='parquet')
    for batch in dataset.scanner(columns=columns, filter=filt, batch_size=batch_size).to_batches():
        yield batch.to_pandas()


def aggregate_pm_features(bundle, points, batches, origins=None, radius_km=75., sigma_km=50.):
    """Pure historical aggregation; accepts DataFrame batches for meaningful tests.

    Point denominators are fixed by geometric ages, not by observation count.
    Availability is rederived only from selected nodes and current raw values.
    """
    if float(radius_km) != 75.:
        raise ValueError('The saved matching pool is 75km. 100km requires rematching, not a larger filter.')
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        raise ValueError('trajectory_sigma_km must be finite and positive.')
    origins = origin_index(bundle) if origins is None else pd.DatetimeIndex(origins).as_unit('ns')
    keys = bundle['nodes'].station_key.astype(str).tolist()
    nodes = pd.Index(keys)
    countries = bundle['nodes'].country.astype(str).to_numpy()
    hour_index = pd.DatetimeIndex(bundle['hourly_index']).as_unit('ns')
    raw = np.asarray(bundle['pm_values'])
    if raw.shape != (len(hour_index), len(nodes)):
        raise ValueError('Raw hourly PM matrix has the wrong shape.')
    p = points.copy().sort_values(['arrival_time_kst', 'lookback_hours']).reset_index(drop=True)
    point_index = pd.Index(p.point_id.astype(str))
    if not point_index.is_unique:
        raise ValueError('Duplicate geometric trajectory points.')
    arrival_positions = origins.get_indexer(pd.DatetimeIndex(p.arrival_time_kst).as_unit('ns'))
    if (arrival_positions < 0).any():
        raise ValueError('Point arrival is outside the requested origin grid.')
    ages = p.lookback_hours.to_numpy(dtype=int)
    if not np.isin(ages, np.arange(1, 49)).all():
        raise ValueError('Invalid trajectory age.')
    band_positions = np.searchsorted(np.array([6, 12, 24, 48]), ages, side='left')
    origin_band = arrival_positions * 4 + band_positions
    per = np.bincount(origin_band, minlength=len(origins) * 4).reshape(len(origins), 4)
    if not np.array_equal(per, np.broadcast_to([6, 6, 12, 24], per.shape)):
        raise ValueError('Incomplete age bands; never divide by the observed subset size.')
    point_pass = pd.DatetimeIndex(p.pass_time_kst).as_unit('ns')
    expected_pass = pd.DatetimeIndex(p.arrival_time_kst) - pd.to_timedelta(ages, unit='h')
    if not point_pass.equals(expected_pass.as_unit('ns')):
        raise ValueError('Point historical pass-time identity is inconsistent.')
    npoint, nnodes = len(p), len(nodes)
    sw = np.zeros((4, npoint), dtype=np.float64)
    swc = np.zeros_like(sw)
    nstation = np.zeros((4, npoint), dtype=np.int32)
    # Compact bitset detects duplicate point/node pairs including across Arrow batches.
    seen_edges = np.zeros((npoint, (nnodes + 7)//8), dtype=np.uint8)
    seen_band_nodes = np.zeros((len(origins)*4, nnodes), dtype=bool)
    diagnostics = {'saved_neighbor_rows_read': 0, 'selected_neighbor_rows': 0,
                   'raw_rejoined_valid_rows': 0, 'saved_pm_vs_current_raw_mismatches': 0,
                   'negative_raw_pm25': 0, 'future_pass_rows': 0}
    for batch in batches:
        diagnostics['saved_neighbor_rows_read'] += len(batch)
        ni = nodes.get_indexer(batch.station_key.astype(str))
        # The target is always separate and must not become an external trajectory donor.
        keep = (ni >= 0) & batch.station_key.astype(str).ne('KR:111121').to_numpy()
        if not keep.any():
            continue
        batch = batch.loc[keep].reset_index(drop=True); ni = ni[keep]
        pi = point_index.get_indexer(batch.point_id.astype(str))
        if (pi < 0).any():
            unknown_origin = origins.get_indexer(pd.DatetimeIndex(batch.arrival_time_kst).as_unit('ns')) < 0
            if ((pi < 0) & ~unknown_origin).any():
                raise ValueError('Selected donor references an unknown point at a requested arrival.')
            keep = pi >= 0
            batch = batch.loc[keep].reset_index(drop=True); ni = ni[keep]; pi = pi[keep]
        if not len(batch):
            continue
        distance = batch.distance_km.to_numpy(dtype=float)
        if not np.isfinite(distance).all() or ((distance < 0) | (distance > radius_km + 1e-6)).any():
            raise ValueError('Saved matching distance outside the declared 75km pool.')
        arrival = pd.DatetimeIndex(batch.arrival_time_kst).as_unit('ns')
        passed = pd.DatetimeIndex(batch.pass_time_kst).as_unit('ns')
        if not passed.equals(point_pass.take(pi)) or not arrival.equals(pd.DatetimeIndex(p.arrival_time_kst).take(pi).as_unit('ns')):
            raise ValueError('Neighbor point/time alignment mismatch.')
        if not np.array_equal(batch.lookback_hours.to_numpy(dtype=int), ages[pi]):
            raise ValueError('Neighbor point/age alignment mismatch.')
        if (passed > arrival).any():
            raise ValueError('Future observation requested by trajectory matching.')
        encoded = pi.astype(np.int64)*nnodes + ni
        if len(np.unique(encoded)) != len(encoded):
            raise ValueError('Duplicate point/station pairs within a matching batch.')
        byte, bit = ni//8, np.left_shift(np.uint8(1), (ni%8).astype(np.uint8))
        if ((seen_edges[pi, byte] & bit) != 0).any():
            raise ValueError('Duplicate point/station pairs across matching batches.')
        np.bitwise_or.at(seen_edges, (pi, byte), bit)
        ti = hour_index.get_indexer(passed)
        values = np.full(len(batch), np.nan, dtype=float)
        known = ti >= 0
        values[known] = raw[ti[known], ni[known]]
        if (values[np.isfinite(values)] < 0).any():
            raise ValueError('Raw node loader supplied negative analysis PM2.5.')
        stored = pd.to_numeric(batch.pm25, errors='coerce').to_numpy(float)
        both = np.isfinite(stored) & np.isfinite(values)
        mismatch = (np.isfinite(stored) != np.isfinite(values)) | (both & ~np.isclose(stored, values, rtol=1e-5, atol=1e-3, equal_nan=True))
        diagnostics['saved_pm_vs_current_raw_mismatches'] += int(mismatch.sum())
        diagnostics['selected_neighbor_rows'] += len(batch)
        valid = np.isfinite(values)
        diagnostics['raw_rejoined_valid_rows'] += int(valid.sum())
        for ci, country in enumerate(COUNTRIES):
            take = valid & (countries[ni] == country)
            if not take.any():
                continue
            target = pi[take]
            weight = np.exp(-.5*(distance[take]/sigma_km)**2)
            np.add.at(sw[ci], target, weight)
            np.add.at(swc[ci], target, weight*values[take])
            np.add.at(nstation[ci], target, 1)
            seen_band_nodes[origin_band[target], ni[take]] = True
    height = p.height_agl_m.to_numpy(float)
    mix = p.diagnostic_mixdepth.to_numpy(float)
    vertical_known = np.isfinite(height) & np.isfinite(mix) & (height >= 0) & (mix >= 0)
    inside = vertical_known & (height <= mix)
    above = vertical_known & (height > mix)
    result = np.full((len(origins), 208), np.nan, dtype=np.float32)
    column = 0
    for ci, country in enumerate(COUNTRIES):
        z = np.divide(swc[ci], sw[ci], out=np.full(npoint, np.nan), where=sw[ci] > 0)
        unique = seen_band_nodes[:, countries == country].sum(axis=1).reshape(len(origins), 4)
        for bi, (lo, hi, band) in enumerate(BANDS):
            idx = np.flatnonzero((ages >= lo) & (ages <= hi))
            zz = z[idx].reshape(len(origins), hi-lo+1)
            ns = nstation[ci, idx].reshape(zz.shape)
            ins = inside[idx].reshape(zz.shape); abv = above[idx].reshape(zz.shape)
            good = np.isfinite(zz); ng = good.sum(axis=1); denom = hi-lo+1
            def mean(mask):
                n = mask.sum(axis=1)
                return np.divide(np.where(mask, zz, 0).sum(axis=1), n,
                                 out=np.full(len(origins), np.nan), where=n > 0)
            avg = mean(good)
            variance = np.divide(np.where(good, (zz-avg[:, None])**2, 0).sum(axis=1), ng,
                                 out=np.full(len(origins), np.nan), where=ng > 0)
            mx = np.where(good, zz, -np.inf).max(axis=1); mx[ng == 0] = np.nan
            nin = ins.sum(axis=1); nabove = abv.sum(axis=1)
            valid_in = (good & ins).sum(axis=1); valid_above = (good & abv).sum(axis=1)
            in_coverage = np.divide(valid_in, nin, out=np.full(len(origins), np.nan), where=nin > 0)
            above_coverage = np.divide(valid_above, nabove, out=np.full(len(origins), np.nan), where=nabove > 0)
            values = (avg, np.sqrt(variance), mx, mean(good & ins), mean(good & abv),
                      ng/denom, ng, ns.sum(axis=1)/denom, unique[:, bi],
                      in_coverage, valid_in, above_coverage, valid_above)
            for value in values:
                result[:, column] = value; column += 1
    if column != 208 or np.isinf(result).any():
        raise ValueError('Incorrect/nonfinite trajectory aggregate output.')
    return result, diagnostics


def build_trajectory_features(project_root, bundle, matching_folder=None, origins=None,
                              radius_km=75., sigma_km=50.):
    root = Path(project_root)
    folder = Path(matching_folder) if matching_folder else root/'Dataset/Trajectory_Matching_Expansion_QC_20261108'
    requested = origin_index(bundle) if origins is None else pd.DatetimeIndex(origins).as_unit('ns')
    names, countries = feature_layout(root)
    neutral = names[:114]
    table = pd.read_parquet(root/'Dataset/Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet',
                            columns=['origin_time_kst']+neutral)
    index = pd.DatetimeIndex(table.origin_time_kst).as_unit('ns')
    positions = index.get_indexer(requested)
    if (positions < 0).any() or not index.is_unique:
        raise ValueError('Neutral trajectory feature times disagree with frozen origins.')
    neutral_values = table.iloc[positions][neutral].to_numpy(np.float32)
    points = read_points(folder, requested)
    pm, diagnostics = aggregate_pm_features(bundle, points, neighbor_batches(folder, requested),
                                             origins=requested, radius_km=radius_km, sigma_km=sigma_km)
    values = np.concatenate([neutral_values, pm], axis=1)
    if values.shape != (len(requested), 322) or np.isinf(values).any():
        raise ValueError('Trajectory context shape/finite-value mismatch.')
    return {'values': values, 'names': names, 'countries': countries,
            'origins': requested, 'diagnostics': diagnostics,
            'version': VERSION, 'radius_km': float(radius_km), 'sigma_km': float(sigma_km)}


def get_cached_trajectory(project_root, bundle, cache_dir, source_signature,
                          radius_km=75., sigma_km=50.):
    """Reuse only a byte-verified cache with identical raw-data/config signatures."""
    root = Path(cache_dir); root.mkdir(parents=True, exist_ok=True)
    contract = {'version': VERSION, 'source_signature': source_signature,
                'radius_km': float(radius_km), 'sigma_km': float(sigma_km),
                'node_order': bundle['nodes'].station_key.tolist(),
                'raw_pm_sha256': array_hash(bundle['pm_values']),
                'origins': [t.isoformat() for t in origin_index(bundle)]}
    signature = canonical_hash(contract)
    prefix = f'trajectory_{signature[:24]}'
    for folder in sorted(root.glob(prefix+'*')):
        metadata = folder/'cache_contract.json'; matrix = folder/'trajectory_features.npz'
        if not metadata.is_file() or not matrix.is_file():
            continue
        m = json.loads(metadata.read_text(encoding='utf-8'))
        if m.get('contract_sha256') != signature:
            continue
        if sha256_file(matrix) != m['npz_sha256']:
            raise ValueError('Trajectory cache bytes changed; refusing silent reuse: '+str(matrix))
        with np.load(matrix, allow_pickle=False) as z:
            values = z['values'].copy()
        names, countries = feature_layout(project_root)
        if m['names'] != names or m['countries'] != countries or values.shape != (len(origin_index(bundle)), 322):
            raise ValueError('Cached trajectory feature schema is inconsistent.')
        return {'values': values, 'names': names, 'countries': countries, 'origins': origin_index(bundle),
                'diagnostics': m['diagnostics'], 'version': VERSION, 'cache_dir': str(folder),
                'cache_reused': True, 'cache_contract_sha256': signature}
    result = build_trajectory_features(project_root, bundle, radius_km=radius_km, sigma_km=sigma_km)
    folder = root/(prefix+'_'+uuid.uuid4().hex[:8]); folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'trajectory_features.npz', values=result['values'])
    metadata = dict(contract_sha256=signature, contract=contract, names=result['names'],
                    countries=result['countries'], diagnostics=result['diagnostics'],
                    npz_sha256=sha256_file(folder/'trajectory_features.npz'))
    (folder/'cache_contract.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result.update(cache_dir=str(folder), cache_reused=False, cache_contract_sha256=signature)
    return result
