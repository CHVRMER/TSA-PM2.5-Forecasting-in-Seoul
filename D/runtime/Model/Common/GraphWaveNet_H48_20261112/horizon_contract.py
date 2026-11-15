"""Immutable model-side extension of the original evaluation grid to 36/48h.

No source Dataset file is written. Original four-horizon labels, conditions and
audited baselines survive bit-for-bit. The new horizons reuse the same origin
calendar, exact target observations and split-boundary rules.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (3, 6, 12, 24, 36, 48)
ORIGINAL_HORIZONS = (3, 6, 12, 24)
SCHEMA_VERSION = 'gwn-h48-grid-v1.0.0'
BASE_GRID_RELATIVE_PATH = 'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'


def _grid_module():
    spec = importlib.util.spec_from_file_location('_gwn_h48_contract_grid', Path(__file__).with_name('grid.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _target(bundle, grid_api):
    times = grid_api.kst(bundle['hourly_index'])
    matrix = np.asarray(bundle['pm_values'])
    if matrix.ndim != 2 or matrix.shape[0] != len(times) or matrix.shape[1] < 1:
        raise ValueError('Bundle target must be pm_values[:,0] aligned to hourly_index.')
    if times.isna().any() or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Target hourly index must be unique, complete timestamps in increasing order.')
    if ((times.minute != 0) | (times.second != 0) | (times.microsecond != 0)).any():
        raise ValueError('Target observations must be indexed on exact hours.')
    values = matrix[:, 0].astype(np.float64)
    values[~np.isfinite(values) | (values < 0)] = np.nan
    return pd.Series(values, index=times, name='pm25')


def _exact_targets(target, requested, grid_api):
    result = target.reindex(grid_api.kst(requested)).to_numpy(float, copy=True)
    times = grid_api.kst(requested)
    in_study = ((times >= pd.Timestamp('2021-01-01', tz=grid_api.TIMEZONE)) &
                (times < pd.Timestamp('2025-01-01', tz=grid_api.TIMEZONE)))
    result[~in_study] = np.nan
    return result


def assert_original_rows_preserved(base_grid, extended_grid):
    """Strict values/dtype/order regression for all original grid columns."""
    api = _grid_module()
    base = api.validate_grid(base_grid)
    old = extended_grid.loc[extended_grid.horizon.isin(ORIGINAL_HORIZONS), base.columns]
    old = old.sort_values(api.KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(base, old, check_exact=True, check_dtype=True)


def _assert_bundle_grid(bundle, extended, api):
    canonical = pd.DatetimeIndex(extended.loc[extended.horizon.eq(3), 'origin_time_kst'])
    if 'origins' in bundle:
        supplied = bundle['origins']
        supplied = supplied['origin_time_kst'] if isinstance(supplied, pd.DataFrame) else supplied
        if not canonical.equals(api.kst(supplied)):
            raise ValueError('Bundle and authoritative evaluation origin calendars differ.')
    if 'horizons' in bundle:
        if tuple(bundle['horizons']) != HORIZONS:
            raise ValueError('The H48 evaluation contract requires all six bundle horizons.')
        if 'y_raw' not in bundle or 'eligible' not in bundle:
            raise ValueError('Declared six-horizon bundles must expose matching y_raw and eligible arrays.')
        for j, horizon in enumerate(HORIZONS):
            part = extended.loc[extended.horizon.eq(horizon)]
            y = np.asarray(bundle['y_raw'])[:, j]
            eligible = np.asarray(bundle['eligible'])[:, j].astype(bool)
            if not np.array_equal(part.y_true.to_numpy(float), y, equal_nan=True) or not np.array_equal(part.eligible.to_numpy(), eligible):
                raise ValueError('Model bundle and evaluation contract labels/eligibility differ at horizon ' + str(horizon))


def build_h48_grid(base_grid, bundle):
    """Return a validated six-horizon table without changing inputs.

    bundle supplies hourly_index, QC-clean pm_values[:,0] and optionally origins,
    horizons/y_raw/eligible. Existing seasonal 12h and 24h requests are identical
    to the new 36h and 48h requests, respectively. Copying those complete audited
    baseline records preserves fallback/source timestamps as well as values.
    """
    api = _grid_module()
    base = api.validate_baselines(api.validate_grid(base_grid))
    if set(base.horizon) != set(ORIGINAL_HORIZONS):
        raise ValueError('Expected the untouched authoritative four-horizon grid.')
    target = _target(bundle, api)
    raw_old = _exact_targets(target, base.target_time_kst, api)
    if not np.array_equal(raw_old, base.y_true.to_numpy(float), equal_nan=True):
        bad = np.flatnonzero(~((raw_old == base.y_true.to_numpy(float)) |
                              (np.isnan(raw_old) & base.y_true.isna().to_numpy())))
        raise ValueError('Bundle target differs from original authoritative labels; first mismatched rows=' + str(bad[:5].tolist()))
    varying = {'horizon', 'target_time_kst', 'y_true', 'eligible', 'y_ge_35'}
    origin_only = [c for c in base if c not in varying and not c.startswith(('persistence_', 'seasonal_'))]
    first = base.loc[base.horizon.eq(3), origin_only].reset_index(drop=True)
    for horizon in ORIGINAL_HORIZONS[1:]:
        other = base.loc[base.horizon.eq(horizon), origin_only].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(first, other, check_exact=True)
        except AssertionError as error:
            raise ValueError('Unexpected target/horizon-dependent metadata must be explicitly rebuilt, not copied from a donor horizon.') from error
    pieces = [base]
    for horizon, source_horizon in ((36, 12), (48, 24)):
        part = base.loc[base.horizon.eq(source_horizon)].copy().reset_index(drop=True)
        part['horizon'] = horizon
        part['target_time_kst'] = part.origin_time_kst + pd.Timedelta(hours=horizon)
        part['y_true'] = _exact_targets(target, part.target_time_kst, api)
        same_split = api.split_for_time(part.target_time_kst) == part.split.to_numpy()
        part['eligible'] = np.isfinite(part.y_true.to_numpy(float)) & same_split
        if 'y_ge_35' in part:
            part['y_ge_35'] = part.y_true.ge(35).astype(base['y_ge_35'].dtype)
        if 'evaluation_contract_version' in part:
            part['evaluation_contract_version'] = SCHEMA_VERSION
        # Persistence uses t; seasonal uses t-12 (36h) or t (48h), just as the
        # source 12h/24h rows. All request/fallback/observation fields stay causal.
        pieces.append(part)
    extended = pd.concat(pieces, ignore_index=True)[base.columns]
    extended = api.validate_baselines(api.validate_grid(extended))
    assert_original_rows_preserved(base, extended)
    if set(extended.horizon) != set(HORIZONS):
        raise AssertionError('Incomplete six-horizon contract.')
    _assert_bundle_grid(bundle, extended, api)
    return extended


def ensure_h48_grid(project_root, output_cache, bundle):
    """Build/reuse a verified immutable model cache; return evaluation_grid Path.

    The authoritative four-horizon grid and every Dataset file are read-only.
    Cache identity includes source grid bytes, exact target history, and both
    contract source files. A corrupt/incomplete existing cache is an error.
    """
    project, cache = Path(project_root).resolve(), Path(output_cache).resolve()
    if cache == project / 'Dataset' or (project / 'Dataset') in cache.parents:
        raise ValueError('Write the extended grid under Model/Cache or a run folder, never Dataset.')
    base_path = project / BASE_GRID_RELATIVE_PATH
    if not base_path.is_file():
        raise FileNotFoundError('The original authoritative grid is required: ' + str(base_path))
    api = _grid_module()
    target = _target(bundle, api)
    digest = hashlib.sha256()
    digest.update(target.index.asi8.astype('<i8').tobytes())
    digest.update(target.to_numpy(dtype='<f8').tobytes())
    source = {'schema_version': SCHEMA_VERSION, 'horizons': list(HORIZONS),
              'base_grid_sha256': _sha(base_path), 'target_sha256': digest.hexdigest(),
              'grid_code_sha256': _sha(Path(__file__).with_name('grid.py')),
              'contract_code_sha256': _sha(__file__)}
    key = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = cache / ('h48_' + key[:20])
    path, audit_path = folder / 'evaluation_grid.parquet', folder / 'horizon_contract_audit.json'
    if folder.exists():
        if not path.is_file() or not audit_path.is_file():
            raise ValueError('Incomplete H48 cache; preserve it and use another output_cache path: ' + str(folder))
        saved = json.loads(audit_path.read_text(encoding='utf-8'))
        if saved.get('sources') != source or saved.get('evaluation_grid_sha256') != _sha(path):
            raise ValueError('Existing H48 cache content changed: ' + str(folder))
        current = api.validate_baselines(api.validate_grid(pd.read_parquet(path)))
        assert_original_rows_preserved(pd.read_parquet(base_path), current)
        if set(current.horizon) != set(HORIZONS):
            raise ValueError('Existing cache has an incomplete horizon calendar.')
        _assert_bundle_grid(bundle, current, api)
        return path
    result = build_h48_grid(pd.read_parquet(base_path), bundle)
    folder.mkdir(parents=True, exist_ok=False)
    # This is a fresh, content-addressed folder. Never replace old cache files.
    result.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(result, restored, check_exact=True)
    audit = {'status': 'passed', 'sources': source, 'evaluation_grid_sha256': _sha(path),
             'original_four_horizon_rows_preserved_exactly': True,
             'source_dataset_modified': False, 'exact_timestamp_target_lookup': True,
             'target_fill_or_interpolation': False, 'prediction_intersection_used': False,
             'seasonal_naive_request': 't+h-24*ceil(h/24); 36h=t-12h, 48h=t',
             'baseline_requests_at_or_before_origin': True,
             'calendar_rows_by_horizon': {str(h): len(g) for h, g in result.groupby('horizon')},
             'eligible_by_split_horizon': {f'{split}/h{h}': int(g.eligible.sum())
                                          for (split, h), g in result.groupby(['split', 'horizon'])}}
    with audit_path.open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    if _sha(base_path) != source['base_grid_sha256']:
        raise ValueError('Original evaluation grid changed during cache construction.')
    return path
