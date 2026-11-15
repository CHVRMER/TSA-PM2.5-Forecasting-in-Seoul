"""Frozen PM2.5 evaluation rows and causal baseline definitions.

Only target availability and the predeclared split determine eligibility.
Model predictions never alter the grid. Time fields are KST.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (3, 6, 12, 24)
TIMEZONE = 'Asia/Seoul'
KEY = ['origin_time_kst', 'horizon']


def kst(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    return (result.tz_localize(TIMEZONE) if result.tz is None else result.tz_convert(TIMEZONE)).as_unit('ns')


def split_for_time(values):
    years = kst(values).year
    return np.where(years <= 2022, 'train', np.where(years == 2023, 'validation', 'test'))


def boolean(values):
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError('Missing eligibility flag.')
        return values.astype(bool)
    mapping = {True: True, False: False, 1: True, 0: False,
               'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}
    result = values.map(mapping)
    if result.isna().any():
        raise ValueError('Eligibility must contain explicit boolean/0/1 values.')
    return result.astype(bool)


def validate_grid(frame):
    """Accept long authoritative rows or expand the original wide origin table."""
    frame = frame.copy()
    if 'horizon' not in frame and 'horizon_h' not in frame:
        pieces = []
        common = [c for c in frame if not any(c == f'{stem}_h{h}'
                  for stem in ('y_pm25', 'eligible', 'target_time') for h in HORIZONS)]
        for horizon in HORIZONS:
            required = [f'y_pm25_h{horizon}', f'eligible_h{horizon}']
            if not set(required) <= set(frame):
                raise ValueError('Grid must be long or contain every y_pm25_h*/eligible_h* column.')
            part = frame[common].copy()
            part['horizon'] = horizon
            part['y_true'] = frame[f'y_pm25_h{horizon}']
            part['eligible'] = frame[f'eligible_h{horizon}']
            if f'target_time_h{horizon}' in frame:
                part['target_time_kst'] = frame[f'target_time_h{horizon}']
            pieces.append(part)
        frame = pd.concat(pieces, ignore_index=True)
    aliases = {'horizon_h': 'horizon', 'origin_split': 'split', 'y': 'y_true'}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame = frame.rename(columns={old: new})
    required = {'origin_time_kst', 'horizon', 'y_true', 'eligible', 'split'}
    if not required <= set(frame):
        raise ValueError(f'Missing grid columns: {sorted(required-set(frame))}')
    frame['origin_time_kst'] = kst(frame.origin_time_kst)
    if frame.origin_time_kst.isna().any():
        raise ValueError('Missing grid origin time.')
    frame['horizon'] = pd.to_numeric(frame.horizon, errors='raise')
    if not frame.horizon.isin(HORIZONS).all():
        raise ValueError('Only 3/6/12/24-hour horizons are supported.')
    frame['horizon'] = frame.horizon.astype(int)
    if frame.duplicated(KEY).any():
        raise ValueError('Duplicate grid origin/horizon.')
    t = pd.DatetimeIndex(frame.origin_time_kst)
    if ((t.hour % 3 != 0) | (t.minute != 0) | (t.second != 0) | (t.microsecond != 0)).any():
        raise ValueError('Evaluation origins must be on the fixed KST three-hour grid.')
    if not frame.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown grid split.')
    if not np.array_equal(frame.split.to_numpy(), split_for_time(t)):
        raise ValueError('Grid splits differ from train 2021–22 / validation 2023 / test 2024.')
    expected_target = frame.origin_time_kst + pd.to_timedelta(frame.horizon, unit='h')
    if 'target_time_kst' in frame:
        if not np.array_equal(kst(frame.target_time_kst).asi8, pd.DatetimeIndex(expected_target).asi8):
            raise ValueError('Grid target time does not equal origin+horizon.')
    frame['target_time_kst'] = expected_target
    frame['eligible'] = boolean(frame.eligible)
    frame['y_true'] = pd.to_numeric(frame.y_true, errors='coerce')
    active = frame.eligible.to_numpy()
    if not np.isfinite(frame.loc[active, 'y_true']).all() or (frame.loc[active, 'y_true'] < 0).any():
        raise ValueError('Every eligible target must be finite, nonnegative and observed.')
    in_study = (expected_target >= pd.Timestamp('2021-01-01', tz=TIMEZONE)) & (expected_target < pd.Timestamp('2025-01-01', tz=TIMEZONE))
    same_split = split_for_time(expected_target) == frame.split.to_numpy()
    if (active & ~(in_study & same_split)).any():
        raise ValueError('Eligible target crosses a split/study boundary.')
    expected_eligible = np.isfinite(frame.y_true.to_numpy(float)) & (frame.y_true.to_numpy(float)>=0) & in_study & same_split
    if not np.array_equal(active, expected_eligible):
        raise ValueError('Eligibility must equal observed finite target AND same split/study; valid target rows cannot be silently excluded.')
    canonical_origins = None
    for horizon, part in frame.groupby('horizon'):
        origins = pd.DatetimeIndex(part.origin_time_kst).sort_values()
        expected_origins = pd.date_range(origins.min(), origins.max(), freq='3h')
        if not origins.equals(expected_origins):
            raise ValueError(f'Grid is missing calendar slots at horizon {horizon}; retain ineligible rows rather than compressing time.')
        if canonical_origins is not None and not origins.equals(canonical_origins):
            raise ValueError('Horizon grids cover different origin calendars.')
        canonical_origins = origins
    return frame.sort_values(KEY).reset_index(drop=True)


def read_frame(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() in {'.parquet', '.pq'} else pd.read_csv(path)


def baseline_at_times(target_hourly, requests, fallback_value=None):
    """Last valid observation at or before each request; never backward-fill.

    The explicit final fallback reproduces the original DCRNN: inverse of the
    2021–22 target log1p mean. It is only used when no earlier measurement exists.
    """
    values = target_hourly.copy()
    if 'time_kst' not in values:
        raise ValueError('Target hourly data need time_kst and pm25 columns.')
    values['time_kst'] = kst(values.time_kst)
    if values.time_kst.duplicated().any():
        raise ValueError('Target hourly data contain duplicate hours; select Junggu only.')
    numeric = pd.to_numeric(values.pm25, errors='coerce').to_numpy(float)
    valid = np.isfinite(numeric) & (numeric >= 0)
    if 'pm25_flag' in values:
        valid &= pd.to_numeric(values.pm25_flag, errors='coerce').eq(0).to_numpy()
    good = values.loc[valid, ['time_kst']].copy()
    good['value'] = numeric[valid]
    good = good.sort_values('time_kst')
    if fallback_value is None:
        train = good.time_kst.between(pd.Timestamp('2021-01-01', tz=TIMEZONE),
                                     pd.Timestamp('2023-01-01', tz=TIMEZONE), inclusive='left')
        if not train.any():
            raise ValueError('No 2021–22 observed target values to fit final baseline fallback.')
        fallback_value = float(np.expm1(np.log1p(good.loc[train, 'value']).mean()))
    if not np.isfinite(fallback_value) or fallback_value < 0:
        raise ValueError('Invalid training-only baseline fallback value.')
    requested = kst(requests)
    source_times = pd.DatetimeIndex(good.time_kst)
    positions = source_times.searchsorted(requested, side='right') - 1
    has_source = positions >= 0
    selected_values = np.full(len(requested), fallback_value, float)
    ages = np.full(len(requested), np.nan, float)
    selected_values[has_source] = good.value.to_numpy()[positions[has_source]]
    ages[has_source] = (requested.asi8[has_source] - source_times.asi8[positions[has_source]]) / 3.6e12
    kind = np.where(~has_source, 'training_log_mean', np.where(ages == 0, 'exact_observation', 'last_past_observation'))
    return pd.DataFrame({'y_hat': selected_values, 'fallback': kind != 'exact_observation',
                         'fallback_kind': kind, 'age_hours': ages,
                         'training_log_mean_fallback_value': fallback_value})


def add_baselines(grid, target_hourly):
    grid = grid.copy()
    for prefix, requests in [('persistence', grid.origin_time_kst),
                             ('seasonal', grid.target_time_kst-pd.Timedelta(hours=24))]:
        result = baseline_at_times(target_hourly, requests)
        for column in result:
            grid[f'{prefix}_{column}'] = result[column].to_numpy()
        grid[f'{prefix}_request_time_kst'] = requests.to_numpy()
    return grid


def validate_baselines(grid):
    needed = {'persistence_y_hat', 'seasonal_y_hat'}
    if not needed <= set(grid):
        raise ValueError('Grid has no audited baselines. Supply --target-hourly (time_kst,pm25) or prepared baseline columns.')
    for column in needed:
        if not np.isfinite(pd.to_numeric(grid.loc[grid.eligible, column], errors='coerce')).all():
            raise ValueError(f'Missing/nonfinite baseline on fixed eligible row: {column}')
    return grid
