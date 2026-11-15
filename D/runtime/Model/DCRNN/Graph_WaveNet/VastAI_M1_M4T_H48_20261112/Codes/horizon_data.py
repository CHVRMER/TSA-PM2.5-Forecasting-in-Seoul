"""Exact observed Junggu targets at six elapsed-hour forecast horizons.

The existing four targets are verified against the raw hourly target and then
preserved bit-for-bit. Added targets are labels only, never model inputs.
Eligibility depends only on observed target quality, study bounds and split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HORIZONS = (3, 6, 12, 24, 36, 48)
LEGACY_HORIZONS = (3, 6, 12, 24)
TIMEZONE = 'Asia/Seoul'
STUDY_START = pd.Timestamp('2021-01-01', tz=TIMEZONE)
STUDY_END = pd.Timestamp('2025-01-01', tz=TIMEZONE)
VERSION = 'exact-junggu-horizons48-v1.0.0'


def kst(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    if result.tz is None:
        result = result.tz_localize(TIMEZONE)
    else:
        result = result.tz_convert(TIMEZONE)
    return result.as_unit('ns')


def split_for_time(values):
    years = kst(values).year
    return np.where(years <= 2022, 'train', np.where(years == 2023, 'validation', 'test'))


def build_exact_labels(hourly_index, target_values, origins, origin_splits, horizons=HORIZONS):
    """Return `(y_raw, eligible, target_times)` using exact timestamps only.

    target_values must already be the QC-filtered Junggu PM2.5 source. This
    routine never forward-fills, interpolates, shifts origin-table row numbers,
    or changes the eligibility of valid observed targets.
    """
    horizons = tuple(int(h) for h in horizons)
    if horizons != HORIZONS:
        raise ValueError('This frozen experiment requires horizons3/6/12/24/36/48.')
    hours, arrivals = kst(hourly_index), kst(origins)
    if hours.has_duplicates or not hours.is_monotonic_increasing or hours.hasnans:
        raise ValueError('Hourly source timestamps must be unique, sorted and finite.')
    if arrivals.has_duplicates or not arrivals.is_monotonic_increasing or arrivals.hasnans:
        raise ValueError('Origin timestamps must be unique, sorted and finite.')
    splits = np.asarray(origin_splits, dtype=str)
    if splits.shape != (len(arrivals),) or not np.array_equal(splits, split_for_time(arrivals)):
        raise ValueError('Origin split labels differ from the frozen21-22/23/24 rule.')
    values = np.asarray(target_values, dtype=np.float64)
    if values.shape != (len(hours),) or np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('Expected one QC-filtered nonnegative/NaN target value per source timestamp.')
    labels = np.full((len(arrivals), len(horizons)), np.nan, dtype=np.float64)
    active = np.zeros(labels.shape, dtype=bool)
    targets = {}
    for column, horizon in enumerate(horizons):
        times = arrivals + pd.Timedelta(hours=horizon)
        positions = hours.get_indexer(times)
        in_study = (times >= STUDY_START) & (times < STUDY_END)
        found = (positions >= 0) & in_study
        labels[found, column] = values[positions[found]]
        active[:, column] = found & np.isfinite(labels[:, column]) & (split_for_time(times) == splits)
        targets[horizon] = times
    return labels, active, targets


def extend_bundle_horizons(bundle, horizons=HORIZONS):
    """Return an independent label/origin view, preserving all input arrays.

    Existing PM/context arrays, their order, and time windows are never mutated.
    Old target values may have up to1e-6 source serialization rounding; they are
    checked to the same old-loader tolerance then carried over exactly.
    """
    if tuple(horizons) != HORIZONS:
        raise ValueError('Unsupported horizon contract.')
    source_horizons = tuple(int(h) for h in bundle['horizons'])
    if source_horizons not in (LEGACY_HORIZONS, HORIZONS):
        raise ValueError('Expected the original four-horizon or verified six-horizon bundle.')
    nodes = bundle['nodes']
    if len(nodes) < 1 or str(nodes.iloc[0].station_key) != 'KR:111121':
        raise ValueError('Junggu target must remain first in the node order.')
    origins = bundle['origins'].copy(deep=True)
    y_raw, eligible, target_times = build_exact_labels(
        bundle['hourly_index'], np.asarray(bundle['pm_values'])[:, 0],
        origins.origin_time_kst, origins.origin_split, horizons)
    saved = np.asarray(bundle['y_raw'], dtype=np.float64)
    old_eligible = np.asarray(bundle['eligible'], dtype=bool)
    if saved.shape != (len(origins), len(source_horizons)) or old_eligible.shape != saved.shape:
        raise ValueError('Existing label shapes do not match the source horizon declaration.')
    for old_column, horizon in enumerate(source_horizons):
        new_column = HORIZONS.index(horizon)
        if not np.allclose(saved[:, old_column], y_raw[:, new_column], rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f'Saved labels disagree with exact raw Junggu hours at h={horizon}.')
        if not np.array_equal(old_eligible[:, old_column], eligible[:, new_column]):
            raise ValueError(f'Saved eligibility differs from observed/same-split checks at h={horizon}.')
        for stem, expected in ((f'y_pm25_h{horizon}', saved[:, old_column]),
                               (f'eligible_h{horizon}', old_eligible[:, old_column])):
            if stem in origins and not np.array_equal(origins[stem].to_numpy(), expected, equal_nan=True):
                raise ValueError('Existing origin column and label array disagree: '+stem)
        time_name = f'target_time_h{horizon}'
        if time_name in origins and not kst(origins[time_name]).equals(target_times[horizon]):
            raise ValueError('Existing target timestamp differs from origin+horizon: '+time_name)
        y_raw[:, new_column] = saved[:, old_column]
    for column, horizon in enumerate(HORIZONS):
        additions = {f'y_pm25_h{horizon}': y_raw[:, column],
                     f'eligible_h{horizon}': eligible[:, column],
                     f'target_time_h{horizon}': target_times[horizon]}
        for name, values in additions.items():
            # Retain verified legacy columns themselves, including their saved
            # datetime resolution/dtype, instead of rewriting equivalent times.
            if horizon in source_horizons and name in origins:
                continue
            origins[name] = values
    result = dict(bundle, origins=origins, y_raw=y_raw, eligible=eligible, horizons=HORIZONS)
    result['horizon_extension_contract'] = {
        'version': VERSION, 'horizons': list(HORIZONS), 'legacy_horizons': list(LEGACY_HORIZONS),
        'target_station_key': 'KR:111121', 'label_rule': 'QC observed Junggu at exact origin+horizon',
        'eligibility_rule': 'finite nonnegative observed target AND same split AND2021-2024 study',
        'legacy_four_values_preserved_exactly': True, 'target_imputation': False,
        'input_arrays_and_windows_changed': False,
        'eligible_by_split': {split: {str(h): int(eligible[origins.origin_split.eq(split).to_numpy(),j].sum())
            for j,h in enumerate(HORIZONS)} for split in ('train','validation','test')}}
    return result
