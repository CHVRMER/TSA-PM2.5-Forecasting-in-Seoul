"""Synthetic six-horizon labels and legacy-data preservation checks, no Torch.

Run this before loading the full dataset. It uses exact timestamps and checks
missing hours, split boundaries, study boundaries, source immutability and
future-label mutations separately from the causal input window.
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd

from horizon_data import (HORIZONS, LEGACY_HORIZONS, build_exact_labels,
                          extend_bundle_horizons, kst, split_for_time)


def fixture():
    hours=pd.date_range('2022-12-28','2023-01-05',freq='h',tz='Asia/Seoul')
    origins=kst(['2022-12-29 00:00','2022-12-30 12:00','2022-12-31 00:00',
                 '2023-01-01 00:00','2023-01-02 00:00'])
    values=np.column_stack((np.arange(len(hours),dtype=np.float64)+1.,
                            np.arange(len(hours),dtype=np.float64)+101.))
    # A target value flagged/missing in the upstream source stays missing here.
    values[hours.get_loc(pd.Timestamp('2023-01-02 12:00',tz='Asia/Seoul')),0]=np.nan
    y,eligible,times=build_exact_labels(hours,values[:,0],origins,split_for_time(origins))
    table=pd.DataFrame({'origin_time_kst':origins,'origin_split':split_for_time(origins)})
    # Simulate permitted source serialization rounding. Old saved values must
    # survive extension exactly instead of being silently recomputed.
    saved=y[:,:4].copy()
    saved[0,0]+=1e-7
    for j,h in enumerate(LEGACY_HORIZONS):
        table[f'y_pm25_h{h}']=saved[:,j]
        table[f'eligible_h{h}']=eligible[:,j]
        # Actual legacy Parquet uses microsecond target timestamp resolution.
        table[f'target_time_h{h}']=times[h].as_unit('us')
    return {'nodes':pd.DataFrame({'station_key':['KR:111121','KR:999999']}),
            'hourly_index':hours,'pm_values':values,'origins':table,
            'horizons':LEGACY_HORIZONS,'y_raw':saved,'eligible':eligible[:,:4].copy(),
            'context_values':np.arange(len(origins)*2,dtype=float).reshape(-1,2),
            'origin_positions':hours.get_indexer(origins),'lookback_hours':48}


def main():
    b=fixture()
    raw=b['pm_values'].copy();old_origins=b['origins'].copy(deep=True)
    old_labels=b['y_raw'].copy();old_elig=b['eligible'].copy()
    extended=extend_bundle_horizons(b)
    assert extended['horizons']==HORIZONS
    assert extended['y_raw'].shape==(len(old_origins),6)
    assert np.array_equal(extended['y_raw'][:,:4],old_labels,equal_nan=True)
    assert np.array_equal(extended['eligible'][:,:4],old_elig)
    pd.testing.assert_frame_equal(extended['origins'][old_origins.columns],old_origins)
    pd.testing.assert_frame_equal(b['origins'],old_origins)
    assert np.array_equal(b['pm_values'],raw,equal_nan=True)
    assert extended['pm_values'] is b['pm_values']
    assert extended['context_values'] is b['context_values']
    assert extended['origin_positions'] is b['origin_positions']
    assert extended['lookback_hours']==b['lookback_hours']
    # The missing target is reached at +36 h from this origin and is never filled.
    assert np.isnan(extended['y_raw'][3,4]) and not extended['eligible'][3,4]
    for j,h in enumerate(HORIZONS):
        expected=kst(old_origins.origin_time_kst)+pd.Timedelta(hours=h)
        assert kst(extended['origins'][f'target_time_h{h}']).equals(expected)
    # Exact year-boundary target is observed but excluded from training.
    assert np.isfinite(extended['y_raw'][1,4]) and not extended['eligible'][1,4]
    assert not extended['eligible'][2,3:].any()
    # Exact lookup remains correct if one source hour is absent entirely.
    hours=b['hourly_index'];removed=hours.get_loc(pd.Timestamp('2023-01-03',tz='Asia/Seoul'))
    keep=np.arange(len(hours))!=removed
    y,e,_=build_exact_labels(hours[keep],raw[keep,0],b['origins'].origin_time_kst,
                             b['origins'].origin_split)
    assert np.isnan(y[4,3]) and not e[4,3]
    # +48 h is still located correctly, proving we did not shift source rows.
    target=pd.Timestamp('2023-01-04',tz='Asia/Seoul')
    assert y[4,5]==raw[hours.get_loc(target),0] and e[4,5]
    # Targets outside2024 are unavailable even when a buffer includes2025.
    end_hours=pd.date_range('2024-12-29','2025-01-03',freq='h',tz='Asia/Seoul')
    y_end,e_end,_=build_exact_labels(end_hours,np.ones(len(end_hours)),
        kst(['2024-12-30 00:00','2024-12-31 00:00']),['test','test'])
    assert np.isfinite(y_end[0,4]) and e_end[0,4]
    assert np.isnan(y_end[0,5]) and not e_end[0,5]
    assert np.isnan(y_end[1,3:]).all() and not e_end[1,3:].any()
    # The caller receives a new label/origin view and extension is idempotent.
    second=extend_bundle_horizons(extended)
    assert np.array_equal(second['y_raw'],extended['y_raw'],equal_nan=True)
    pd.testing.assert_frame_equal(second['origins'],extended['origins'])
    # Future observations can change future labels, never the input window.
    chosen=3;origin_pos=b['origin_positions'][chosen]
    altered=raw.copy();altered[origin_pos+1:,0]=123456.
    ym,em,_=build_exact_labels(hours,altered[:,0],b['origins'].origin_time_kst,
                               b['origins'].origin_split)
    assert ym[chosen,5]!=extended['y_raw'][chosen,5] and em[chosen,5]
    start=origin_pos-b['lookback_hours']+1
    assert np.array_equal(raw[start:origin_pos+1],altered[start:origin_pos+1],equal_nan=True)
    # Mismatched old labels or eligibility must stop the run.
    for key in ('y_raw','eligible'):
        corrupt=dict(b)
        corrupt[key]=b[key].copy()
        if key=='y_raw':corrupt[key][0,0]+=2.
        else:corrupt[key][0,0]=not corrupt[key][0,0]
        try:extend_bundle_horizons(corrupt)
        except ValueError:pass
        else:raise AssertionError('Corrupted legacy target contract was accepted: '+key)
    print(json.dumps({'tests':'passed','horizons':list(HORIZONS),
        'legacy_four_arrays_and_wide_columns_preserved_exactly':True,
        'new_labels_exact_elapsed_hours':True,'missing_source_hour_not_row_shifted':True,
        'missing_target_not_imputed':True,'cross_split_targets_ineligible':True,
        'study_end_targets_unavailable':True,'future_label_mutation_input_window_unchanged':True,
        'source_arrays_unmodified':True,'extension_idempotent':True,
        'invalid_legacy_labels_or_eligibility_rejected':True}))


if __name__=='__main__':main()
