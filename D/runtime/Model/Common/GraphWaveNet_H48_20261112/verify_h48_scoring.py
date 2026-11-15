"""CPU-only regression/causality checks for the private H48 common scorer.

Run with --original-common pointing at the read-only, original Model/Common.
No model training or 2024 performance analysis is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_error(action, text=None):
    try:
        action()
    except (ValueError, FileNotFoundError) as error:
        if text and text not in str(error):
            raise AssertionError('Unexpected failure: ' + str(error)) from error
        return
    raise AssertionError('Expected a strict contract rejection.')


def fixture(old_grid, start='2022-12-29', end='2024-01-03 21:00'):
    origin = pd.date_range(start, end, freq='3h', tz='Asia/Seoul')
    hourly = pd.date_range('2021-01-01', '2025-01-05 23:00', freq='h', tz='Asia/Seoul')
    n = np.arange(len(hourly))
    values = np.round(22 + 8*np.sin(n/31) + 5*np.cos(n/113)).astype(float)
    values[(hourly >= pd.Timestamp('2023-02-02', tz='Asia/Seoul')) &
           (hourly < pd.Timestamp('2023-02-04', tz='Asia/Seoul'))] = np.nan
    values[hourly >= pd.Timestamp('2025-01-01', tz='Asia/Seoul')] = 999999.0
    target = pd.Series(values, index=hourly)
    pieces = []
    for horizon in (3, 6, 12, 24):
        requested = origin + pd.Timedelta(hours=horizon)
        label = target.reindex(requested).to_numpy(float, copy=True)
        label[requested >= pd.Timestamp('2025-01-01', tz='Asia/Seoul')] = np.nan
        split = old_grid.split_for_time(origin)
        eligible = np.isfinite(label) & (split == old_grid.split_for_time(requested))
        frame = pd.DataFrame({'origin_time_kst': origin, 'split': split,
                              'season': np.where(origin.month.isin((12, 1, 2)), 'DJF', 'other'),
                              'route_group': np.where(np.arange(len(origin)) % 2, 'west', 'other'),
                              'horizon': horizon, 'y_true': label, 'eligible': eligible,
                              'target_time_kst': requested})
        pieces.append(frame)
    base = old_grid.add_baselines(old_grid.validate_grid(pd.concat(pieces, ignore_index=True)),
                                  pd.DataFrame({'time_kst': hourly, 'pm25': values}))
    for prefix in ('persistence', 'seasonal'):
        base[prefix + '_observation_time_kst'] = (base[prefix + '_request_time_kst'] -
                                                  pd.to_timedelta(base[prefix + '_age_hours'], unit='h'))
    bundle = {'hourly_index': hourly, 'pm_values': values[:, None],
              'origins': pd.DataFrame({'origin_time_kst': origin})}
    return base, bundle


def predictions(grid, models=('M1', 'M2', 'M3', 'M4', 'M3T', 'M4T'), seeds=(11,)):
    active = grid.loc[grid.split.eq('validation') & grid.eligible].copy()
    frames = []
    for i, model in enumerate(models):
        for seed in seeds:
            frame = active[['origin_time_kst', 'horizon', 'target_time_kst', 'split']].copy()
            frame['model'], frame['input_condition'], frame['seed'] = model, 'all_conditions', str(seed)
            frame['prediction_kind'], frame['data_version'] = 'seed', 'synthetic-h48'
            frame['y_hat'] = active.y_true.to_numpy() + (i+1)*.4*np.sin(np.arange(len(active))*.13) + (seed-11)*.03
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-common', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    neural = load(folder/'neural_evaluation.py', '_h48_verify_neural')
    scorer, new_grid, _ = neural._load_common('.', folder)
    old_scorer, old_grid, _ = neural._load_common('.', args.original_common)
    contract = load(folder/'horizon_contract.py', '_h48_verify_contract')
    protected = {str(p): sha(p) for p in (args.original_common/'grid.py', args.original_common/'score.py',
                                          args.original_common/'neural_evaluation.py')}
    base, bundle = fixture(old_grid)
    extended = contract.build_h48_grid(base, bundle)
    contract.assert_original_rows_preserved(base, extended)
    assert set(extended.horizon) == {3, 6, 12, 24, 36, 48}
    assert extended.groupby('horizon').size().nunique() == 1
    for horizon, offset in ((36, -12), (48, 0)):
        part = extended.loc[extended.horizon.eq(horizon)]
        pd.testing.assert_index_equal(new_grid.kst(part.seasonal_request_time_kst),
                                     new_grid.kst(part.origin_time_kst + pd.Timedelta(hours=offset)), check_names=False)
        assert (new_grid.kst(part.seasonal_request_time_kst) <= new_grid.kst(part.origin_time_kst)).all()
    # Exact missing labels stay missing, and annual train/validation/test crossings are ineligible.
    gaps = extended.target_time_kst.between(pd.Timestamp('2023-02-02', tz='Asia/Seoul'),
                                           pd.Timestamp('2023-02-04', tz='Asia/Seoul'), inclusive='left')
    assert extended.loc[gaps, 'y_true'].isna().all() and not extended.loc[gaps, 'eligible'].any()
    crossings = new_grid.split_for_time(extended.target_time_kst) != extended.split.to_numpy()
    assert crossings.any() and not extended.loc[crossings, 'eligible'].any()
    base_end, bundle_end = fixture(old_grid, '2024-12-30', '2024-12-31 21:00')
    end = contract.build_h48_grid(base_end, bundle_end)
    outside = end.target_time_kst >= pd.Timestamp('2025-01-01', tz='Asia/Seoul')
    assert outside.any() and not end.loc[outside, 'eligible'].any() and end.loc[outside, 'y_true'].isna().all()
    # Future measurements cannot influence any baseline at an origin.
    t = pd.Timestamp('2023-01-05 09:00', tz='Asia/Seoul')
    one = extended.loc[extended.origin_time_kst.eq(t)].copy()
    raw = pd.DataFrame({'time_kst': bundle['hourly_index'], 'pm25': bundle['pm_values'][:, 0]})
    original_baselines = new_grid.add_baselines(one, raw)
    poisoned = raw.copy()
    poisoned.loc[poisoned.time_kst.gt(t), 'pm25'] = 99999999.
    poison_baselines = new_grid.add_baselines(one, poisoned)
    baseline_columns = [c for c in original_baselines if c.startswith(('persistence_', 'seasonal_'))]
    pd.testing.assert_frame_equal(original_baselines[baseline_columns], poison_baselines[baseline_columns], check_exact=True)
    future_grid = extended.copy()
    ix = future_grid.index[future_grid.horizon.eq(36)][0]
    future_grid.loc[ix, 'seasonal_request_time_kst'] = future_grid.loc[ix, 'origin_time_kst'] + pd.Timedelta(hours=12)
    require_error(lambda: new_grid.validate_baselines(future_grid), 'baseline request')
    # Metrics and score-row field contracts regress exactly on the original four horizons.
    p6 = predictions(extended)
    p4 = p6.loc[p6.horizon.isin((3, 6, 12, 24))].copy()
    pairs = [('M2','M1'), ('M3','M2'), ('M4','M3'), ('M3T','M3'), ('M4T','M4'), ('M3','M1'), ('M4','M1')]
    old = old_scorer.score_frames(base, p4, pairs=pairs, bootstrap_reps=20, bootstrap_seed=23)
    new4 = scorer.score_frames(extended.loc[extended.horizon.le(24)], p4, pairs=pairs, bootstrap_reps=20, bootstrap_seed=23)
    new6 = scorer.score_frames(extended, p6, pairs=pairs, bootstrap_reps=20, bootstrap_seed=23)
    for name in ('metrics', 'seed_metric_summary', 'tests', 'conditions', 'score_rows'):
        pd.testing.assert_frame_equal(old[name], new4[name], check_exact=True)
    for name in ('metrics', 'seed_metric_summary', 'tests', 'conditions', 'score_rows'):
        assert list(old[name].columns) == list(new6[name].columns), name
        assert old[name].dtypes.astype(str).to_dict() == new6[name].dtypes.astype(str).to_dict(), name
    old_metrics_from_six = new6['metrics'].loc[new6['metrics'].horizon.le(24)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old['metrics'], old_metrics_from_six, check_exact=True)
    assert set(old['audit']) == set(new6['audit'])
    assert len(new6['tests']) == 84
    assert new6['tests'].loc[new6['tests'].horizon.eq(36), 'hac_lag_slots'].eq(11).all()
    assert new6['tests'].loc[new6['tests'].horizon.eq(48), 'hac_lag_slots'].eq(15).all()
    assert new6['tests'].bootstrap_block_slots.eq(56).all()
    require_error(lambda: scorer.score_frames(extended, p6.iloc[1:], bootstrap_reps=10), 'missing')
    bad = p6.copy(); bad.loc[0, 'y_hat'] = np.nan
    require_error(lambda: scorer.score_frames(extended, bad, bootstrap_reps=10), 'nonfinite')
    require_error(lambda: scorer.score_frames(extended, p6, split='test', bootstrap_reps=10), 'requires')
    p_multi = predictions(extended, models=('M1',), seeds=(11,22))
    ensemble = scorer.score_frames(extended, p_multi, bootstrap_reps=10)
    assert len(ensemble['metrics'].loc[ensemble['metrics'].prediction_kind.eq('seed')]) == 12
    assert len(ensemble['metrics'].loc[ensemble['metrics'].prediction_kind.eq('ensemble')]) == 6
    rows = ensemble['score_rows']
    expected = rows.loc[rows.prediction_kind.eq('seed')].groupby(['origin_time_kst','horizon']).y_hat_raw.mean()
    actual = rows.loc[rows.prediction_kind.eq('ensemble')].set_index(['origin_time_kst','horizon']).y_hat_raw
    np.testing.assert_allclose(actual.sort_index(), expected.sort_index(), rtol=0, atol=1e-14)
    # Full wrapper output names/schema and collision-free new invocation folders.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='h48_common_', dir=args.report.parent) as tmp:
        project = Path(tmp)/'TSA'
        private = project/'Model/Common/GraphWaveNet_H48_20261112'
        private.mkdir(parents=True)
        for name in ('grid.py','score.py','neural_evaluation.py','horizon_contract.py'):
            shutil.copy2(folder/name, private/name)
        base_path = project/contract.BASE_GRID_RELATIVE_PATH
        base_path.parent.mkdir(parents=True)
        base.to_parquet(base_path,index=False)
        cached = contract.ensure_h48_grid(project, project/'Model/Cache/evaluation', bundle)
        before = sha(cached)
        assert contract.ensure_h48_grid(project, project/'Model/Cache/evaluation', bundle) == cached
        assert sha(cached) == before
        wrong_bundle = dict(bundle, origins=bundle['origins'].iloc[:-1])
        require_error(lambda: contract.ensure_h48_grid(project, project/'Model/Cache/evaluation', wrong_bundle), 'calendars differ')
        require_error(lambda: contract.ensure_h48_grid(project, project/'Dataset/derived', bundle), 'never Dataset')
        run = project/'Result/GraphWaveNet_H48/run'
        run.mkdir(parents=True)
        export = run/'Exports/validation_predictions_by_seed.parquet'
        export.parent.mkdir()
        p6.to_parquet(export,index=False)
        metadata = {'horizons':list(new_grid.HORIZONS),'evaluation_grid_path':str(cached)}
        specs=[{'model':name,'graph':'distance'} for name in ('M1','M2','M3','M4','M3T','M4T')]
        plan = neural.write_neural_scoring_plan(run,'GraphWaveNet_H48','all_conditions',specs,[11],metadata,pairs)
        assert plan['training_metadata']['evaluation_grid_sha256'] == sha(cached)
        missing_h48 = p6.loc[p6.horizon.ne(48)]
        require_error(lambda: neural._validate_predictions(missing_h48,scorer,plan,'validation'), 'six horizons')
        result = neural.score_neural_run(project,run,'GraphWaveNet_H48','all_conditions',bootstrap_reps=10,make_plots=False)
        saved_files = {name:sha(Path(result['output_dir'])/name) for name in
                       ('metrics.csv','seed_metric_summary.csv','tests.csv','conditions.csv','score_rows.parquet','scoring_audit.json')}
        repeated = neural.score_neural_run(project,run,'GraphWaveNet_H48','all_conditions',bootstrap_reps=10,make_plots=False)
        assert repeated['output_dir'] != result['output_dir']
        assert all(sha(Path(result['output_dir'])/name)==digest for name,digest in saved_files.items())
        for name in ('metrics','seed_metric_summary','tests','conditions'):
            assert list(pd.read_csv(Path(result['output_dir'])/(name+'.csv')).columns) == list(old[name].columns)
        assert list(pd.read_parquet(Path(result['output_dir'])/'score_rows.parquet').columns) == list(old['score_rows'].columns)
        names=sorted(saved_files)
    assert all(sha(path)==digest for path,digest in protected.items())
    report = {'status':'passed','horizons':list(new_grid.HORIZONS),
              'original_four_grid_rows_exactly_preserved':True,'original_four_metrics_exactly_preserved':True,
              'original_four_scores_equal_when_scored_as_four':True,
              'holm_note':'Adding horizons expands the predeclared test family; adjusted p-values are not asserted unchanged.',
              'exact_target_lookup_missing_split_and_study_boundaries_tested':True,
              'future_seasonal_poison_invariant':True,'future_baseline_request_rejected':True,
              'fixed_rows_missing_or_nonfinite_prediction_rejected':True,
              'hac_lags':{'36':11,'48':15},'bootstrap_block_slots':56,
              'six_models_seed11_seven_input_contrasts_test_rows':len(new6['tests']),
              'individual_seed_and_raw_unit_ensemble_separate':True,
              'standard_output_filenames':names,'column_order_and_dtypes_match_original':True,
              'repeated_scoring_creates_new_folder_without_overwriting':True,
              'derived_grid_cached_under_model_only':True,'old_source_sha256_unchanged':protected}
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
