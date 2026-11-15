"""Strict shared scoring: fixed rows, causal baselines, seed/ensemble separation.

No model-dependent row intersection. Eligible prediction omissions/nonfinite
values are errors. Temporal resampling and HAC retain the real 3-hour timeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from grid import HORIZONS, KEY, TIMEZONE, add_baselines, kst, read_frame, validate_baselines, validate_grid

CONDITIONS = ('season', 'route_group', 'route_history12h_sector', 'wind_sector', 'traj_wind_sector', 'cold_season')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')


def metric_values(y, yhat, persistence):
    y, yhat, persistence = [np.asarray(a, dtype=float) for a in (y, yhat, persistence)]
    if not len(y) or not all(np.isfinite(a).all() for a in (y, yhat, persistence)):
        raise ValueError('Metrics require a nonempty fixed sample with every prediction finite.')
    error, ref_error = yhat-y, persistence-y
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    ref_rmse, ref_mae = float(np.sqrt(np.mean(ref_error**2))), float(np.mean(np.abs(ref_error)))
    sst = float(np.sum((y-y.mean())**2))
    return {'n': len(y), 'RMSE': rmse, 'MSE': float(np.mean(error**2)), 'MAE': mae,
            'MBE': float(error.mean()), 'R2': 1-float(np.sum(error**2))/sst if sst > 0 else np.nan,
            'persistence_RMSE': ref_rmse, 'persistence_MAE': ref_mae,
            'RMSE_improvement_pct': 100*(1-rmse/ref_rmse) if ref_rmse > 0 else np.nan,
            'MAE_improvement_pct': 100*(1-mae/ref_mae) if ref_mae > 0 else np.nan}


def calendar_errors(times, e1, e2, full_origins):
    """Embed eligible errors on the complete original calendar grid, leaving gaps."""
    times, complete = kst(times), kst(full_origins)
    if complete.has_duplicates or not complete.is_monotonic_increasing:
        raise ValueError('Full origin grid must be sorted and unique.')
    if len(complete) > 1 and not np.all(np.diff(complete.asi8) == pd.Timedelta(hours=3).value):
        raise ValueError('Full origin grid must preserve every actual three-hour slot.')
    if times.has_duplicates:
        raise ValueError('Duplicate eligible error timestamps.')
    positions = complete.get_indexer(times)
    if (positions < 0).any():
        raise ValueError('Error timestamp is outside full calendar grid.')
    a, b = np.full(len(complete), np.nan), np.full(len(complete), np.nan)
    a[positions], b[positions] = np.asarray(e1, float), np.asarray(e2, float)
    if not np.isfinite(a[positions]).all() or not np.isfinite(b[positions]).all():
        raise ValueError('Eligible errors must be finite.')
    return a, b


def dm_calendar(e1, e2, horizon, lag=None):
    """Squared-loss DM with Bartlett HAC at exact calendar lags.

    Default lag restores the original overlap rule max(h/3-1,0). Missing slots
    have zero centered score; they do not become adjacent observations. The
    variance of the mean uses n_observed**2, not the compressed array length.
    """
    minimum = max(int(horizon)//3-1, 0)
    lag = minimum if lag is None else int(lag)
    if lag < minimum:
        raise ValueError(f'HAC lag {lag} is shorter than the overlap minimum {minimum}.')
    a, b = np.asarray(e1, float), np.asarray(e2, float)
    active = np.isfinite(a) & np.isfinite(b)
    n = int(active.sum())
    if n < 3:
        return {'dm_stat': np.nan, 'p': np.nan, 'hac_lag_slots': lag, 'hac_pairs_by_lag': [], 'dm_n': n}
    d = np.where(active, a*a-b*b, 0.0)
    mean = float(d[active].mean())
    centered = np.where(active, d-mean, 0.0)
    total = float(centered @ centered)
    pairs = []
    for k in range(1, lag+1):
        if k < len(d):
            n_pairs = int((active[k:] & active[:-k]).sum())
            term = float(centered[k:] @ centered[:-k])
        else:
            n_pairs, term = 0, 0.0
        total += 2*(1-k/(lag+1))*term
        pairs.append(n_pairs)
    variance = total/(n*n)
    if variance <= np.finfo(float).eps:
        # A constant nonzero loss difference has no estimable stochastic
        # variance, so it is not assigned an artificial p=0.
        stat, p = (0.0, 1.0) if abs(mean) < 1e-12 else (np.nan, np.nan)
    else:
        stat = mean/math.sqrt(variance)
        p = math.erfc(abs(stat)/math.sqrt(2))
    return {'dm_stat': stat, 'p': p, 'hac_lag_slots': lag, 'hac_pairs_by_lag': pairs, 'dm_n': n}


def bootstrap_calendar(e1, e2, reps=1000, block_slots=56, seed=0):
    """Moving 7-day calendar blocks; missing hours remain missing in each block.

    Samples full calendar slots, never contiguous runs of eligible rows. Each
    draw truncates to the original calendar length and averages finite pairs.
    """
    a, b = np.asarray(e1, float), np.asarray(e2, float)
    if len(a) != len(b) or not len(a):
        raise ValueError('Bootstrap needs equally sized nonempty calendar arrays.')
    if reps < 1 or block_slots < 1:
        raise ValueError('Positive bootstrap reps and block slots required.')
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError('Paired bootstrap errors have different availability masks.')
    n, length = len(a), min(int(block_slots), len(a))
    rng = np.random.default_rng(seed)
    starts = np.arange(n-length+1)
    blocks = math.ceil(n/length)
    values, sample_sizes = [], []
    for _ in range(int(reps)):
        positions = (rng.choice(starts, blocks)[:, None] + np.arange(length)[None, :]).ravel()[:n]
        aa, bb = a[positions], b[positions]
        good = np.isfinite(aa) & np.isfinite(bb)
        if good.any():
            values.append(float(np.sqrt(np.mean(aa[good]**2))-np.sqrt(np.mean(bb[good]**2))))
            sample_sizes.append(int(good.sum()))
    interval = np.percentile(values, [2.5, 97.5]) if len(values) >= max(10, reps*.8) else [np.nan, np.nan]
    return {'ci95_low': float(interval[0]), 'ci95_high': float(interval[1]),
            'bootstrap_valid_reps': len(values), 'bootstrap_reps': int(reps),
            'bootstrap_block_slots': length, 'bootstrap_block_hours': 3*length,
            'bootstrap_calendar_slots': n,
            'bootstrap_min_observed': min(sample_sizes) if sample_sizes else 0,
            'bootstrap_max_observed': max(sample_sizes) if sample_sizes else 0}


def holm(values):
    p = np.asarray(values, float)
    result = np.full(len(p), np.nan)
    eligible = np.flatnonzero(np.isfinite(p))
    order = eligible[np.argsort(p[eligible])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order)-rank)*p[index]))
        result[index] = running
    return result


def normalize_seed(value):
    if pd.isna(value) or str(value).strip() in {'', 'None', 'nan'}:
        return ''
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (ValueError, TypeError):
        pass
    return str(value)


def prepare_predictions(frame):
    frame = frame.copy()
    for old, new in [('horizon_h', 'horizon'), ('y_pred', 'y_hat'), ('origin_split', 'split'),
                     ('input_set', 'input_condition'), ('dataset_version', 'data_version')]:
        if new not in frame and old in frame:
            frame = frame.rename(columns={old: new})
    required = {'origin_time_kst', 'horizon', 'model', 'y_hat'}
    if not required <= set(frame):
        raise ValueError(f'Missing prediction columns: {sorted(required-set(frame))}')
    frame['origin_time_kst'] = kst(frame.origin_time_kst)
    frame['horizon'] = pd.to_numeric(frame.horizon, errors='raise').astype(int)
    frame['y_hat'] = pd.to_numeric(frame.y_hat, errors='coerce')
    if 'seed' not in frame:
        frame['seed'] = ''
    frame['seed'] = frame.seed.map(normalize_seed)
    if 'prediction_kind' not in frame:
        frame['prediction_kind'] = np.where(frame.seed.ne(''), 'seed', 'single')
    if not frame.prediction_kind.isin(['seed', 'single', 'ensemble']).all():
        raise ValueError('prediction_kind must be seed, single or ensemble.')
    if ((frame.prediction_kind == 'seed') & frame.seed.eq('')).any():
        raise ValueError('Seed predictions require a concrete seed.')
    if ((frame.prediction_kind != 'seed') & frame.seed.ne('')).any():
        raise ValueError('Single/ensemble prediction rows must not carry an individual seed.')
    if 'input_condition' not in frame:
        frame['input_condition'] = 'default'
    frame['input_condition'] = frame.input_condition.fillna('default').astype(str)
    if 'data_version' not in frame:
        frame['data_version'] = 'unspecified'
    frame['data_version'] = frame.data_version.fillna('unspecified').astype(str)
    frame['model'] = frame.model.astype(str)
    run_columns = ['model', 'input_condition', 'prediction_kind', 'seed']
    if frame.duplicated([*run_columns, *KEY]).any():
        raise ValueError('Duplicate predictions within model/input/kind/seed/origin/horizon; do not pool seeds.')
    return frame


def _run_id(model, condition, kind, seed):
    return f'{model}|{condition}|{kind}'+(f'|seed={seed}' if seed else '')


def prediction_runs(predictions, grid, models=None, make_ensembles=True):
    """Return strict complete runs on fixed eligible rows, plus raw-unit ensembles."""
    active = grid.loc[grid.eligible].copy().set_index(KEY)
    source = prepare_predictions(predictions)
    all_keys = pd.MultiIndex.from_frame(grid[KEY])
    if models:
        missing_models = set(models)-set(source.model)
        if missing_models:
            raise ValueError(f'Requested models absent from prediction inputs: {sorted(missing_models)}')
        source = source.loc[source.model.isin(models)].copy()
    source_keys = pd.MultiIndex.from_frame(source[KEY])
    if not source_keys.isin(all_keys).all():
        raise ValueError('Predictions include origin/horizon keys outside the fixed scoring split grid.')
    runs = {}
    for (model, condition, kind, seed), part in source.groupby(['model', 'input_condition', 'prediction_kind', 'seed'], dropna=False, sort=True):
        indexed = part.set_index(KEY)
        missing = active.index.difference(indexed.index)
        if len(missing):
            raise ValueError(f'{model}/{condition}/{kind}/{seed}: missing {len(missing)} eligible predictions; sample={missing[:3].tolist()}')
        ordered = indexed.loc[active.index]
        if not np.isfinite(ordered.y_hat.to_numpy(float)).all():
            raise ValueError(f'{model}/{condition}/{kind}/{seed}: nonfinite eligible prediction; rows must not be dropped.')
        for truth_column in ['y_true', 'y']:
            if truth_column in ordered:
                truth = pd.to_numeric(ordered[truth_column], errors='coerce').to_numpy(float)
                if not np.allclose(truth, active.y_true.to_numpy(float), atol=1e-6, rtol=0, equal_nan=False):
                    raise ValueError(f'{model}: prediction file truth differs from authoritative fixed targets.')
        if 'split' in ordered and not ordered.split.eq(active.split.to_numpy()).all():
            raise ValueError(f'{model}: prediction split does not match fixed grid.')
        ident = _run_id(model, condition, kind, seed)
        versions = ordered.data_version.unique()
        if len(versions) != 1:
            raise ValueError(f'{model}/{condition}/{kind}/{seed}: mixed data versions inside one run.')
        runs[ident] = {'model': model, 'input_condition': condition, 'prediction_kind': kind,
                       'seed': seed, 'run_id': ident, 'raw': ordered.y_hat.to_numpy(float),
                       'ensemble_size': 1, 'ensemble_rule': '', 'data_version': versions[0]}
    if make_ensembles:
        families = sorted({(r['model'], r['input_condition']) for r in runs.values() if r['prediction_kind']=='seed'})
        for model, condition in families:
            members = [r for r in runs.values() if r['model']==model and r['input_condition']==condition and r['prediction_kind']=='seed']
            if len(members) < 2:
                continue
            versions = {r['data_version'] for r in members}
            if len(versions) != 1:
                raise ValueError(f'{model}/{condition}: cannot ensemble seeds trained on different data versions.')
            if any(r['model']==model and r['input_condition']==condition and r['prediction_kind']=='ensemble' for r in runs.values()):
                raise ValueError(f'{model}/{condition}: an ensemble is supplied and auto-generated; use --no-ensemble or omit supplied ensemble.')
            ident = _run_id(model, condition, 'ensemble', '')
            runs[ident] = {'model': model, 'input_condition': condition, 'prediction_kind': 'ensemble', 'seed': '',
                           'run_id': ident, 'raw': np.mean(np.stack([r['raw'] for r in members]), axis=0),
                           'ensemble_size': len(members), 'ensemble_rule': 'arithmetic mean of seed forecasts in ug/m3 before final nonnegative clipping',
                           'data_version': next(iter(versions))}
    for r in runs.values():
        r['y_hat'] = np.maximum(r['raw'], 0.0)
    return active, runs


def _resolve_model_alias(alias, runs):
    if alias in runs:
        return alias
    matches = [key for key, run in runs.items() if run['model']==alias]
    if len(matches)==1:
        return matches[0]
    ensembles = [key for key in matches if runs[key]['prediction_kind']=='ensemble']
    if len(ensembles)==1:
        return ensembles[0]
    raise ValueError(f'Comparison alias {alias!r} is ambiguous/absent; supply an exact run_id.')


def score_frames(grid, predictions, split='validation', models=None, pairs=(), final=False,
                 bootstrap_reps=1000, bootstrap_seed=0, hac_lag=None, make_ensembles=True):
    if split=='test' and not final:
        raise ValueError('2024 test scoring requires --final after settings are frozen; this is not a claim that 2024 has never been inspected.')
    frozen = validate_baselines(validate_grid(grid))
    fixed = frozen.loc[frozen.split.eq(split)].copy()
    if fixed.empty:
        raise ValueError(f'No grid rows for {split}.')
    source = prepare_predictions(predictions)
    origin_set = pd.Index(fixed.origin_time_kst.unique())
    if 'split' in source:
        source = source.loc[source.split.eq(split)].copy()
    else:
        source = source.loc[source.origin_time_kst.isin(origin_set)].copy()
    active, runs = prediction_runs(source, fixed, models, make_ensembles)
    if active.empty:
        raise ValueError('No fixed eligible rows in scoring split.')
    for baseline, column in [('persistence', 'persistence_y_hat'), ('seasonal_naive', 'seasonal_y_hat')]:
        ident = _run_id(baseline, 'common_baseline', 'single', '')
        if any(r['model']==baseline for r in runs.values()):
            raise ValueError('Baselines are generated from the authoritative target series; do not supply external baseline predictions.')
        raw = active[column].to_numpy(float)
        runs[ident] = {'model': baseline, 'input_condition': 'common_baseline', 'prediction_kind': 'single',
                       'seed': '', 'run_id': ident, 'raw': raw, 'y_hat': np.maximum(raw, 0),
                       'ensemble_size': 1, 'ensemble_rule': '', 'data_version': 'authoritative_target_hourly'}
    persistence_key = _run_id('persistence', 'common_baseline', 'single', '')
    ref = runs[persistence_key]['y_hat']
    comparison_pairs = [(key, persistence_key) for key in runs if key!=persistence_key]
    for pair in pairs:
        if isinstance(pair, str):
            if ':' not in pair:
                raise ValueError('Extra comparisons use MODEL_A:MODEL_B.')
            a, b = pair.split(':', 1)
        else:
            a, b = pair
        resolved = (_resolve_model_alias(a, runs), _resolve_model_alias(b, runs))
        if resolved not in comparison_pairs:
            comparison_pairs.append(resolved)
    metric_rows, condition_rows, test_rows, row_records = [], [], [], []
    active_reset = active.reset_index()
    for horizon in sorted(fixed.horizon.unique()):
        hmask = active_reset.horizon.eq(horizon).to_numpy()
        hg = active_reset.loc[hmask]
        if hg.empty:
            continue
        all_times = pd.DatetimeIndex(fixed.loc[fixed.horizon.eq(horizon), 'origin_time_kst']).sort_values()
        truth = hg.y_true.to_numpy(float)
        for ident, run in runs.items():
            meta = {k: run[k] for k in ('model', 'input_condition', 'prediction_kind', 'seed', 'run_id', 'ensemble_size', 'ensemble_rule', 'data_version')}
            values = run['y_hat'][hmask]
            raw = run['raw'][hmask]
            metric = {'split': split, 'horizon': int(horizon), **meta, **metric_values(truth, values, ref[hmask]),
                      'negative_predictions_clipped': int((raw<0).sum()), 'fixed_eligible_rows': len(hg)}
            for b in ['persistence', 'seasonal']:
                if f'{b}_fallback' in hg:
                    metric[f'{b}_fallback_n'] = int(hg[f'{b}_fallback'].sum())
            metric_rows.append(metric)
            record = hg.copy()
            for key, value in meta.items():
                record[key] = value
            record['y_hat_raw'], record['y_hat'] = raw, values
            row_records.append(record)
            for col in CONDITIONS:
                if col not in hg:
                    continue
                for level, part in hg.groupby(col, dropna=False):
                    indexer = hg.index.get_indexer(part.index)
                    condition_rows.append({'split': split, 'horizon': int(horizon), **meta, 'condition': col,
                                           'level': str(level), **metric_values(truth[indexer], values[indexer], ref[hmask][indexer])})
            high = truth >= 35
            if high.any():
                condition_rows.append({'split': split, 'horizon': int(horizon), **meta, 'condition': 'y_ge_35',
                                       'level': 'high', **metric_values(truth[high], values[high], ref[hmask][high])})
        for a, b in comparison_pairs:
            e1, e2 = runs[a]['y_hat'][hmask]-truth, runs[b]['y_hat'][hmask]-truth
            ca, cb = calendar_errors(hg.origin_time_kst, e1, e2, all_times)
            stat = dm_calendar(ca, cb, int(horizon), lag=hac_lag)
            boot = bootstrap_calendar(ca, cb, reps=bootstrap_reps, seed=bootstrap_seed)
            stat['hac_pairs_by_lag'] = json.dumps(stat['hac_pairs_by_lag'])
            test_rows.append({'split': split, 'horizon': int(horizon), 'run_id': a, 'versus_run_id': b,
                              'n': len(hg), 'rmse_diff': float(np.sqrt(np.mean(e1**2))-np.sqrt(np.mean(e2**2))), **stat, **boot})
    metrics = pd.DataFrame(metric_rows)
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests['p_holm'] = holm(tests.p)
        tests['improved'] = (tests.ci95_high < 0) & (tests.p_holm < .05)
    seeds = metrics.loc[metrics.prediction_kind.eq('seed')]
    summaries = []
    for keys, part in seeds.groupby(['split', 'horizon', 'model', 'input_condition']):
        record = dict(zip(['split','horizon','model','input_condition'], keys))
        record.update(seed_count=len(part), seeds=','.join(sorted(part.seed.astype(str))),
                      summary_kind='mean_and_sample_std_of_individual_seed_metrics_not_ensemble')
        for col in ['RMSE','MAE','MSE','MBE','R2','RMSE_improvement_pct','MAE_improvement_pct']:
            record[col+'_mean'] = float(part[col].mean())
            record[col+'_std'] = float(part[col].std(ddof=1)) if len(part)>1 else np.nan
        summaries.append(record)
    audit = {'protocol_version': 'fixed-calendar-seed-h48-v1.0.0', 'split': split,
             'fixed_eligible_by_horizon': {str(int(h)): int(g.eligible.sum()) for h,g in fixed.groupby('horizon')},
             'fixed_calendar_slots_by_horizon': {str(int(h)): len(g) for h,g in fixed.groupby('horizon')},
             'rows_selected_using_predictions': False, 'model_prediction_intersection_used': False,
             'negative_prediction_rule': 'clip each scored forecast at zero; count per run and horizon',
             'seed_score_summary_is_not_prediction_ensemble': True,
             'ensemble_rule': 'raw-unit arithmetic mean, then nonnegative clip',
             'hac_default': 'max(horizon/3-1,0), Bartlett, exact elapsed-time slots, gaps retained',
             'hac_lag_override': hac_lag, 'bootstrap': 'moving 7-day=56 calendar-slot blocks; gaps remain inside blocks',
             'bootstrap_reps': int(bootstrap_reps), 'bootstrap_seed': int(bootstrap_seed),
             'holm_family': 'all finite p-values in this declared scoring invocation (all included horizons/runs/pairs)',
             'test_guard': bool(final), 'does_not_train_models': True,
             'inference_caveat': 'Overlap minimum HAC lag need not capture all additional serial dependence; predeclare lag/block sensitivity, do not tune on significance.'}
    return {'metrics': metrics, 'seed_metric_summary': pd.DataFrame(summaries),
            'tests': tests, 'conditions': pd.DataFrame(condition_rows),
            'score_rows': pd.concat(row_records, ignore_index=True), 'audit': audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True,
                        help='Derived six-horizon grid produced by horizon_contract.ensure_h48_grid; the old Dataset grid remains unchanged.')
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--target-hourly', type=Path, nargs='+', help='Hourly PM2.5 file(s), including optional history buffer; target station is selected when station_id exists.')
    parser.add_argument('--target-station-id', default='111121')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--pairs', nargs='*', default=[])
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--no-ensemble', action='store_true')
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--hac-lag', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.split=='test' and not args.final:
        parser.error('2024 test is locked. Use --final after the experiment specification is frozen.')
    g = validate_grid(read_frame(args.grid))
    if set(g.horizon) != set(HORIZONS):
        parser.error('The H48 scoring CLI requires the derived six-horizon evaluation grid.')
    if args.target_hourly:
        target = pd.concat([read_frame(p) for p in args.target_hourly], ignore_index=True)
        if 'station_id' in target:
            target = target.loc[target.station_id.astype(str).eq(args.target_station_id)].copy()
        if target.empty:
            parser.error('No target observations after station_id selection.')
        g = add_baselines(g, target)
    frames, provenance = [], []
    for path in args.predictions:
        frames.append(read_frame(path))
        sidecar = path.with_suffix('.meta.json')
        provenance.append({'path': str(path.resolve()), 'sha256': sha256(path),
                           'sidecar': json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.exists() else None})
    result = score_frames(g, pd.concat(frames, ignore_index=True), args.split, args.models, args.pairs,
                          args.final, args.bootstrap_reps, args.bootstrap_seed, args.hac_lag, not args.no_ensemble)
    # Keep every invocation in a new folder; the six v5 output schemas below
    # and all calculation functions above remain unchanged.
    import uuid
    base_output = args.output_dir
    base_output.mkdir(parents=True, exist_ok=True)
    while True:
        run_name = datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]
        candidate_output = base_output / run_name
        try:
            candidate_output.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        args.output_dir = candidate_output
        break
    for name in ['metrics','seed_metric_summary','tests','conditions']:
        result[name].to_csv(args.output_dir/f'{name}.csv', index=False, encoding='utf-8-sig')
    result['score_rows'].to_parquet(args.output_dir/'score_rows.parquet', index=False)
    result['audit'].update(written_kst=datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                           grid_path=str(args.grid.resolve()), grid_sha256=sha256(args.grid), prediction_sources=provenance)
    if args.target_hourly:
        result['audit']['target_hourly'] = [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in args.target_hourly]
        result['audit']['target_station_id'] = args.target_station_id
    json_write(args.output_dir/'scoring_audit.json', result['audit'])
    print(result['metrics'].to_string(index=False))
    print('Saved scoring output:', args.output_dir)


if __name__=='__main__':
    main()
