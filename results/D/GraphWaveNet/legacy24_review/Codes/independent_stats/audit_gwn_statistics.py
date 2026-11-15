"""Read-only verification of archived Graph WaveNet predictions and common scores."""
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
sys.path.insert(0, str(PROJECT / 'Model/Common'))
import score

ARCHIVES = [Path('C:/Users/kimjy/Downloads') / f'VastAI-20261111T003043Z-1-00{i}.zip' for i in (1, 2)]
BATCH = 'VastAI/20261111_014538_406520_eaa72758/'
handles = [zipfile.ZipFile(p) for p in ARCHIVES]
members = {}
for z in handles:
    for i in z.infolist():
        if i.filename in members:
            prev = members[i.filename]
            assert (prev[1].file_size, prev[1].CRC) == (i.file_size, i.CRC)
        members[i.filename] = (z, i)

def read(name):
    z, i = members[name]
    return z.read(i)

def find(suffix, prefix=BATCH):
    ns = [n for n in members if n.startswith(prefix) and n.endswith(suffix)]
    assert len(ns) == 1, ns
    return ns[0]

def jread(name):
    return json.loads(read(name))

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))

grid_path = PROJECT / 'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
grid = pd.read_parquet(grid_path)
audit = {'batch': BATCH, 'archives': list(map(str, ARCHIVES)), 'grid': str(grid_path),
         'grid_sha256': hashlib.sha256(grid_path.read_bytes()).hexdigest(),
         'scorer_sha256': hashlib.sha256(Path(score.__file__).read_bytes()).hexdigest(),
         'splits': {}}
keys = ['origin_time_kst', 'horizon']
model = 'GraphWaveNet_distance_fixed_plus_adaptive'
ids = {c: f'{model}|{c}|seed|seed=11' for c in ['M1', 'M2', 'M3', 'M4', 'M3T', 'M4T']}
# Actual origin concentration reconstructed from authoritative labels at their target time.
now = grid[['target_time_kst', 'y_true']].drop_duplicates().dropna()
assert not now.duplicated('target_time_kst').any()
current = now.set_index('target_time_kst').y_true

for split in ['validation', 'test']:
    prefix = BATCH + f'Common_Scoring/Combined/{split}/'
    archived = {}
    for name in ['metrics', 'tests', 'conditions', 'seed_metric_summary']:
        archived[name] = pd.read_csv(io.BytesIO(read(find('/' + name + '.csv', prefix))))
        archived[name].to_csv(OUT / f'{split}_original_{name}.csv', index=False)
    archived_rows = pd.read_parquet(io.BytesIO(read(find('/score_rows.parquet', prefix))))
    plan = jread(find('/input_condition_comparison_plan.json', prefix))
    archived_audit = jread(find('/scoring_audit.json', prefix))
    assert archived_audit['grid_sha256'] == audit['grid_sha256']
    assert set(plan['common_code_sha256'].values()).issuperset({audit['scorer_sha256']})
    frames = []
    source_audit = {}
    for condition in ids:
        p = find(f'/Exports/{split}_predictions_by_seed.parquet', BATCH + 'Conditions/' + condition + '/')
        data = read(p)
        f = pd.read_parquet(io.BytesIO(data))
        assert f.input_condition.eq(condition).all()
        assert f.seed.astype(str).str.replace('.0', '', regex=False).eq('11').all()
        assert not f.duplicated(keys).any()
        assert np.isfinite(f.y_hat).all()
        assert (((f.target_time_kst-f.origin_time_kst)/pd.Timedelta(hours=1)) == f.horizon).all()
        frames.append(f)
        source_audit[condition] = {'path': p, 'sha256': hashlib.sha256(data).hexdigest(), 'rows': len(f)}
    predictions = pd.concat(frames, ignore_index=True)
    results = score.score_frames(grid, predictions, split=split, pairs=plan['predeclared_pairs'],
                                 final=split == 'test', bootstrap_reps=1000,
                                 bootstrap_seed=0, make_ensembles=False)
    checks = {}
    for name, cols, index in [
        ('metrics', ['n', 'RMSE', 'MSE', 'MAE', 'MBE', 'R2', 'RMSE_improvement_pct',
                     'negative_predictions_clipped'], ['run_id', 'horizon']),
        ('tests', ['n', 'rmse_diff', 'p', 'p_holm', 'ci95_low', 'ci95_high', 'improved'],
         ['run_id', 'versus_run_id', 'horizon'])]:
        a = archived[name].set_index(index).sort_index()
        b = results[name].set_index(index).sort_index()
        assert a.index.equals(b.index)
        differences = {}
        for col in cols:
            np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float),
                                       atol=1e-10, rtol=1e-10, equal_nan=True)
            differences[col] = float(np.max(np.abs(a[col].to_numpy(float)-b[col].to_numpy(float))))
        checks[name] = differences
    # Every recorded forecast, truth and time is exactly recoverable from raw exports + frozen grid.
    row_index = ['run_id', *keys]
    a = archived_rows.set_index(row_index).sort_index()
    b = results['score_rows'].set_index(row_index).sort_index()
    assert a.index.equals(b.index)
    for col in ['y_true', 'y_hat', 'y_hat_raw', 'persistence_y_hat', 'seasonal_y_hat']:
        np.testing.assert_allclose(a[col], b[col], atol=0, rtol=0)
    assert (a.target_time_kst == b.target_time_kst).all()
    metrics = results['metrics']
    neural = metrics.loc[metrics.prediction_kind.eq('seed')]
    for metric in ['RMSE', 'MAE', 'MBE', 'R2', 'RMSE_improvement_pct']:
        neural.pivot(index='input_condition', columns='horizon', values=metric).to_csv(OUT / f'{split}_{metric.lower()}_pivot.csv')
    contrasts = results['tests'].loc[~results['tests'].versus_run_id.str.startswith('persistence|')].copy()
    contrasts['candidate'] = contrasts.run_id.str.split('|').str[1]
    contrasts['reference'] = contrasts.versus_run_id.str.split('|').str[1]
    lookup = neural.set_index(['input_condition', 'horizon'])
    contrasts['reference_RMSE'] = [lookup.loc[(r, h), 'RMSE'] for r, h in zip(contrasts.reference, contrasts.horizon)]
    contrasts['candidate_RMSE'] = [lookup.loc[(r, h), 'RMSE'] for r, h in zip(contrasts.candidate, contrasts.horizon)]
    contrasts['gain_pct'] = -100*contrasts.rmse_diff/contrasts.reference_RMSE
    contrasts['significant_worsening'] = (contrasts.ci95_low > 0) & (contrasts.p_holm < .05)
    contrasts.to_csv(OUT / f'{split}_planned_input_comparisons.csv', index=False)
    # One extra all-observations vs domestic comparison; not in archived predeclared family.
    exploratory = []
    all_rows = results['score_rows']
    parts = {c: all_rows.loc[all_rows.run_id.eq(ids[c])].set_index(keys).sort_index() for c in ids}
    pair_list = [('M4', 'M1'), ('M3T', 'M1'), ('M4T', 'M1')]
    for candidate, reference in pair_list:
        ca, cb = parts[candidate], parts[reference]
        assert ca.index.equals(cb.index)
        for h in [3, 6, 12, 24]:
            x = ca.xs(h, level='horizon')
            y = cb.xs(h, level='horizon')
            truth = x.y_true.to_numpy(float)
            ea = x.y_hat.to_numpy(float)-truth
            eb = y.y_hat.to_numpy(float)-truth
            full = grid.loc[grid.split.eq(split) & grid.horizon.eq(h), 'origin_time_kst'].sort_values()
            aa, bb = score.calendar_errors(x.index, ea, eb, full)
            row = {'split': split, 'candidate': candidate, 'reference': reference, 'horizon': h,
                   'n': len(x), 'rmse_diff': float(np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2))),
                   'gain_pct': float(100*(1-np.sqrt(np.mean(ea**2)/np.mean(eb**2)))),
                   **score.dm_calendar(aa, bb, h), **score.bootstrap_calendar(aa, bb, reps=1000, seed=0)}
            exploratory.append(row)
    extra = pd.DataFrame(exploratory)
    extra['p_holm_exploratory12'] = score.holm(extra.p)
    extra['improved_exploratory12'] = (extra.ci95_high < 0) & (extra.p_holm_exploratory12 < .05)
    extra.to_csv(OUT / f'{split}_exploratory_vs_domestic_12tests.csv', index=False)
    # Descriptive strata only: no conditional significance or claim of source contribution.
    cond_rows = []
    for candidate, reference in [('M4', 'M1'), ('M3T', 'M3'), ('M4T', 'M4')]:
        a = parts[candidate].reset_index()
        b = parts[reference].reset_index()
        assert a[keys].equals(b[keys])
        a['reference_y_hat'] = b.y_hat.to_numpy()
        a['current_pm25'] = a.origin_time_kst.map(current)
        a['current_pm25_bin'] = pd.cut(a.current_pm25, [-np.inf, 15, 35, 75, np.inf],
                                         labels=['<15', '15-<35', '35-<75', '>=75'], right=False).astype(str)
        a['future_pm25_bin'] = np.where(a.y_true >= 35, '>=35', '<35')
        for col in ['season', 'cold_season', 'route_group', 'wind_sector', 'traj_wind_sector',
                    'current_pm25_bin', 'future_pm25_bin']:
            for (h, level), f in a.groupby(['horizon', col], dropna=False):
                truth = f.y_true.to_numpy(float)
                va = score.metric_values(truth, f.y_hat, f.persistence_y_hat)
                vb = score.metric_values(truth, f.reference_y_hat, f.persistence_y_hat)
                cond_rows.append({'split': split, 'candidate': candidate, 'reference': reference,
                    'horizon': int(h), 'condition': col, 'level': str(level), 'n': len(f),
                    'candidate_RMSE': va['RMSE'], 'reference_RMSE': vb['RMSE'],
                    'gain_pct': 100*(1-va['RMSE']/vb['RMSE']),
                    'candidate_MAE': va['MAE'], 'reference_MAE': vb['MAE'],
                    'candidate_MBE': va['MBE'], 'reference_MBE': vb['MBE']})
    pd.DataFrame(cond_rows).to_csv(OUT / f'{split}_conditional_descriptive_comparisons.csv', index=False)
    audit['splits'][split] = {'source_predictions': source_audit, 'checks_max_abs_difference': checks,
         'fixed_rows': results['audit']['fixed_eligible_by_horizon'],
         'original_family_count': len(results['tests']),
         'original_input_comparison_count': len(contrasts),
         'input_significant_improvements': int(contrasts.improved.sum()),
         'input_significant_worsenings': int(contrasts.significant_worsening.sum()),
         'scored_rows_exactly_reproduced': True,
         'posthoc_additional_12_tests_separate_family': True}
    print(split, neural.pivot(index='input_condition', columns='horizon', values='RMSE').round(5).to_string())
    print('planned input improvements', contrasts.loc[contrasts.improved, ['candidate', 'reference', 'horizon', 'gain_pct', 'p_holm']].to_dict('records'))
    print('planned input worsens', contrasts.loc[contrasts.significant_worsening, ['candidate', 'reference', 'horizon', 'gain_pct', 'p_holm']].to_dict('records'))
    print('exploratory', extra[['candidate', 'reference', 'horizon', 'gain_pct', 'p_holm_exploratory12', 'improved_exploratory12']].to_dict('records'))

(OUT/'statistics_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=json_default)+'\n', encoding='utf-8')
for z in handles:
    z.close()
