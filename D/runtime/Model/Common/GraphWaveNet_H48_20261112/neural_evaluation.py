"""Automatic six-horizon neural evaluation with the v5 output schemas.

The scoring plan is written before fitting. Full exports are scored on the
authoritative fixed calendar; pilot outputs are explicitly skipped. This module
does not fit models, impute targets, change baseline definitions, or compress gaps.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCHEMA_VERSION = 'neural-evaluation-h48-v1.0.0'
HORIZONS = (3, 6, 12, 24, 36, 48)
GRAPH_PAIRS = (('trajectory_directed', 'distance'),
               ('trajectory_symmetric', 'distance'),
               ('trajectory_directed', 'trajectory_symmetric'))
COMPARABILITY_FIELDS = ('target_station_key', 'horizons', 'lookback_hours',
                        'target_transform', 'data_signature', 'preprocessing_signature',
                        'train_split', 'validation_split', 'prediction_unit')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _unique_invocation_directory(base):
    """Create a new child atomically; no scoring call reuses prior outputs."""
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
        path = base / (timestamp + '_' + uuid.uuid4().hex[:8])
        path.resolve().relative_to(base)
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a unique scoring invocation directory.')


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _canonical(_read(path)) != _canonical(value):
            raise ValueError(f'Frozen plan differs: {path}. Use a new run directory.')
    else:
        _json(path, value)


def _load_common(project_root, common_dir=None):
    """Import the caller's private score/grid pair without global module leaks."""
    common = Path(common_dir) if common_dir is not None else Path(project_root) / 'Model/Common/GraphWaveNet_H48_20261112'
    common = common.resolve()
    paths = {name: common / (name + '.py') for name in ('grid', 'score')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Upload the common scoring code: {path}')
    hashes = {str(path): _sha(path) for path in paths.values()}
    grid_spec = importlib.util.spec_from_file_location('_neural_fixed_grid', paths['grid'])
    grid = importlib.util.module_from_spec(grid_spec)
    grid_spec.loader.exec_module(grid)
    old_grid = sys.modules.get('grid')
    try:
        sys.modules['grid'] = grid
        score_spec = importlib.util.spec_from_file_location('_neural_fixed_score', paths['score'])
        scorer = importlib.util.module_from_spec(score_spec)
        score_spec.loader.exec_module(scorer)
    finally:
        if old_grid is None:
            sys.modules.pop('grid', None)
        else:
            sys.modules['grid'] = old_grid
    return scorer, grid, hashes


def _normalize_specs(model_specs):
    rows = [dict(row) for row in model_specs]
    names = [str(row.get('model', '')) for row in rows]
    if not names or any(not name or '|' in name or ':' in name for name in names) or len(set(names)) != len(names):
        raise ValueError('Declare unique nonempty model names without | or : characters.')
    for row, name in zip(rows, names):
        row['model'] = name
        row.setdefault('graph', '')
        row.setdefault('adaptive_mode', '')
    return rows


def _model_pairs(specs, pair_specs=None):
    names = {row['model'] for row in specs}
    if pair_specs is None:
        pairs = []
        # Default graph-only comparisons keep all declared non-graph model
        # settings identical. A different architecture/profile needs an explicit
        # model comparison, not an automatic claim about graph direction alone.
        def comparison_group(row):
            return _canonical({key: value for key, value in row.items()
                               if key not in {'model', 'graph', 'label', 'description'}})
        modes = dict.fromkeys(comparison_group(row) for row in specs)
        for mode in modes:
            groups = {}
            for row in specs:
                if comparison_group(row) == mode:
                    graph = str(row['graph'])
                    if graph in groups:
                        raise ValueError('Ambiguous graph model inside an adaptive mode; pass explicit pair_specs.')
                    groups[graph] = row['model']
            for candidate, reference in GRAPH_PAIRS:
                if candidate in groups and reference in groups:
                    pairs.append([groups[candidate], groups[reference]])
    else:
        pairs = []
        for pair in pair_specs:
            if isinstance(pair, dict):
                pair = (pair['candidate'], pair['reference'])
            if len(pair) != 2:
                raise ValueError('Every model pair needs candidate and reference.')
            pairs.append([str(pair[0]), str(pair[1])])
    if any(a not in names or b not in names or a == b for a, b in pairs):
        raise ValueError('A comparison refers to an undeclared or identical model.')
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('Duplicate model comparisons.')
    return pairs


def write_neural_scoring_plan(run_root, model_family, input_condition, model_specs,
                              seeds, training_metadata, pair_specs=None):
    """Freeze input/model/seed/comparison settings before any fitting or scoring.

    model_specs is a list of dictionaries containing at least model and graph.
    Graph comparisons are generated within the same adaptive_mode. Architecture
    and training settings may be recorded as additional JSON-compatible fields.
    Repeated calls are accepted only when the declared content is identical.
    """
    root = Path(run_root).resolve()
    specs = _normalize_specs(model_specs)
    normalized_seeds = [str(int(seed)) for seed in seeds]
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError('Declare a nonempty unique seed list.')
    condition = str(input_condition)
    if not condition or '|' in condition or ':' in condition:
        raise ValueError('Input condition must be a stable identifier without | or :.')
    metadata = dict(training_metadata)
    if tuple(metadata.get('horizons', HORIZONS)) != HORIZONS:
        raise ValueError('The H48 common contract requires horizons 3,6,12,24,36,48.')
    metadata['horizons'] = list(HORIZONS)
    metadata.setdefault('prediction_unit', 'ug/m3')
    metadata.setdefault('prediction_export', 'raw_unclipped_original_units')
    if metadata['prediction_unit'] != 'ug/m3' or metadata['prediction_export'] != 'raw_unclipped_original_units':
        raise ValueError('Export raw, unclipped original-unit predictions for the common scorer.')
    if not metadata.get('evaluation_grid_path'):
        raise ValueError('Record evaluation_grid_path from ensure_h48_grid before freezing the six-horizon plan.')
    declared_grid = Path(metadata['evaluation_grid_path'])
    if declared_grid.is_absolute():
        if not declared_grid.is_file():
            raise FileNotFoundError('Build the H48 evaluation grid before freezing the scoring plan: ' + str(declared_grid))
        actual_grid_hash = _sha(declared_grid)
        if metadata.get('evaluation_grid_sha256', actual_grid_hash) != actual_grid_hash:
            raise ValueError('Declared H48 grid hash does not match the derived grid.')
        metadata['evaluation_grid_sha256'] = actual_grid_hash
    elif not metadata.get('evaluation_grid_sha256'):
        raise ValueError('Relative H48 grid paths require evaluation_grid_sha256 in the frozen metadata.')
    declaration = {'schema_version': SCHEMA_VERSION, 'model_family': str(model_family),
                   'input_condition': condition, 'model_specs': specs,
                   'seeds': normalized_seeds, 'training_metadata': metadata,
                   'model_pairs': _model_pairs(specs, pair_specs),
                   'comparison_sign': 'candidate RMSE minus reference RMSE; negative means lower error',
                   'fixed_grid_relative_path': str(metadata['evaluation_grid_path']),
                   'ensemble_rule': 'arithmetic mean of unclipped forecasts in ug/m3, then clip at zero',
                   'seed_score_mean_is_not_ensemble_score': True,
                   'validation_inference': '2023 early-stopping data; descriptive diagnostics, not independent final superiority'}
    path = root / 'scoring_plan.json'
    _immutable_json(path, declaration)
    return declaration


def _run_id(model, condition, kind, seed=''):
    return f'{model}|{condition}|{kind}' + (f'|seed={seed}' if seed else '')


def _resolved_pairs(plan):
    rows = []
    for candidate, reference in plan['model_pairs']:
        for seed in plan['seeds']:
            rows.append((_run_id(candidate, plan['input_condition'], 'seed', seed),
                         _run_id(reference, plan['input_condition'], 'seed', seed)))
        if len(plan['seeds']) >= 2:
            rows.append((_run_id(candidate, plan['input_condition'], 'ensemble'),
                         _run_id(reference, plan['input_condition'], 'ensemble')))
    return rows


def _prediction_path(root, split, prediction_path):
    if prediction_path is not None:
        path = Path(prediction_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    choices = (root / 'Exports' / f'{split}_predictions_by_seed.parquet',
               root / 'Summary/Common_Exports' / f'{split}_predictions_by_seed.parquet')
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError('Full seed predictions are unavailable: ' + ' or '.join(map(str, choices)))


def _validate_predictions(frame, scorer, plan, split):
    data = scorer.prepare_predictions(frame)
    if set(data.prediction_kind) != {'seed'}:
        raise ValueError('Pass only seed exports; the common scorer creates the prediction ensemble.')
    if not data.input_condition.eq(plan['input_condition']).all():
        raise ValueError('Prediction input conditions do not match the frozen plan.')
    if 'split' in data and not data.split.eq(split).all():
        raise ValueError('Export the selected split separately.')
    if 'experiment_mode' in data and not data.experiment_mode.eq('full').all():
        raise ValueError('Full-grid scoring does not accept pilot exports.')
    if not np.isfinite(data.y_hat.to_numpy(float)).all():
        raise ValueError('Nonfinite forecasts must be fixed, not removed from the evaluation rows.')
    if 'common_export_unclipped' in data and not data.common_export_unclipped.eq(True).all():
        raise ValueError('Use the raw-unit common export rather than the legacy clipped export.')
    if 'target_time_kst' in data:
        actual = pd.DatetimeIndex(pd.to_datetime(data.target_time_kst, utc=True)).tz_convert('Asia/Seoul').as_unit('ns')
        expected = pd.DatetimeIndex(data.origin_time_kst + pd.to_timedelta(data.horizon, unit='h')).as_unit('ns')
        if not np.array_equal(actual.asi8, expected.asi8):
            raise ValueError('Prediction target time differs from origin+horizon.')
    names = {row['model'] for row in plan['model_specs']}
    if set(data.model) != names:
        raise ValueError('Exported models differ from the predeclared model list.')
    for name in names:
        part = data.loc[data.model.eq(name)]
        if set(part.seed) != set(plan['seeds']) or set(part.horizon) != set(HORIZONS):
            raise ValueError('Each model must export the same declared seeds and six horizons.')
    return data


def _output_tables(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('metrics', 'seed_metric_summary', 'tests', 'conditions'):
        result[name].to_csv(directory / (name + '.csv'), index=False, encoding='utf-8-sig')
    # Match the reference ZIP CLI exactly, including Parquet's default codec.
    result['score_rows'].to_parquet(directory / 'score_rows.parquet', index=False)
    additional = directory / 'Additional_Analysis'
    additional.mkdir(exist_ok=True)
    metrics = result['metrics']
    metrics.loc[metrics.prediction_kind.eq('seed')].to_csv(additional / 'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    metrics.loc[metrics.prediction_kind.eq('ensemble')].to_csv(additional / 'metrics_ensemble.csv', index=False, encoding='utf-8-sig')
    metrics.loc[metrics.input_condition.eq('common_baseline')].to_csv(additional / 'metrics_baselines.csv', index=False, encoding='utf-8-sig')


def _comparison_metrics(metrics, pairs):
    rows = []
    indexed = metrics.set_index(['run_id', 'horizon'])
    for candidate, reference in pairs:
        for horizon in HORIZONS:
            a, b = indexed.loc[(candidate, horizon)], indexed.loc[(reference, horizon)]
            rows.append({'candidate_run_id': candidate, 'reference_run_id': reference,
                         'horizon': horizon, 'prediction_kind': a.prediction_kind, 'seed': a.seed,
                         'candidate_RMSE': a.RMSE, 'reference_RMSE': b.RMSE,
                         'RMSE_difference_candidate_minus_reference': a.RMSE - b.RMSE,
                         'RMSE_gain_reference_minus_candidate': b.RMSE - a.RMSE,
                         'RMSE_gain_pct': 100 * (b.RMSE - a.RMSE) / b.RMSE if b.RMSE else np.nan,
                         'MAE_difference_candidate_minus_reference': a.MAE - b.MAE,
                         'n': int(a.n)})
    return pd.DataFrame(rows)


def _plots(metrics, seed_summary, comparisons, directory, run_root, learning_histories=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    figure_dir = directory / 'Figures'
    figure_dir.mkdir(exist_ok=True)
    names = metrics.loc[metrics.prediction_kind.eq('seed'), 'model'].drop_duplicates().tolist()
    colors = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for i, name in enumerate(names):
        part = seed_summary.loc[seed_summary.model.eq(name)].sort_values('horizon')
        axes[0].errorbar(part.horizon, part.RMSE_mean, yerr=part.RMSE_std.fillna(0),
                         marker='o', capsize=3, color=colors(i % 10), label=name)
        ensemble = metrics.loc[metrics.model.eq(name) & metrics.prediction_kind.eq('ensemble')].sort_values('horizon')
        if len(ensemble):
            axes[1].plot(ensemble.horizon, ensemble.RMSE, marker='o', color=colors(i % 10), label=name)
    for ax in axes:
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Forecast horizon (hours)')
        ax.set_ylabel('RMSE (µg/m³)')
        ax.grid(alpha=.2)
    axes[0].set_title('Individual seed scores: mean ± sample SD (not CI)')
    axes[1].set_title('Ensemble: mean predictions, separately scored')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / 'rmse_seed_scores_and_ensemble.png', dpi=170)
    plt.close(fig)
    if len(comparisons):
        selected = comparisons.loc[comparisons.prediction_kind.eq('ensemble')]
        if selected.empty:
            selected = comparisons
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for (candidate, reference), part in selected.groupby(['candidate_run_id', 'reference_run_id'], sort=False):
            label = candidate.split('|')[0] + ' vs ' + reference.split('|')[0]
            part = part.sort_values('horizon')
            ax.plot(part.horizon, part.RMSE_gain_reference_minus_candidate, marker='o', label=label)
        ax.axhline(0, color='#666', lw=.8)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Forecast horizon (hours)')
        ax.set_ylabel('RMSE(reference) − RMSE(candidate), µg/m³')
        ax.set_title('Predeclared comparisons: positive values mean lower candidate RMSE')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / 'graph_or_model_comparison_gains.png', dpi=170)
        plt.close(fig)
    paths = [Path(p) for p in (learning_histories or [])]
    if not paths:
        paths = sorted(set(Path(run_root).rglob('training_history.csv')) |
                       set(Path(run_root).rglob('history.csv')) |
                       set(Path(run_root).rglob('learning_curve.csv')))
    histories = []
    for path in paths:
        if path.suffix.lower() != '.csv' or not path.is_file():
            continue
        frame = pd.read_csv(path)
        x_name = next((name for name in ('epoch', 'epoch_number') if name in frame), None)
        metric = next((name for name in ('val_rmse_h6', 'val6hRMSE', 'validation_rmse_h6', 'val_h6_rmse', 'validation_RMSE_h6') if name in frame), None)
        lr_name = next((name for name in ('learning_rate_used', 'learning_rate', 'lr', 'lr_before_step') if name in frame), None)
        loss_name = next((name for name in ('train_loss', 'training_loss') if name in frame), None)
        if x_name and (metric or lr_name):
            histories.append((path, frame, x_name, metric, lr_name, loss_name))
    if histories:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for path, frame, x_name, metric, lr_name, loss_name in histories:
            label = path.parent.parent.name + '/' + path.parent.name
            if loss_name:
                axes[0].plot(frame[x_name], frame[loss_name], lw=1, label=label)
            if metric:
                axes[1].plot(frame[x_name], frame[metric], lw=1, label=label)
            if lr_name:
                axes[2].plot(frame[x_name], frame[lr_name], lw=1, label=label)
        axes[0].set_ylabel('Training loss (configured scale)')
        axes[1].set_ylabel('Validation 6-hour RMSE (µg/m³)')
        axes[2].set_ylabel('Learning rate used')
        axes[2].set_yscale('log')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
        if len(histories) <= 12:
            axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figure_dir / 'learning_curves_and_learning_rate.png', dpi=170)
        plt.close(fig)


def score_neural_run(project_root, run_root, model_family, input_condition,
                     split='validation', pair_specs=None, experiment_mode='full',
                     final=False, common_dir=None, grid_path=None, prediction_path=None,
                     bootstrap_reps=1000, bootstrap_seed=0, hac_lag=None,
                     make_plots=True, learning_histories=None, output_root=None):
    """Automatically score a completed full neural run and all graph comparisons.

    The caller writes scoring_plan.json before training and exports only seed
    forecasts. output_root directs scoring artifacts to a separate Drive Result
    folder; the training plan and prediction source stay under run_root.
    No node count, country name or model architecture is hardcoded.
    """
    project, root = Path(project_root).resolve(), Path(run_root).resolve()
    if experiment_mode != 'full':
        base = Path(output_root).resolve() if output_root is not None else root / 'Scoring'
        directory = _unique_invocation_directory(base / split)
        result = {'status': 'skipped_pilot', 'reason': 'Pilot predictions do not cover the authoritative full evaluation calendar.',
                  'experiment_mode': experiment_mode, 'full_grid_rules_weakened': False,
                  'output_dir': str(directory)}
        _json(directory / 'pilot_scoring_skipped.json', result)
        return result
    if split not in ('validation', 'test'):
        raise ValueError('Only validation or test can be scored.')
    if split == 'test' and not final:
        raise ValueError('Test scoring requires final=True after settings are frozen.')
    plan_path = root / 'scoring_plan.json'
    if not plan_path.is_file():
        raise FileNotFoundError('Write the scoring plan before training: ' + str(plan_path))
    plan = _read(plan_path)
    if plan['model_family'] != str(model_family) or plan['input_condition'] != str(input_condition):
        raise ValueError('Scoring arguments differ from the frozen plan.')
    if pair_specs is not None and _model_pairs(plan['model_specs'], pair_specs) != plan['model_pairs']:
        raise ValueError('Comparison pairs changed after the scoring plan was frozen.')
    scorer, grid_module, common_hashes = _load_common(project, common_dir)
    declared_grid = plan.get('training_metadata', {}).get('evaluation_grid_path')
    if not declared_grid:
        raise ValueError('Six-horizon plans must declare evaluation_grid_path; the old four-horizon grid cannot be substituted.')
    grid_file = Path(grid_path or declared_grid)
    grid_file = (grid_file if grid_file.is_absolute() else project / grid_file).resolve()
    declared_path = Path(declared_grid)
    declared_path = (declared_path if declared_path.is_absolute() else project / declared_path).resolve()
    if grid_file != declared_path:
        raise ValueError('Scoring grid differs from the six-horizon grid frozen before training.')
    if not grid_file.is_file():
        raise FileNotFoundError('Upload evaluation_grid.parquet: ' + str(grid_file))
    if _sha(grid_file) != plan['training_metadata'].get('evaluation_grid_sha256'):
        raise ValueError('The H48 evaluation grid changed after the scoring plan was frozen.')
    source = _prediction_path(root, split, prediction_path)
    protected = {**common_hashes, str(grid_file): _sha(grid_file), str(source): _sha(source), str(plan_path): _sha(plan_path)}
    frame = _validate_predictions(pd.read_parquet(source), scorer, plan, split)
    resolved = _resolved_pairs(plan)
    base = Path(output_root).resolve() if output_root is not None else root / 'Scoring'
    directory = _unique_invocation_directory(base / split)
    additional = directory / 'Additional_Analysis'
    invocation = {'schema_version': SCHEMA_VERSION, 'model_family': model_family, 'input_condition': input_condition,
                  'split': split, 'final': bool(final), 'bootstrap_reps': int(bootstrap_reps),
                  'bootstrap_seed': int(bootstrap_seed), 'hac_lag': hac_lag,
                  'plan_sha256': protected[str(plan_path)], 'prediction_sha256': protected[str(source)],
                  'grid_sha256': protected[str(grid_file)], 'common_source_sha256': common_hashes,
                  'resolved_comparison_pairs': [list(pair) for pair in resolved]}
    _immutable_json(additional / 'evaluation_invocation.json', invocation)
    frozen_grid = grid_module.validate_grid(grid_module.read_frame(grid_file))
    if set(frozen_grid.horizon) != set(HORIZONS):
        raise ValueError('The frozen H48 grid must include all six horizons.')
    result = scorer.score_frames(frozen_grid, frame, split=split,
                                 pairs=resolved, final=final, bootstrap_reps=bootstrap_reps,
                                 bootstrap_seed=bootstrap_seed, hac_lag=hac_lag, make_ensembles=True)
    _output_tables(result, directory)
    comparisons = _comparison_metrics(result['metrics'], resolved)
    comparisons.to_csv(additional / 'graph_comparison_metrics.csv', index=False, encoding='utf-8-sig')
    selected_tests = result['tests'].loc[[tuple(row) in set(resolved)
                                        for row in result['tests'][['run_id', 'versus_run_id']].itertuples(index=False, name=None)]].copy()
    selected_tests.to_csv(additional / 'graph_comparison_tests.csv', index=False, encoding='utf-8-sig')
    sidecar_path = source.with_suffix('.meta.json')
    if sidecar_path.exists():
        sidecar = _read(sidecar_path)
        sidecar_sha256 = _sha(sidecar_path)
    else:
        # A previous model export is read-only. Supplemental provenance belongs
        # to this new scoring invocation rather than the old prediction folder.
        sidecar = None
        sidecar_sha256 = None
        supplemental = {'schema_version': SCHEMA_VERSION, 'model_family': model_family, 'input_condition': input_condition,
                   'prediction_kind': 'seed', 'split': split, 'seeds': plan['seeds'],
                   'prediction_unit': 'ug/m3', 'prediction_export': 'raw_unclipped_original_units',
                   'prediction_sha256': protected[str(source)], 'scoring_plan': plan,
                   'scoring_plan_sha256': protected[str(plan_path)]}
        _json(additional / 'prediction_source_metadata.json', supplemental)
    if make_plots:
        _plots(result['metrics'], result['seed_metric_summary'], comparisons, additional, root, learning_histories)
    if any(_sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A frozen scorer, grid, plan, or prediction file changed during scoring.')
    # These are the original ZIP CLI's audit keys and provenance item shape.
    # Keep all automation-specific metadata in a separate optional artifact.
    audit = dict(result['audit'])
    audit.update(written_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                 grid_path=str(grid_file.resolve()), grid_sha256=_sha(grid_file),
                 prediction_sources=[{'path': str(source.resolve()), 'sha256': _sha(source), 'sidecar': sidecar}])
    _json(directory / 'scoring_audit.json', audit)
    extra_audit = dict(status='passed', schema_version=SCHEMA_VERSION, model_family=model_family,
                 input_condition=input_condition, written_kst=audit['written_kst'],
                 protected_source_sha256=protected, prediction_path=str(source),
                 prediction_sidecar_sha256=sidecar_sha256, prediction_sidecar=sidecar,
                 predeclared_model_pairs=plan['model_pairs'], resolved_graph_comparison_pairs=[list(p) for p in resolved],
                 graph_comparison_rows=len(comparisons), graph_comparison_test_rows=len(selected_tests),
                 same_seed_sets_verified=True, fixed_evaluation_rules_modified=False,
                 validation_diagnostics_nonindependent=split == 'validation',
                 use_validation_for_final_superiority=False, test2024_scored=split == 'test')
    extra_audit['standard_output_contract'] = 'Six reference ZIP CLI files retain names, schemas, dtypes and core audit structure.'
    extra_audit['core_audit_sha256'] = _sha(directory / 'scoring_audit.json')
    _json(additional / 'neural_scoring_audit.json', extra_audit)
    return {'status': 'passed', 'output_dir': str(directory), 'additional_analysis_dir': str(additional),
            'metric_rows': len(result['metrics']),
            'seed_metric_rows': int(result['metrics'].prediction_kind.eq('seed').sum()),
            'ensemble_metric_rows': int(result['metrics'].prediction_kind.eq('ensemble').sum()),
            'graph_comparison_rows': len(comparisons), 'test2024_scored': split == 'test',
            'validation_diagnostics_nonindependent': split == 'validation'}


def _comparable_metadata(left, right):
    a, b = left['training_metadata'], right['training_metadata']
    absent = [key for key in COMPARABILITY_FIELDS if key not in a or key not in b]
    if absent:
        raise ValueError('Cross-model comparison metadata is incomplete: ' + ', '.join(absent))
    mismatches = [key for key in COMPARABILITY_FIELDS if _canonical(a[key]) != _canonical(b[key])]
    if left['input_condition'] != right['input_condition']:
        mismatches.append('input_condition')
    if set(left['seeds']) != set(right['seeds']):
        mismatches.append('seeds')
    if mismatches:
        raise ValueError('Cross-model comparisons require matching inputs/targets/preprocessing: ' + ', '.join(mismatches))
    return {key: a[key] for key in COMPARABILITY_FIELDS}


def write_cross_model_comparison_plan(left_run_root, right_run_root, output_root,
                                       model_pairs, analysis_status='predeclared'):
    """Freeze explicit family-to-family pairs; never choose models by their scores."""
    if analysis_status not in ('predeclared', 'exploratory'):
        raise ValueError('Declare the analysis as predeclared or exploratory.')
    left_root, right_root = Path(left_run_root).resolve(), Path(right_run_root).resolve()
    left, right = _read(left_root / 'scoring_plan.json'), _read(right_root / 'scoring_plan.json')
    comparable = _comparable_metadata(left, right)
    left_names = {row['model'] for row in left['model_specs']}
    right_names = {row['model'] for row in right['model_specs']}
    if left_names & right_names:
        raise ValueError('Model family exports need disjoint model names.')
    pairs = [[str(a), str(b)] for a, b in model_pairs]
    if not pairs or len({tuple(row) for row in pairs}) != len(pairs):
        raise ValueError('Declare unique cross-family model comparisons.')
    if any(a not in left_names or b not in right_names for a, b in pairs):
        raise ValueError('Cross-model pairs must be (left-family model, right-family model).')
    result = {'schema_version': SCHEMA_VERSION, 'analysis_status': analysis_status,
              'left_run_root': str(left_root), 'right_run_root': str(right_root),
              'left_scoring_plan_sha256': _sha(left_root / 'scoring_plan.json'),
              'right_scoring_plan_sha256': _sha(right_root / 'scoring_plan.json'),
              'input_condition': left['input_condition'], 'seeds': left['seeds'],
              'model_pairs': pairs, 'comparable_input_target_preprocessing': comparable,
              'architecture_and_training_settings_may_differ': True,
              'left_model_specs': left['model_specs'], 'right_model_specs': right['model_specs'],
              'left_training_metadata': left['training_metadata'], 'right_training_metadata': right['training_metadata'],
              'interpretation': 'Full model configurations are compared; this is not a graph-only or architecture-only causal effect.'}
    _immutable_json(Path(output_root).resolve() / 'cross_model_plan.json', result)
    return result


def compare_neural_runs(project_root, left_run_root, right_run_root, output_root,
                         split='validation', final=False, common_dir=None,
                         bootstrap_reps=1000, bootstrap_seed=0, hac_lag=None, make_plots=True,
                         scoring_output_root=None):
    """Score two predeclared model-family runs on exactly the same calendar rows."""
    if split not in ('validation', 'test'):
        raise ValueError('Only validation or test can be scored.')
    if split == 'test' and not final:
        raise ValueError('Test scoring requires final=True before reading test predictions.')
    output = Path(output_root).resolve()
    declaration = _read(output / 'cross_model_plan.json')
    left_root, right_root = Path(left_run_root).resolve(), Path(right_run_root).resolve()
    if declaration['left_run_root'] != str(left_root) or declaration['right_run_root'] != str(right_root):
        raise ValueError('Cross-model run roots differ from the frozen comparison plan.')
    left_path, right_path = left_root / 'scoring_plan.json', right_root / 'scoring_plan.json'
    if (_sha(left_path) != declaration['left_scoring_plan_sha256'] or
            _sha(right_path) != declaration['right_scoring_plan_sha256']):
        raise ValueError('A cross-model scoring plan changed.')
    left, right = _read(left_path), _read(right_path)
    _comparable_metadata(left, right)
    comparison_run = _unique_invocation_directory(output)
    scorer, _, _ = _load_common(project_root, common_dir)
    frames = []
    for root, plan in ((left_root, left), (right_root, right)):
        path = _prediction_path(root, split, None)
        frames.append(_validate_predictions(pd.read_parquet(path), scorer, plan, split))
    combined_specs = left['model_specs'] + right['model_specs']
    metadata = dict(left['training_metadata'], cross_model_analysis_status=declaration['analysis_status'],
                    model_setting_difference_disclosure=declaration['interpretation'])
    write_neural_scoring_plan(comparison_run, 'CrossModel', left['input_condition'], combined_specs,
                              left['seeds'], metadata, pair_specs=declaration['model_pairs'])
    export = comparison_run / 'Exports' / f'{split}_predictions_by_seed.parquet'
    export.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    if export.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(export), combined, check_exact=True)
        except AssertionError as exc:
            raise ValueError('Existing cross-model predictions differ; use a new comparison run.') from exc
    else:
        combined.to_parquet(export, index=False, compression='zstd')
    result = score_neural_run(project_root, comparison_run, 'CrossModel', left['input_condition'],
                              split=split, final=final, common_dir=common_dir,
                              bootstrap_reps=bootstrap_reps, bootstrap_seed=bootstrap_seed,
                              hac_lag=hac_lag, make_plots=make_plots,
                              output_root=scoring_output_root)
    _json(Path(result['additional_analysis_dir']) / 'cross_model_comparison.json', declaration)
    result['comparison_run_root'] = str(comparison_run)
    return result
