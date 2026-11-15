"""Fixed-universe M1--M4T data contracts for the Graph WaveNet H48 cloud runner.

The 635 node order and declared graph stay fixed. Each condition changes only
allowed PM observations and explicitly declared trajectory context. Every mask
is applied before the preserved 2021--2022 normalizer is fitted.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd

try:
    from .trajectory_features import (array_hash, canonical_hash, feature_layout,
        get_cached_trajectory, origin_index, sha256_file)
except ImportError:
    from trajectory_features import (array_hash, canonical_hash, feature_layout,
        get_cached_trajectory, origin_index, sha256_file)

try:
    from .horizon_data import HORIZONS, extend_bundle_horizons
except ImportError:
    from horizon_data import HORIZONS, extend_bundle_horizons

VERSION = 'gwn-h48-m1-m4t-fixed635-v1.0.0'
TARGET = 'KR:111121'
CONDITION_ORDER = ('M1', 'M2', 'M3', 'M4', 'M3T', 'M4T')


def _settings(config):
    data = dict(config.get('data', {}))
    return dict(lookback_hours=int(data.get('lookback_hours', 48)),
                near_china_km=float(data.get('near_china_km', 600.)),
                trajectory_radius_km=float(data.get('trajectory_radius_km', 75.)),
                trajectory_sigma_km=float(data.get('trajectory_sigma_km', 50.)))


def _graph_kind(config):
    kind = str(config.get('model', {}).get('graph', 'distance'))
    if kind != 'distance':
        raise ValueError('This M1--M4T information experiment freezes model.graph=distance.')
    return kind


def _load_adapter(project_root):
    path = Path(project_root)/'Model/DCRNN/Neighbor_Expansion_QC_20261108/Codes/neighbor_adapter.py'
    if not path.is_file():
        raise FileNotFoundError('Preserved neighbor adapter unavailable: '+str(path))
    spec = importlib.util.spec_from_file_location('_gwn_h48_preserved_neighbor_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def haversine_km(longitude, latitude, target_longitude, target_latitude):
    lon, lat = np.deg2rad(np.asarray(longitude, float)), np.deg2rad(np.asarray(latitude, float))
    tlon, tlat = np.deg2rad(float(target_longitude)), np.deg2rad(float(target_latitude))
    value = np.sin((lat-tlat)/2)**2+np.cos(lat)*np.cos(tlat)*np.sin((lon-tlon)/2)**2
    return 6371.0088*2*np.arctan2(np.sqrt(np.clip(value, 0, 1)), np.sqrt(np.clip(1-value, 0, 1)))


def define_conditions(bundle, near_china_km=600.):
    nodes = bundle['nodes']
    keys = nodes.station_key.astype(str).tolist()
    if len(keys) != len(set(keys)) or len(keys) != 635 or keys[0] != TARGET:
        raise ValueError('Expected the preserved 635-node universe with Junggu first.')
    target = nodes.iloc[0]
    distance = haversine_km(nodes.longitude, nodes.latitude, target.longitude, target.latitude)
    groups = {c: set(nodes.loc[nodes.country.eq(c) & nodes.station_key.ne(TARGET), 'station_key'])
              for c in ('KR', 'CN', 'JP', 'MN')}
    if {c: len(v) for c, v in groups.items()} != {'KR': 514, 'CN': 77, 'JP': 42, 'MN': 1}:
        raise ValueError('Country scopes differ from the frozen F634_BOTH_W24 universe.')
    if not np.isfinite(near_china_km) or near_china_km <= 0:
        raise ValueError('near_china_km must be finite and positive.')
    near = set(nodes.loc[nodes.country.eq('CN') & (distance <= near_china_km), 'station_key'])
    if float(near_china_km) == 600. and len(near) != 21:
        raise ValueError('The current 600km Chinese selection must contain 21 series.')
    selected = {'M1': groups['KR'], 'M2': groups['KR'] | near,
                'M3': groups['KR'] | groups['CN'],
                'M4': set().union(*groups.values())}
    selected['M3T'] = selected['M3']; selected['M4T'] = selected['M4']
    result = {}
    for name in CONDITION_ORDER:
        allowed = selected[name]
        result[name] = {'condition': name, 'base_observation_condition': name.rstrip('T'),
                        'station_keys': [k for k in keys if k in allowed],
                        'allowed_external_series': len(allowed), 'fixed_node_count': 635,
                        'trajectory_enabled': name.endswith('T'),
                        'trajectory_countries': ['KR', 'CN'] if name == 'M3T' else
                            ['KR', 'CN', 'JP', 'MN'] if name == 'M4T' else [],
                        'near_china_km': float(near_china_km),
                        'near_china_keys': sorted(near) if name == 'M2' else [],
                        'distance_definition': 'great_circle_km_from_saved_Junggu_node_coordinates'}
    return result


def _source_paths(project_root, bundle, adapter_path, build_trajectory,
                  build_transport_graph=False):
    root = Path(project_root).resolve(); dataset = root/'Dataset'
    paths = {adapter_path.resolve()}
    preserved = root/'Model/DCRNN/V3_Data_20261107/Codes'
    paths.update((preserved/name).resolve() for name in ('original_data.py', 'original_graphs.py', 'v3_adapter.py'))
    config = root/'Model/Common/Neighbor_Experiments_QC_20261108'
    paths.update((config/name).resolve() for name in ('experiments.json', 'station_registry.csv'))
    for record in bundle['source_fingerprints']:
        path = Path(record['path'])
        if not path.is_absolute():
            path = Path(record.get('source_root', dataset))/path
        paths.add(path.resolve())
    paths.add((dataset/'Trajectory_Matching_National_v4/station_registry.csv').resolve())
    paths.add((dataset/'Origin_Features_2021_2024_v3/feature_dictionary.csv').resolve())
    paths.add((dataset/'Evaluation_Contract_20261107/evaluation_grid.parquet').resolve())
    if build_trajectory:
        paths.add((dataset/'Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet').resolve())
    if build_trajectory or build_transport_graph:
        matching = Path(bundle['neighbor_matching_folder']).resolve()
        paths.update((matching/name).resolve() for name in (
            'matching_manifest.json', 'trajectory_point_coverage.parquet',
            'trajectory_station_neighbors.parquet'))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def prepare_universe(project_root, config, cache_dir, build_trajectory=True):
    """Load preserved inputs, fixed distance graph and reusable T cache.

    The graph uses only station coordinates and the preserved500km/median-
    distance Gaussian rule. Targets/PM/validation trajectories never define it.
    """
    root = Path(project_root).expanduser().resolve(); settings = _settings(config)
    if settings['trajectory_radius_km'] != 75.:
        raise ValueError('Saved trajectory matching supports 75km only; regenerate matching for 100km.')
    if settings['lookback_hours'] != 48:
        raise ValueError('This frozen M1--M4T experiment requires lookback_hours=48.')
    adapter, adapter_path = _load_adapter(root)
    data_root = root/'Dataset'
    neighbor_config = root/'Model/Common/Neighbor_Experiments_QC_20261108'
    bundle = adapter.load_neighbor_bundle(data_root, condition='F634_BOTH_W24',
                config_dir=neighbor_config, lookback_hours=settings['lookback_hours'])
    api = adapter.preserved_neighbor_api(data_root)
    bundle = extend_bundle_horizons(bundle)
    if tuple(bundle['horizons']) != HORIZONS:
        raise ValueError('Expected all six forecast horizons through48hours.')
    if any(not (n.startswith('asos') or n.startswith('cal_')) for n in bundle['context_names']):
        raise ValueError('Unexpected regional observations in the common ASOS/calendar context.')
    conditions = define_conditions(bundle, settings['near_china_km'])
    graph_kind = _graph_kind(config)
    graphs, graph_reports = adapter.build_neighbor_graphs(
        bundle, include_trajectory=False)
    adjacency = np.asarray(graphs[graph_kind], dtype=np.float32)
    if adjacency.shape != (635, 635) or not np.isfinite(adjacency).all():
        raise ValueError('The fixed adjacency is not a finite 635x635 matrix.')
    if (adjacency < 0).any():
        raise ValueError('Negative fixed graph weights are not permitted.')
    graph_report = graph_reports[graph_kind]
    files = _source_paths(root, bundle, adapter_path, build_trajectory,
                          build_transport_graph=False)
    fingerprints = []
    for path in files:
        fingerprints.append({'path': path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
                             'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    source_contract = {'version': VERSION, 'settings': settings,
        'universe_id': 'F634_BOTH_W24', 'nodes': 635, 'external_series': 634,
        'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ('data_experiments.py', 'trajectory_features.py', 'horizon_data.py')},
        'node_order': bundle['nodes'].station_key.tolist(), 'source_files': fingerprints,
        'graph_sha256': array_hash(adjacency), 'raw_pm_sha256': array_hash(bundle['pm_values']),
        'base_context_sha256': array_hash(bundle['context_values']),
        'labels_sha256': array_hash(bundle['y_raw']), 'eligible_sha256': array_hash(bundle['eligible']),
        'horizons': list(HORIZONS), 'horizon_extension_contract': bundle['horizon_extension_contract'],
        'origin_times': [t.isoformat() for t in origin_index(bundle)],
        'build_trajectory': bool(build_trajectory), 'graph_kind': graph_kind,
        'graph_report': graph_report,
        'country_external_counts': {'KR': 514, 'CN': 77, 'JP': 42, 'MN': 1}}
    source_contract['source_signature'] = canonical_hash(source_contract)
    if build_trajectory:
        trajectory = get_cached_trajectory(root, bundle, Path(cache_dir)/'trajectory',
                    source_contract['source_signature'], radius_km=settings['trajectory_radius_km'],
                    sigma_km=settings['trajectory_sigma_km'])
    else:
        names, countries = feature_layout(root)
        trajectory = {'values': np.full((len(bundle['origins']), 322), np.nan, dtype=np.float32),
                      'names': names, 'countries': countries, 'built': False,
                      'cache_reused': False, 'cache_contract_sha256': None}
    baseline_preprocessing = api['fit_dcrnn_preprocessors'](bundle)
    return {'bundle': bundle, 'adjacency': adjacency, 'api': api, 'source_contract': source_contract,
            'conditions': conditions, 'trajectory': trajectory,
            'graph_report': graph_report, 'base_context_count': len(bundle['context_names']),
            'baseline_preprocessing': baseline_preprocessing, 'project_root': root}


def prepare_condition(universe, condition, config):
    """Apply both masks before fitting; append a fixed 322-context block in all arms."""
    if condition not in universe['conditions']:
        raise ValueError('Unknown M1--M4T condition: '+str(condition))
    if _settings(config) != universe['source_contract']['settings']:
        raise ValueError('Data settings changed after the universe/cache contract was created.')
    if _graph_kind(config) != universe['source_contract']['graph_kind']:
        raise ValueError('Graph kind changed after the universe/cache contract was created.')
    if array_hash(universe['adjacency']) != universe['source_contract']['graph_sha256']:
        raise ValueError('Fixed graph was mutated after the source contract was created.')
    source = universe['bundle']; spec = universe['conditions'][condition]
    if array_hash(source['pm_values']) != universe['source_contract']['raw_pm_sha256']:
        raise ValueError('Raw universe was mutated after loading.')
    if array_hash(source['context_values']) != universe['source_contract']['base_context_sha256']:
        raise ValueError('Common context was mutated after loading.')
    keys = source['nodes'].station_key.astype(str).tolist()
    selected = set(spec['station_keys']) | {TARGET}
    mask = np.array([k in selected for k in keys], dtype=bool)
    raw = np.asarray(source['pm_values']).copy(); raw[:, ~mask] = np.nan
    t = universe['trajectory']
    if spec['trajectory_enabled'] and not universe['source_contract']['build_trajectory']:
        raise ValueError('Trajectory features were disabled at prepare_universe; T conditions cannot run.')
    tcountries = np.array(t['countries'])
    tmask = (np.isin(tcountries, ['NEUTRAL']+spec['trajectory_countries'])
             if spec['trajectory_enabled'] else np.zeros(322, dtype=bool))
    tvalues = np.asarray(t['values']).copy(); tvalues[:, ~tmask] = np.nan
    base_count = universe['base_context_count']
    bundle = dict(source, pm_values=raw,
                  context_values=np.concatenate([source['context_values'], tvalues], axis=1),
                  context_names=list(source['context_names'])+list(t['names']),
                  gwn_input_condition=condition)
    api = universe['api']
    preprocessing = api['fit_dcrnn_preprocessors'](bundle)
    arrays = api['transform_dcrnn_bundle'](bundle, preprocessing)
    if np.asarray(preprocessing['pm_valid_count'])[~mask].any():
        raise ValueError('Excluded PM reached the fitted normalizer.')
    if not np.all(arrays['hourly_x'][:, ~mask, 0] == 0) or not np.all(arrays['hourly_x'][:, ~mask, 1] == 1):
        raise ValueError('Excluded PM values or availability leaked into the tensor.')
    ncontext = len(bundle['context_names'])
    bad_context = np.r_[np.zeros(base_count, bool), ~tmask]
    if np.asarray(preprocessing['context_valid_count'])[bad_context].any():
        raise ValueError('Excluded trajectory context reached fitted moments.')
    if not np.all(arrays['context_x'][:, :ncontext][:, bad_context] == 0) or not np.all(arrays['context_x'][:, ncontext:][:, bad_context] == 1):
        raise ValueError('Excluded trajectory values/availability leaked into model context.')
    baseline = universe['baseline_preprocessing']
    for key in ('context_mean', 'context_std', 'context_valid_count'):
        if preprocessing[key][:base_count] != baseline[key]:
            raise ValueError('Base ASOS/calendar normalizer changed across input arms.')
    for key in ('pm_mean', 'pm_std', 'pm_valid_count'):
        if preprocessing[key][0] != baseline[key][0]:
            raise ValueError('Junggu target normalizer changed across input arms.')
    for key in ('y_raw', 'eligible', 'origin_positions'):
        if not np.array_equal(bundle[key], source[key], equal_nan=True):
            raise ValueError('Targets, eligibility or origin grid changed: '+key)
    contract = {'version': VERSION, 'condition': condition, 'source_signature': universe['source_contract']['source_signature'],
        'horizons': list(HORIZONS),
        'settings': _settings(config), 'fixed_node_count': 635,
        'graph_kind': universe['source_contract']['graph_kind'],
        'fixed_graph_sha256': array_hash(universe['adjacency']),
        'node_order_sha256': canonical_hash(keys), 'allowed_external_series': len(selected)-1,
        'allowed_station_keys': [k for k in keys if k in selected and k != TARGET],
        'excluded_station_keys': [k for k in keys if k not in selected],
        'trajectory_enabled': spec['trajectory_enabled'], 'trajectory_countries': spec['trajectory_countries'],
        'trajectory_feature_names': list(t['names']), 'trajectory_allowed_names': [name for name,keep in zip(t['names'], tmask) if keep],
        'trajectory_context_size': 322, 'base_context_size': base_count, 'raw_context_size': ncontext,
        'transformed_context_size': arrays['context_x'].shape[1],
        'mask_applied_before_normalization': True, 'targets_and_common_context_preserved': True,
        'train_fit_start_inclusive': preprocessing['fit_start_inclusive'],
        'train_fit_end_exclusive': preprocessing['fit_end_exclusive'],
        'observation_mask_sha256': array_hash(mask), 'trajectory_mask_sha256': array_hash(tmask),
        'condition_raw_pm_sha256': array_hash(raw), 'condition_context_sha256': array_hash(bundle['context_values']),
        'trajectory_cache_contract_sha256': t.get('cache_contract_sha256'),
        'no_gases_or_pm10_added': True, 'excluded_nodes_remain_latent_graph_relays': True}
    contract['condition_signature'] = canonical_hash(contract)
    bundle['gwn_condition_contract'] = contract
    return {'bundle': bundle, 'preprocessing': preprocessing, 'arrays': arrays, 'contract': contract}
