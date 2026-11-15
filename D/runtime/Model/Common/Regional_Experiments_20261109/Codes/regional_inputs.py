"""Raw-observation masks for a fixed 635-node regional forecast experiment.

Models, graph edges, labels, common context and evaluation rows do not change.
No source data are overwritten. Excluded observations are removed before the
preserved train-only normalizer is fitted; every condition must be retrained.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

SCHEMA_VERSION='regional29-fixed635-v1.0.0'
TARGET_KEY='KR:111121'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def array_sha256(values):
    values=np.ascontiguousarray(values)
    if values.dtype.hasobject:raise TypeError('Object arrays need an explicit canonical representation.')
    h=hashlib.sha256();h.update(str(values.dtype).encode());h.update(str(values.shape).encode())
    h.update(memoryview(values).cast('B'));return h.hexdigest()


def node_order_sha256(keys):
    return hashlib.sha256(json.dumps(list(keys),ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def target_context_preprocessing_sha256(preprocessing):
    p=preprocessing
    return canonical_sha256({'target_key':TARGET_KEY,
        'fit_start_inclusive':p['fit_start_inclusive'],'fit_end_exclusive':p['fit_end_exclusive'],
        'target_pm_mean':p['pm_mean'][0],'target_pm_std':p['pm_std'][0],
        'target_pm_valid_count':p['pm_valid_count'][0],
        'context_mean':p['context_mean'],'context_std':p['context_std'],
        'context_valid_count':p['context_valid_count'],'context_names':p['context_names']})


def load_regional_configuration(config_dir):
    folder=Path(config_dir)
    records={name:json.loads((folder/(name+'.json')).read_text(encoding='utf-8-sig'))
             for name in ('experiments','regions','comparisons')}
    e=records['experiments'];regions=records['regions']['regions'];comparisons=records['comparisons']['comparisons']
    if e['schema_version']!=SCHEMA_VERSION:raise ValueError('Unexpected regional configuration version.')
    if e['target']['station_key']!=TARGET_KEY or e['horizons']!=[3,6,12,24] or e['primary_horizon']!=6 or e['universe_id']!='F634_BOTH_W24':raise ValueError('Frozen regional target/horizons/universe changed.')
    if e['expected_node_count']!=635 or e['external_series_count']!=634:raise ValueError('Expected the fixed 635-node universe.')
    conditions=e['conditions'];universe=set(conditions['F']['station_keys'])
    if len(conditions)!=29 or 'D67' in conditions or set(e['condition_order'])!=set(conditions):raise ValueError('Exactly 29 conditions without D67 are required.')
    if len(universe)!=634 or TARGET_KEY in universe:raise ValueError('Invalid external universe.')
    if len(regions)!=17 or len(comparisons)!=28:raise ValueError('Expected 17 regions and 28 comparisons.')
    allkeys=[k for item in regions.values() for k in item['station_keys']]
    if len(allkeys)!=len(set(allkeys)) or set(allkeys)!=universe:raise ValueError('Regional station sets must be disjoint and exhaustive.')
    domestic=set(k for item in regions.values() if item['country']=='KR' for k in item['station_keys'])
    if len(domestic)!=514 or set(conditions['DN']['station_keys'])!=domestic:raise ValueError('Domestic baseline must contain 514 stations.')
    for region,item in regions.items():
        keys=set(item['station_keys'])
        if len(keys)!=len(item['station_keys']) or any(not key.startswith(item['country']+':') for key in keys):raise ValueError('Region country/station ownership mismatch: '+region)
        if set(conditions['DROP_'+region]['station_keys'])!=universe-keys:raise ValueError('Invalid drop condition: '+region)
        if item['country']!='KR' and set(conditions['ADD_'+region]['station_keys'])!=domestic|keys:raise ValueError('Invalid addition condition: '+region)
    for name,item in conditions.items():
        keys=item['station_keys']
        if len(keys)!=len(set(keys)) or not set(keys).issubset(universe) or len(keys)!=item['allowed_external_series']:raise ValueError('Invalid selected station list: '+name)
    if len({row['comparison_id'] for row in comparisons})!=28:raise ValueError('Duplicate comparison IDs.')
    for item in comparisons:
        if item['candidate'] not in conditions or item['reference'] not in conditions:raise ValueError('A declared comparison references an unknown condition.')
    expected_pairs={('F','DN')}
    expected_pairs.update(('ADD_'+r,'DN') for r,item in regions.items() if item['country']!='KR')
    expected_pairs.update(('F','DROP_'+r) for r in regions)
    if {(item['candidate'],item['reference']) for item in comparisons}!=expected_pairs:raise ValueError('Regional comparison directions differ from the 28 frozen contrasts.')
    return dict(e,regions=regions,comparisons=comparisons,config_dir=str(folder.resolve()),
                config_sha256=canonical_sha256(records))


def regional_universe_fingerprints(bundle,graphs):
    keys=bundle['nodes'].station_key.astype(str).tolist()
    if not keys or keys[0]!=TARGET_KEY or len(keys)!=len(set(keys)):raise ValueError('Target must be first in a unique fixed node order.')
    shape=(len(keys),len(keys))
    if not graphs or any(np.asarray(a).shape!=shape or not np.isfinite(a).all() for a in graphs.values()):raise ValueError('Graphs must cover every fixed node with finite weights.')
    times=pd.DatetimeIndex(bundle['origins'].origin_time_kst)
    origin_record={'origin_time_kst':[t.isoformat() for t in times],
                   'origin_split':bundle['origins'].origin_split.astype(str).tolist(),
                   'horizons':list(bundle['horizons']),'lookback_hours':int(bundle['lookback_hours']),
                   'hourly_index':[t.isoformat() for t in bundle['hourly_index']]}
    invariant={'node_order_sha256':node_order_sha256(keys),
        'graph_sha256':{name:array_sha256(a) for name,a in graphs.items()},
        'target_observations_sha256':array_sha256(np.asarray(bundle['pm_values'])[:,0]),
        'context_values_sha256':array_sha256(bundle['context_values']),
        'context_names_sha256':canonical_sha256(list(bundle['context_names'])),
        'origin_grid_sha256':canonical_sha256(origin_record),
        'y_raw_sha256':array_sha256(bundle['y_raw']),'eligible_sha256':array_sha256(bundle['eligible']),
        'origin_positions_sha256':array_sha256(bundle['origin_positions'])}
    invariant['source_universe_signature']=canonical_sha256(dict(invariant,raw_pm_sha256=array_sha256(bundle['pm_values'])))
    return invariant


def prepare_regional_condition(universe_bundle,condition_id,config_dir,api,graphs,universe_fingerprints=None):
    """Return (masked_bundle, preprocessing, arrays, contract); never mutate source.

    api supplies the unchanged fit_dcrnn_preprocessors and transform_dcrnn_bundle.
    Conditions change only external PM observation availability. Excluded graph
    nodes remain as latent relays. The target and ASOS/calendar context survive.
    """
    config=load_regional_configuration(config_dir) if not isinstance(config_dir,dict) else config_dir
    if condition_id not in config['conditions']:raise ValueError('Unknown regional condition: '+str(condition_id))
    nodes=universe_bundle['nodes'];keys=nodes.station_key.astype(str).tolist()
    expected=set(config['conditions']['F']['station_keys'])|{TARGET_KEY}
    if len(keys)!=635 or set(keys)!=expected or keys[0]!=TARGET_KEY:raise ValueError('Load F634_BOTH_W24 first; every condition needs the same 635-node universe.')
    if tuple(universe_bundle['horizons'])!=(3,6,12,24):raise ValueError('Regional prediction horizons changed.')
    # Frozen context permits only the original local ASOS/calendar information.
    if any(not (name.startswith('asos') or name.startswith('cal_')) for name in universe_bundle['context_names']):raise ValueError('Unexpected regional or pollutant context would leak excluded observations.')
    fingerprints=regional_universe_fingerprints(universe_bundle,graphs)
    if universe_fingerprints is not None and canonical_sha256(fingerprints)!=canonical_sha256(universe_fingerprints):
        raise ValueError('The fixed source universe changed after its fingerprints were frozen.')
    selected=set(config['conditions'][condition_id]['station_keys'])|{TARGET_KEY}
    mask=np.array([key in selected for key in keys],dtype=bool)
    raw=np.asarray(universe_bundle['pm_values'])
    if raw.shape!=(len(universe_bundle['hourly_index']),len(keys)):raise ValueError('Hourly PM matrix shape mismatch.')
    masked=raw.copy();masked[:,~mask]=np.nan
    bundle=dict(universe_bundle,pm_values=masked,regional_input_condition=condition_id)
    preprocessing=api['fit_dcrnn_preprocessors'](bundle)
    arrays=api['transform_dcrnn_bundle'](bundle,preprocessing)
    counts=np.asarray(preprocessing['pm_valid_count'])
    if np.any(counts[~mask]!=0):raise ValueError('Excluded observations reached the fitted normalizer.')
    if not np.all(arrays['hourly_x'][:,~mask,0]==0) or not np.all(arrays['hourly_x'][:,~mask,1]==1):raise ValueError('Excluded node values/availability are not constant.')
    contract=dict(fingerprints,schema_version=SCHEMA_VERSION,condition_id=condition_id,
        universe_id=config['universe_id'],expected_node_count=635,external_series_count=634,
        allowed_external_series=len(selected)-1,config_sha256=config['config_sha256'],
        allowed_station_keys=[key for key in keys if key in selected and key!=TARGET_KEY],
        excluded_station_keys=[key for key in keys if key not in selected],
        observation_mask_sha256=array_sha256(mask),
        target_context_preprocessing_sha256=target_context_preprocessing_sha256(preprocessing),
        observation_mask_applied_before_fit=True,excluded_values_are_normalized_zero_and_missing_one=True,
        fixed_node_order_and_graphs=True,target_context_labels_preserved=True,
        latent_relay_nodes_retained=True,preprocessor_fit_start=preprocessing['fit_start_inclusive'],
        preprocessor_fit_end=preprocessing['fit_end_exclusive'],
        data_signature_allowed_to_change_by_condition=True,preprocessing_signature_allowed_to_change_by_condition=True)
    bundle['regional_condition_contract']=contract
    return bundle,preprocessing,arrays,contract
