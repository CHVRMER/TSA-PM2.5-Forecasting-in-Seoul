"""Data-only neighbor expansion for the preserved 601-node Seoul DCRNN.

The original loader, graph functions, model and training implementation are
unchanged. Eight explicit observation selections are supplied by the shared
Neighbor_Experiments_QC configuration. This module can run after the original
notebook definitions, or import their unchanged data/graph definitions lazily.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


BASE_CONDITION = 'F600_W24'
NATIONAL_REGISTRY = 'Trajectory_Matching_National_v4/station_registry.csv'
BASE_MATCHING = 'Trajectory_Matching_National_v4'
EXPANSION_MATCHING = 'Trajectory_Matching_Expansion_QC_20261108'
EXTRA_FOLDER = 'Neighbors_Expansion_QC_20261108'
NEW_SOURCES = {
    'China_East_QC': (
        EXTRA_FOLDER+'/china_east_hourly_qc_2021_2024_kst.parquet',
        EXTRA_FOLDER+'/china_east_context_buffer_qc.parquet'),
    'Japan_East': (
        EXTRA_FOLDER+'/japan_east18_hourly_2021_2024_kst.parquet',
        EXTRA_FOLDER+'/japan_east18_context_buffer.parquet'),
}
WEST_SOURCE = {'Japan': ('Japan/japan_west_hourly_2021_2024_kst.parquet',
                         'Japan/japan_west_context_buffer.parquet')}
_PRESERVED_API_CACHE = {}


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2*1024*1024),b''):value.update(block)
    return value.hexdigest()


def preserved_neighbor_api(dataset_root, preserved_code_dir=None):
    """Return the existing data/graph namespace, without rewriting its files.

    In a self-contained notebook the original definitions can already exist in
    this global namespace. A standalone module resolves the original three
    source files under Model/DCRNN/V3_Data_20261107/Codes instead.
    """
    required=('load_dcrnn_expanded_bundle','_expanded_fill_panel',
              'build_preserved_dcrnn_graphs','fit_dcrnn_preprocessors',
              'transform_dcrnn_bundle','DCRNNOriginDataset')
    if all(callable(globals().get(name)) for name in required):
        return globals()
    directory=(Path(preserved_code_dir) if preserved_code_dir is not None else
               Path(dataset_root).resolve().parent/'Model/DCRNN/V3_Data_20261107/Codes').resolve()
    key=str(directory)
    if key not in _PRESERVED_API_CACHE:
        names=('original_data.py','original_graphs.py','v3_adapter.py')
        namespace={'__name__':'_preserved_seoul_dcrnn_data_graphs'}
        hashes={}
        for name in names:
            path=directory/name
            if not path.is_file():raise FileNotFoundError('Preserved DCRNN source unavailable: '+str(path))
            hashes[name]=_sha(path)
            exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace,namespace)
        missing=[name for name in required if not callable(namespace.get(name))]
        if missing:raise ValueError('Preserved API missing: '+','.join(missing))
        namespace['_neighbor_preserved_source_hashes']=hashes
        _PRESERVED_API_CACHE[key]=namespace
    return _PRESERVED_API_CACHE[key]


def _configuration(dataset_root,config_dir):
    if config_dir is None:
        module_file=globals().get('__file__')
        candidate=(Path(module_file).resolve().parents[3]/'Common/Neighbor_Experiments_QC_20261108'
                   if module_file else Path(dataset_root).resolve().parent/'Model/Common/Neighbor_Experiments_QC_20261108')
        if not (candidate/'experiments.json').is_file():
            candidate=Path(dataset_root).resolve().parent/'Model/Common/Neighbor_Experiments_QC_20261108'
    else:candidate=Path(config_dir)
    config=json.loads((candidate/'experiments.json').read_text(encoding='utf-8-sig'))
    registry=pd.read_csv(candidate/'station_registry.csv',dtype=str,keep_default_na=False)
    if registry.station_key.duplicated().any():raise ValueError('Duplicate expansion registry station key.')
    if len(config['conditions'])!=8:raise ValueError('Expected eight frozen neighbor conditions.')
    return candidate.resolve(),config,registry


def load_neighbor_bundle(dataset_root, condition=BASE_CONDITION, config_dir=None,
                         extra_data_root=None, lookback_hours=48,
                         preserved_code_dir=None):
    """Load one explicit data experiment, keeping target/context/labels intact.

    extra_data_root is a staging Dataset directory when new QC files have not
    been installed yet. Baseline source files always come from dataset_root.
    New western Japanese nodes use existing Japan hourly data and buffer.
    """
    root=Path(dataset_root).resolve()
    extra_root=Path(extra_data_root).resolve() if extra_data_root is not None else root
    folder,config,registry=_configuration(root,config_dir)
    if condition not in config['conditions']:raise ValueError('Unknown neighbor condition: '+str(condition))
    spec=config['conditions'][condition]
    selected=list(spec['station_keys'])
    if len(selected)!=len(set(selected)) or 'KR:111121' in selected:
        raise ValueError('External station selection duplicates or includes the target.')
    if len(selected)!=int(spec['expected_external_series']):
        raise ValueError('Condition external-series count mismatch.')
    if not set(selected).issubset(set(registry.station_key)):
        raise ValueError('Condition references an unregistered station.')
    api=preserved_neighbor_api(root,preserved_code_dir)
    bundle=api['load_dcrnn_expanded_bundle'](
        root,lookback_hours=lookback_hours,node_set='national_v3',national_registry=NATIONAL_REGISTRY)
    base_nodes=bundle['nodes']
    base_external=set(base_nodes.station_key)-{'KR:111121'}
    if len(base_nodes)!=601 or base_external!=set(config['conditions'][BASE_CONDITION]['station_keys']):
        raise ValueError('Current 601-node reference does not match frozen F600_W24.')
    if not base_nodes.iloc[0].station_key=='KR:111121':
        raise ValueError('Preserved DCRNN target must stay at node index zero.')
    reference={'condition':condition,'config_version':config['schema_version'],
               'config_directory':str(folder),'baseline_condition':BASE_CONDITION,
               'target_context_labels_preserved':True,'DCRNN_architecture_modified':False,
               'preserved_source_hashes':api.get('_neighbor_preserved_source_hashes',{}),
               'external_series':len(selected),'node_count':len(selected)+1,
               'factors':dict(spec['factors'])}
    if condition==BASE_CONDITION:
        # All original arrays and node ordering are returned without selection,
        # recasting, copying, source replacement or altered graph construction.
        bundle['neighbor_expansion_contract']=reference
        bundle['neighbor_input_condition']=condition
        bundle['neighbor_matching_folder']=str(root/BASE_MATCHING)
        return bundle

    target_and_selected=set(selected)|{'KR:111121'}
    keep=base_nodes.station_key.isin(target_and_selected).to_numpy()
    retained=base_nodes.loc[keep].copy()
    new_keys=sorted(set(selected)-base_external)
    indexed=registry.set_index('station_key',drop=False)
    extra=indexed.loc[new_keys].copy() if new_keys else registry.iloc[:0].copy()
    # Metadata for retained nodes remain original; only genuinely new keys use
    # the newly audited registry. Original relative node order is preserved.
    nodes=pd.concat([retained,extra],ignore_index=True)
    if set(nodes.station_key)!=target_and_selected or nodes.station_key.duplicated().any():
        raise ValueError('Selected observation scope and graph nodes differ.')
    if not nodes.pm25_unit.eq('ug/m3').all():raise ValueError('Mixed PM2.5 units.')
    if not nodes.station_key.eq(nodes.country+':'+nodes.station_id.astype(str)).all():
        raise ValueError('Station identity metadata mismatch.')
    for name in ('latitude','longitude'):
        nodes[name]=pd.to_numeric(nodes[name],errors='raise')
    if not np.isfinite(nodes[['latitude','longitude']]).all().all():raise ValueError('Missing node coordinates.')
    if not nodes.latitude.between(-90,90).all() or not nodes.longitude.between(-180,180).all():
        raise ValueError('Invalid coordinates.')
    nodes['node_index']=np.arange(len(nodes))
    panel=np.full((len(bundle['hourly_index']),len(nodes)),np.nan,dtype=np.float32)
    panel[:,:len(retained)]=bundle['pm_values'][:,keep]
    extra_rows=nodes.iloc[len(retained):].copy()
    if not extra_rows.source_dataset.isin({'Japan',*NEW_SOURCES}).all():
        raise ValueError('Unsupported source for additional neighbor nodes.')
    new_audit=[]
    for source in ('Japan',*NEW_SOURCES):
        part=extra_rows.loc[extra_rows.source_dataset.eq(source)].copy()
        if part.empty:continue
        positions=part.node_index.to_numpy(int)
        part['node_index']=np.arange(len(part))
        source_root=root if source=='Japan' else extra_root
        paths=WEST_SOURCE[source] if source=='Japan' else NEW_SOURCES[source]
        values,audit=api['_expanded_fill_panel'](source_root,part,bundle['hourly_index'],{source:paths})
        panel[:,positions]=values
        for item in audit:
            item=dict(item,source_dataset=source,source_root=str(source_root),
                      neighbor_condition=condition,loaded_station_keys=part.station_key.tolist())
            new_audit.append(item)
            path=source_root/item['path']
            fingerprint={'path':item['path'],'source_root':str(source_root),
                         'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
            bundle['source_fingerprints'].append(fingerprint)
    bundle['nodes']=nodes
    bundle['pm_values']=panel
    bundle['source_audit'].extend(new_audit)
    # legacy_nodes is intentionally the original116 (not current601 or new
    # selected nodes); preserved graph code obtains the frozen sigma from it.
    if len(bundle['legacy_nodes'])!=116:raise ValueError('Legacy distance reference changed.')
    bundle['reference_node_set']=bundle['node_set']
    bundle['reference_data_expansion_contract']=dict(bundle['data_expansion_contract'])
    bundle['node_set']='neighbor_'+condition
    bundle['data_expansion_contract']=dict(bundle['reference_data_expansion_contract'],
        node_set=bundle['node_set'],nodes=len(nodes),
        country_counts=nodes.groupby('country').size().to_dict(),
        japan_selection=dict(spec['factors']),
        city_proxy_nodes=int(nodes.source_dataset.isin({'China_Border_Proxy','China_East_QC'}).sum()),
        neighbor_condition=condition,architecture_modified=False)
    bundle['neighbor_expansion_contract']=dict(reference,
        retained_reference_nodes=len(retained),added_node_keys=new_keys,
        removed_reference_node_keys=base_nodes.loc[~keep,'station_key'].tolist(),
        selected_scope_applied_before_geographic_nearest_assignment=True,
        original_relative_node_order_preserved=True,
        city_average_new_nodes=int(nodes.source_dataset.eq('China_East_QC').sum()),
        no_future_observation_or_label_input=True)
    bundle['neighbor_input_condition']=condition
    bundle['neighbor_matching_folder']=str(extra_root/EXPANSION_MATCHING)
    return bundle


def build_neighbor_graphs(bundle, include_trajectory=True, matching_folder=None,
                           distance_radius_km=500., radius_km=75., max_gap_hours=6.,
                           bridge_unmatched=True, preserved_code_dir=None):
    """Call the preserved graph algorithms with a condition-bound source pool.

    The baseline must use exact original V4. Other conditions use the expanded
    union pool and the original selected-node filter before nearest assignment.
    """
    if 'neighbor_input_condition' not in bundle:raise ValueError('Use load_neighbor_bundle first.')
    expected=Path(bundle['neighbor_matching_folder']).resolve()
    if matching_folder is not None:
        provided=Path(matching_folder)
        if not provided.is_absolute():provided=expected.parent/provided
        if provided.resolve()!=expected:
            raise ValueError('Matching folder differs from the frozen condition source.')
    if include_trajectory and not expected.is_dir():
        raise FileNotFoundError('Condition matching source unavailable: '+str(expected))
    api=preserved_neighbor_api(bundle['dataset_root'],preserved_code_dir)
    graphs,reports=api['build_preserved_dcrnn_graphs'](
        bundle,include_trajectory=include_trajectory,matching_folder=expected,
        distance_radius_km=distance_radius_km,radius_km=radius_km,
        max_gap_hours=max_gap_hours,bridge_unmatched=bridge_unmatched)
    for report in reports.values():
        report.update(neighbor_condition=bundle['neighbor_input_condition'],
                      DCRNN_architecture_modified=False,graph_algorithm_changed=False)
    return graphs,reports
