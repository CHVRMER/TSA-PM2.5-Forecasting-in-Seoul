"""Data-only expansion for the unchanged original Seoul DCRNN implementation.

Execute after the original data.py and graphs.py definitions (the notebook does
this automatically). The model/training functions and graph weighting functions
remain unchanged; only hourly-source loading and selected-neighbor routing differ.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


V3_EXTRA_SOURCES = {
    'Japan': ('Japan/japan_west_hourly_2021_2024_kst.parquet',
              'Japan/japan_west_context_buffer.parquet'),
    'Mongolia': ('Mongolia/mongolia_ub_embassy_hourly_2021_2024_kst.parquet',
                 'Mongolia/mongolia_ub_embassy_context_buffer.parquet'),
    'China_Border_Proxy': ('China_Border_Proxy/china_border_cities_hourly_2021_2024_kst.parquet',
                         'China_Border_Proxy/china_border_cities_context_buffer.parquet'),
}


def _expanded_fill_panel(root, nodes, hourly_index, source_files):
    panel = np.full((len(hourly_index), len(nodes)), np.nan, dtype=np.float32)
    seen = np.zeros(panel.shape, dtype=bool)
    audit = []
    for source, relatives in source_files.items():
        subset = nodes.loc[nodes.source_dataset.eq(source)]
        if subset.empty:
            continue
        mapping = dict(zip(subset.station_id.astype(str), subset.node_index.astype(int)))
        for relative in relatives:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing hourly data: {relative}')
            frame = pq.read_table(path, columns=['time_kst', 'station_id', 'pm25', 'pm25_flag'],
                                  filters=[('station_id', 'in', list(mapping))]).to_pandas()
            frame['time_kst'] = _dcrnn_kst(frame.time_kst)
            frame['station_id'] = frame.station_id.astype(str)
            if frame.duplicated(['time_kst', 'station_id']).any():
                raise ValueError(f'Duplicate station/hour: {relative}')
            time_pos = hourly_index.get_indexer(frame.time_kst)
            take = time_pos >= 0
            frame, time_pos = frame.loc[take], time_pos[take]
            node_pos = frame.station_id.map(mapping).to_numpy(dtype=int)
            if seen[time_pos, node_pos].any():
                raise ValueError(f'Overlapping observations across source files: {relative}')
            values = pd.to_numeric(frame.pm25, errors='coerce').to_numpy(float)
            flags = pd.to_numeric(frame.pm25_flag, errors='coerce').to_numpy(float)
            valid = (flags == 0) & np.isfinite(values) & (values >= 0)
            if np.any((flags == 0) & ~valid):
                raise ValueError(f'Invalid flag-zero PM2.5: {relative}')
            panel[time_pos, node_pos] = np.where(valid, values, np.nan)
            seen[time_pos, node_pos] = True
            audit.append({'path': relative, 'rows_in_hourly_grid': int(len(frame)),
                          'valid_pm25_rows': int(valid.sum()), 'bytes': path.stat().st_size,
                          'mtime_ns': path.stat().st_mtime_ns})
    return panel, audit


def load_dcrnn_expanded_bundle(dataset_root, lookback_hours=48, node_set='v3_154',
                               national_registry=None):
    """Keep legacy values/context/labels unchanged; append explicitly chosen nodes.

    legacy116: exact original loader.
    v3_154: legacy116 + Japanese balanced24 + Ulaanbaatar1 + Chinese city13.
    national_v3: explicit separate data-expansion experiment; supply an audited
      registry with the normal identity/coordinate/source columns. This does not
      silently turn on when nationwide files appear on disk.
    """
    if node_set not in {'legacy116', 'v3_154', 'national_v3'}:
        raise ValueError('node_set must be legacy116, v3_154, or national_v3')
    bundle = load_dcrnn_bundle(dataset_root, lookback_hours=lookback_hours)
    bundle['legacy_node_count'] = len(bundle['nodes'])
    bundle['legacy_nodes'] = bundle['nodes'].copy()
    bundle['node_set'] = node_set
    if node_set == 'legacy116':
        return bundle
    root = bundle['dataset_root']
    registry_path = root/'Trajectory_Matching_2021_2024_v3/station_registry.csv'
    balanced_path = root/'Origin_Features_2021_2024_v3/japan_balanced_stations.csv'
    registry = pd.read_csv(registry_path, dtype=str).fillna('')
    japan = pd.read_csv(balanced_path, dtype=str).fillna('')
    legacy_keys = set(bundle['nodes'].station_key)
    selected = registry.country.ne('JP') | registry.station_key.isin(set(japan.station_key))
    extra = registry.loc[selected & ~registry.station_key.isin(legacy_keys)].copy()
    if len(extra) != 38 or len(japan) != 24:
        raise ValueError('Expected 24 Japanese + 1 Mongolian + 13 Chinese city additions')
    if not extra.source_dataset.isin(V3_EXTRA_SOURCES).all():
        raise ValueError('Unsupported source in v3 additions')
    extra = extra.sort_values('station_key', kind='stable')
    nodes = pd.concat([bundle['nodes'], extra], ignore_index=True)
    if node_set == 'national_v3':
        if national_registry is None:
            raise ValueError('national_v3 requires an explicitly audited national_registry')
        national_registry=Path(national_registry)
        if not national_registry.is_absolute():
            national_registry=root/national_registry
        national = pd.read_csv(national_registry, dtype=str).fillna('')
        required = {'station_key','station_id','country','source_dataset','pm25_unit','latitude','longitude'}
        if not required.issubset(national):
            raise ValueError(f'National registry missing {sorted(required-set(national))}')
        national = national.loc[national.country.eq('KR')].copy()
        if 'primary_model_eligible' in national:
            use = national.primary_model_eligible.astype(str).str.lower().isin(['true','1','yes'])
            national = national.loc[use].copy()
        elif 'train_quality_eligible' in national:
            use = national.train_quality_eligible.astype(str).str.lower().isin(['true','1','yes'])
            national = national.loc[use].copy()
        else:
            raise ValueError('National model registry needs train-only primary_model_eligible or train_quality_eligible')
        national = national.loc[~national.station_key.isin(set(nodes.station_key))]
        nodes = pd.concat([nodes, national.sort_values('station_key')], ignore_index=True)
    if nodes.station_key.duplicated().any() or not nodes.pm25_unit.eq('ug/m3').all():
        raise ValueError('Duplicate node identity or inconsistent PM2.5 unit')
    if not (nodes.station_key == nodes.country + ':' + nodes.station_id).all():
        raise ValueError('station_key must equal country:station_id')
    for field in ['latitude','longitude']:
        nodes[field] = pd.to_numeric(nodes[field], errors='raise')
    if not np.isfinite(nodes[['latitude','longitude']]).all().all():
        raise ValueError('Coordinates unavailable; do not silently invent locations')
    if not nodes.latitude.between(-90,90).all() or not nodes.longitude.between(-180,180).all():
        raise ValueError('Out-of-range coordinates')
    nodes['node_index'] = np.arange(len(nodes))
    nodes['observation_type'] = np.where(nodes.source_dataset.eq('China_Border_Proxy'),
                                           'Chinese_city_average_proxy', 'station_observation')
    nodes['coordinate_interpretation'] = np.where(nodes.source_dataset.eq('China_Border_Proxy'),
        'approximate_city_location_not_measurement_station', 'source_station_metadata')
    new_nodes = nodes.iloc[bundle['legacy_node_count']:].copy()
    new_nodes['node_index'] = np.arange(len(new_nodes))
    sources = dict(V3_EXTRA_SOURCES)
    sources['Korea_National'] = (
        'Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet',
        'Korea_National_2021_2024/korea_national_pm25_context_buffer.parquet')
    sources['Korea_National_2021_2024'] = sources['Korea_National']
    new_panel, audit = _expanded_fill_panel(root, new_nodes, bundle['hourly_index'], sources)
    bundle['pm_values'] = np.concatenate([bundle['pm_values'], new_panel], axis=1)
    bundle['nodes'] = nodes
    bundle['source_audit'].extend(audit)
    for path in [registry_path, balanced_path, *[root/a['path'] for a in audit]]:
        bundle['source_fingerprints'].append({'path': str(path.relative_to(root)),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
    bundle['data_expansion_contract'] = {
        'node_set': node_set, 'legacy_prefix_preserved': True,
        'legacy_node_count': bundle['legacy_node_count'], 'nodes': len(nodes),
        'target_context_and_labels': 'original v2 frozen allowlist and eligibility unchanged',
        'japan_selection': 'existing balanced24 file; no newly fitted station selection',
        'country_counts': nodes.groupby('country').size().to_dict(),
        'russian_observation_nodes': 0,
        'city_proxy_nodes': int(nodes.source_dataset.eq('China_Border_Proxy').sum()),
        'national_is_separate_experiment': node_set == 'national_v3',
    }
    return bundle


def expanded_trajectory_adjacency(bundle, matching_folder=None, radius_km=75.,
                                  max_gap_hours=6., bridge_unmatched=True):
    """Read selected nodes from a wider matching pool; use original graph functions."""
    root = bundle['dataset_root']
    default = ('Trajectory_Matching_2021_2024' if bundle['node_set']=='legacy116'
               else 'Trajectory_Matching_2021_2024_v3')
    folder = root/default if matching_folder is None else Path(matching_folder)
    if not folder.is_absolute():
        folder=root/folder
    manifest_path=folder/'matching_manifest.json'
    declared_radius=75.
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        declared_radius=float(manifest.get('parameters',{}).get('radius_km',manifest.get('radius_km',75.)))
    if not np.isfinite(radius_km) or radius_km<=0 or radius_km>declared_radius+1e-9:
        raise ValueError(f'Matching supports at most {declared_radius:g} km; requested {radius_km}')
    start, end = _graph_timestamp('2021-01-01'), _graph_timestamp('2023-01-01')
    filters = [('arrival_time_kst','>=',start.to_pydatetime()),
               ('arrival_time_kst','<',end.to_pydatetime())]
    point_path, edge_path = folder/'trajectory_point_coverage.parquet', folder/'trajectory_station_neighbors.parquet'
    points = pq.read_table(point_path, columns=['point_id','arrival_time_kst','lookback_hours'], filters=filters).to_pandas()
    edge_columns=['point_id','arrival_time_kst','lookback_hours','station_key','distance_km']
    external_keys = set(bundle['nodes'].station_key)-{'KR:111121'}
    source_neighbor_rows=0
    streamed=edge_path.stat().st_size>100_000_000
    if streamed:
        # Keep only the closest candidate in each batch, then apply the original
        # closest-candidate function globally. Lexicographic min is associative,
        # so batches cannot change the chosen node; peak RAM stays bounded for
        # nationwide neighbor tables with millions of overlapping nearby nodes.
        pieces=[]
        for batch in pq.ParquetFile(edge_path).iter_batches(batch_size=131072,columns=edge_columns):
            chunk=_graph_training_rows(batch.to_pandas(),start,end)
            chunk=chunk.loc[chunk.station_key.isin(external_keys)]
            source_neighbor_rows+=len(chunk)
            if chunk.empty:
                continue
            if not np.isfinite(chunk.distance_km).all() or (chunk.distance_km<0).any():
                raise ValueError('Invalid matching distance')
            chunk=chunk.loc[chunk.distance_km<=float(radius_km)+1e-9]
            pieces.append(chunk.sort_values(['distance_km','station_key'],kind='mergesort').drop_duplicates('point_id'))
        edges=(pd.concat(pieces,ignore_index=True).sort_values(['distance_km','station_key'],kind='mergesort')
               .drop_duplicates('point_id')) if pieces else pd.DataFrame(columns=edge_columns)
    else:
        edges = pq.read_table(edge_path, columns=edge_columns,filters=filters).to_pandas()
        edges = _graph_training_rows(edges,start,end)
        edges = edges.loc[edges.station_key.isin(external_keys)].copy()
        source_neighbor_rows=len(edges)
    points = _graph_training_rows(points,start,end)
    if points.point_id.duplicated().any():
        raise ValueError('Duplicated trajectory point identities')
    per_arrival = points.groupby('arrival_time_kst').lookback_hours.agg(['count','nunique'])
    if not ((per_arrival['count']==48)&(per_arrival['nunique']==48)).all():
        raise ValueError('Expected 48 hourly points for every training arrival')
    # A wider source pool does not silently add nodes to the selected experiment.
    if not np.isfinite(edges.distance_km).all() or (edges.distance_km < 0).any():
        raise ValueError('Invalid matching distance')
    ref = points.rename(columns={'arrival_time_kst':'point_arrival','lookback_hours':'point_age'})
    check = edges.merge(ref,on='point_id',how='left',validate='many_to_one')
    if check.point_arrival.isna().any() or not check.arrival_time_kst.eq(check.point_arrival).all() or not check.lookback_hours.eq(check.point_age).all():
        raise ValueError('Neighbor identity/time mismatch')
    selected = nearest_trajectory_assignments(edges,bundle['nodes'],radius_km=radius_km)
    adjacency, report = trajectory_adjacency_from_assignments(selected,bundle['nodes'],
        all_arrivals=per_arrival.index,max_gap_hours=max_gap_hours,bridge_unmatched=bridge_unmatched)
    report.update({'matching_folder':str(folder),'matching_radius_km':float(radius_km),
        'saved_matching_radius_km':declared_radius,
        'graph_algorithm_changed':False,'selected_node_set':bundle['node_set'],
        'training_arrivals_only':True,'pollution_or_target_values_used':False,
        'filtered_source_neighbor_rows':source_neighbor_rows,'selected_geographic_points':len(selected),
        'reader':'streamed_exact_nearest_min' if streamed else 'parquet_filtered_table'})
    return adjacency, report


def build_preserved_dcrnn_graphs(bundle, include_trajectory=True, matching_folder=None,
                                 distance_radius_km=500., radius_km=75., max_gap_hours=6.,
                                 bridge_unmatched=True):
    """Keep the legacy numerical distance scale when only the node set expands."""
    _, legacy = distance_adjacency(bundle['legacy_nodes'],radius_km=distance_radius_km)
    sigma = legacy['sigma_km']
    adjacency, report = distance_adjacency(bundle['nodes'],radius_km=distance_radius_km,sigma_km=sigma)
    report.update({'sigma_basis':'fixed legacy116 positive distances within 500km',
                   'graph_algorithm_changed':False,'data_expansion_only':bundle['node_set']!='legacy116'})
    graphs, reports = {'distance':adjacency}, {'distance':report}
    if include_trajectory:
        graphs['trajectory_directed'], reports['trajectory_directed'] = expanded_trajectory_adjacency(
            bundle,matching_folder=matching_folder,radius_km=radius_km,
            max_gap_hours=max_gap_hours,bridge_unmatched=bridge_unmatched)
        graphs['trajectory_symmetric'], reports['trajectory_symmetric'] = symmetric_trajectory_adjacency(
            graphs['trajectory_directed'],bundle['nodes'])
    return graphs,reports


def dcrnn_common_inverse_unclipped(values, preprocessing):
    """Raw inverse for COMMON scoring only; native training inverse is unchanged."""
    mean=float(preprocessing['pm_mean'][0])
    std=float(preprocessing['pm_std'][0])
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.expm1(np.asarray(values,dtype=np.float64)*std+mean)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite common forecast; fixed evaluation rows cannot be dropped')
    return result


@torch.no_grad()
def predict_dcrnn_common_unclipped(model, loader, bundle, preprocessing, device,
                                    model_name, seed, data_version, experiment_mode):
    """Separate inference pass; no fitting and no changes to native model selection.

    Legacy exported y_hat has already been clipped by original_training.py and
    cannot reveal its old negative forecasts. Always infer again from the saved
    best weights, retaining raw inverse predictions for the common scorer.
    """
    model.eval()
    outputs,indices=[],[]
    for x,context,_target,_mask,origin_index in loader:
        prediction=model(x.to(device),context.to(device))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite common model output')
        outputs.append(prediction.cpu().numpy())
        indices.extend(origin_index.numpy().tolist())
    if not outputs:
        raise ValueError('No common prediction batches')
    indices=np.asarray(indices,dtype=int)
    if len(indices)!=len(np.unique(indices)):
        raise ValueError('Duplicate common prediction origins')
    raw=dcrnn_common_inverse_unclipped(np.concatenate(outputs),preprocessing)
    persistence,fallback=dcrnn_persistence(bundle,preprocessing)
    rows=[]
    for row,index in enumerate(indices):
        origin=bundle['origins'].iloc[index]
        for j,horizon in enumerate(bundle['horizons']):
            value=float(raw[row,j])
            rows.append({'origin_time_kst':origin['origin_time_kst'],
                'target_time_kst':origin['origin_time_kst']+pd.Timedelta(hours=horizon),
                'horizon':int(horizon),'horizon_h':int(horizon),'model':model_name,
                'split':origin['origin_split'],'y_true':float(bundle['y_raw'][index,j]),
                'y_hat':value,'y_pred':max(value,0.),'y_hat_legacy_clipped':max(value,0.),
                'eligible':bool(bundle['eligible'][index,j]),'seed':int(seed),
                'persistence_pred':float(persistence[index]),'persistence_fallback':bool(fallback[index]),
                'dataset_version':data_version,'experiment_mode':experiment_mode,
                'common_export_unclipped':True,
                'inference_source':'same_best_weights_separate_unclipped_inverse_no_fitting'})
    return pd.DataFrame(rows),indices


def collect_dcrnn_common_from_best(run_summaries,bundle,arrays,preprocessing,graphs,
                                    config,split,device):
    """Reload saved best checkpoints for raw common exports; never train."""
    if split not in {'validation','test'}:
        raise ValueError('Common export split must be validation or test')
    if split=='test' and config['mode']!='full':
        raise ValueError('No pilot test inference')
    ids=split_origin_indices(bundle,split,require_any_label=False)
    if split=='validation' and config.get('max_val_origins'):
        ids=ids[:int(config['max_val_origins'])]
    dataset=DCRNNOriginDataset(bundle,arrays,ids,config['lookback_hours'])
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0)
    frames=[]
    for summary in run_summaries:
        graph_name=summary['graph']
        adjacency=graphs[graph_name]
        checkpoint=torch.load(Path(summary['run_directory'])/'best.pt',map_location=device,weights_only=False)
        if checkpoint['contract']['data']!=dcrnn_data_signature(bundle):
            raise ValueError('Best checkpoint and common-export data disagree')
        graph_signature=hashlib.sha256(np.ascontiguousarray(adjacency,dtype=np.float32).tobytes()).hexdigest()
        if checkpoint['contract']['graph']!=graph_signature:
            raise ValueError('Best checkpoint and common-export graph disagree')
        model=DCRNNMultiHorizon(adjacency,input_dim=2,hidden_dim=config['hidden_dim'],
            diffusion_steps=config['diffusion_steps'],context_dim=arrays['context_x'].shape[1],
            horizons=bundle['horizons'],dropout=config.get('dropout',0.)).to(device)
        model.load_state_dict(checkpoint['model'])
        table,produced=predict_dcrnn_common_unclipped(model,loader,bundle,preprocessing,device,
            'DCRNN_'+graph_name,summary['seed'],summary['data_signature'],config['mode'])
        if not np.array_equal(produced,ids):
            raise AssertionError('Common inference did not cover its prescribed origin grid')
        frames.append(table)
        del model
    return frames


def export_dcrnn_common_predictions(prediction_tables, output_dir, node_set, split,
                                     data_version, experiment_mode):
    """Export seed predictions and RAW-unit arithmetic ensembles separately."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for table in prediction_tables:
        frame=table.copy()
        if 'common_export_unclipped' not in frame or not frame.common_export_unclipped.eq(True).all():
            raise ValueError('Legacy clipped exports cannot recover raw predictions; infer again from best weights')
        frame['horizon']=frame['horizon'].astype(int)
        frame['y_hat']=pd.to_numeric(frame['y_hat'],errors='raise')
        frame['model']=frame['model'].astype(str)+'__'+node_set
        frame['prediction_kind']='seed'
        frame['input_condition']=node_set
        frame['data_version']=data_version
        frame['experiment_mode']=experiment_mode
        frames.append(frame)
    all_seed=pd.concat(frames,ignore_index=True)
    keys=['origin_time_kst','horizon','model']
    if all_seed.duplicated(keys+['seed']).any():
        raise ValueError('Duplicate model/seed/origin/horizon predictions')
    expected=all_seed.groupby('model').seed.nunique()
    sizes=all_seed.groupby(keys).seed.nunique()
    if any(n!=expected.loc[key[2]] for key,n in sizes.items()):
        raise ValueError('Cannot ensemble different row sets or silently drop a missing seed')
    for key,g in all_seed.groupby(keys,sort=False):
        if g.eligible.nunique()!=1 or g.y_true.nunique(dropna=False)!=1:
            raise ValueError('Seed target/eligibility disagreement')
    if not np.isfinite(all_seed.y_hat).all():
        raise ValueError('Nonfinite seed predictions')
    seed_path=out/f'{split}_predictions_by_seed.parquet'
    all_seed.to_parquet(seed_path,index=False)
    ensemble=all_seed.groupby(keys,as_index=False).agg(
        y_hat=('y_hat','mean'),target_time_kst=('target_time_kst','first'),
        split=('split','first'),y_true=('y_true','first'),eligible=('eligible','first'),
        n_seeds=('seed','nunique'))
    ensemble['prediction_kind']='ensemble'
    ensemble['seed']=''
    ensemble['ensemble_method']='raw_unit_arithmetic_mean'
    ensemble['input_condition']=node_set
    ensemble['data_version']=data_version
    ensemble['experiment_mode']=experiment_mode
    ensemble['common_export_unclipped']=True
    ensemble['y_hat_legacy_clipped']=all_seed.groupby(keys,as_index=False).y_hat_legacy_clipped.mean().y_hat_legacy_clipped
    ensemble['legacy_clipped_comparison_mean']=ensemble.pop('y_hat_legacy_clipped')
    ensemble_path=out/f'{split}_predictions_ensemble_raw_mean.parquet'
    ensemble.to_parquet(ensemble_path,index=False)
    metadata={'node_set':node_set,'split':split,'data_version':data_version,
              'ensemble_rule':'arithmetic mean of seed forecasts AFTER inverse transform in ug/m3',
              'mean_of_seed_scores_is_not_ensemble_score':True,
              'common_y_hat_is_unclipped_inverse':True,
              'negative_clipping_stage':'Common score.py after seed raw export / after raw ensemble mean',
              'legacy_training_validation_clipping_unchanged':True,
              'pilot_is_not_fixed_full_grid_score':experiment_mode=='pilot',
              'seed_rows':len(all_seed),'ensemble_rows':len(ensemble)}
    (out/f'{split}_prediction_export.meta.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    for path,kind in [(seed_path,'seed'),(ensemble_path,'ensemble')]:
        specific=dict(metadata,prediction_file=path.name,prediction_kind=kind)
        path.with_suffix('.meta.json').write_text(json.dumps(specific,ensure_ascii=False,indent=2),encoding='utf-8')
    return all_seed,ensemble
