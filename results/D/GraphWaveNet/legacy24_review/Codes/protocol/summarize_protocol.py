from pathlib import Path
import json,csv,math
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'protocol_audit.json').read_text(encoding='utf-8'))
rows=[]
invariants=[]
for c,d in r['conditions'].items():
    s=next(j for n,j in d['json'].items() if n.endswith('run_summary.json'))
    curve=next(iter(d['curves'].values()))
    vals=[float(a['validation_rmse_h6']) for a in curve]
    train=[float(a['train_loss']) for a in curve]
    contract=d['condition_contract']
    rows.append({'condition':c,'trained_epochs':s['trained_epochs'],'best_epoch':s['best_epoch'],
        'val6_rmse':s['validation_rmse_h6'],'max_val6_rmse':max(vals),'first_val6_rmse':vals[0],
        'last_val6_rmse':vals[-1],'min_train_loss':min(train),'max_train_loss':max(train),'last_train_loss':train[-1],
        'nonfinite_curve_cells':sum(not math.isfinite(x) for x in vals+train),
        'parameter_count':s['parameter_count'],'current_session_training_seconds':s['session_training_seconds'],
        'curve_epoch_seconds_sum':sum(float(a['seconds']) for a in curve),
        'allowed_external_series':contract['allowed_external_series'],
        'trajectory_countries':'|'.join(contract['trajectory_countries']),
        'trajectory_allowed_names':len(contract['trajectory_allowed_names']),
        'trajectory_feature_names':len(contract['trajectory_feature_names']),
        'raw_context_size':contract['raw_context_size'],'transformed_context_size':contract['transformed_context_size'],
        'train_start':contract['train_fit_start_inclusive'],'train_end':contract['train_fit_end_exclusive'],
        'mask_before_normalization':contract['mask_applied_before_normalization'],
        'graph_sha256':contract['fixed_graph_sha256'],'node_order_sha256':contract['node_order_sha256'],
        'has_policy_revision':'stopping_policy_revision' in s,
        'last_bad_epochs':curve[-1]['bad_epochs']})
    invariants.append({k:contract[k] for k in ('fixed_node_count','fixed_graph_sha256','node_order_sha256','base_context_size','transformed_context_size','targets_and_common_context_preserved','excluded_nodes_remain_latent_graph_relays')})
    allowed=contract['allowed_station_keys'];excluded=contract['excluded_station_keys']
    assert len(allowed)==contract['allowed_external_series']
    assert not set(allowed)&set(excluded)
    assert len(set(allowed)|set(excluded))==634
    assert all(n in contract['trajectory_feature_names'] for n in contract['trajectory_allowed_names'])
    assert (contract['trajectory_countries']==['KR','CN'] and all('_all_' not in n and '_jp_' not in n and '_mn_' not in n for n in contract['trajectory_allowed_names'])) if c=='M3T' else True
with (HERE/'training_protocol_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,ensure_ascii=False,indent=2))
print('invariants_equal',all(v==invariants[0] for v in invariants))
print('source', {k:v for k,v in r['source_contract'].items() if k not in ('node_order','source_files','implementation_sha256','origin_times')})
print('contract settings',r['conditions']['M3T']['condition_contract']['settings'])
