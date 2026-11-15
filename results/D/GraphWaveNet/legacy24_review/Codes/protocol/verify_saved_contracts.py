from pathlib import Path
import zipfile,json,hashlib,io
import numpy as np

HERE=Path(__file__).resolve().parent
zs=[zipfile.ZipFile(Path('C:/Users/kimjy/Downloads')/f'VastAI-20261111T003043Z-1-00{i}.zip') for i in (1,2)]
members={i.filename:(z,i) for z in zs for i in z.infolist()}
root='VastAI/20261111_014538_406520_eaa72758/'
def read(n):z,i=members[n];return z.read(i)
def js(n):return json.loads(read(n))
results=[];base=None;graphbytes=None
for c in ('M1','M2','M3','M4','M3T','M4T'):
    paths=[n for n in members if n.startswith(root+f'Conditions/{c}/')]
    sm=next(n for n in paths if n.endswith('/run_summary.json'));s=js(sm)
    folder=sm.rsplit('/',1)[0]
    p=js(folder+'/preprocessing.json')
    cc=js(root+f'Condition_Contracts/{c}.json')
    fingerprint={
      'target_pm_mean':p['pm_mean'][0],'target_pm_std':p['pm_std'][0],'target_valid':p['pm_valid_count'][0],
      'base_context_mean':p['context_mean'][:53],'base_context_std':p['context_std'][:53],
      'base_context_valid_count':p['context_valid_count'][:53], 'base_context_names':p['context_names'][:53],
      'fit_start':p['fit_start_inclusive'],'fit_end':p['fit_end_exclusive']}
    if base is None:base=fingerprint
    assert fingerprint==base
    norder=next(n for n in paths if n.endswith('/node_order.csv') and '/Models/' not in n)
    # Saved preprocessor excludes every unused PM and trajectory feature.
    import csv
    nodes=list(csv.DictReader(io.StringIO(read(norder).decode('utf-8-sig'))))
    keys=[n['station_key'] for n in nodes]
    excluded=set(cc['excluded_station_keys'])
    assert all(p['pm_valid_count'][j]==0 for j,k in enumerate(keys) if k in excluded)
    tallowed=set(cc['trajectory_allowed_names'])
    assert all(p['context_valid_count'][53+j]==0 for j,k in enumerate(cc['trajectory_feature_names']) if k not in tallowed)
    bn=folder+'/best.pt'; sha=hashlib.sha256(read(bn)).hexdigest()
    assert sha==s['checkpoint_sha256']
    graph=next(n for n in paths if n.endswith('/Graphs/distance.npy'))
    gb=read(graph)
    if graphbytes is None:graphbytes=gb
    assert gb==graphbytes
    a=np.load(io.BytesIO(gb),allow_pickle=False)
    assert a.shape==(635,635) and np.isfinite(a).all()
    adaptive=np.load(io.BytesIO(read(folder+'/adaptive_adjacency.npy')),allow_pickle=False)
    assert adaptive.shape==a.shape and np.isfinite(adaptive).all()
    results.append({'condition':c,'target_base_preprocessor_identical':True,'excluded_preprocessor_counts_zero':True,
       'best_pt_sha_matches_summary':True,'best_pt_sha256':sha,'same_fixed_distance_graph':True,
       'fixed_nonzero_edges':int(np.count_nonzero(a)), 'adaptive_nonzero_edges':int(np.count_nonzero(adaptive)),
       'adaptive_row_sum_min':float(adaptive.sum(axis=1).min()),'adaptive_row_sum_max':float(adaptive.sum(axis=1).max())})
(HERE/'saved_contract_verification.json').write_text(json.dumps({'results':results,'base_context_names':base['base_context_names']},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
