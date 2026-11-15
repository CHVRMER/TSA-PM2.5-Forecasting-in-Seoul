"""One-shot, prediction-before-label final evaluation; shared production/synthetic engine."""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,platform,uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.metadata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from src.adaptive import condition_frame,apply_horizon

PROJECT=Path(__file__).resolve().parents[1]
BASE=['M1','M2','M3','M4','M3T','M4T']
MODELS=[*BASE,'Adaptive_wind','Adaptive_route'];HS=[3,6,12,24]
COUNTS=dict(zip(BASE,[1142,1478,2118,2518,2424,2924]))
ANCHOR='0d260f45a5190d26c8a3270a1ef0db55382bc131b313c14431de79121190dcd9'
DATA_DIR='Dataset/Origin_Features_2021_2024_v3'
DATA_FILE=DATA_DIR+'/origin_features_2021_2024_kst.parquet'
OUTPUT='results/test_2024/execution'
PRIMARY='results/final_test_2024_pipeline.lock';SECONDARY='results/final_test_2024.lock'
COMPARISONS=[(m,'persistence') for m in MODELS]+[(a,b) for i,a in enumerate(MODELS) for b in MODELS[:i]]
CORE=[(m,'persistence') for m in MODELS]+[('M2','M1'),('M3','M2'),('M3','M1'),('M4','M3'),('M3T','M3'),('M4T','M4'),('M4T','M3T'),('Adaptive_wind','M1'),('Adaptive_wind','M3'),('Adaptive_route','M1'),('Adaptive_route','M3'),('Adaptive_route','Adaptive_wind')]
CODE=['src/final_test.py','scripts/run_final_test.py','scripts/freeze_final_test.py','scripts/check_final_test.py']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def canonical(d):return json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def map_sha(d):return hashlib.sha256(canonical(d)).hexdigest()
def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
def relative(root,path):return str(Path(path).resolve().relative_to(Path(root).resolve()))
def safe(root,rel):
    if not isinstance(rel,str) or Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Unsafe artifact path')
    p=(Path(root)/rel).resolve();p.relative_to(Path(root).resolve());return p

def policy():
    return {'models':MODELS,'horizons':HS,'primary_horizon':6,'year':2024,'origin_split':'test','origin_frequency_hours':3,'expected_origin_rows':2928,'comparisons':[list(x) for x in COMPARISONS],'holm_family':[f'{h}:{a}:{b}' for h in HS for a,b in COMPARISONS],'holm_family_size':144,'core_display_comparisons':[list(x) for x in CORE],'core_display_tests':80,'core_separate_holm':False,'bootstrap_reps':1000,'bootstrap_seed':0,'bootstrap_block_slots':56,'DM':'squared errors; Bartlett HAC lag=h/3 on full calendar grid','improvement':'RMSE_difference_CI95_upper<0 AND Holm_p<0.05','prediction':'mean(expm1(seed predictions)); blend raw M1/M3 then clip only at scoring','seeds':[0,1,2],'inference_threads':2,'refit':False,'weight_fit_year':2023,'reestimate_adaptive':False,'eligible_h_required':True,'target_alignment':'exact origin+h; origin and target both2024','missing_target':'never impute','missing_inputs':'LightGBM native NaN; unknown adaptive condition -> frozen season weight','negative_predictions':'clip to0 for scoring; count negatives','output':OUTPUT,'primary_lock':PRIMARY,'secondary_lock':SECONDARY}
def versions():return {'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','lightgbm','scipy']}}

def validate_plan(plan,root,synthetic=False):
    root=Path(root).resolve()
    if synthetic and root==PROJECT:raise PermissionError('Synthetic execution cannot use the real project root')
    if plan.get('schema')!=1 or plan.get('mode')!=('synthetic' if synthetic else 'production'):raise PermissionError('Wrong execution mode')
    if plan.get('project_root')!=str(root) or plan.get('policy')!=policy():raise ValueError('Frozen scope/pairs/evaluation policy mismatch')
    if plan.get('versions')!=versions():raise ValueError('Environment differs from frozen versions')
    if plan.get('dataset',{}).get('file')!=DATA_FILE:raise ValueError('Unexpected input source')
    if set(plan['base_models'])!=set(BASE) or set(plan['adaptive_weights'])!=set(MODELS[6:]):raise ValueError('Incomplete model registry')
    static=plan['static_artifacts_sha256'];preserve=plan['preserve_sha256'];modelmap=plan['model_config_sha256']
    if not set(modelmap)<=set(static) or any(static[k]!=v for k,v in modelmap.items()):raise ValueError('Model commitment not bound to static ledger')
    if map_sha(modelmap)!=plan['model_config_composite_sha256']:raise ValueError('Model commitment mismatch')
    if not synthetic and (len(modelmap)!=147 or plan['model_config_composite_sha256']!=ANCHOR):raise ValueError('Approved147 artifacts changed')
    # Only static models/configuration and explicitly2023 outputs are permitted in pre-data ledgers.
    for ledger in [static,preserve,modelmap]:
        for rel,digest in ledger.items():
            p=safe(root,rel)
            if p==safe(root,DATA_FILE) or rel.startswith(('Models/work/','Models/predictions/')):raise ValueError('Mixed-year data in pre-access ledger')
            if sha(p)!=digest:raise ValueError('Frozen artifact changed: '+rel)
    for rel in [*CODE,'src/common_scoring.py','src/adaptive.py']:
        if rel not in static or static[rel]!=sha(PROJECT/rel):raise ValueError('Runner code not frozen')
    meta_path=DATA_DIR+'/feature_manifest.json';dict_path=DATA_DIR+'/feature_dictionary.csv'
    if meta_path not in static or dict_path not in static:raise ValueError('Missing frozen feature metadata')
    metadata=json.loads(safe(root,meta_path).read_text())
    if metadata!=plan['dataset']['metadata']:raise ValueError('Dataset metadata differs from frozen plan')
    with safe(root,dict_path).open(encoding='utf-8-sig') as f:roles={x['name']:x['role'] for x in csv.DictReader(f)}
    required=set(CODE+[meta_path,dict_path]);keys={str(h) for h in HS}
    for model in BASE:
        if set(plan['base_models'][model])!=keys:raise ValueError('Horizon registry mismatch')
        for hs,entry in plan['base_models'][model].items():
            names=entry['features']
            if not names or len(set(names))!=len(names) or any(roles.get(c)!='feature' for c in names):raise ValueError('Unsafe feature allowlist')
            if not synthetic and len(names)!=COUNTS[model]:raise ValueError('Changed model feature count')
            if len(entry['model_files'])!=3:raise ValueError('Missing seed model')
            bound=[*entry['model_files'],entry['ensemble_manifest'],entry['config']]
            if not set(bound)<=set(modelmap):raise ValueError('Model registry escaped approved commitment')
            if entry['config']!=f'configs/{model.lower()}_{hs}h.json':raise ValueError('Wrong frozen configuration path')
            folder=Path(entry['ensemble_manifest']).parent
            if entry['model_files']!=[str(folder/f'{model}_h{hs}_seed{seed}.txt') for seed in [0,1,2]]:raise ValueError('Wrong seed/horizon model binding')
            manifest=json.loads(safe(root,entry['ensemble_manifest']).read_text());config=json.loads(safe(root,entry['config']).read_text())
            if manifest['model']!=model or manifest['horizon']!=int(hs) or manifest['feature_names']!=names or manifest['seeds']!=[0,1,2]:raise ValueError('Wrong saved model registry')
            if config['model']!=model or config['horizon']!=int(hs):raise ValueError('Wrong model configuration')
            required.update(entry['model_files']+[entry['ensemble_manifest'],entry['config']])
    for model in MODELS[6:]:
        if set(plan['adaptive_weights'][model])!=keys:raise ValueError('Missing Adaptive horizon')
        for hs,rel in plan['adaptive_weights'][model].items():
            if rel!=f'Models/{model}/h{hs}/weights.json' or rel not in modelmap:raise ValueError('Wrong frozen weight path')
            required.add(rel);w=json.loads(safe(root,rel).read_text())
            if w['model']!=model or w['horizon']!=int(hs) or w['fitted_year']!=2023:raise ValueError('Wrong frozen weight source')
            if len(w['season_rules'])!=2 or len(w['cell_rules'])!=(10 if model=='Adaptive_wind' else 8):raise ValueError('Incomplete frozen rules')
            if any(not 0<=row['final_weight']<=1 for row in w['season_rules']+w['cell_rules']):raise ValueError('Invalid frozen weights')
    if not required<=set(static):raise ValueError('Unhashed required artifact')
    for rel,digest in preserve.items():
        if rel in static and static[rel]!=digest:raise ValueError('Contradictory hash ledgers')
    if not synthetic:
        record=json.loads(safe(root,plan['dataset']['hash_record']).read_text())
        if record['original_inputs_sha256'][DATA_FILE]!=plan['dataset']['expected_sha256']:raise ValueError('Input hash differs from existing validation record')
        report=safe(root,plan['runner_validation']['file'])
        if sha(report)!=plan['runner_validation']['sha256']:raise ValueError('Synthetic validation receipt changed')
        v=json.loads(report.read_text())
        if not v['passed'] or v['real_2024_data_used'] or v['code_sha256']!={c:static[c] for c in CODE}:raise ValueError('Runner has not passed frozen-code validation')
    return {'static_artifacts':len(static),'preserved_artifacts':len(preserve),'model_config_composite_sha256':plan['model_config_composite_sha256'],'raw_data_opened':False}

class Gate:
    def __init__(self,root,plan_hash):
        self.root=Path(root);self.plan_hash=plan_hash;self.run_id=uuid.uuid4().hex;self.state='claimed';self.out=safe(root,OUTPUT);self.source=safe(root,DATA_FILE);self.expected_indices={};self.row_audit={}
    def assert_claimed(self):
        receipt=json.loads(safe(self.root,PRIMARY).read_text())
        if receipt.get('run_id')!=self.run_id or receipt.get('plan_sha256')!=self.plan_hash:raise PermissionError('Input access without matching pipeline lock')
    def assert_source_unchanged(self):
        self.assert_claimed();st=self.source.stat()
        if (st.st_size,st.st_mtime_ns,st.st_ino)!=self.source_signature:raise ValueError('Input changed during the locked run')
    def event(self,name,**fields):
        self.assert_claimed()
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:json.dump({'time_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'event':name,'state':self.state,**fields},f,ensure_ascii=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    def dataset_hash(self):
        self.assert_claimed();self.event('dataset_checksum_access');return sha(self.source)
    def read_features(self,features):
        self.assert_claimed()
        if self.state!='features':raise PermissionError('Wrong feature-access phase')
        self.assert_source_unchanged()
        cols=list(dict.fromkeys(['origin_time_kst','origin_split','feature_cutoff_time_kst','route_history12h_sector','asos108_wind_u_now','asos108_wind_v_now','asos108_wind_calm_now',*features]))
        if any(c.startswith(('y_pm25_','eligible_h','target_time_','target_split_')) for c in cols):raise ValueError('Target column in prediction request')
        self.event('feature_columns_read',columns=cols)
        d=pq.read_table(self.source,columns=cols,filters=[('origin_split','=','test')],use_threads=False).to_pandas();self.assert_source_unchanged();times=pd.DatetimeIndex(d.origin_time_kst)
        calendar=pd.date_range('2024-01-01','2025-01-01',inclusive='left',freq='3h',tz='Asia/Seoul')
        if str(times.tz)!='Asia/Seoul' or times.tolist()!=calendar.tolist() or not times.is_unique:raise ValueError('Incomplete/duplicate/misaligned origin grid')
        if not d.origin_split.eq('test').all() or not d.feature_cutoff_time_kst.eq(d.origin_time_kst).all():raise ValueError('Feature cutoff or split mismatch')
        if np.isinf(d[features].to_numpy(dtype=float)).any():raise ValueError('Infinite model input')
        return d
    def label_access(self,h):
        self.assert_claimed()
        if self.state!='scoring' or not safe(self.root,SECONDARY).exists():raise PermissionError('Target access before scoring lock')
        if sha(self.out/'frozen_predictions.json')!=self.prediction_manifest_sha:raise ValueError('Prediction freeze changed')
        self.assert_source_unchanged();self.event('label_columns_read',horizon=h)

def scorer_copy(root,gate):
    path=PROJECT/'src/common_scoring.py';spec=importlib.util.spec_from_file_location('_locked_final_'+uuid.uuid4().hex,path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    c.ROOT=Path(root);c.FEATURES=safe(root,DATA_DIR);c.FILE=gate.source;c.PRED=gate.out/'predictions'
    original_sha=c.sha
    def checked_sha(path):
        if Path(path).resolve()==gate.source:gate.assert_claimed()
        return original_sha(path)
    c.sha=checked_sha;original_load=c.load_grid
    def checked_grid(split,h):
        gate.label_access(h);g=original_load(split,h);gate.assert_source_unchanged()
        idx=pd.DatetimeIndex(g.index)
        if not idx.is_unique or not idx.is_monotonic_increasing or not idx.year.astype(int).tolist()==[2024]*len(idx):raise ValueError('Invalid target row grid')
        if not g.target_time_kst.eq(pd.Series(idx,index=g.index)+pd.Timedelta(hours=h)).all():raise ValueError('Target time is not exact origin+h')
        if not np.isfinite(g.y).all():raise ValueError('Eligible target missing/nonfinite')
        expected=g.index[np.isfinite(g.seoul_pm25_now)];gate.expected_indices[h]=expected.tolist();gate.row_audit[h]={'eligible_rows':len(g),'persistence_missing_rows':int((~np.isfinite(g.seoul_pm25_now)).sum()),'common_rows':len(expected)}
        return g
    c.load_grid=checked_grid
    return c

def execute(plan_file,root,*,final=False,expected_sha=None,synthetic=False,fault=None):
    """Only --final plus the reviewed plan digest may enter the dataset-access phase."""
    if not final:raise PermissionError('No final execution approval flag; no input opened')
    if not expected_sha or len(expected_sha)!=64:raise PermissionError('Explicit reviewed plan SHA256 required')
    root=Path(root).resolve();plan_file=Path(plan_file).resolve();plan_file.relative_to(root)
    if sha(plan_file)!=expected_sha:raise ValueError('Plan digest mismatch')
    plan=json.loads(plan_file.read_text());validate_plan(plan,root,synthetic)
    if fault is not None and not synthetic:raise PermissionError('Fault hooks are synthetic-only')
    primary=safe(root,PRIMARY);secondary=safe(root,SECONDARY);out=safe(root,OUTPUT)
    if primary.exists() or secondary.exists() or out.exists():raise FileExistsError('Final run already claimed/output exists; never auto-reset')
    gate=Gate(root,expected_sha);primary.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive claim is made before checksumming or opening the mixed-year input file.
    write_json(primary,{'mode':plan['mode'],'run_id':gate.run_id,'plan_sha256':expected_sha,'status':'claimed','automatic_reset':False})
    def hook(stage):
        if fault is not None:fault(stage,gate)
    try:
        out.mkdir(parents=True,exist_ok=False);gate.event('pipeline_claimed',plan_sha256=expected_sha);hook('after_lock')
        if gate.dataset_hash()!=plan['dataset']['expected_sha256']:raise ValueError('Dataset checksum differs from frozen historical record')
        stat=gate.source.stat();gate.source_signature=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        gate.state='features';union=list(dict.fromkeys(c for model in BASE for hs in map(str,HS) for c in plan['base_models'][model][hs]['features']));data=gate.read_features(union);hook('after_features')
        scoring=scorer_copy(root,gate);prediction_dir=out/'predictions';prediction_dir.mkdir();predictions={};model_details=[]
        gate.state='predicting'
        for model in BASE:
            parts=[]
            for h in HS:
                entry=plan['base_models'][model][str(h)];X=data[entry['features']];per_seed=[]
                for rel in entry['model_files']:
                    booster=lgb.Booster(model_file=str(safe(root,rel)))
                    if booster.feature_name()!=entry['features']:raise ValueError('Saved feature order differs from frozen plan')
                    per_seed.append(np.expm1(booster.predict(X,num_threads=2,num_iteration=-1)))
                raw=np.mean(per_seed,axis=0)
                if not np.isfinite(raw).all():raise ValueError('Nonfinite prediction')
                parts.append(pd.DataFrame({'origin_time_kst':data.origin_time_kst,'horizon':h,'model':model,'split':'test','y_hat':raw}))
            predictions[model]=pd.concat(parts,ignore_index=True)
        context=condition_frame(data[['origin_time_kst','route_history12h_sector','asos108_wind_u_now','asos108_wind_v_now','asos108_wind_calm_now']])
        for model,kind in [('Adaptive_wind','wind'),('Adaptive_route','route')]:
            parts=[]
            for h in HS:
                frame=context.copy()
                for base in ['M1','M3']:frame[base+'_raw']=predictions[base].loc[predictions[base].horizon.eq(h),'y_hat'].to_numpy()
                frozen=json.loads(safe(root,plan['adaptive_weights'][model][str(h)]).read_text());detail=apply_horizon(frame,pd.DataFrame(frozen['season_rules']),pd.DataFrame(frozen['cell_rules']),kind)
                if not detail.weight.between(0,1).all():raise ValueError('Invalid applied weight')
                part=pd.DataFrame({'origin_time_kst':data.origin_time_kst,'horizon':h,'model':model,'split':'test','y_hat':detail.y_hat_raw});parts.append(part)
                detail.insert(0,'origin_time_kst',data.origin_time_kst);detail['model']=model;detail['horizon']=h;model_details.append(detail)
            predictions[model]=pd.concat(parts,ignore_index=True)
        for model in MODELS:
            frame=predictions[model]
            if len(frame)!=11712 or frame.duplicated(['origin_time_kst','horizon']).any() or not np.isfinite(frame.y_hat).all():raise ValueError('Incomplete prediction coverage')
            scoring.export_predictions(frame,prediction_dir/f'{model}.parquet',{'seeds':[0,1,2],'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,'test_only':True,'validation_only':False,'weights_fit_year':2023 if model.startswith('Adaptive_') else None,'plan_sha256':expected_sha,'mode':plan['mode'],'frozen_source_artifacts':plan['adaptive_weights'][model] if model.startswith('Adaptive_') else plan['base_models'][model]})
            loaded=pq.read_table(prediction_dir/f'{model}.parquet').to_pandas();pd.testing.assert_frame_equal(loaded,frame.assign(horizon=frame.horizon.astype('int8')),check_dtype=True)
        pd.concat(model_details,ignore_index=True).to_parquet(out/'adaptive_application_details.parquet',index=False)
        gate.event('predictions_saved',models=8,rows_per_model=11712);hook('after_predictions')
        artifact_hashes=dict(plan['static_artifacts_sha256']);artifact_hashes[relative(root,plan_file)]=expected_sha;artifact_hashes[DATA_FILE]=plan['dataset']['expected_sha256']
        for file in prediction_dir.iterdir():artifact_hashes[relative(root,file)]=sha(file)
        frozen_path=out/'frozen_predictions.json';write_json(frozen_path,{'models':MODELS,'horizons':HS,'no_refit_on_2023':True,'plan_sha256':expected_sha,'policy':policy(),'artifacts_sha256':artifact_hashes,'mode':plan['mode']})
        gate.prediction_manifest_sha=sha(frozen_path);gate.state='predictions_frozen';gate.event('predictions_frozen',sha256=gate.prediction_manifest_sha);hook('before_scoring')
        if sha(plan_file)!=expected_sha:raise ValueError('Plan changed during inference')
        gate.state='scoring';gate.event('scoring_requested')
        score_dir,scores,tests=scoring.evaluate('test',MODELS,HS,pairs=[f'{a}:{b}' for a,b in COMPARISONS if b!='persistence'],final=True,frozen=frozen_path,output_root=out/'common_scores',reps=1000,seed=0,prediction_dir=prediction_dir)
        hook('after_scoring')
        saved=pd.read_parquet(score_dir/'score_rows_test.parquet')
        if len(tests)!=144 or len(scores)!=36 or not np.allclose(tests.p_holm,scoring.holm(tests.p),equal_nan=True):raise ValueError('Wrong Holm family/results')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):raise ValueError('Wrong improvement rule')
        saved_policy=json.loads((score_dir/'policy.meta.json').read_text())
        if saved_policy['holm_family']!=policy()['holm_family'] or saved_policy['bootstrap_reps']!=1000 or saved_policy['seed']!=0:raise ValueError('Scorer policy changed')
        for h,g in saved.groupby('horizon'):
            if g.origin_time_kst.tolist()!=gate.expected_indices[h]:raise ValueError('Unexpected common-row exclusion')
            for name in ['persistence',*MODELS]:
                observed=scores[(scores.model==name)&scores.horizon.eq(h)].iloc[0];actual=scoring.metrics(g.y.to_numpy(),g[name].to_numpy(),g.persistence.to_numpy())
                if not np.allclose([observed[k] for k in ['RMSE','MAE','MBE']],[actual[k] for k in ['RMSE','MAE','MBE']]):raise ValueError('Metric recomputation failed')
        annotated=tests.copy();annotated['core_hypothesis']=[(a,b) in CORE for a,b in zip(tests.model,tests.versus)];annotated['primary_horizon']=annotated.horizon.eq(6);annotated.to_csv(out/'hypothesis_comparisons_test.csv',index=False)
        if int(annotated.core_hypothesis.sum())!=80:raise ValueError('Wrong core display subset')
        for ledger in [plan['preserve_sha256'],plan['static_artifacts_sha256']]:
            for rel,digest in ledger.items():
                if sha(safe(root,rel))!=digest:raise ValueError('Original artifact changed: '+rel)
        if gate.dataset_hash()!=plan['dataset']['expected_sha256']:raise ValueError('Dataset changed during final run')
        files={relative(root,p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='events.jsonl'}
        gate.state='completed';gate.event('completed',common_rows=gate.row_audit)
        result={'status':'completed','mode':plan['mode'],'run_id':gate.run_id,'plan_sha256':expected_sha,'prediction_manifest_sha256':gate.prediction_manifest_sha,'models':MODELS,'horizons':HS,'holm_tests':144,'core_display_tests':80,'common_scores':relative(root,score_dir),'row_audit':gate.row_audit,'base_model_refit':False,'adaptive_refit':False,'originals_preserved':len(plan['preserve_sha256']),'artifacts_sha256':files,'event_log_sha256':sha(out/'events.jsonl')}
        write_json(out/'manifest.json',result);return result
    except BaseException as exc:
        # Never delete either lock or overwrite partial predictions after a failure.
        gate.state='failed'
        try:
            gate.event('failed',error_type=type(exc).__name__,error=str(exc));write_json(out/'failure.json',{'status':'failed','mode':plan['mode'],'run_id':gate.run_id,'plan_sha256':expected_sha,'error_type':type(exc).__name__,'error':str(exc),'locks_retained':True,'automatic_retry':False})
        except BaseException:pass
        raise
