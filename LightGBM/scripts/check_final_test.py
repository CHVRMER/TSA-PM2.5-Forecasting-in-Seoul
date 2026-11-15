"""All datasets/labels here are fabricated. Real production entry/locks are never invoked."""
from pathlib import Path
import os,sys,json,shutil,tempfile,subprocess,threading
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/final_test_runner_validation/.matplotlib_cache'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from unittest.mock import patch
from src import final_test as f
from src.adaptive import fit_horizon
from scripts.freeze_final_test import make_plan,EXTRA
REPORT=ROOT/'results/final_test_runner_validation'

def build_fixture(root):
    root=Path(root);rng=np.random.default_rng(813)
    for rel in set([*f.CODE,*EXTRA]):
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        # Existing code is static; fake metadata/configuration is written below.
        if (ROOT/rel).exists():shutil.copyfile(ROOT/rel,dest)
    features=['seoul_pm25_now','synthetic_signal'];n=2928;index=pd.date_range('2024-01-01',periods=n,freq='3h',tz='Asia/Seoul');hour=np.arange(n)*3
    truth=lambda h:20+3*np.sin(h/13)+.7*np.cos(h/2)
    data=pd.DataFrame({'origin_time_kst':index,'origin_split':'test','feature_cutoff_time_kst':index,'route_history12h_sector':np.array(['NW','N','E','LOCAL'])[np.arange(n)%4],'asos108_wind_u_now':np.cos(hour/3),'asos108_wind_v_now':np.sin(hour/3),'asos108_wind_calm_now':(np.arange(n)%20==0).astype(float),'seoul_pm25_now':truth(hour),'synthetic_signal':np.sin(hour/5)})
    data.loc[30,'seoul_pm25_now']=np.nan;data.loc[50,'synthetic_signal']=np.nan;data.loc[data.index%100==10,['asos108_wind_u_now','asos108_wind_v_now']]=np.nan;data.loc[data.index%91==1,'route_history12h_sector']='UNDEFINED'
    for h in f.HS:
        target=index+pd.Timedelta(hours=h);eligible=target.year==2024;eligible=np.array(eligible);eligible[70]=False
        data[f'target_time_h{h}']=target;data[f'eligible_h{h}']=eligible;data[f'y_pm25_h{h}']=np.where(eligible,truth(hour+h),np.nan)
    source=root/f.DATA_FILE;source.parent.mkdir(parents=True,exist_ok=True);data.to_parquet(source,index=False)
    (root/f.DATA_DIR/'feature_manifest.json').write_text(json.dumps({'version':'SYNTHETIC_ONLY','created_kst':'SYNTHETIC_NO_REAL_DATA','rows':n,'columns':len(data.columns)}))
    pd.DataFrame([{'name':c,'role':'feature' if c in features else 'label'} for c in data.columns]).to_csv(root/f.DATA_DIR/'feature_dictionary.csv',index=False)
    (root/f.DATA_DIR/'feature_sets.json').write_text(json.dumps({'sets':{'synthetic_features':features}}));(root/'configs/feature_groups.json').write_text('{}')
    X=pd.DataFrame({'seoul_pm25_now':rng.uniform(15,25,80),'synthetic_signal':rng.uniform(-1,1,80)})
    for j,model in enumerate(f.BASE):
        for h in f.HS:
            folder=root/f'Models/{model}/h{h}';folder.mkdir(parents=True,exist_ok=True)
            cfg={'model':model,'horizon':h,'feature_set':'synthetic_features','expected_features':2,'seeds':[0,1,2],'excluded_features':[]};(root/f'configs/{model.lower()}_{h}h.json').write_text(json.dumps(cfg))
            (folder/'ensemble_manifest.json').write_text(json.dumps({'model':model,'horizon':h,'seeds':[0,1,2],'feature_names':features}))
            for seed in [0,1,2]:
                y=(-.2+.01*X.synthetic_signal if model=='M1' else 3+.08*X.synthetic_signal+.025*j+.002*h)+.025*seed
                booster=lgb.train({'objective':'regression','verbosity':-1,'num_threads':2,'num_leaves':3,'min_data_in_leaf':3,'seed':seed,'deterministic':True,'force_col_wise':True},lgb.Dataset(X,label=y),num_boost_round=4);booster.save_model(str(folder/f'{model}_h{h}_seed{seed}.txt'))
    # Frozen toy weights are fitted on synthetic2023 only, then cannot be refitted by execute().
    wind=(['west']*300+['north']*200+['east']*180+['south']*90+['calm']*30)*2;route=(['west']*500+['north']*200+['east']*90+['local']*10)*2
    fit=pd.DataFrame({'origin_time_kst':pd.date_range('2023-01-01',periods=1600,freq='3h',tz='Asia/Seoul'),'season':['cold']*800+['warm']*800,'wind_sector':wind,'route_group':route,'M1_raw':-1.,'M3_raw':[-1.]*800+[2.]*800,'y':[-1.]*800+[.2]*800})
    for h in f.HS:
        seasons,cells=fit_horizon(fit,h,1e-4)
        for name,kind in [('Adaptive_wind','wind'),('Adaptive_route','route')]:
            folder=root/f'Models/{name}/h{h}';folder.mkdir(parents=True,exist_ok=True);record={'model':name,'horizon':h,'fitted_year':2023,'season_rules':seasons.astype(object).where(pd.notna(seasons),None).to_dict('records'),'cell_rules':cells[cells.kind==kind].astype(object).where(pd.notna(cells[cells.kind==kind]),None).to_dict('records')};(folder/'weights.json').write_text(json.dumps(record,allow_nan=False));(folder/'config.json').write_text(json.dumps({'synthetic':True}))
    (root/'SYNTHETIC_ONLY.txt').write_text('All observations/targets/models/weights in this fixture are artificial. Calendar2024 is used only to test leap-year alignment. No real Test data used.\n')
    (root/'baseline_preserve.txt').write_text('must remain unchanged')

def plan_for(root,mutate=None):
    if mutate is not None:
        p=root/f.DATA_FILE;d=pd.read_parquet(p);d=mutate(d);d.to_parquet(p,index=False)
    plan=make_plan(root,'synthetic',f.sha(root/f.DATA_FILE),{'baseline_preserve.txt':f.sha(root/'baseline_preserve.txt')});p=root/'plan.json';f.write_json(p,plan);return p,f.sha(p)

class AccessTrace:
    def __init__(self,root):self.root=Path(root).resolve();self.source=self.root/f.DATA_FILE;self.calls=[];self.original=pq.read_table
    def __call__(self,path,*args,**kw):
        p=Path(path).resolve()
        if p.is_relative_to(ROOT/'Dataset'):raise AssertionError('REAL DATA ACCESS FORBIDDEN')
        if not p.is_relative_to(self.root):raise AssertionError('Dataset read escaped synthetic root')
        if p==self.source:
            assert (self.root/f.PRIMARY).exists();columns=kw.get('columns');assert columns is not None,'Full raw table decode is forbidden'
            label=any(c.startswith(('y_pm25_','eligible_h','target_time_')) for c in columns)
            if label:assert (self.root/f.SECONDARY).exists() and (self.root/f.OUTPUT/'frozen_predictions.json').exists()
            self.calls.append({'columns':list(columns),'labels':label,'primary_lock':True,'secondary_lock':(self.root/f.SECONDARY).exists(),'prediction_freeze':(self.root/f.OUTPUT/'frozen_predictions.json').exists()})
        return self.original(path,*args,**kw)

def run_checked(root,plan,digest,**kwargs):
    trace=AccessTrace(root)
    with patch.object(pq,'read_table',side_effect=trace),patch('src.adaptive.fit_horizon',side_effect=AssertionError('No adaptive fitting during final run')),patch('src.adaptive.fit_weight',side_effect=AssertionError('No weight estimation during final run')),patch.object(lgb,'train',side_effect=AssertionError('No training during final run')):
        result=f.execute(plan,root,expected_sha=digest,synthetic=True,**kwargs)
    return result,trace

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    receipt=REPORT/'verification.json'
    if receipt.exists():
        if (ROOT/f.PRIMARY).exists() or (ROOT/f.SECONDARY).exists():raise FileExistsError('Never revalidate during/after a real final run')
        receipt.rename(REPORT/f'verification_attempt_{f.sha(receipt)[:12]}.json')
    production_locks_before={p:(ROOT/p).exists() for p in [f.PRIMARY,f.SECONDARY]};assert not any(production_locks_before.values())
    checks=[]
    with tempfile.TemporaryDirectory(prefix='final_test_synthetic_') as tmp:
        tmp=Path(tmp);template=tmp/'template';template.mkdir();build_fixture(template)
        def case(name):
            root=tmp/name;shutil.copytree(template,root);return root
        # Entire success path is retained as explicitlysynthetic evidence, outside production output.
        import uuid
        success=REPORT/'runs'/uuid.uuid4().hex/'synthetic_success';success.parent.mkdir(parents=True,exist_ok=False);shutil.copytree(template,success);plan,digest=plan_for(success)
        result,trace=run_checked(success,plan,digest,final=True)
        assert result['status']=='completed' and result['mode']=='synthetic' and len([x for x in trace.calls if x['labels']])==4
        assert all(not x['labels'] for x in trace.calls[:1]);scores=pd.read_csv(success/result['common_scores']/'score_metrics_test.csv');tests=pd.read_csv(success/result['common_scores']/'score_tests_test.csv');assert len(tests)==144 and scores[scores.model.eq('M1')].negative_predictions_clipped.gt(0).all()
        details=pd.read_parquet(success/f.OUTPUT/'adaptive_application_details.parquet');assert details.fallback_reason.eq('unknown_condition').any() and details.season_fallback_used.any() and details.near_zero_exception_used.any()
        m1=pd.read_parquet(success/f.OUTPUT/'predictions/M1.parquet');m3=pd.read_parquet(success/f.OUTPUT/'predictions/M3.parquet')
        for model in ['Adaptive_wind','Adaptive_route']:
            actual=pd.read_parquet(success/f.OUTPUT/f'predictions/{model}.parquet');dd=details[details.model.eq(model)]
            assert np.allclose(actual.y_hat,(1-dd.weight.to_numpy())*m1.y_hat+dd.weight.to_numpy()*m3.y_hat)
        x=pd.read_parquet(success/f.DATA_FILE).iloc[:10][['seoul_pm25_now','synthetic_signal']];raw=[lgb.Booster(model_file=str(success/f'Models/M3/h6/M3_h6_seed{s}.txt')).predict(x,num_threads=2) for s in [0,1,2]];wanted=np.mean(np.expm1(raw),axis=0);wrong=np.expm1(np.mean(raw,axis=0));assert np.max(abs(wanted-wrong))>1e-5;assert np.allclose(wanted,m3[m3.horizon.eq(6)].y_hat.iloc[:10])
        checks.append({'case':'full_success_8models_4h_144tests_1000bootstrap','passed':True,'rows_per_prediction':11712,'label_read_trace':trace.calls,'raw_mixture_and_negative_clipping_verified':True,'unknown_small_cell_near_zero_fallbacks_verified':True})
        try:f.execute(plan,success,final=True,expected_sha=digest,synthetic=True)
        except FileExistsError:checks.append({'case':'repeat_execution_blocked','passed':True})
        else:raise AssertionError('Second final run accepted')
        # Pre-access guards must not create locks or decode any data.
        for name,change in [('no_final',None),('no_plan_digest',None),('wrong_plan_digest',None),('changed_pairs',lambda p:p['policy']['comparisons'].pop()),('changed_horizons',lambda p:p['policy'].update(horizons=[6])),('changed_bootstrap',lambda p:p['policy'].update(bootstrap_reps=10)),('changed_seed',lambda p:p['policy'].update(bootstrap_seed=9)),('unsafe_path',lambda p:p['base_models']['M1']['3'].update(config='../outside.json')),('target_in_allowlist',lambda p:p['base_models']['M1']['3']['features'].append('y_pm25_h3')),('wrong_seed_horizon',lambda p:p['base_models']['M1']['3'].update(model_files=p['base_models']['M1']['6']['model_files'])),('changed_family_order',lambda p:p['policy']['holm_family'].reverse()),('reordered_model_features',lambda p:p['base_models']['M1']['3']['features'].reverse())]:
            root=case(name);p,d=plan_for(root)
            if change:
                obj=json.loads(p.read_text());change(obj);p.write_text(json.dumps(obj));d=f.sha(p)
            if name=='wrong_plan_digest':d='0'*64
            if name=='no_plan_digest':d=None
            trace=AccessTrace(root)
            with patch.object(pq,'read_table',side_effect=trace):
                try:f.execute(p,root,final=name!='no_final',expected_sha=d,synthetic=True)
                except (PermissionError,ValueError,FileNotFoundError):pass
                else:raise AssertionError(name+' not blocked')
            assert not trace.calls and not (root/f.PRIMARY).exists();checks.append({'case':name,'passed':True,'source_reads':0})
        # Static artifact mutation cannot reach data.
        root=case('weight_mutation');p,d=plan_for(root);w=root/'Models/Adaptive_wind/h3/weights.json';w.write_text(w.read_text()+'\n')
        try:run_checked(root,p,d,final=True)
        except ValueError:pass
        else:raise AssertionError('Changed weights accepted')
        assert not (root/f.PRIMARY).exists();checks.append({'case':'changed_frozen_weight_blocked_before_source','passed':True})
        # Exceptions after a claim preserve locks and partial outputs, never retry.
        for stage in ['after_lock','after_features','after_predictions','before_scoring','after_scoring']:
            root=case('fault_'+stage);p,d=plan_for(root)
            def fail(here,gate):
                if here==stage:raise RuntimeError('injected_'+stage)
            try:run_checked(root,p,d,final=True,fault=fail)
            except RuntimeError:pass
            else:raise AssertionError('Injected failure not retained')
            assert (root/f.PRIMARY).exists() and (root/f.OUTPUT/'failure.json').exists() and (root/f.SECONDARY).exists()==(stage=='after_scoring')
            try:f.execute(p,root,final=True,expected_sha=d,synthetic=True)
            except FileExistsError:pass
            else:raise AssertionError('Failed run retried')
            checks.append({'case':'failure_'+stage,'passed':True,'primary_lock_retained':True,'automatic_retry_blocked':True})
        root=case('secondary_already_consumed');p,d=plan_for(root);(root/f.SECONDARY).parent.mkdir(parents=True,exist_ok=True);(root/f.SECONDARY).write_text('{}')
        try:run_checked(root,p,d,final=True)
        except FileExistsError:pass
        else:raise AssertionError('Existing scoring lock ignored')
        assert not (root/f.PRIMARY).exists();checks.append({'case':'existing_scoring_lock_blocks_pipeline','passed':True})
        root=case('dataset_checksum_mismatch');p,d=plan_for(root);source=root/f.DATA_FILE;source.write_bytes(source.read_bytes()+b'changed');trace=AccessTrace(root)
        with patch.object(pq,'read_table',side_effect=trace):
            try:f.execute(p,root,final=True,expected_sha=d,synthetic=True)
            except ValueError:pass
            else:raise AssertionError('Changed data accepted')
        assert (root/f.PRIMARY).exists() and not trace.calls;checks.append({'case':'changed_source_hash_blocks_column_decode','passed':True})
        root=case('keyboard_interrupt');p,d=plan_for(root)
        def interrupt(stage,gate):
            if stage=='after_lock':raise KeyboardInterrupt('synthetic interrupt')
        try:run_checked(root,p,d,final=True,fault=interrupt)
        except KeyboardInterrupt:pass
        else:raise AssertionError('Interrupt was not propagated')
        assert (root/f.PRIMARY).exists() and json.loads((root/f.OUTPUT/'failure.json').read_text())['error_type']=='KeyboardInterrupt';checks.append({'case':'interrupt_retains_lock_and_failure_record','passed':True})
        # Bad feature tables fail before labels, after the primary lock.
        mutations={
            'duplicate_origin':lambda d:d.assign(origin_time_kst=d.origin_time_kst.mask(d.index==1,d.origin_time_kst.iloc[0])),
            'missing_origin':lambda d:d.iloc[1:].reset_index(drop=True),
            'future_cutoff':lambda d:d.assign(feature_cutoff_time_kst=d.feature_cutoff_time_kst+pd.Timedelta(hours=1)),
            'infinite_input':lambda d:d.assign(synthetic_signal=d.synthetic_signal.mask(d.index==20,np.inf)),
        }
        for name,mutate in mutations.items():
            root=case(name);p,d=plan_for(root,mutate);trace=AccessTrace(root)
            with patch.object(pq,'read_table',side_effect=trace):
                try:f.execute(p,root,final=True,expected_sha=d,synthetic=True)
                except ValueError:pass
                else:raise AssertionError(name+' accepted')
            assert (root/f.PRIMARY).exists() and not any(x['labels'] for x in trace.calls);assert not (root/f.SECONDARY).exists();checks.append({'case':name,'passed':True,'no_label_decode':True,'lock_retained':True})
        # A target alignment defect is detectable only after predictions are frozen and scoring locked.
        def wrong_target(d):
            d.loc[0,'target_time_h6']+=pd.Timedelta(hours=1);return d
        root=case('wrong_target_time');p,d=plan_for(root,wrong_target);trace=AccessTrace(root)
        with patch.object(pq,'read_table',side_effect=trace):
            try:f.execute(p,root,final=True,expected_sha=d,synthetic=True)
            except ValueError as e:assert 'exact origin+h' in str(e)
            else:raise AssertionError('Bad target alignment accepted')
        assert (root/f.PRIMARY).exists() and (root/f.SECONDARY).exists() and (root/f.OUTPUT/'failure.json').exists();assert all(x['prediction_freeze'] and x['secondary_lock'] for x in trace.calls if x['labels']);checks.append({'case':'target_alignment_failure_after_both_locks','passed':True})
        # Tamper with saved predictions after their freeze; no target may be decoded.
        root=case('prediction_tamper');p,d=plan_for(root)
        def tamper(stage,gate):
            if stage=='before_scoring':
                x=gate.out/'predictions/M1.parquet';x.write_bytes(x.read_bytes()+b'changed')
        trace=AccessTrace(root)
        with patch.object(pq,'read_table',side_effect=trace):
            try:f.execute(p,root,final=True,expected_sha=d,synthetic=True,fault=tamper)
            except ValueError:pass
            else:raise AssertionError('Tampered prediction accepted')
        assert not any(x['labels'] for x in trace.calls) and (root/f.PRIMARY).exists();checks.append({'case':'prediction_freeze_tamper_blocks_labels','passed':True})
        # Simultaneous attempts: exactly one exclusive claim, even before any data read.
        root=case('concurrent_claim');p,d=plan_for(root);claimed=threading.Event();release=threading.Event()
        def pause(stage,gate):
            if stage=='after_lock':claimed.set();assert release.wait(timeout=30);raise RuntimeError('stop_after_concurrent_claim_check')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(f.execute,p,root,final=True,expected_sha=d,synthetic=True,fault=pause);assert claimed.wait(timeout=30)
            try:f.execute(p,root,final=True,expected_sha=d,synthetic=True)
            except FileExistsError:pass
            else:raise AssertionError('Concurrent claim accepted')
            finally:release.set()
            try:future.result(timeout=30)
            except RuntimeError:pass
        assert (root/f.PRIMARY).exists() and not (root/f.SECONDARY).exists();checks.append({'case':'concurrent_single_claim','passed':True})
        # CLI without approval and CLI misuse of a synthetic plan must fail without real locks.
        for args in [[],['--final','--plan',str(plan),'--plan-sha256',digest]]:
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_final_test.py'),*args],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});assert run.returncode!=0
        checks.append({'case':'CLI_missing_approval_and_synthetic_plan_rejected','passed':True})
        try:f.execute(plan,ROOT,final=True,expected_sha=digest,synthetic=True)
        except PermissionError:pass
        else:raise AssertionError('Synthetic execution allowed real root')
        checks.append({'case':'synthetic_cannot_use_real_project_root','passed':True})
    # Load all72 actual frozen models on fabricated rows (not real2024 data); never score them.
    from scripts.freeze_final_test import registry
    base,adaptive,before=registry(ROOT);assert len(before)==147 and f.map_sha(before)==f.ANCHOR
    loaded=0
    for model in f.BASE:
        for h in f.HS:
            entry=base[model][str(h)];X=pd.DataFrame(np.zeros((3,len(entry['features']))),columns=entry['features']);X.iloc[1,:]=np.nan;X.iloc[2,:]=1.
            for rel in entry['model_files']:
                booster=lgb.Booster(model_file=str(ROOT/rel));assert booster.feature_name()==entry['features'];y=np.expm1(booster.predict(X,num_threads=2));assert np.isfinite(y).all();loaded+=1
    assert loaded==72;assert all(f.sha(ROOT/rel)==digest for rel,digest in before.items());checks.append({'case':'actual72_frozen_models_on_fabricated_0_NaN_1_inputs','passed':True,'real_observations_used':False,'models_modified':False})
    previous=json.loads((ROOT/'results/validation_2023/Adaptive/preflight.json').read_text())['protected_sha256'];preserved=0
    for rel,digest in previous.items():
        if rel.startswith(('Models/work/','Models/predictions/')):continue
        assert f.sha(ROOT/rel)==digest,rel;preserved+=1
    assert {p:(ROOT/p).exists() for p in production_locks_before}==production_locks_before;assert not (ROOT/f.OUTPUT).exists()
    result={'passed':True,'mode':'synthetic_validation_only','real_2024_data_used':False,'code_sha256':{rel:f.sha(ROOT/rel) for rel in f.CODE},'versions':f.versions(),'cases':checks,'case_count':len(checks),'production_locks_created':False,'production_execution_output_created':False,'actual_frozen_models_verified_with_artificial_inputs':72,'model_config_composite_sha256':f.map_sha(before),'old_protected_files_preserved':preserved,'synthetic_success_root':str(success.relative_to(ROOT)),'full_success_receipt':result}
    f.write_json(receipt,result);print(json.dumps({'passed':True,'case_count':len(checks),'real_2024_data_used':False,'production_locks_created':False,'receipt':str(receipt)},ensure_ascii=False))
if __name__=='__main__':main()
