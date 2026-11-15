"""Synthetic scoring-contract regression; no model training or real 2024 scoring."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import regional_evaluation as evaluation

PROJECT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
COMMON=PROJECT/'Model/Common'
sys.path.insert(0,str(COMMON))
from neural_evaluation import write_neural_scoring_plan

BASE=Path.cwd()/'tmp/regional_evaluation_tests'
REPORT=Path(__file__).resolve().parents[1]/'Results/Verification/regional_evaluation_verification.json'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def fixture(root,full_config=None):
    keys=['KR:111121','KR:domestic','CN:a','CN:b']
    conditions={'DN':{'station_keys':['KR:domestic']},'F':{'station_keys':keys[1:]},
        'ADD_CN_A':{'station_keys':['KR:domestic','CN:a']},
        'DROP_CN_A':{'station_keys':['KR:domestic','CN:b']}}
    config={'experiments':{'conditions':conditions},'regions':{'regions':{
        'KR':{'country':'KR','station_keys':['KR:domestic']},'CN_A':{'country':'CN','station_keys':['CN:a']},
        'CN_B':{'country':'CN','station_keys':['CN:b']}}},
        'comparisons':{'comparisons':[
            {'comparison_id':'FOREIGN_ALL','kind':'foreign_all','region_id':'ALL_FOREIGN','candidate':'F','reference':'DN'},
            {'comparison_id':'ADD_CN_A','kind':'add','region_id':'CN_A','candidate':'ADD_CN_A','reference':'DN'},
            {'comparison_id':'DROP_CN_A','kind':'drop','region_id':'CN_A','candidate':'F','reference':'DROP_CN_A'}]}}
    if full_config is not None:
        config=copy.deepcopy(full_config);conditions=config['experiments']['conditions']
        keys=['KR:111121']+list(conditions['F']['station_keys'])
    graph=np.roll(np.eye(len(keys),dtype=np.float32),1,axis=1)
    times=pd.date_range('2023-03-01',periods=90,freq='3h',tz='Asia/Seoul')
    parts=[]
    for horizon in (3,6,12,24):
        truth=20+np.sin(np.arange(90)/5)*5+horizon/10
        eligible=np.ones(90,dtype=bool);eligible[30:45]=False
        parts.append(pd.DataFrame({'origin_time_kst':times,'target_time_kst':times+pd.Timedelta(hours=horizon),
            'horizon':horizon,'split':'validation','y_true':np.where(eligible,truth,np.nan),'eligible':eligible,
            'persistence_y_hat':truth+4,'seasonal_y_hat':truth+5,'persistence_fallback':False,
            'seasonal_fallback':False,'season':'spring'}))
    grid=pd.concat(parts,ignore_index=True)
    grid_path=root/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
    grid_path.parent.mkdir(parents=True);grid.to_parquet(grid_path,index=False)
    records=[]
    families=['DCRNN','GraphWaveNet']
    for family in families:
        for condition,spec in conditions.items():
            outer=root/'sources'/family/condition;modelroot=outer/'Training'/'original_run_id'
            modelroot.mkdir(parents=True)
            pd.DataFrame({'station_key':keys,'node_index':range(len(keys))}).to_csv(modelroot/'node_order.csv',index=False)
            (modelroot/'Graphs').mkdir();np.save(modelroot/'Graphs/trajectory_directed.npy',graph)
            selected=set(spec['station_keys'])
            preprocess={'fit_start_inclusive':'2021-01-01T00:00:00+09:00','fit_end_exclusive':'2023-01-01T00:00:00+09:00',
                'node_order':keys,'context_names':['asos108_wind_u_now'],'pm_mean':[2.]+[1.]*(len(keys)-1),
                'pm_std':[.5]+[1.]*(len(keys)-1),'pm_valid_count':[100]+[100 if k in selected else 0 for k in keys[1:]],
                'context_mean':[0.],'context_std':[1.],'context_valid_count':[100]}
            write(modelroot/'preprocessing.json',preprocess)
            mask=np.asarray([k=='KR:111121' or k in selected for k in keys],dtype=bool)
            contract={'universe_id':config['experiments'].get('universe_id','SYNTHETIC_FIXED4'),'node_order_sha256':evaluation.node_order_sha256(keys),
                'graph_sha256':{'trajectory_directed':evaluation.array_sha256(graph)},
                'target_observations_sha256':'fixed_target','context_values_sha256':'fixed_context',
                'context_names_sha256':'fixed_context_names','origin_grid_sha256':'fixed_origins',
                'y_raw_sha256':'fixed_y','eligible_sha256':'fixed_eligible','origin_positions_sha256':'fixed_positions',
                'source_universe_signature':'fixed_universe','config_sha256':evaluation.canonical_sha256(config),
                'target_context_preprocessing_sha256':evaluation.target_context_preprocessing_sha256(preprocess),
                'condition_id':condition,'allowed_station_keys':spec['station_keys'],
                'excluded_station_keys':[k for k in keys[1:] if k not in selected],
                'observation_mask_sha256':evaluation.array_sha256(mask),
                'observation_mask_applied_before_fit':True,'excluded_values_are_normalized_zero_and_missing_one':True,
                'fixed_node_order_and_graphs':True,'target_context_labels_preserved':True,'latent_relay_nodes_retained':True}
            write(outer/'regional_contract.json',contract)
            model=family+'_trajectory_directed'
            metadata={'target_station_key':'KR:111121','horizons':[3,6,12,24],'lookback_hours':48,
                'target_transform':'log1p + train-only standardization','data_signature':'masked_data_'+condition,
                'preprocessing_signature':evaluation.canonical_sha256(preprocess),'prediction_unit':'ug/m3',
                'train_split':'2021-2022','validation_split':'2023',
                'training_configuration':{'input_condition':condition,'data_signature':'masked_data_'+condition,
                    'learning_rate':.001,'seeds':[11,22],'model_family':family}}
            write_neural_scoring_plan(modelroot,family,condition,[{'model':model,'graph':'trajectory_directed','hidden_dim':4}],
                [11,22],metadata,pair_specs=[])
            write(modelroot/'completion.json',{'status':'passed','mode':'full','test_scored':False})
            write(modelroot/'execution_status.json',{'status':'passed'})
            frames=[]
            for seed in [11,22]:
                frame=grid[['origin_time_kst','target_time_kst','horizon','split','y_true','eligible']].copy()
                value=3. if condition=='DN' else .8 if condition=='F' else 2. if condition.startswith('ADD_') else 1.5
                error=value+(-2. if seed==11 else 2.)+.15*np.sin(np.arange(len(frame))/7)
                # Finite forecasts also exist on unavailable truth rows.
                truth=20+np.sin(np.tile(np.arange(90),4)/5)*5+frame.horizon.to_numpy()/10
                frame['y_hat']=truth+error
                frame['model']=model;frame['input_condition']=condition;frame['prediction_kind']='seed';frame['seed']=seed
                frame['data_version']='masked_data_'+condition;frame['experiment_mode']='full';frame['common_export_unclipped']=True
                frames.append(frame)
            export=modelroot/'Exports/validation_predictions_by_seed.parquet';export.parent.mkdir()
            pd.concat(frames,ignore_index=True).to_parquet(export,index=False)
            write(export.with_suffix('.meta.json'),{'synthetic':True,'source_export_unclipped':True})
            records.append({'condition':condition,'model_family':family,'run_root':str(modelroot),
                'regional_contract_path':str(outer/'regional_contract.json')})
    return config,records,grid


class RegionalScoringTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=BASE)
        self.root=Path(self.temp.name)
        self.config,self.records,self.grid=fixture(self.root)

    def tearDown(self):self.temp.cleanup()

    def score(self,**kwargs):
        options=dict(project_root=self.root,completed_runs=self.records,experiment_config=self.config,
            output_root=self.root/'Result/Common_Scoring/Regional/synthetic_batch',common_dir=COMMON,
            bootstrap_reps=20,make_plots=False)
        options.update(kwargs)
        return evaluation.score_regional_runs(**options)

    def modify_json(self,path,function):
        value=json.loads(Path(path).read_text(encoding='utf-8'));function(value);write(path,value)

    def test_complete_flow_schema_gain_sign_ci_and_plots(self):
        before=hashes(self.root/'sources')
        result=self.score(make_plots=True)
        directory=Path(result['output_dir']);additional=directory/'Additional_Analysis'
        self.assertEqual(result['status'],'passed');self.assertEqual(result['job_count'],8)
        self.assertEqual(result['regional_test_rows'],3*2*3*4)
        core={'metrics.csv','seed_metric_summary.csv','tests.csv','conditions.csv','score_rows.parquet','scoring_audit.json'}
        self.assertEqual({p.name for p in directory.iterdir()},core|{'Additional_Analysis'})
        scorer,_,_=evaluation._load_common(self.root,COMMON)
        frames=[pd.read_parquet(Path(r['run_root'])/'Exports/validation_predictions_by_seed.parquet') for r in self.records]
        plan=json.loads((additional/'regional_comparison_plan.json').read_text())
        expected=scorer.score_frames(self.grid,pd.concat(frames,ignore_index=True),pairs=plan['resolved_run_id_pairs'],bootstrap_reps=20)
        for name in ('metrics','seed_metric_summary','tests','conditions'):
            actual=pd.read_csv(directory/(name+'.csv'))
            tmp=self.root/('expected_'+name+'.csv');expected[name].to_csv(tmp,index=False)
            pd.testing.assert_frame_equal(actual,pd.read_csv(tmp))
        expected_path=self.root/'reference.parquet';expected['score_rows'].to_parquet(expected_path,index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(directory/'score_rows.parquet'),pd.read_parquet(expected_path))
        self.assertEqual(pq.read_schema(directory/'score_rows.parquet'),pq.read_schema(expected_path))
        self.assertEqual(pq.ParquetFile(directory/'score_rows.parquet').metadata.row_group(0).column(0).compression,
                         pq.ParquetFile(expected_path).metadata.row_group(0).column(0).compression)
        audit=json.loads((directory/'scoring_audit.json').read_text())
        self.assertEqual(list(audit),list(expected['audit'])+['written_kst','grid_path','grid_sha256','prediction_sources'])
        gains=pd.read_csv(additional/'regional_gains.csv');tests=pd.read_csv(directory/'tests.csv')
        self.assertEqual(len(gains),72)
        for row in gains.itertuples():
            test=tests.loc[tests.run_id.eq(row.candidate_run_id)&tests.versus_run_id.eq(row.reference_run_id)&tests.horizon.eq(row.horizon)].iloc[0]
            self.assertAlmostEqual(row.RMSE_gain,-test.rmse_diff,places=12)
            self.assertAlmostEqual(row.gain_ci95_low,-test.ci95_high,places=12)
            self.assertAlmostEqual(row.gain_ci95_high,-test.ci95_low,places=12)
            self.assertAlmostEqual(row.p_holm,test.p_holm,places=12)
        drop=gains.loc[gains.kind.eq('drop')].iloc[0]
        self.assertAlmostEqual(drop.RMSE_gain_pct,100*drop.RMSE_gain/drop.candidate_RMSE)
        metrics=pd.read_csv(directory/'metrics.csv');summary=pd.read_csv(directory/'seed_metric_summary.csv')
        ensemble=metrics.loc[metrics.model.eq('DCRNN_trajectory_directed')&metrics.input_condition.eq('F')&metrics.prediction_kind.eq('ensemble')&metrics.horizon.eq(6)].iloc[0]
        seed_mean=summary.loc[summary.model.eq('DCRNN_trajectory_directed')&summary.input_condition.eq('F')&summary.horizon.eq(6)].iloc[0]
        self.assertNotAlmostEqual(ensemble.RMSE,seed_mean.RMSE_mean,places=3)
        self.assertEqual(len(list((additional/'Figures').glob('*.png'))),4)
        self.assertEqual(before,hashes(self.root/'sources'))

    def test_repeat_creates_new_directory_preserves_previous(self):
        first=Path(self.score()['output_dir']);before=hashes(first)
        second=Path(self.score()['output_dir'])
        self.assertNotEqual(first,second);self.assertEqual(first.parent,second.parent)
        self.assertEqual(before,hashes(first))

    def test_missing_completed_job_rejected(self):
        with self.assertRaisesRegex(ValueError,'all predeclared jobs'):self.score(completed_runs=self.records[:-1])
        self.assertFalse((self.root/'Result').exists())

    def test_duplicate_job_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.score(completed_runs=self.records+[self.records[0]])

    def test_graph_tampering_rejected(self):
        path=Path(self.records[0]['run_root'])/'Graphs/trajectory_directed.npy';a=np.load(path);a[0,0]=1;np.save(path,a)
        with self.assertRaisesRegex(ValueError,'Actual graph supports'):self.score()

    def test_excluded_normalizer_observations_rejected(self):
        p=Path(self.records[0]['run_root'])/'preprocessing.json'
        self.modify_json(p,lambda v:v['pm_valid_count'].__setitem__(2,10))
        with self.assertRaisesRegex(ValueError,'Excluded-region observations'):self.score()

    def test_changed_target_context_rejected(self):
        p=Path(self.records[0]['run_root'])/'preprocessing.json'
        self.modify_json(p,lambda v:v['context_mean'].__setitem__(0,1.))
        with self.assertRaisesRegex(ValueError,'Target/context preprocessing'):self.score()

    def test_condition_mask_mismatch_rejected(self):
        p=Path(self.records[0]['regional_contract_path'])
        self.modify_json(p,lambda v:v['allowed_station_keys'].append('CN:a'))
        with self.assertRaisesRegex(ValueError,'Allowed regional station list'):self.score()

    def test_changed_training_protocol_rejected(self):
        p=Path(self.records[1]['run_root'])/'scoring_plan.json'
        self.modify_json(p,lambda v:v['training_metadata']['training_configuration'].__setitem__('learning_rate',.002))
        with self.assertRaisesRegex(ValueError,'Model/training settings changed'):self.score()

    def test_missing_eligible_prediction_rejected(self):
        p=Path(self.records[0]['run_root'])/'Exports/validation_predictions_by_seed.parquet'
        f=pd.read_parquet(p);f.iloc[1:].to_parquet(p,index=False)
        with self.assertRaisesRegex(ValueError,'missing 1 eligible predictions'):self.score()

    def test_forecast_condition_contamination_rejected(self):
        p=Path(self.records[0]['run_root'])/'Exports/validation_predictions_by_seed.parquet'
        f=pd.read_parquet(p);f.loc[0,'input_condition']='F';f.to_parquet(p,index=False)
        with self.assertRaisesRegex(ValueError,'true condition IDs'):self.score()

    def test_data_version_mismatch_rejected(self):
        p=Path(self.records[0]['run_root'])/'Exports/validation_predictions_by_seed.parquet'
        f=pd.read_parquet(p);f['data_version']='wrong';f.to_parquet(p,index=False)
        with self.assertRaisesRegex(ValueError,'data version differs'):self.score()

    def test_failed_job_rejected(self):
        p=Path(self.records[0]['run_root'])/'execution_status.json';write(p,{'status':'failed'})
        with self.assertRaisesRegex(ValueError,'completed full model job'):self.score()

    def test_raw_mask_guarantee_required(self):
        p=Path(self.records[0]['regional_contract_path'])
        self.modify_json(p,lambda v:v.__setitem__('observation_mask_applied_before_fit',False))
        with self.assertRaisesRegex(ValueError,'must guarantee'):self.score()

    def test_test_guard_precedes_test_export_read(self):
        with self.assertRaisesRegex(ValueError,'requires final=True'):self.score(split='test')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RegionalScoringTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'synthetic_only':True,
        'model_training_performed':False,'real_2024_scored':False,
        'source_sha256':hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':['global Holm before regional filtering','same seed sets','raw-unit ensemble separate from score mean',
                  'fixed-node/graph/target/context invariants','excluded observations absent from fitted normalizer',
                  'full declared matrix required','same-family training profile fixed','repeat output preservation',
                  'standard six output schemas/dtypes/Parquet codec','regional gain and CI sign',
                  'missing/contaminated prediction rejection','test guard before reading test export']}
    if result.wasSuccessful():
        config_dir=Path(__file__).resolve().parents[1]
        raw,_,_,_=evaluation._configuration(config_dir)
        smoke=evaluation._unique(Path.cwd()/'output/regional29_20261109/verification/full29_synthetic_scoring')
        config,records,grid=fixture(smoke,full_config=raw)
        scored=evaluation.score_regional_runs(smoke,records,config,
            smoke/'Result/Common_Scoring/Regional/synthetic_full29',common_dir=COMMON,
            bootstrap_reps=20,make_plots=True)
        assert scored['job_count']==58 and scored['condition_count']==29 and scored['comparison_count']==28
        assert scored['regional_test_rows']==28*2*3*4
        report['full_29_condition_635_node_synthetic_scoring']={**scored,
            'seed_count':2,'node_count':635,'synthetic_forecasts_not_model_results':True,
            'model_training_performed':False}
    REPORT.parent.mkdir(parents=True,exist_ok=True);write(REPORT,report)
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
