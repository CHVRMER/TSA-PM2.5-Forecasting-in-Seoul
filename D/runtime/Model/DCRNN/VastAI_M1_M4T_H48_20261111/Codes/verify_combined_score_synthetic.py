"""Exercise the private six-horizon scorer through the Vast.ai batch wrapper.

No observation data or model training are used. This checks fixed rows, exact
input-condition contrast IDs, seed/ensemble separation and the six schemas.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd

import run_experiments as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    root=args.project_root.resolve()
    common=root/'Model/Common/DCRNN_H48_20261111'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dcrnn_h48_synthetic_score_',dir=args.report.parent if args.report else None) as temporary:
        test=Path(temporary)
        copied=test/'Model/Common/DCRNN_H48_20261111'
        copied.mkdir(parents=True)
        for name in ('score.py','grid.py','neural_evaluation.py'):
            shutil.copy2(common/name,copied/name)
        _,_,scorer,grid_module,_=runner.common_modules(test)
        origins=pd.date_range('2023-12-16', '2024-01-15',freq='3h',tz='Asia/Seoul')
        hours=pd.date_range('2021-01-01','2024-01-18 23:00',freq='h',tz='Asia/Seoul')
        n=np.arange(len(hours),dtype=float)
        target_hourly=pd.DataFrame({'time_kst':hours,'pm25':30+10*np.sin(n/23)+3*np.cos(n/117)})
        target_lookup=target_hourly.set_index('time_kst').pm25
        parts=[]
        for horizon in (3,6,12,24,36,48):
            target=origins+pd.Timedelta(hours=horizon)
            x=np.arange(len(origins),dtype=float)
            truth=target_lookup.reindex(target).to_numpy()
            parts.append(pd.DataFrame({'origin_time_kst':origins,'horizon':horizon,
                'target_time_kst':target,'y_true':truth,'eligible':origins.year==target.year,
                'split':np.where(origins.year==2023,'validation','test'),
                'season':'winter','route_group':'west','wind_sector':'west'}))
        grid=grid_module.add_baselines(grid_module.validate_grid(pd.concat(parts,ignore_index=True)),target_hourly)
        grid=grid_module.validate_baselines(grid)
        grid_path=test/'Model/Cache/H48/evaluation_grid.parquet'
        grid_path.parent.mkdir(parents=True);grid.to_parquet(grid_path,index=False)
        completed={};seeds=[11]
        for i,condition in enumerate(runner.CONDITIONS):
            export=test/'Source_Runs'/condition/'Exports';export.mkdir(parents=True)
            completed[condition]=str(export.parent)
            for split in ('validation','test'):
                pieces=[]
                g=grid.loc[grid.split.eq(split)].copy()
                x=np.arange(len(g),dtype=float)
                for seed in seeds:
                    pieces.append(pd.DataFrame({'origin_time_kst':g.origin_time_kst,
                        'horizon':g.horizon,'split':split,'model':runner.MODEL_NAME,
                        'input_condition':condition,'prediction_kind':'seed','seed':seed,
                        'data_version':f'synthetic-{condition}',
                        'y_hat':g.y_true.to_numpy()+(.7-i*.06)*np.sin(x/9+seed/10)}))
                pd.concat(pieces,ignore_index=True).to_parquet(export/f'{split}_predictions_by_seed.parquet',index=False)
        config={'scoring':{'bootstrap_reps':20,'bootstrap_seed':3}}
        initial=runner.combined_score(test,completed,runner.CONDITIONS,seeds,test/'Result',config,final=False,grid_path=grid_path)
        assert set(initial)=={'validation'},'The test split must stay locked without the final flag.'
        first=Path(initial['validation'])
        previous={name:runner.hashlib.sha256((first/name).read_bytes()).hexdigest() for name in runner.STANDARD_OUTPUTS}
        second=runner.combined_score(test,completed,runner.CONDITIONS,seeds,test/'Result',config,final=True,grid_path=grid_path)
        assert set(second)=={'validation','test'}
        assert second['validation']!=initial['validation'],'Repeated scoring must use a new output folder.'
        assert previous=={name:runner.hashlib.sha256((first/name).read_bytes()).hexdigest() for name in runner.STANDARD_OUTPUTS}
        _,_,scorer,_,_=runner.common_modules(test)
        expected_pairs=set(runner.comparison_run_pairs(runner.CONDITIONS,seeds))
        assert len(expected_pairs)==7 # seven contrasts × seed11
        for split,directory in second.items():
            directory=Path(directory)
            for name in runner.STANDARD_OUTPUTS:assert (directory/name).is_file(),name
            tests=pd.read_csv(directory/'tests.csv')
            assert len(tests)==84 # (six models + seasonal vs persistence + seven input contrasts) × six horizons
            assert set(tests.horizon)=={3,6,12,24,36,48}
            assert tests.loc[tests.horizon.eq(36),'hac_lag_slots'].eq(11).all()
            assert tests.loc[tests.horizon.eq(48),'hac_lag_slots'].eq(15).all()
            observed=set(tests[['run_id','versus_run_id']].itertuples(index=False,name=None))
            assert expected_pairs<=observed,'The exact same-seed and ensemble condition contrasts must exist.'
            predictions=pd.concat([pd.read_parquet(Path(completed[c])/'Exports'/f'{split}_predictions_by_seed.parquet') for c in runner.CONDITIONS],ignore_index=True)
            expected=scorer.score_frames(grid,predictions,split=split,pairs=list(expected_pairs),final=split=='test',bootstrap_reps=20,bootstrap_seed=3)
            for table in ('metrics','seed_metric_summary','tests','conditions'):
                actual=pd.read_csv(directory/f'{table}.csv')
                assert actual.columns.tolist()==expected[table].columns.tolist(),f'{table} column order changed.'
                assert len(actual)==len(expected[table]),f'{table} row count changed.'
            actual=pd.read_parquet(directory/'score_rows.parquet')
            assert actual.columns.tolist()==expected['score_rows'].columns.tolist()
            assert len(actual)==len(expected['score_rows'])
            assert actual.groupby(['run_id','horizon']).size().equals(expected['score_rows'].groupby(['run_id','horizon']).size())
            audit=json.loads((directory/'scoring_audit.json').read_text(encoding='utf-8'))
            assert not audit['model_prediction_intersection_used']
            assert not audit['rows_selected_using_predictions']
            assert audit['test_guard']==(split=='test')
        # Test multiple seeds separately: individual seed scores must never be
        # substituted for a raw-unit prediction ensemble.
        extra=predictions.copy();extra['seed']=22;extra['y_hat']=extra.y_hat+.1
        multi=pd.concat([predictions,extra],ignore_index=True)
        multi_pairs=runner.comparison_run_pairs(runner.CONDITIONS,[11,22])
        assert len(multi_pairs)==21
        multi_result=scorer.score_frames(grid,multi,split=split,pairs=multi_pairs,final=True,bootstrap_reps=10)
        assert len(multi_result['metrics'].loc[multi_result['metrics'].prediction_kind.eq('seed')])==72
        assert len(multi_result['metrics'].loc[multi_result['metrics'].prediction_kind.eq('ensemble')])==36
        report={'synthetic_combined_scoring':'passed','standard_outputs':list(runner.STANDARD_OUTPUTS),
                          'same_seed_and_ensemble_contrasts':len(expected_pairs),'test_lock_checked':True,
                          'horizons':[3,6,12,24,36,48],'seed11_family_test_rows':84,
                          'multi_seed_contrasts':len(multi_pairs),'individual_seeds_and_ensemble_separate':True,
                          'new_output_folders_checked':True,'source_outputs_preserved':True,
                          'real_model_training_tested':False}
        if args.report:
            args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
