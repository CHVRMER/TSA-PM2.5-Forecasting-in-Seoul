"""Small real-Torch train/export/private-common-score integration; synthetic only."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,tempfile
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'vendor'),str(HERE)]
import neural_training as trainer
from graph_wavenet_model import GraphWaveNetMultiHorizon
import run_experiments as runner

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=HERE.parents[4])
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='gwn_h48_e2e_',dir=args.report.parent) as tmp:
        project=Path(tmp);common=project/'Model/Common/GraphWaveNet_H48_20261112'
        common.mkdir(parents=True)
        for name in ('score.py','grid.py','neural_evaluation.py','horizon_contract.py'):
            shutil.copy2(args.project_root/'Model/Common/GraphWaveNet_H48_20261112'/name,common/name)
        _,evaluation,scorer,grid_module,_=runner.common_modules(project)
        hours=pd.date_range('2021-01-01','2024-02-01',freq='h',tz='Asia/Seoul')
        raw=20+2*np.sin(np.arange(len(hours),dtype=float)/17)
        panel=np.stack((raw,raw+2,raw-1),axis=1).astype(np.float32)
        training_origins=pd.date_range('2021-01-05',periods=4,freq='3h',tz='Asia/Seoul')
        evaluation_origins=pd.date_range('2023-12-28','2024-01-03',freq='3h',tz='Asia/Seoul')
        origins=training_origins.append(evaluation_origins)
        split=np.where(origins.year<2023,'train',np.where(origins.year==2023,'validation','test'))
        y=np.stack([pd.Series(panel[:,0],index=hours).reindex(origins+pd.Timedelta(hours=h)).to_numpy()
                    for h in runner.HORIZONS],axis=1)
        eligible=np.isfinite(y)
        for column,h in enumerate(runner.HORIZONS):
            target=origins+pd.Timedelta(hours=h)
            eligible[:,column] &= np.where(target.year<2023,'train',np.where(target.year==2023,'validation','test'))==split
        rng=np.random.default_rng(314)
        context=rng.normal(0,.05,(len(origins),375)).astype(np.float32)
        bundle={'nodes':pd.DataFrame({'station_key':['KR:111121','KR:SYNTH1','CN:SYNTH2']}),
                'hourly_index':hours,'origins':pd.DataFrame({'origin_time_kst':origins,'origin_split':split}),
                'pm_values':panel,'context_values':context,'horizons':runner.HORIZONS,'lookback_hours':48,
                'y_raw':y,'eligible':eligible,'origin_positions':hours.get_indexer(origins)}
        preprocessing={'pm_mean':[3.]*3,'pm_std':[.2]*3}
        arrays={'hourly_x':np.stack(((np.log1p(panel)-3)/.2,np.zeros_like(panel)),axis=-1),
                'context_x':np.concatenate((context,np.zeros_like(context)),axis=1),
                'y_scaled':((np.log1p(y)-3)/.2).astype(np.float32),'eligible':eligible}
        class Dataset(torch.utils.data.Dataset):
            def __init__(self,b,a,ids,lookback):self.b,self.a,self.ids,self.lookback=b,a,np.asarray(ids),lookback
            def __len__(self):return len(self.ids)
            def __getitem__(self,item):
                i=int(self.ids[item]);end=int(self.b['origin_positions'][i])+1
                return tuple(torch.as_tensor(v) for v in (
                    self.a['hourly_x'][end-self.lookback:end],self.a['context_x'][i],
                    self.a['y_scaled'][i],self.a['eligible'][i],np.int64(i)))
        def indices(b,which,require_any_label):
            keep=b['origins'].origin_split.eq(which).to_numpy(copy=True)
            if require_any_label:keep=keep & b['eligible'].any(axis=1)
            return np.flatnonzero(keep)
        api={'DCRNNOriginDataset':Dataset,'split_origin_indices':indices}
        parts=[]
        for column,h in enumerate(runner.HORIZONS):
            part=pd.DataFrame({'origin_time_kst':origins,'target_time_kst':origins+pd.Timedelta(hours=h),
                'horizon':h,'y_true':y[:,column],'eligible':eligible[:,column],'split':split,
                'season':'winter','route_group':'west','wind_sector':'west'})
            parts.append(part.loc[part.split.ne('train')])
        grid=grid_module.add_baselines(grid_module.validate_grid(pd.concat(parts,ignore_index=True)),
            pd.DataFrame({'time_kst':hours,'pm25':panel[:,0]}))
        grid=grid_module.validate_baselines(grid)
        grid_path=project/'Model/Cache/evaluation_grid.parquet';grid_path.parent.mkdir(parents=True)
        grid.to_parquet(grid_path,index=False)
        adjacency=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.float32)
        spec={'model':runner.MODEL_NAME,'graph':'distance','adaptive_mode':'fixed_plus_adaptive'}
        def factory(graph,_spec,context_dim):
            return GraphWaveNetMultiHorizon(graph,horizons=runner.HORIZONS,output_steps=48,context_dim=context_dim,
                residual_channels=4,dilation_channels=4,skip_channels=8,end_channels=16,
                blocks=4,layers=4,context_dropout=.2,architecture_profile='project48')
        cfg=dict(model_family='GraphWaveNet',experiment_bundle=runner.EXPERIMENT_BUNDLE,input_condition='M1',
            mode='full',seeds=[11],max_epochs=2,patience=10,micro_batch_size=2,effective_batch_size=4,
            eval_batch_size=2,optimizer='adam',learning_rate=.001,optimizer_epsilon=1e-8,weight_decay=1e-4,
            scheduler='plateau',lr_patience=3,lr_gamma=.5,min_learning_rate=1e-5,selection_horizon=6,
            max_grad_norm=5.,loss='mae_original',device='cpu',use_amp=False,resume=True,run_final_test=True,
            checkpoint_policy={'snapshot_every_epochs':5,'keep_recent':2},bootstrap_reps=10,
            evaluation_grid_path=str(grid_path),evaluation_grid_sha256=scorer.sha256(grid_path))
        summaries=trainer.run_neural_experiment(bundle,{'distance':adjacency},preprocessing,arrays,[spec],cfg,
            project/'Result/GraphWaveNet/VastAI_H48/Conditions/M1',factory,api,project,common_dir=common)
        summary=summaries[0];run=Path(summary['experiment_run_root'])
        assert summary['test_scored'] and summary['validation_scored']
        for which in ('validation','test'):
            frame=pd.read_parquet(run/'Exports'/f'{which}_predictions_by_seed.parquet')
            assert len(frame)==int((split==which).sum())*6 and set(frame.horizon)==set(runner.HORIZONS)
            assert np.isfinite(frame.y_hat).all()
            folder=Path(summary['scoring_locations'][which])
            assert all((folder/name).is_file() for name in runner.STANDARD_OUTPUTS)
            assert len(pd.read_csv(folder/'tests.csv'))==12
            score_rows=pd.read_parquet(folder/'score_rows.parquet')
            expected_counts=grid.loc[grid.split.eq(which)&grid.eligible].groupby('horizon').size()
            for (_,h),count in score_rows.groupby(['run_id','horizon']).size().items():
                assert count==expected_counts.loc[h]
        old=Path(summary['scoring_locations']['test'])
        hashes={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in runner.STANDARD_OUTPUTS}
        again=evaluation.score_neural_run(project,run,'GraphWaveNet','M1',split='test',experiment_mode='full',
            final=True,common_dir=common,bootstrap_reps=10,make_plots=False)
        assert Path(again['output_dir'])!=old
        assert all(hashlib.sha256((old/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    report=dict(status='passed',torch=torch.__version__,device='cpu',synthetic_only=True,
        real_635_node_training_tested=False,trained_epochs=2,horizons=list(runner.HORIZONS),
        validation_export_rows=int((split=='validation').sum())*6,test_export_rows=int((split=='test').sum())*6,
        model_train_export_private_score_pipeline=True,
        both_splits_common_six_outputs=True,individual_condition_tests=12,repeated_scoring_no_overwrite=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
