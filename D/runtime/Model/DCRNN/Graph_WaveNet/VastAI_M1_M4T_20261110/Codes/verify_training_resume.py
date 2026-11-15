"""Exercise the actual trainer through an epoch-boundary interruption on CPU."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import neural_training as trainer
from checkpoint_store import CheckpointStore,sha256_file


class TinyForecast(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(2,5,batch_first=True)
        self.dropout=nn.Dropout(.25);self.head=nn.Linear(5,4)
        self.architecture_config={'synthetic':'GRU5-dropout-head4'}
    def forward(self,x,c):
        y,_=self.gru(x[:,:,0,:]);return self.head(self.dropout(y[:,-1]))


class ToyOrigins(Dataset):
    def __init__(self,bundle,arrays,indices,lookback_hours):self.a=arrays;self.ids=indices
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        k=self.ids[i]
        return (torch.tensor(self.a['x'][k]),torch.zeros(1),torch.tensor(self.a['y'][k]),
                torch.tensor(self.a['mask'][k]),torch.tensor(k))


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b,(a,b)


def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(731)
    x=rng.normal(size=(12,3,1,2)).astype('float32')
    y=(.15*x[:,-1,0,0,None]+np.array([[1.,1.1,1.2,1.3]])).astype('float32')
    arrays={'x':x,'y':y,'mask':np.ones((12,4),bool),'context_x':np.zeros((12,1),dtype='float32')}
    times=pd.DatetimeIndex(list(pd.date_range('2022-02-01',periods=8,freq='3h',tz='Asia/Seoul'))+
                           list(pd.date_range('2023-02-01',periods=4,freq='3h',tz='Asia/Seoul')))
    bundle={'origins':pd.DataFrame({'origin_time_kst':times,'origin_split':['train']*8+['validation']*4}),
            'lookback_hours':3,'horizons':(3,6,12,24),'y_raw':np.expm1(y).astype(float),
            'eligible':arrays['mask']}
    api={'DCRNNOriginDataset':ToyOrigins,'split_origin_indices':lambda b,s,require_any_label=False:
         np.flatnonzero(b['origins'].origin_split.eq(s).to_numpy())}
    config={'seed':11,'mode':'pilot','input_condition':'M1','device':'cpu','data_signature':'synthetic-12origins',
            'max_epochs':8,'patience':20,'micro_batch_size':2,'effective_batch_size':4,'eval_batch_size':2,
            'optimizer':'adam','learning_rate':.002,'weight_decay':.0001,'scheduler':'multistep',
            'lr_milestones':[4,6],'lr_gamma':.5,'loss':'mse_log_z','max_grad_norm':5.,
            'checkpoint_policy':{'snapshot_every_epochs':5,'keep_recent':2},'resume':True,
            'max_train_origins':8,'max_val_origins':4}
    spec={'model':'synthetic_forecast','graph':'distance'}
    preprocessing={'pm_mean':[0.],'pm_std':[1.]}
    factory=lambda a,s,c:TinyForecast()
    def fit(path):return trainer.train_one_neural_run(bundle,arrays,preprocessing,np.eye(1,dtype='float32'),
                                                    spec,config,path,factory,api)
    with tempfile.TemporaryDirectory(prefix='vast_trainer_resume_',dir=Path.cwd()) as name:
        root=Path(name)
        trainer.CHECKPOINT_EVENT_CALLBACK=None
        _,summary_a,pred_a=fit(root/'continuous')
        def stop_after_three(event):
            if event['epoch']==3 and not event.get('final'):raise KeyboardInterrupt('test interruption')
        trainer.CHECKPOINT_EVENT_CALLBACK=stop_after_three
        try:fit(root/'interrupted')
        except KeyboardInterrupt:pass
        else:raise AssertionError('Test interruption was not raised')
        suffix='Models/synthetic_forecast/seed11'
        source=root/'interrupted'/suffix
        hashes={p.relative_to(source).as_posix():sha256_file(p) for p in source.rglob('*') if p.is_file()}
        CheckpointStore(source).copy_to(root/'resumed'/suffix)
        trainer.CHECKPOINT_EVENT_CALLBACK=None
        _,summary_b,pred_b=fit(root/'resumed')
        assert hashes=={p.relative_to(source).as_posix():sha256_file(p) for p in source.rglob('*') if p.is_file()}
        a,_=CheckpointStore(root/'continuous'/suffix).load_latest(None)
        b,_=CheckpointStore(root/'resumed'/suffix).load_latest(None)
        for field in ('model','best_model','optimizer','scheduler','scaler','epoch','best_epoch','best','bad_epochs','global_step','rng'):
            equal(a[field],b[field])
        for xa,xb in zip(a['history'],b['history']):
            equal({k:v for k,v in xa.items() if k!='seconds'},{k:v for k,v in xb.items() if k!='seconds'})
        pd.testing.assert_frame_equal(pred_a,pred_b,check_exact=True)
        assert summary_a['best_epoch']==summary_b['best_epoch']
        assert summary_b['trained_epochs']==8
        print(json.dumps({'actual_trainer_resume':'passed','continuous_epochs':8,'interrupted_after':3,
            'weights_optimizer_scheduler_rng_best_predictions_bitwise_equal':True,
            'source_run_preserved':True,'paid_resources_used':False}),flush=True)


if __name__=='__main__':main()
