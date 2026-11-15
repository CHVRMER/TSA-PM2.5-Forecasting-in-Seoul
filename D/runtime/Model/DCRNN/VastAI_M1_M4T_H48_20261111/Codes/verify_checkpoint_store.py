"""CPU-only failure-injection tests for the strict Vast.ai checkpoint store.

Run this before renting/starting expensive training. No PM2.5 observations or
paid resources are used. Subprocesses deliberately terminate while saving.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from checkpoint_store import (CheckpointStore, CheckpointError,
                              CheckpointContractMismatch, NoValidCheckpoint,
                              POINTER_NAME, STATE_NAME, MANIFEST_NAME, COMMIT_NAME,
                              sha256_file)

CONTRACT={'model':'synthetic-dropout-net','data':'synthetic-only','seed':11,
          'max_epochs':8,'optimizer':'Adam','scheduler':'MultiStepLR'}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())


def fixture(completed, best_epoch=None):
    best_epoch=completed if best_epoch is None else best_epoch
    history=[{'epoch':epoch,'validation_rmse_h6':float(100-epoch if epoch<=best_epoch else 120+epoch),
              'global_step':epoch*3} for epoch in range(1,completed+1)]
    return {'contract':CONTRACT,'epoch':completed-1,'best_epoch':best_epoch,
            'model':{'weight':torch.tensor([float(completed)])},
            'best_model':{'weight':torch.tensor([float(best_epoch)])},
            'best':float(100-best_epoch),'optimizer':{'synthetic':completed},
            'scheduler':{'last_epoch':completed},'scaler':{},
            'bad_epochs':completed-best_epoch,'global_step':completed*3,
            'history':history,'rng':rng_state()}


def file_fingerprints(folder):
    return {str(path.relative_to(folder)):sha256_file(path)
            for path in Path(folder).rglob('*') if path.is_file()}


def nested_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertIsInstance(b,torch.Tensor);test.assertTrue(torch.equal(a,b))
    elif isinstance(a,np.ndarray):
        test.assertTrue(np.array_equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for key in a:nested_equal(test,a[key],b[key])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):nested_equal(test,x,y)
    else:test.assertEqual(a,b)


def train_tiny(seed_folder, stop_after=8, *, restore=False):
    """Real dropout+Adam+scheduler+gradient accumulation, exact epoch resume."""
    random.seed(11);np.random.seed(11);torch.manual_seed(11)
    torch.use_deterministic_algorithms(True)
    model=nn.Sequential(nn.Linear(5,9),nn.Tanh(),nn.Dropout(.35),nn.Linear(9,2))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,5],gamma=.5)
    scaler={};history=[];start=0;best=float('inf');best_epoch=0;best_model=None;step=0;bad=0
    store=CheckpointStore(seed_folder,snapshot_every=5,keep_recent=2)
    if restore:
        state,_=store.load_latest(CONTRACT)
        if state is None:raise AssertionError('A deterministic resume must restore existing state.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler=state['scaler']
        history=copy.deepcopy(state['history']);start=state['epoch']+1;best=state['best']
        best_epoch=state['best_epoch'];best_model=copy.deepcopy(state['best_model'])
        step=state['global_step'];bad=state['bad_epochs'];restore_rng(state['rng'])
    # A private data RNG keeps feature/target construction independent of the
    # restored global RNG used by shuffling, dropout and random diagnostics.
    generator=torch.Generator().manual_seed(888)
    x=torch.randn(24,5,generator=generator)
    y=torch.stack([x[:,0]*.4+x[:,1]*.7,x[:,2]*-.3],dim=1)
    loader=DataLoader(TensorDataset(x,y),batch_size=4,shuffle=True,num_workers=0)
    validation=DataLoader(TensorDataset(x,y),batch_size=6,shuffle=False,num_workers=0)
    for epoch in range(start,stop_after):
        model.train();optimizer.zero_grad();loss_sum=0.;count=0
        for batch,(features,truth) in enumerate(loader):
            prediction=model(features)
            loss=(prediction-truth).square().mean()/2
            loss.backward();loss_sum+=float(loss.detach());count+=1
            if batch%2==1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();optimizer.zero_grad();step+=1
        model.eval();errors=[]
        with torch.no_grad():
            for features,truth in validation:errors.append((model(features)-truth).square())
        score=float(torch.sqrt(torch.cat(errors).mean()))
        # Exercise Python and NumPy RNG restoration in addition to dropout.
        diagnostic=random.random()+float(np.random.random())
        if score<best:
            best=score;best_epoch=epoch+1;bad=0
            best_model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:bad+=1
        scheduler.step()
        history.append({'epoch':epoch+1,'validation_rmse_h6':score,
                        'global_step':step,'loss':loss_sum/count,
                        'diagnostic':diagnostic,'lr':optimizer.param_groups[0]['lr']})
        payload={'contract':CONTRACT,'model':model.state_dict(),'best_model':best_model,
                 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'scaler':scaler,'epoch':epoch,'best_epoch':best_epoch,'best':best,
                 'bad_epochs':bad,'global_step':step,'history':history,'rng':rng_state()}
        store.save(payload)
    final,_=store.load_latest(CONTRACT)
    model.load_state_dict(final['best_model']);model.eval()
    with torch.no_grad():predictions=model(x).clone()
    return final,predictions


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='strict_checkpoint_test_')
        self.root=Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()

    def test_fresh_and_self_contained_best(self):
        store=CheckpointStore(self.root/'seed')
        self.assertIsNone(store.load_latest(CONTRACT)[0])
        store.save(fixture(4,best_epoch=2))
        state,info=store.load_latest(CONTRACT)
        self.assertEqual(state['epoch'],3);self.assertEqual(state['best_epoch'],2)
        self.assertEqual(state['model']['weight'].item(),4)
        self.assertEqual(state['best_model']['weight'].item(),2)
        self.assertFalse(info['fallback_used'])

    def test_retention_every_five_recent_two_and_final_pin(self):
        store=CheckpointStore(self.root/'seed',snapshot_every=5,keep_recent=2)
        for completed in range(1,13):store.save(fixture(completed,min(completed,7)))
        store.save(fixture(12,best_epoch=7),pin=True)
        manifests=[json.loads((folder/MANIFEST_NAME).read_text()) for folder in store._candidates()]
        self.assertEqual({row['completed_epochs'] for row in manifests},{5,10,11,12})
        self.assertTrue(all(row['snapshot'] for row in manifests if row['completed_epochs'] in {5,10}))
        self.assertTrue(any(row['pinned'] and row['completed_epochs']==12 for row in manifests))
        self.assertEqual(store.load_latest(CONTRACT)[0]['best_epoch'],7)

    def test_corrupt_latest_falls_back_and_all_corrupt_fail_closed(self):
        store=CheckpointStore(self.root/'seed')
        for completed in range(1,5):store.save(fixture(completed))
        latest=store._candidates()[0]
        with (latest/STATE_NAME).open('ab') as stream:stream.write(b'corrupted')
        state,info=store.load_latest(CONTRACT)
        self.assertEqual(state['epoch'],2);self.assertTrue(info['fallback_used'])
        self.assertEqual(len(info['rejected_generations']),1)
        previous=Path(info['generation_dir'])
        (previous/MANIFEST_NAME).write_text('{broken',encoding='utf-8')
        with self.assertRaises(NoValidCheckpoint):store.load_latest(CONTRACT)

    def test_wrong_contract_never_falls_back_or_starts_fresh(self):
        store=CheckpointStore(self.root/'seed');store.save(fixture(1))
        wrong=dict(CONTRACT,seed=22)
        with self.assertRaises(CheckpointContractMismatch):store.load_latest(wrong)
        payload=fixture(2);payload['contract']=wrong
        with self.assertRaises(CheckpointContractMismatch):store.save(payload)

    def test_pointer_is_a_hint_and_manifest_commit_are_verified(self):
        store=CheckpointStore(self.root/'seed')
        store.save(fixture(1));store.save(fixture(2))
        (store.root/POINTER_NAME).write_text('{broken',encoding='utf-8')
        state,info=store.load_latest(CONTRACT)
        self.assertEqual(state['epoch'],1);self.assertTrue(info['pointer_stale'])
        self.assertIsNotNone(info['pointer_error'])
        (store._candidates()[0]/COMMIT_NAME).unlink()
        state,info=store.load_latest(CONTRACT)
        self.assertEqual(state['epoch'],0);self.assertTrue(info['fallback_used'])

    def test_only_pending_or_legacy_artifacts_fail_closed(self):
        store=CheckpointStore(self.root/'pending')
        (store.root/'.pending_interrupted').mkdir()
        with self.assertRaises(NoValidCheckpoint):store.load_latest(CONTRACT)
        legacy=CheckpointStore(self.root/'legacy')
        (legacy.seed_folder/'last.pt').write_bytes(b'old independent pair')
        with self.assertRaisesRegex(NoValidCheckpoint,'Legacy'):legacy.load_latest(CONTRACT)

    def test_copy_verifies_bytes_preserves_source_and_rejects_nonempty_destination(self):
        source=CheckpointStore(self.root/'source')
        for completed in range(1,7):source.save(fixture(completed))
        before=file_fingerprints(source.root)
        result=source.copy_to(self.root/'destination')
        self.assertEqual(before,file_fingerprints(source.root))
        destination=CheckpointStore(self.root/'destination')
        a,_=source.load_latest(CONTRACT);b,_=destination.load_latest(CONTRACT)
        nested_equal(self,a,b)
        self.assertFalse(result['source_files_modified'])
        with self.assertRaises(CheckpointError):source.copy_to(self.root/'destination')

    def test_copy_rejects_bad_generation_and_copies_fallback(self):
        source=CheckpointStore(self.root/'source')
        source.save(fixture(1));source.save(fixture(2))
        (source._candidates()[0]/STATE_NAME).write_bytes(b'truncated')
        result=source.copy_to(self.root/'destination')
        self.assertEqual(len(result['rejected_generations']),1)
        state,_=CheckpointStore(self.root/'destination').load_latest(CONTRACT)
        self.assertEqual(state['epoch'],0)

    def test_failed_disk_write_preserves_previous_committed_state(self):
        store=CheckpointStore(self.root/'seed');store.save(fixture(1))
        def fail_save(payload,stream):
            stream.write(b'partial data');stream.flush();raise OSError('synthetic disk full')
        with mock.patch('checkpoint_store.torch.save',side_effect=fail_save):
            with self.assertRaises(OSError):store.save(fixture(2))
        state,info=store.load_latest(CONTRACT)
        self.assertEqual(state['epoch'],0)
        self.assertEqual(len(info['ignored_pending_generations']),1)

    def test_payload_best_history_and_weights_are_consistent(self):
        store=CheckpointStore(self.root/'seed')
        wrong=fixture(1);wrong['best_model']['weight'].fill_(99)
        with self.assertRaisesRegex(ValueError,'newly selected best'):store.save(wrong)
        wrong=fixture(1);wrong['best']=42
        with self.assertRaisesRegex(ValueError,'Best score'):store.save(wrong)

    def test_hard_process_crash_at_each_commit_boundary(self):
        stages={'after_state_fsync':1,'after_manifest_fsync':1,'after_commit_marker':1,
                'after_generation_rename':2,'after_pointer_replace':2,'before_prune':2}
        for stage,expected in stages.items():
            with self.subTest(stage=stage):
                folder=self.root/stage
                store=CheckpointStore(folder);store.save(fixture(1))
                command=[sys.executable,'-B',str(Path(__file__).resolve()),
                         '--crash-worker',str(folder),stage]
                result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,timeout=60,check=False)
                self.assertEqual(result.returncode,73,result.stdout)
                state,_=store.load_latest(CONTRACT)
                self.assertEqual(state['epoch']+1,expected)
                self.assertEqual(state['best_epoch'],expected)
                self.assertEqual(state['model']['weight'].item(),expected)
                self.assertEqual(state['best_model']['weight'].item(),expected)

    def test_actual_training_resumes_identical_weights_optimizer_scheduler_rng_and_best(self):
        torch.set_num_threads(1)
        uninterrupted,prediction_a=train_tiny(self.root/'uninterrupted',8)
        train_tiny(self.root/'interrupted',3)
        CheckpointStore(self.root/'interrupted').copy_to(self.root/'resumed_new_run')
        resumed,prediction_b=train_tiny(self.root/'resumed_new_run',8,restore=True)
        for key in ('model','best_model','optimizer','scheduler','scaler','rng','history',
                    'epoch','best_epoch','best','bad_epochs','global_step'):
            with self.subTest(state=key):nested_equal(self,uninterrupted[key],resumed[key])
        self.assertTrue(torch.equal(prediction_a,prediction_b))
        # A completed run must perform no further updates on resume.
        before=copy.deepcopy(resumed['optimizer'])
        finished,finished_prediction=train_tiny(self.root/'resumed_new_run',8,restore=True)
        nested_equal(self,before,finished['optimizer'])
        self.assertTrue(torch.equal(prediction_b,finished_prediction))


def crash_worker(folder,stage):
    class CrashStore(CheckpointStore):
        def _fault_hook(self,current,directory):
            if current==stage:
                print('synthetic hard exit at '+stage,flush=True)
                os._exit(73)
    CrashStore(folder).save(fixture(2))
    raise AssertionError('Requested crash hook did not execute.')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--crash-worker':
        crash_worker(sys.argv[2],sys.argv[3])
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--report',type=Path)
        options,rest=parser.parse_known_args()
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointTests)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        report={'checkpoint_store_tests':'passed' if result.wasSuccessful() else 'failed',
                'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                'hard_process_crash_stages':6,'actual_optimizer_rng_resume_equivalence':True,
                'paid_resources_used':False,'real_pm25_model_training_tested':False}
        if options.report:
            options.report.parent.mkdir(parents=True,exist_ok=True)
            options.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report),flush=True)
        raise SystemExit(0 if result.wasSuccessful() else 1)
