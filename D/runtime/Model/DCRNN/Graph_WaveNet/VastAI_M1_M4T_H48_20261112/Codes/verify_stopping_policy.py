"""H48 strict stopping-setting compatibility; no PyTorch or cloud calls."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import stopping_policy as policy

class StrictStoppingTests(unittest.TestCase):
    def setUp(self):
        self.training={'max_epochs':75,'patience':10,'optimizer':'adam','learning_rate':0.001}

    def test_identical_settings_allowed(self):
        self.assertEqual(policy.stopping_reduction(self.training,copy.deepcopy(self.training)),{})

    def test_limit_reductions_and_increases_rejected(self):
        for key,values in {'max_epochs':[25,100],'patience':[5,20]}.items():
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'H48 stopping settings differ'):
                    policy.stopping_reduction(self.training,dict(self.training,**{key:value}))

    def test_optimizer_and_lr_changes_rejected(self):
        for item in [{'optimizer':'adamw'},{'learning_rate':0.01}]:
            with self.subTest(item=item),self.assertRaisesRegex(ValueError,'Non-stopping training settings differ'):
                policy.stopping_reduction(self.training,dict(self.training,**item))

    def test_invalid_limits_rejected(self):
        for key in ('max_epochs','patience'):
            for value in (0,-1,True,1.5):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    policy.stopping_reduction(self.training,dict(self.training,**{key:value}))

    def test_copy_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source=Path(folder)/'source';source.mkdir();destination=Path(folder)/'new'
            for name in ('contract.json','preprocessing.json','learning_curve.csv'):
                (source/name).write_text(name,encoding='utf-8')
            calls=[]
            def copy_to(dest):
                calls.append(dest);return {'status':'copied','source_files_modified':False}
            store=SimpleNamespace(seed_folder=source,
                load_latest=lambda _:({'contract':{'training':dict(self.training)},'epoch':4},{}),
                copy_to=copy_to)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            result=policy.import_seed_checkpoint(store,destination,self.training,None,None,None)
            self.assertEqual(result['status'],'copied')
            self.assertEqual(calls,[destination])
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertEqual(before,{p.name:p.read_bytes() for p in destination.iterdir()})

    def test_changed_limits_never_copy(self):
        calls=[]
        store=SimpleNamespace(seed_folder='unused',
            load_latest=lambda _:({'contract':{'training':dict(self.training)}},{}),
            copy_to=lambda _:calls.append('copy'))
        with self.assertRaisesRegex(ValueError,'H48 stopping settings differ'):
            policy.import_seed_checkpoint(store,'unused',dict(self.training,max_epochs=25),None,None,None)
        self.assertEqual(calls,[])

    def test_no_completed_epoch_records_fresh_restart(self):
        store=SimpleNamespace(seed_folder='unused',load_latest=lambda _:(None,{}))
        result=policy.import_seed_checkpoint(store,'unused',self.training,None,None,None)
        self.assertEqual(result['status'],'fresh_restart_no_completed_epoch')
        self.assertIs(result['source_files_modified'],False)

if __name__=='__main__':unittest.main(verbosity=2)
