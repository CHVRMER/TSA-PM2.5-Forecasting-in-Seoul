"""Offline fallback restore checks; only synthetic owned checkpoints are used."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import random
import numpy as np
import torch
from checkpoint_store import CheckpointStore,sha256_file
from drive_backup import RcloneDriveBackup
from restore_verified_backup import (MODEL,TRAINING_VERSION,RestoreError,
                                     restore_verified_backup,_digest)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def fixture(raw):
    run=raw/'Conditions/M1/run_20261110_01'
    seed=run/'Models'/MODEL/'seed11';seed.mkdir(parents=True)
    p={'node_order':['KR:111121']+[f'KR:S{i}' for i in range(634)],
       'context_names':[f'context_{i}' for i in range(375)],
       'pm_mean':[0.]*635,'pm_std':[1.]*635,'pm_valid_count':[2]*635,
       'context_mean':[0.]*375,'context_std':[1.]*375,'context_valid_count':[2]*375}
    contract={'version':TRAINING_VERSION,'data':'1'*64,'graph':'2'*64,'model':MODEL,
        'architecture':{'implementation':'dcrnn-seoul-paper-aligned-v2','input_dim':2,'context_dim':750,
                       'supervised_node':0,'horizons':[3,6,12,24,36,48],'hidden_dim':64,
                       'encoder_layers':2,'decoder_layers':2,'decoder_steps':48,'support_self_loops':True},
        'training':{'input_condition':'M1','seed':11,'mode':'full','experiment_bundle':'dcrnn-vast-m1-m4t-h48-v1'},'preprocessing':_digest(p,compact=False)}
    write(seed/'contract.json',contract);write(seed/'preprocessing.json',p)
    write(run/'experiment_config.json',{'version':TRAINING_VERSION,'training':{'input_condition':'M1','experiment_bundle':'dcrnn-vast-m1-m4t-h48-v1'},
                                      'data':contract['data'],'preprocessing':contract['preprocessing']})
    write(raw/'resolved_configuration.json',{'schema_version':'dcrnn-vast-m1-m4t-h48-checkpoint-v1',
                                           'selected_conditions':['M1','M2']})
    write(raw/'batch_manifest.json',{'status':'running','selected_conditions':['M1','M2']})
    write(raw/'Condition_Contracts/M1.json',{'condition':'M1','fixed_node_count':635})
    (raw/'console.log').write_text('excluded live log')
    store=CheckpointStore(seed,snapshot_every=5,keep_recent=3)
    def save(completed):
        payload={'contract':contract,'epoch':completed-1,'best_epoch':completed,
            'model':{'weight':torch.tensor([float(completed)])},
            'best_model':{'weight':torch.tensor([float(completed)])},'best':float(100-completed),
            'optimizer':{},'scheduler':None,'scaler':{},'bad_epochs':0,'global_step':completed,
            'history':[{'epoch':i,'validation_rmse_h6':float(100-i),'global_step':i}
                       for i in range(1,completed+1)],
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                   'torch':torch.get_rng_state(),'cuda':None}}
        return store.save(payload)
    fifth=save(5);tenth=save(10)
    backup=RcloneDriveBackup({'backup':{}},raw,'pilot')
    files=backup._file_manifest()
    marker={'schema_version':'drive-verified-publication-v1','backup_id':'synthetic_10',
        'main_copy_verified':True,'restore_requires_file_manifest_match':True,
        'files':files,'manifest_sha256':_digest(files),'reason':'checkpoint_epoch_10'}
    marker_path=raw/'Backup_receipts/Published/test/verified.json';write(marker_path,marker)
    return seed,store,save,fifth,tenth,marker_path,marker


def fingerprints(root):
    return {p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob('*') if p.is_file()}


def fails(call):
    try:call()
    except (RestoreError,FileExistsError):pass
    else:raise AssertionError('Unsafe restore was accepted.')


def main():
    test_parent=Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='.restore_backup_test_',dir=test_parent) as temporary:
        base=Path(temporary).resolve();assert base.is_relative_to(test_parent)
        raw=base/'downloaded';raw.mkdir()
        seed,store,save,fifth,tenth,marker_path,marker=fixture(raw)
        fifteenth=save(15)  # Present in cloud download, but absent from selected publication.
        Path(tenth['state_path']).write_bytes(b'failed later copy / corrupted state')
        (raw/'batch_manifest.json').write_text('{partial mutable metadata')
        before=fingerprints(raw)
        output=base/'restored'
        report=restore_verified_backup(raw,marker_path,output)
        assert fingerprints(raw)==before
        assert not report['all_listed_files_hash_valid']
        assert report['restored_checkpoints'][0]['completed_epochs']==5
        assert report['not_started_conditions']==['M2']
        assert report['unlisted_generations_not_copied']==[
            str(Path(fifteenth['generation_dir']).relative_to(raw)).replace('\\','/')]
        restored_seed=output/seed.relative_to(raw)
        valid,_=CheckpointStore(restored_seed).load_latest(None)
        assert valid['epoch']==4 and valid['model']['weight'].item()==5
        assert len(list((restored_seed/'Checkpoints').glob('epoch_*')))==1
        assert not (output/'console.log').exists()
        assert json.loads((output/'batch_manifest.json').read_text())['conditions']['M2']['status']=='not_started'
        assert json.loads((output/'batch_manifest.json').read_text())['mode']=='full'
        assert json.loads((output/'batch_manifest.json').read_text())['experiment_schema']=='dcrnn-vast-m1-m4t-h48-v1'
        fails(lambda:restore_verified_backup(raw,marker_path,output))
        # Initial publication has no model state and cannot be advertised as resumable.
        initial=dict(marker,files=[row for row in marker['files'] if not '/Checkpoints/' in row['path']])
        initial['manifest_sha256']=_digest(initial['files'])
        initial_path=base/'initial.json';write(initial_path,initial)
        fails(lambda:restore_verified_backup(raw,initial_path,base/'initial_restore'))
        assert not (base/'initial_restore').exists()
        # Hash-valid but semantically wrong model/condition contracts must fail hard.
        contract_path=seed/'contract.json'
        contract=json.loads(contract_path.read_text());contract['training']['input_condition']='M2'
        write(contract_path,contract)
        changed=json.loads(json.dumps(marker))
        for row in changed['files']:
            if row['path']==contract_path.relative_to(raw).as_posix():
                row['bytes']=contract_path.stat().st_size;row['sha256']=sha256_file(contract_path)
        changed['manifest_sha256']=_digest(changed['files'])
        bad_contract=base/'bad_contract_marker.json';write(bad_contract,changed)
        fails(lambda:restore_verified_backup(raw,bad_contract,base/'wrong_contract_restore'))
        assert not (base/'wrong_contract_restore').exists()
        traversal=json.loads(json.dumps(marker));traversal['files'][0]['path']='../outside'
        traversal['manifest_sha256']=_digest(traversal['files'])
        traversal_path=base/'traversal.json';write(traversal_path,traversal)
        fails(lambda:restore_verified_backup(raw,traversal_path,base/'unsafe'))
    print(json.dumps({'restore_backup_tests':'passed','corrupt_epoch10_falls_back_to_verified_epoch5':True,
        'unlisted_epoch15_not_imported':True,'mutable_pointer_and_batch_metadata_rebuilt':True,
        'no_source_mutation':True,'existing_output_rejected':True,'initial_only_backup_rejected':True,
        'critical_model_contract_mismatch_rejected':True,'unsafe_paths_rejected':True}))


if __name__=='__main__':main()
