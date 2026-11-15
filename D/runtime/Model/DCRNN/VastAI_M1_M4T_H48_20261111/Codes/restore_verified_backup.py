"""Restore owned Drive training artifacts into a NEW, locally verified batch.

Only files listed by one published backup marker are eligible. Mutable cloud
files can be newer/partial: immutable checkpoint triples are independently
verified, so an intact epoch 5 can survive a failed later cloud copy. No source
file is modified. Use only your own trusted training backups: PyTorch checkpoint
payloads are loaded by CheckpointStore after both external and internal hashes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid

from checkpoint_store import (CheckpointStore, CheckpointContractMismatch,
    SCHEMA_VERSION, STATE_NAME, MANIFEST_NAME, COMMIT_NAME, contract_sha256, sha256_file)

CONDITIONS = ('M1','M2','M3','M4','M3T','M4T')
MODEL = 'DCRNN_trajectory_directed_H48'
TRAINING_VERSION = 'seoul-dcrnn-m1-m4t-h48-training-20261111-v1'
TRIPLE = {STATE_NAME, MANIFEST_NAME, COMMIT_NAME}


class RestoreError(RuntimeError):
    pass


def _write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _digest(value, *, compact=True):
    kwargs={'sort_keys':True,'ensure_ascii':False,'allow_nan':False,'default':str}
    if compact:kwargs['separators']=(',',':')
    return hashlib.sha256(json.dumps(value,**kwargs).encode('utf-8')).hexdigest()


def _relative(value):
    if (not isinstance(value,str) or not value or '\\' in value or ':' in value
            or value.startswith('/') or any(p in ('','.','..') for p in value.split('/'))):
        raise RestoreError('Marker contains an unsafe relative file path.')
    return PurePosixPath(value)


def _load_marker(marker_path):
    marker=json.loads(Path(marker_path).read_text(encoding='utf-8'))
    if (marker.get('schema_version')!='drive-verified-publication-v1'
            or marker.get('main_copy_verified') is not True
            or marker.get('restore_requires_file_manifest_match') is not True
            or not isinstance(marker.get('files'),list) or not marker['files']):
        raise RestoreError('Unknown/incomplete published backup marker schema.')
    seen=set()
    for item in marker['files']:
        if not isinstance(item,dict):raise RestoreError('Invalid marker file record.')
        relative=_relative(item.get('path'))
        key=str(relative).casefold()
        if key in seen:raise RestoreError('Duplicate or case-colliding marker file paths.')
        seen.add(key)
        if (not isinstance(item.get('bytes'),int) or isinstance(item['bytes'],bool) or item['bytes']<0
                or not re.fullmatch('[0-9a-f]{64}',str(item.get('sha256','')))):
            raise RestoreError('Invalid marker file size/SHA256 schema.')
    if _digest(marker['files'])!=marker.get('manifest_sha256'):
        raise RestoreError('Published marker manifest SHA256 does not match its listed records.')
    return marker


def _file_status(source, item):
    path=source/str(_relative(item['path']))
    if not path.is_file():return 'missing'
    if path.is_symlink() or not path.resolve().is_relative_to(source):
        raise RestoreError('A listed backup source escapes the downloaded batch or is a symlink.')
    for parent in path.parents:
        if parent==source:break
        if parent.is_symlink():raise RestoreError('A listed source has a symlink parent.')
    if path.stat().st_size!=item['bytes']:return 'size_mismatch'
    return 'verified' if sha256_file(path)==item['sha256'] else 'sha256_mismatch'


def _generation(path):
    parts=PurePosixPath(path).parts
    if 'Checkpoints' not in parts:return None
    pos=parts.index('Checkpoints')
    if len(parts)>pos+1 and parts[pos+1].startswith('epoch_'):
        return '/'.join(parts[:pos+2])
    return None


def _generation_metadata(relative):
    parts=PurePosixPath(relative).parts
    if (len(parts)!=8 or parts[0]!='Conditions' or parts[1] not in CONDITIONS
            or parts[3]!='Models' or parts[4]!=MODEL or not re.fullmatch('seed[0-9]+',parts[5])
            or parts[6]!='Checkpoints' or not parts[7].startswith('epoch_')):
        raise RestoreError('Checkpoint location differs from the frozen VastAI model layout: '+relative)
    return {'condition':parts[1],'run':'/'.join(parts[:3]),'seed':'/'.join(parts[:6]),
            'seed_number':int(parts[5][4:]),'generation':parts[7]}


def _required_json(source, verified, relative):
    if relative not in verified:
        raise RestoreError('A required fixed model contract is missing/corrupt or unlisted: '+relative)
    value=json.loads((source/relative).read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise RestoreError('A fixed model contract must be a JSON mapping: '+relative)
    return value


def _model_contract(source, verified, meta):
    seed=meta['seed'];run=meta['run']
    c=_required_json(source,verified,seed+'/contract.json')
    p=_required_json(source,verified,seed+'/preprocessing.json')
    r=_required_json(source,verified,run+'/experiment_config.json')
    required={'version','data','graph','model','architecture','training','preprocessing'}
    if not required<=set(c) or c['version']!=TRAINING_VERSION or c['model']!=MODEL:
        raise RestoreError('Critical model contract version/model schema mismatch.')
    a,t=c['architecture'],c['training']
    if (not isinstance(a,dict) or not isinstance(t,dict)
            or a.get('implementation')!='dcrnn-seoul-paper-aligned-v2'
            or a.get('input_dim')!=2 or a.get('context_dim')!=750
            or a.get('hidden_dim')!=64 or a.get('encoder_layers')!=2 or a.get('decoder_layers')!=2
            or a.get('horizons')!=[3,6,12,24,36,48] or a.get('supervised_node')!=0
            or a.get('decoder_steps')!=48 or a.get('support_self_loops') is not True
            or t.get('experiment_bundle')!='dcrnn-vast-m1-m4t-h48-v1'
            or t.get('input_condition')!=meta['condition'] or t.get('seed')!=meta['seed_number']
            or t.get('mode') not in ('full','pilot')):
        raise RestoreError('Critical model/condition/seed/input tensor schema mismatch.')
    if any(not re.fullmatch('[0-9a-f]{64}',str(c[k])) for k in ('data','graph','preprocessing')):
        raise RestoreError('Critical model contract fingerprint schema mismatch.')
    if (not isinstance(p.get('node_order'),list) or len(p['node_order'])!=635
            or len(set(p['node_order']))!=635 or p['node_order'][0]!='KR:111121'
            or not isinstance(p.get('context_names'),list) or len(p['context_names'])!=375
            or len(set(p['context_names']))!=375):
        raise RestoreError('Critical preprocessing node/context schema mismatch.')
    for key,size in (('pm_mean',635),('pm_std',635),('pm_valid_count',635),
                     ('context_mean',375),('context_std',375),('context_valid_count',375)):
        if not isinstance(p.get(key),list) or len(p[key])!=size:
            raise RestoreError('Critical preprocessing moment schema mismatch: '+key)
    # The trainer uses the normal JSON separators for this particular signature.
    if _digest(p,compact=False)!=c['preprocessing']:
        raise RestoreError('Preprocessing content differs from the checkpoint contract fingerprint.')
    if (r.get('version')!=c['version'] or r.get('data')!=c['data']
            or r.get('preprocessing')!=c['preprocessing']
            or r.get('training',{}).get('experiment_bundle')!='dcrnn-vast-m1-m4t-h48-v1'
            or r.get('training',{}).get('input_condition')!=meta['condition']):
        raise RestoreError('Run and seed model contract schemas disagree.')
    return c


def restore_verified_backup(downloaded_root, marker_path, output_root):
    """Return a detailed restore report. Never modify input or overwrite output."""
    source_arg=Path(downloaded_root).expanduser()
    if source_arg.is_symlink():raise RestoreError('Downloaded batch root cannot be a symlink.')
    source=source_arg.resolve();output=Path(output_root).expanduser().resolve()
    marker_path=Path(marker_path).expanduser().resolve()
    if not source.is_dir():raise RestoreError('Downloaded batch root does not exist.')
    if output.exists():raise FileExistsError('Output must be a NEW folder; existing output is never overwritten.')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise RestoreError('Source and restored output must be independent folders.')
    marker=_load_marker(marker_path)
    verified={};failures=[]
    for item in marker['files']:
        status=_file_status(source,item)
        if status=='verified':verified[item['path']]=item
        else:failures.append({'path':item['path'],'reason':status})
    listed={item['path']:item for item in marker['files']}
    generations=sorted({g for name in listed if (g:=_generation(name)) is not None})
    if not generations:raise RestoreError('This marker contains no model checkpoint; an initial backup alone cannot resume training.')
    grouped={};rejected=[];contracts={}
    for generation in generations:
        meta=_generation_metadata(generation)
        required={generation+'/'+name for name in TRIPLE}
        if not required<=set(listed) or not required<=set(verified):
            rejected.append({'generation':generation,'reason':'incomplete_or_hash_invalid_triple',
                             'missing_or_invalid':sorted(required-set(verified))})
            continue
        if meta['seed'] not in contracts:
            contracts[meta['seed']]=_model_contract(source,verified,meta)
        # Unknown schema / different model settings are not simple corruption.
        manifest=_required_json(source,verified,generation+'/'+MANIFEST_NAME)
        commit=_required_json(source,verified,generation+'/'+COMMIT_NAME)
        if manifest.get('schema_version')!=SCHEMA_VERSION or commit.get('schema_version')!=SCHEMA_VERSION:
            raise RestoreError('Critical checkpoint storage schema mismatch.')
        if manifest.get('contract_sha256')!=contract_sha256(contracts[meta['seed']]):
            raise RestoreError('Critical checkpoint/model contract mismatch; refusing unrelated fallback.')
        grouped.setdefault(meta['seed'],[]).append((generation,meta))
    if not grouped:raise RestoreError('No listed, hash-valid checkpoint triple remains; source preserved.')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.parent/('.pending_restore_'+output.name+'_'+uuid.uuid4().hex[:8])
    staging.mkdir(exist_ok=False)
    copied=set();valid_generations=[];restored=[];skipped=[]
    try:
        for seed,items in grouped.items():
            store=CheckpointStore(staging/seed)
            for generation,meta in items:
                destination=staging/generation;destination.mkdir(parents=True,exist_ok=False)
                for filename in TRIPLE:
                    relative=generation+'/'+filename
                    shutil.copy2(source/relative,staging/relative)
                try:
                    state,manifest=store._validate_directory(destination,contracts[seed])
                except CheckpointContractMismatch as error:
                    raise RestoreError('Critical committed model contract mismatch: '+str(error)) from error
                except Exception as error:
                    rejected.append({'generation':generation,'reason':'internal_checkpoint_validation_failed',
                                     'error_type':type(error).__name__})
                    resolved=destination.resolve()
                    if resolved.parent!=store.root.resolve() or not resolved.is_relative_to(staging):
                        raise RestoreError('Unsafe rejected generation cleanup target.')
                    shutil.rmtree(resolved)
                    continue
                valid_generations.append({'path':generation,'completed_epochs':state['epoch']+1,
                    'best_epoch':state['best_epoch'],'state_sha256':manifest['state_sha256'],
                    'condition':meta['condition'],'seed':meta['seed_number'],'run':meta['run']})
                copied.update(generation+'/'+name for name in TRIPLE)
            state,info=store.load_latest(contracts[seed])
            if state is not None:
                store._write_pointer(info['generation'])
                restored.append({'condition':items[0][1]['condition'],'seed':items[0][1]['seed_number'],
                    'run':items[0][1]['run'],'seed_folder':seed,'generation':info['generation'],
                    'completed_epochs':info['completed_epochs'],'best_epoch':info['best_epoch'],
                    'state_sha256':info['state_sha256']})
            else:
                empty_seed=store.seed_folder.resolve()
                if not empty_seed.is_relative_to(staging):raise RestoreError('Unsafe empty seed cleanup target.')
                shutil.rmtree(empty_seed)
                parent=empty_seed.parent
                while parent!=staging and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir();parent=parent.parent
        if not restored:raise RestoreError('No internally valid checkpoint remains; source preserved.')
        valid_runs={item['run'] for item in restored}
        for relative,item in verified.items():
            parts=PurePosixPath(relative).parts
            if _generation(relative) is not None:continue
            if (relative=='batch_manifest.json' or 'Checkpoints' in parts or relative.endswith('.log')
                    or parts[-1] in ('last.pt','best.pt','execution_status.json','checkpoint_recovery.json')):
                skipped.append({'path':relative,'reason':'mutable_pointer_or_derived_state_not_reused'})
                continue
            if len(parts)>=3 and parts[0]=='Conditions' and '/'.join(parts[:3]) not in valid_runs:
                skipped.append({'path':relative,'reason':'run_has_no_restored_checkpoint'})
                continue
            destination=staging/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/relative,destination)
            if sha256_file(destination)!=item['sha256']:raise RestoreError('Local restore copy checksum mismatch: '+relative)
            copied.add(relative)
        settings=_required_json(source,verified,'resolved_configuration.json')
        if settings.get('schema_version')!='dcrnn-vast-m1-m4t-h48-checkpoint-v1':
            raise RestoreError('The verified resolved configuration belongs to a different experiment schema.')
        restored_modes={contracts[item['seed_folder']]['training']['mode'] for item in restored}
        if len(restored_modes)!=1:
            raise RestoreError('One restored batch cannot mix pilot and full checkpoint modes.')
        restored_mode=next(iter(restored_modes))
        selected=settings.get('selected_conditions',list(CONDITIONS))
        if not isinstance(selected,list) or not selected or not set(selected)<=set(CONDITIONS):
            raise RestoreError('Selected condition schema in the verified configuration is invalid.')
        selected=list(dict.fromkeys(selected+[item['condition'] for item in restored]))
        conditions={}
        for condition in selected:
            runs=sorted({item['run'] for item in restored if item['condition']==condition})
            conditions[condition]={'status':'restored_checkpoint' if runs else 'not_started',
                'run_root':str(output/runs[-1]) if runs else None,
                'restored_seeds':[item for item in restored if item['condition']==condition]}
        unlisted_generations=[]
        for directory in source.rglob('epoch_*'):
            if directory.is_dir() and directory.parent.name=='Checkpoints':
                relative=directory.relative_to(source).as_posix()
                if relative not in generations:unlisted_generations.append(relative)
        report={'schema_version':'verified-drive-restore-v1','restored_utc':datetime.now(timezone.utc).isoformat(),
            'status':'restored_with_differences' if failures or rejected else 'restored_verified_checkpoints',
            'downloaded_root':str(source),'output_root':str(output),'source_files_modified':False,
            'marker_path':str(marker_path),'marker_sha256':sha256_file(marker_path),
            'backup_id':marker.get('backup_id'),'backup_reason':marker.get('reason'),
            'all_listed_files_hash_valid':not failures,'listed_files':len(listed),
            'hash_verified_files':len(verified),'copied_verified_files':len(copied),
            'missing_or_changed_files':failures,'skipped_verified_files':skipped,
            'rejected_generations':rejected,'unlisted_generations_not_copied':sorted(unlisted_generations),
            'verified_checkpoint_generations':valid_generations,'restored_checkpoints':restored,
            'not_started_conditions':[c for c,item in conditions.items() if item['status']=='not_started'],
            'no_unlisted_source_files_copied':True,'mutable_status_and_current_pointers_rebuilt':True}
        _write(staging/'Restoration/published_marker.json',marker)
        _write(staging/'restore_report.json',report)
        _write(staging/'batch_manifest.json',{'status':'restored_for_resume','batch_root':str(output),
            'mode':restored_mode,'validate_only':False,'resume_source':None,
            'model_family':'DCRNN','experiment_schema':'dcrnn-vast-m1-m4t-h48-v1',
            'seeds':sorted({item['seed'] for item in restored}),'final_test':False,
            'selected_conditions':selected,'conditions':conditions,'restoration':{
                'backup_id':marker.get('backup_id'),'report':'restore_report.json',
                'all_listed_files_hash_valid':not failures,'restored_checkpoint_count':len(restored)}})
        if output.exists():raise FileExistsError('Output appeared during verification; refusing overwrite.')
        os.rename(staging,output)
        return report
    except BaseException:
        resolved=staging.resolve()
        if resolved.exists():
            if resolved.parent!=output.parent or not resolved.name.startswith('.pending_restore_'):
                raise RestoreError('Unsafe restore staging cleanup target.')
            shutil.rmtree(resolved)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloaded-root',required=True,type=Path)
    parser.add_argument('--marker',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    report=restore_verified_backup(args.downloaded_root,args.marker,args.output_root)
    print(json.dumps({'status':report['status'],'output_root':report['output_root'],
        'all_listed_files_hash_valid':report['all_listed_files_hash_valid'],
        'restored_checkpoints':report['restored_checkpoints'],
        'missing_or_changed_files':len(report['missing_or_changed_files']),
        'rejected_generations':len(report['rejected_generations']),
        'report':str(args.output_root/'restore_report.json')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
