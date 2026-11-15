"""Crash-consistent, checksummed epoch checkpoints for one model/seed writer.

An immutable generation contains both the current training state and the best
model selected so far. CURRENT.json is only a hint: committed generations are
scanned when it is stale or broken. A failed or partial write cannot replace the
previous committed generation. Local checkpoints cannot survive loss of their
underlying disk; independent backups are a separate concern.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import torch

SCHEMA_VERSION = 'immutable-epoch-current-and-best-v1'
STATE_NAME = 'state.pt'
MANIFEST_NAME = 'manifest.json'
COMMIT_NAME = 'COMMITTED.json'
POINTER_NAME = 'CURRENT.json'


class CheckpointError(RuntimeError):
    """No unsafe automatic fresh start is allowed after this error."""


class CheckpointContractMismatch(CheckpointError):
    pass


class NoValidCheckpoint(CheckpointError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=str)


def contract_sha256(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _sync_directory(directory):
    # Linux/Vast.ai supports directory fsync. Windows rejects opening directories
    # this way; files are still fsynced and os.replace remains atomic there.
    if os.name == 'nt':
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_fsync(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                default=str)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}.')
    return value


def _validate_payload(state, expected_contract=None):
    required = {'contract', 'model', 'best_model', 'optimizer', 'scheduler', 'scaler',
                'epoch', 'best_epoch', 'best', 'bad_epochs', 'global_step', 'history', 'rng'}
    if not isinstance(state, dict) or not required <= set(state):
        raise ValueError('Incomplete self-contained checkpoint payload: '+str(sorted(required-set(state or {}))))
    if not isinstance(state['contract'], dict):
        raise ValueError('Checkpoint contract must be a mapping.')
    if expected_contract is not None and contract_sha256(state['contract']) != contract_sha256(expected_contract):
        raise CheckpointContractMismatch('Checkpoint data/model/optimization contract differs from this run.')
    epoch = _integer(state['epoch'], 'epoch')
    best_epoch = _integer(state['best_epoch'], 'best_epoch', 1)
    if best_epoch > epoch+1:
        raise ValueError('Best epoch cannot be newer than the completed training epoch.')
    if not math.isfinite(float(state['best'])):
        raise ValueError('Best validation score must be finite.')
    _integer(state['bad_epochs'], 'bad_epochs')
    _integer(state['global_step'], 'global_step')
    history = state['history']
    if not isinstance(history, list) or len(history) != epoch+1:
        raise ValueError('History must cover every completed epoch exactly once.')
    if [row.get('epoch') for row in history] != list(range(1, epoch+2)):
        raise ValueError('History epoch labels must be one-based and contiguous.')
    selected_row = history[best_epoch-1]
    selection_horizon = int(state['contract'].get('training', {}).get('selection_horizon', 6))
    if 'selection_horizon' in selected_row and int(selected_row['selection_horizon']) != selection_horizon:
        raise ValueError('Selected horizon differs between checkpoint contract and history.')
    score_key = ('validation_selection_score' if 'validation_selection_score' in selected_row
                 else 'validation_rmse_h6')
    if selection_horizon != 6 and score_key != 'validation_selection_score':
        raise ValueError('A non-6h selection requires its explicit validation selection score.')
    if score_key in selected_row and float(selected_row[score_key]) != float(state['best']):
        raise ValueError('Best score does not match the selected epoch in learning history.')
    if 'global_step' in history[-1] and history[-1]['global_step'] != state['global_step']:
        raise ValueError('Global optimizer step does not match the completed learning history.')
    models = state['model'], state['best_model']
    if not all(isinstance(model, dict) and model for model in models) or set(models[0]) != set(models[1]):
        raise ValueError('Current and best models must contain the same nonempty state keys.')
    for key in models[0]:
        a, b = models[0][key], models[1][key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('Model state entries must be tensors: '+key)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Current/best tensor schema mismatch: '+key)
        if b.device.type != 'cpu':
            raise ValueError('best_model must be a detached CPU clone before save: '+key)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError('Nonfinite current/best model tensor: '+key)
        if best_epoch == epoch+1 and not torch.equal(a.detach().cpu(), b):
            raise ValueError('A newly selected best model must equal this completed epoch: '+key)
    rng = state['rng']
    if not isinstance(rng, dict) or not {'python','numpy','torch','cuda'} <= set(rng):
        raise ValueError('Python, NumPy, Torch and CUDA RNG states are all required.')
    if not isinstance(state['optimizer'], dict) or not isinstance(state['scaler'], dict):
        raise ValueError('Optimizer and AMP scaler states must be mappings.')
    return state


class CheckpointStore:
    """Single-writer storage for one seed; directory names are never reused."""
    def __init__(self, seed_folder, snapshot_every=5, keep_recent=2):
        self.seed_folder = Path(seed_folder).expanduser().resolve()
        self.root = self.seed_folder/'Checkpoints'
        self.snapshot_every = _integer(snapshot_every, 'snapshot_every', 1)
        self.keep_recent = _integer(keep_recent, 'keep_recent', 1)
        self.root.mkdir(parents=True, exist_ok=True)

    def _fault_hook(self, stage, directory):
        """No-op extension point used only by subprocess crash-injection tests."""

    def _candidates(self):
        # Generation names start with a zero-padded completed-epoch count, then
        # nanosecond time and random suffix; different saves never overwrite.
        return sorted((p for p in self.root.glob('epoch_*') if p.is_dir()),
                      key=lambda p:p.name, reverse=True)

    def _validate_directory(self, folder, expected_contract=None, map_location='cpu'):
        folder = Path(folder)
        folder.resolve().relative_to(self.root.resolve())
        if folder.is_symlink():
            raise ValueError('Checkpoint generation directories cannot be symlinks.')
        marker = _read_json(folder/COMMIT_NAME)
        if marker.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unknown checkpoint commit schema.')
        if sha256_file(folder/MANIFEST_NAME) != marker.get('manifest_sha256'):
            raise ValueError('Checkpoint manifest checksum mismatch.')
        manifest = _read_json(folder/MANIFEST_NAME)
        if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('generation') != folder.name:
            raise ValueError('Checkpoint manifest schema/generation mismatch.')
        if marker.get('generation') != folder.name:
            raise ValueError('Checkpoint commit generation mismatch.')
        if expected_contract is not None and manifest.get('contract_sha256') != contract_sha256(expected_contract):
            raise CheckpointContractMismatch('Committed checkpoint contract differs; refusing cross-run fallback.')
        path = folder/STATE_NAME
        if path.is_symlink() or path.stat().st_size != manifest.get('state_bytes'):
            raise ValueError('Checkpoint state size mismatch.')
        if sha256_file(path) != manifest.get('state_sha256'):
            raise ValueError('Checkpoint state checksum mismatch.')
        # These files are our own trusted training artifacts. Never load an
        # untrusted external .pt merely because it has a matching filename.
        state = torch.load(path, map_location='cpu', weights_only=False)
        _validate_payload(state, expected_contract)
        metadata = state.get('_checkpoint_store', {})
        for key in ('generation','schema_version','epoch','best_epoch','contract_sha256'):
            if metadata.get(key) != manifest.get(key):
                raise ValueError('Checkpoint payload/manifest metadata mismatch: '+key)
        if state['epoch'] != manifest['epoch'] or state['best_epoch'] != manifest['best_epoch'] or float(state['best']) != manifest['best_score']:
            raise ValueError('Checkpoint epoch/best selection disagrees with its manifest.')
        if contract_sha256(state['contract']) != manifest['contract_sha256']:
            raise CheckpointContractMismatch('Payload contract and committed manifest differ.')
        if map_location != 'cpu':
            # Loading once more preserves torch.load's standard map_location
            # semantics for CUDA and callable/device maps. Validation remains on
            # CPU so the embedded best model can safely stay a CPU clone.
            state = torch.load(path, map_location=map_location, weights_only=False)
            state['best_model'] = {key:value.detach().cpu().clone()
                                   for key,value in state['best_model'].items()}
        return state, manifest

    def _pointer(self):
        path = self.root/POINTER_NAME
        if not path.exists():
            return None, 'missing' if self._candidates() else None
        try:
            pointer = _read_json(path)
            if not isinstance(pointer.get('generation'), str) or pointer.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('Invalid checkpoint pointer schema.')
            return pointer, None
        except (OSError, ValueError, TypeError) as error:
            return None, f'{type(error).__name__}: {error}'

    def load_latest(self, expected_contract, map_location='cpu'):
        failures = []
        candidates = self._candidates()
        pointer, pointer_error = self._pointer()
        pending = sorted(p.name for p in self.root.glob('.pending_*') if p.is_dir())
        for folder in candidates:
            try:
                state, manifest = self._validate_directory(folder, expected_contract, map_location)
            except CheckpointContractMismatch:
                # Different data/seed/model/training settings are not corruption
                # and must never trigger fallback to an unrelated older run.
                raise
            except Exception as error:
                failures.append({'generation':folder.name,'error_type':type(error).__name__,'reason':str(error)})
                continue
            pointer_stale = pointer is None or pointer.get('generation') != folder.name
            info = {'status':'restored','schema_version':SCHEMA_VERSION,
                    'generation':folder.name,'generation_dir':str(folder),'state_path':str(folder/STATE_NAME),
                    'state_sha256':manifest['state_sha256'],'epoch':state['epoch'],
                    'completed_epochs':state['epoch']+1,'best_epoch':state['best_epoch'],
                    'best_score':float(state['best']),'fallback_used':bool(failures),
                    'rejected_generations':failures,'pointer_stale':pointer_stale,
                    'pointer_error':pointer_error,'ignored_pending_generations':pending,
                    'snapshot':manifest['snapshot'],'pinned':manifest['pinned']}
            if failures or pointer_stale:
                print(f'[checkpoint recovery] restored epoch {state["epoch"]+1}; '
                      f'rejected={len(failures)}; pointer_stale={pointer_stale}; {folder}',flush=True)
            return state, info
        if (self.seed_folder/'last.pt').exists() or (self.seed_folder/'best.pt').exists():
            raise NoValidCheckpoint('Legacy separate last.pt/best.pt cannot prove a crash-consistent current/best pair. '
                                    'Use a new strict run or an explicitly verified migration; automatic import is disabled.')
        existing = list(self.root.iterdir())
        if existing:
            raise NoValidCheckpoint('Checkpoint artifacts exist but no valid committed state remains. '
                                    'Refusing an automatic fresh start. Rejections='+canonical_json(failures)+
                                    '; pending='+canonical_json(pending))
        return None, {'status':'fresh','schema_version':SCHEMA_VERSION,'fallback_used':False,
                      'rejected_generations':[],'pointer_stale':False,'ignored_pending_generations':[]}

    def _write_pointer(self, generation):
        temp = self.root/('.pointer_'+uuid.uuid4().hex+'.tmp')
        _write_json_fsync(temp, {'schema_version':SCHEMA_VERSION,'generation':generation,
                                'written_utc':datetime.now(timezone.utc).isoformat()})
        os.replace(temp, self.root/POINTER_NAME)
        _sync_directory(self.root)

    def save(self, payload, pin=False):
        _validate_payload(payload)
        previous, _ = self.load_latest(payload['contract'])
        if previous is not None and payload['epoch'] < previous['epoch']:
            raise CheckpointError('Cannot append an older epoch behind the latest verified checkpoint.')
        completed = payload['epoch']+1
        name = f'epoch_{completed:06d}_{time.time_ns():020d}_{uuid.uuid4().hex[:8]}'
        pending = self.root/('.pending_'+uuid.uuid4().hex)
        pending.mkdir(exist_ok=False)
        contract_hash = contract_sha256(payload['contract'])
        metadata = {'schema_version':SCHEMA_VERSION,'generation':name,'epoch':payload['epoch'],
                    'best_epoch':payload['best_epoch'],'contract_sha256':contract_hash}
        state = dict(payload, _checkpoint_store=metadata)
        with (pending/STATE_NAME).open('xb') as stream:
            torch.save(state, stream)
            stream.flush(); os.fsync(stream.fileno())
        self._fault_hook('after_state_fsync', pending)
        manifest = dict(metadata, completed_epochs=completed, best_score=float(payload['best']),
                        state_bytes=(pending/STATE_NAME).stat().st_size,
                        state_sha256=sha256_file(pending/STATE_NAME),
                        snapshot=completed % self.snapshot_every == 0, pinned=bool(pin),
                        created_utc=datetime.now(timezone.utc).isoformat(), writer_pid=os.getpid())
        _write_json_fsync(pending/MANIFEST_NAME, manifest)
        self._fault_hook('after_manifest_fsync', pending)
        _write_json_fsync(pending/COMMIT_NAME, {'schema_version':SCHEMA_VERSION,'generation':name,
                                               'manifest_sha256':sha256_file(pending/MANIFEST_NAME)})
        _sync_directory(pending)
        self._fault_hook('after_commit_marker', pending)
        final = self.root/name
        os.replace(pending, final)
        _sync_directory(self.root)
        self._fault_hook('after_generation_rename', final)
        # Validate the serialized bytes before publishing the pointer/pruning.
        self._validate_directory(final, payload['contract'])
        self._write_pointer(name)
        self._fault_hook('after_pointer_replace', final)
        self._fault_hook('before_prune', final)
        pruned = self._prune(payload['contract'])
        self._fault_hook('after_prune', final)
        return {'generation':name,'generation_dir':str(final),'state_path':str(final/STATE_NAME),
                'state_sha256':manifest['state_sha256'],'completed_epochs':completed,
                'best_epoch':payload['best_epoch'],'snapshot':manifest['snapshot'],
                'pinned':manifest['pinned'],'pruned_generations':pruned}

    def _prune(self, expected_contract):
        candidates = self._candidates(); keep=set(); recent_epochs=set(); records=[]
        for folder in candidates:
            try:
                # Preserve corrupt artifacts for diagnosis. Only verified files
                # written by this store may be automatically retained/pruned.
                _, manifest = self._validate_directory(folder, expected_contract)
            except CheckpointContractMismatch:
                raise
            except Exception:
                keep.add(folder.name); continue
            records.append((folder, manifest))
            if len(recent_epochs) < self.keep_recent or manifest['epoch'] in recent_epochs:
                recent_epochs.add(manifest['epoch']); keep.add(folder.name)
            if manifest['snapshot'] or manifest['pinned']:
                keep.add(folder.name)
        pruned=[]
        for folder, _ in records:
            if folder.name in keep:
                continue
            resolved=folder.resolve();resolved.relative_to(self.root.resolve())
            if resolved.parent != self.root.resolve() or not folder.name.startswith('epoch_'):
                raise CheckpointError('Unsafe checkpoint retention target.')
            shutil.rmtree(resolved); pruned.append(folder.name)
        if pruned:_sync_directory(self.root)
        return pruned

    def copy_to(self, destination_seed_folder):
        """Copy verified immutable generations into an empty new seed folder."""
        state, latest_info = self.load_latest(None)
        if state is None:
            raise NoValidCheckpoint('No committed checkpoint exists to copy.')
        contract=state['contract']
        destination=CheckpointStore(destination_seed_folder,self.snapshot_every,self.keep_recent)
        if destination.root == self.root or list(destination.root.iterdir()):
            raise CheckpointError('Resume destination must be a different empty Checkpoints directory.')
        copied=[]; rejected=[]
        for source in reversed(self._candidates()):
            try:
                _, manifest=self._validate_directory(source,contract)
            except CheckpointContractMismatch:
                raise
            except Exception as error:
                rejected.append({'generation':source.name,'reason':str(error)});continue
            pending=destination.root/('.pending_'+uuid.uuid4().hex);pending.mkdir(exist_ok=False)
            for filename in (STATE_NAME,MANIFEST_NAME,COMMIT_NAME):
                path=pending/filename
                shutil.copy2(source/filename,path)
                # Windows requires a writable descriptor for FlushFileBuffers;
                # Linux also accepts this mode without changing file contents.
                with path.open('r+b') as stream:os.fsync(stream.fileno())
            _sync_directory(pending)
            final=destination.root/source.name
            os.replace(pending,final);_sync_directory(destination.root)
            destination._validate_directory(final,contract)
            copied.append({'generation':source.name,'state_sha256':manifest['state_sha256']})
        restored, info=destination.load_latest(contract)
        if restored is None or info['state_sha256'] != latest_info['state_sha256']:
            raise CheckpointError('Copied resume state differs from the verified source generation.')
        destination._write_pointer(info['generation'])
        report={'schema_version':SCHEMA_VERSION,'source_seed_folder':str(self.seed_folder),
                'destination_seed_folder':str(destination.seed_folder),'source_files_modified':False,
                'copied_generations':copied,'rejected_generations':rejected,
                'source_recovery':latest_info,'restored_generation':info['generation']}
        _write_json_fsync(destination.root/('IMPORT_'+uuid.uuid4().hex+'.json'),report)
        _sync_directory(destination.root)
        return report
