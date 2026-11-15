"""Narrow, audited stopping-limit reductions on preserved resume checkpoints.

No weights, optimizer state, RNG or observations are changed. Rebound state is
written to a NEW empty store; the original source generations remain intact.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

STOP_KEYS = ('max_epochs', 'patience')


def _limits(training):
    result = {}
    for key in STOP_KEYS:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + ' must be a positive integer.')
        result[key] = value
    return result


def stopping_reduction(old_training, new_training):
    """Only lowering the two stopping controls is compatible with this update."""
    old, new = _limits(old_training), _limits(new_training)
    old_frozen = {k: v for k, v in old_training.items() if k not in STOP_KEYS}
    new_frozen = {k: v for k, v in new_training.items() if k not in STOP_KEYS}
    if old_frozen != new_frozen:
        raise ValueError('Non-stopping training settings differ; cannot reuse this checkpoint.')
    if any(new[key] > old[key] for key in STOP_KEYS):
        raise ValueError('Automatic stopping-policy revision permits reductions only.')
    return {key: {'old': old[key], 'new': new[key]} for key in STOP_KEYS if old[key] != new[key]}


def import_seed_checkpoint(source_store, destination, config, store_factory, write_json, write_curve):
    """Copy a compatible seed, or rebind ONLY its stopping limits in a fresh store."""
    destination = Path(destination)
    previous, recovery = source_store.load_latest(None)
    source = Path(source_store.seed_folder)
    if previous is None:
        return {'source_seed_folder': str(source), 'destination_seed_folder': str(destination),
                'status': 'fresh_restart_no_completed_epoch', 'source_files_modified': False}
    old_contract = previous['contract']
    old_training = old_contract['training']
    limits = _limits(config)
    revised_training = dict(old_training, **limits)
    changes = stopping_reduction(old_training, revised_training)
    if not changes:
        report = source_store.copy_to(destination)
        for name in ('contract.json', 'preprocessing.json', 'learning_curve.csv'):
            path = source / name
            if path.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination / name)
        return report

    selected = previous
    selected_generation = recovery.get('generation')
    rejected = []
    if previous['epoch'] + 1 > limits['max_epochs']:
        selected = None
        for folder in source_store._candidates():
            try:
                # Skip clearly newer immutable generations before loading tensors.
                count = int(folder.name.split('_')[1])
                if count > limits['max_epochs']:
                    continue
                candidate, manifest = source_store._validate_directory(folder, old_contract)
                if candidate['epoch'] + 1 <= limits['max_epochs']:
                    selected = candidate
                    selected_generation = manifest['generation']
                    break
            except Exception as error:
                if type(error).__name__ == 'CheckpointContractMismatch':
                    raise
                rejected.append({'generation': folder.name, 'error': str(error)})
        if selected is None:
            raise ValueError('No verified checkpoint at or below the new max_epochs; '
                             'refusing to label a later model as a shorter run.')

    policy = config.get('checkpoint_policy', {})
    target = store_factory(destination, snapshot_every=int(policy.get('snapshot_every_epochs', 5)),
                           keep_recent=int(policy.get('keep_recent', 2)))
    if any(target.root.iterdir()):
        raise ValueError('Stopping-policy revision requires a fresh empty destination store.')
    contract = copy.deepcopy(selected['contract'])
    contract['training'].update(limits)
    rebound = dict(selected, contract=contract)
    rebound.pop('_checkpoint_store', None)
    commit = target.save(rebound)
    revision = {'kind': 'prospective_stopping_limit_reduction', 'changes': changes,
                'source_completed_epochs': previous['epoch'] + 1,
                'selected_completed_epochs': selected['epoch'] + 1,
                'selected_source_generation': selected_generation,
                'bounded_by_new_epoch_limit': previous['epoch'] + 1 > limits['max_epochs'],
                'bad_epochs_preserved': selected['bad_epochs'], 'best_epoch_preserved': selected['best_epoch'],
                'effective_limits': limits, 'source_files_modified': False,
                'optimizer_rng_weights_history_preserved': True,
                'patience_is_prospective_not_retroactive_replay': True,
                'rejected_bounded_generations': rejected}
    write_json(destination / 'contract.json', contract)
    write_json(destination / 'stopping_policy_revision.json', revision)
    write_curve(destination / 'learning_curve.csv', selected['history'])
    if (source / 'preprocessing.json').is_file():
        shutil.copy2(source / 'preprocessing.json', destination / 'preprocessing.json')
    return {'source_seed_folder': str(source), 'destination_seed_folder': str(destination),
            'status': 'stopping_policy_revised', 'source_files_modified': False,
            'source_recovery': recovery, 'imported_commit': commit, 'stopping_policy_revision': revision}
