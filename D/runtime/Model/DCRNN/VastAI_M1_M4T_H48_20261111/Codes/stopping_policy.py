"""Strict H48 stopping settings on preserved resume checkpoints.

No weights, optimizer state, RNG or observations are changed. H48 resume never
rebinds epoch/patience limits from a different training configuration.
"""
from __future__ import annotations

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
    """Compatibility API retained; H48 permits no stopping-limit changes."""
    old, new = _limits(old_training), _limits(new_training)
    old_frozen = {k: v for k, v in old_training.items() if k not in STOP_KEYS}
    new_frozen = {k: v for k, v in new_training.items() if k not in STOP_KEYS}
    if old_frozen != new_frozen:
        raise ValueError('Non-stopping training settings differ; cannot reuse this checkpoint.')
    if old != new:
        raise ValueError('H48 stopping settings differ; start a new experiment instead of reusing this checkpoint.')
    return {}


def import_seed_checkpoint(source_store, destination, config, store_factory, write_json, write_curve):
    """Copy an identical H48 seed into a new store; never rebind its settings."""
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
    stopping_reduction(old_training, revised_training)
    report = source_store.copy_to(destination)
    for name in ('contract.json', 'preprocessing.json', 'learning_curve.csv'):
        path = source / name
        if path.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / name)
    return report

