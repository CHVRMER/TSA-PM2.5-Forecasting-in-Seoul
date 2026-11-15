"""Verified, synchronous Google Drive backup through externally configured rclone.

Only ``rclone copy`` and a one-way checksum ``rclone check`` write/check results.
There is no remote sync, delete, purge, credential generation, or credential log.
Training must stop if an enabled backup fails; committed local checkpoints remain.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid


DEFAULT_REMOTE = 'gdrive:Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI_H48'
EXCLUDES = ('.pending*/**', '**/.pending*/**', '*.tmp', '*.log', 'Backup_receipts/**')
_SENSITIVE_KEYS = {'token', 'access_token', 'refresh_token', 'api_key', 'password',
                   'client_secret', 'private_key', 'credentials', 'rclone_config',
                   'config_text', 'config_contents'}


class DriveBackupError(RuntimeError):
    """A backup/preflight failed; its message deliberately omits command output."""


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _assert_no_credentials(value):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if str(key).lower() in _SENSITIVE_KEYS:
            raise DriveBackupError('Inline backup credentials are forbidden; configure rclone externally.')
        if isinstance(item, dict):
            _assert_no_credentials(item)


class RcloneDriveBackup:
    """Back up one unique execution folder without overwriting other executions.

    ``config['backup']`` accepts enabled (default true), required_for_full (true),
    remote, binary, timeout_seconds (1..600), backup_in_pilot (false), retries (2),
    low_level_retries (3), transfers (4), and checkers (4). Credentials are read by
    rclone from its existing external config / RCLONE_CONFIG environment only.

    The caller chooses checkpoint intervals. Every ``backup`` call synchronously
    copies and verifies the whole committed batch before returning ``success``.
    ``force`` is recorded for final saves; it never enables pilot cloud access.
    Optional injected process/which callables exist solely for offline tests.
    """

    def __init__(self, config, batch_root, mode, *, process_runner=None, which=None):
        self.config = dict(config.get('backup', {}))
        _assert_no_credentials(self.config)
        self.batch_root = Path(batch_root).expanduser().resolve()
        self.mode = str(mode).lower()
        if self.mode not in ('pilot', 'full'):
            raise ValueError('Backup mode must be pilot or full.')
        self.enabled = bool(self.config.get('enabled', True))
        self.required_for_full = bool(self.config.get('required_for_full', True))
        self.backup_in_pilot = bool(self.config.get('backup_in_pilot', False))
        self.active = self.enabled and (self.mode == 'full' or self.backup_in_pilot)
        self.remote = str(self.config.get('remote', DEFAULT_REMOTE)).strip()
        self.binary = str(self.config.get('binary', 'rclone')).strip()
        self.timeout_seconds = int(self.config.get('timeout_seconds', 600))
        if not 1 <= self.timeout_seconds <= 600:
            raise ValueError('Backup timeout_seconds must lie between 1 and 600.')
        self._run_process = process_runner or subprocess.run
        self._which = which or shutil.which
        self._preflight_ok = False
        self._resolved_binary = None
        self._remote_name = None
        self.destination = None
        self.receipts_root = self.batch_root/'Backup_receipts'
        self.last_successful = None
        pointer = self.receipts_root/'last_successful.json'
        if pointer.is_file():
            try:
                previous = json.loads(pointer.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                previous = {}
            if (previous.get('operation') == 'backup' and previous.get('status') == 'success'
                    and previous.get('batch_name') == self.batch_root.name):
                self.last_successful = previous

    def _record(self, record):
        """Unique receipts; only the explicit last-success pointer is replaced."""
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        record = dict(record, timestamp_utc=_stamp(), mode=self.mode,
                      enabled=self.active, required_for_full=self.required_for_full,
                      batch_name=self.batch_root.name)
        if self.destination is not None:
            record['destination'] = self.destination
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]+'.json'
        path = self.receipts_root/name
        payload = json.dumps(record, ensure_ascii=False, indent=2)+'\n'
        path.write_text(payload, encoding='utf-8')
        if record.get('operation') == 'backup' and record.get('status') == 'success':
            pointer = self.receipts_root/'last_successful.json'
            temporary = self.receipts_root/(uuid.uuid4().hex+'.tmp')
            temporary.write_text(payload, encoding='utf-8')
            os.replace(temporary, pointer)
            self.last_successful = record
        return record

    def _flags(self):
        values = {}
        for key, default in (('retries', 2), ('low_level_retries', 3),
                             ('transfers', 4), ('checkers', 4)):
            value = int(self.config.get(key, default))
            if not 1 <= value <= 16:
                raise ValueError('Backup retry/concurrency settings must lie between 1 and 16.')
            values[key] = value
        return ['--ask-password=false', '--contimeout=30s', '--timeout=60s',
                '--retries='+str(values['retries']),
                '--low-level-retries='+str(values['low_level_retries']),
                '--retries-sleep=5s', '--transfers='+str(values['transfers']),
                '--checkers='+str(values['checkers'])]

    def _command(self, operation, arguments, *, output_needed=False):
        argv = [self._resolved_binary, operation]+list(arguments)+self._flags()
        try:
            result = self._run_process(argv, capture_output=True, text=True,
                                       timeout=self.timeout_seconds, check=False, shell=False)
        except subprocess.TimeoutExpired:
            raise DriveBackupError('rclone '+operation+' timed out; no verified cloud save was recorded.') from None
        except OSError:
            raise DriveBackupError('rclone '+operation+' could not start; check the external rclone installation.') from None
        if result.returncode != 0:
            # rclone errors may contain auth tokens or request URLs. Never echo/store them.
            raise DriveBackupError('rclone '+operation+' failed (exit '+str(result.returncode)+'); cloud backup is not verified.')
        return result.stdout if output_needed else None

    def _file_manifest(self):
        """Inventory the exact main-copy files; omit transient/receipt files."""
        records = []
        for directory, dirs, files in os.walk(self.batch_root, topdown=True, followlinks=False):
            base = Path(directory)
            allowed = []
            for name in dirs:
                if (name.startswith('.pending') or name.endswith(('.tmp', '.log'))
                        or (base == self.batch_root and name == 'Backup_receipts')):
                    continue
                if (base/name).is_symlink():
                    raise DriveBackupError('Backup outputs contain a directory symlink; refusing an ambiguous restore manifest.')
                allowed.append(name)
            dirs[:] = allowed
            for name in sorted(files):
                if name.endswith(('.tmp', '.log')):
                    continue
                path = base/name
                if path.is_symlink():
                    raise DriveBackupError('Backup outputs contain a file symlink; refusing an ambiguous restore manifest.')
                before = path.stat()
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(8*1024*1024), b''):
                        digest.update(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise DriveBackupError('A committed output changed while hashing; retry after its writer finishes.')
                records.append({'path':path.relative_to(self.batch_root).as_posix(),
                                'bytes':after.st_size, 'sha256':digest.hexdigest()})
        return sorted(records, key=lambda item:item['path'])

    def preflight(self):
        """Fail before full training unless rclone and the configured Drive work."""
        try:
            if self.mode == 'full' and (not self.enabled or not self.required_for_full):
                raise DriveBackupError('Full experiments require enabled, mandatory Google Drive backup.')
            if not self.active:
                return self._record({'operation':'preflight', 'status':'disabled_for_pilot',
                                     'reason':'backup_in_pilot is false; pilot remains local'})
            if not self.batch_root.is_dir():
                raise DriveBackupError('The committed batch output directory does not exist.')
            match = re.fullmatch(r'([A-Za-z0-9_-]+):(.*)', self.remote)
            if not match or '\x00' in self.remote or '..' in match.group(2).replace('\\','/').split('/'):
                raise DriveBackupError('Set backup.remote to a named rclone remote and safe folder path.')
            self._remote_name = match.group(1)+':'
            self.destination = self.remote.rstrip('/')+'/'+self.batch_root.name
            self._resolved_binary = self._which(self.binary)
            if not self._resolved_binary:
                raise DriveBackupError('rclone is not installed or the configured binary cannot be found.')
            remotes = self._command('listremotes', [], output_needed=True)
            if self._remote_name not in {line.strip() for line in remotes.splitlines()}:
                raise DriveBackupError('The requested rclone remote is not externally configured.')
            # Read-only access check at remote root, so a new experiment folder need not exist.
            self._command('lsf', [self._remote_name, '--max-depth=1', '--dirs-only'])
            self._preflight_ok = True
            return self._record({'operation':'preflight', 'status':'success',
                                 'configured_remote':self._remote_name, 'remote_access_checked':True})
        except (DriveBackupError, ValueError) as error:
            self._preflight_ok = False
            self._record({'operation':'preflight', 'status':'failure', 'error':str(error)})
            raise

    def backup(self, reason, force=False):
        """Return success only after checksum verification; otherwise raise."""
        if not self.active:
            if self.mode == 'full':
                self.preflight()  # Full may never silently downgrade to local-only.
            return self._record({'operation':'backup', 'status':'disabled_for_pilot',
                                 'reason':str(reason), 'force':bool(force)})
        if not self._preflight_ok:
            self.preflight()
        stage = 'copy'
        try:
            manifest = self._file_manifest()
            filters = [item for pattern in EXCLUDES for item in ('--exclude', pattern)]
            self._command('copy', [str(self.batch_root), self.destination]+filters)
            stage = 'check'
            # rclone check compares common file hashes; local files + Drive use MD5.
            # --one-way tolerates old files already in this batch; nothing is deleted.
            self._command('check', [str(self.batch_root), self.destination, '--one-way']+filters)
            if self._file_manifest() != manifest:
                raise DriveBackupError('Committed outputs changed during backup; no verified publication was recorded.')
            manifest_hash = hashlib.sha256(json.dumps(manifest, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
            backup_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
            marker_folder = self.receipts_root/'Published'/backup_id
            marker_folder.mkdir(parents=True, exist_ok=False)
            marker = {'schema_version':'drive-verified-publication-v1', 'backup_id':backup_id,
                'timestamp_utc':_stamp(), 'batch_name':self.batch_root.name,
                'destination':self.destination, 'reason':str(reason), 'main_copy_verified':True,
                'verification':'rclone check --one-way (common hashes; Google Drive MD5)',
                'excluded_patterns':list(EXCLUDES), 'files':manifest, 'manifest_sha256':manifest_hash,
                'restore_requires_file_manifest_match':True,
                'restore_note':'A later failed copy can change mutable files in this batch. Verify every listed SHA256 before treating this publication as a restorable complete batch.'}
            (marker_folder/'verified.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            marker_remote = self.destination+'/Backup_receipts/Published/'+backup_id
            stage = 'publish_receipt'
            self._command('copy', [str(marker_folder), marker_remote])
            stage = 'verify_receipt'
            self._command('check', [str(marker_folder), marker_remote, '--one-way'])
            record = self._record({'operation':'backup', 'status':'success',
                'reason':str(reason), 'force':bool(force), 'copy_verified':True,
                'verification':'rclone check --one-way (common hashes; Google Drive MD5)',
                'excluded_patterns':list(EXCLUDES), 'remote_deletion_used':False,
                'published_receipt_verified':True, 'backup_id':backup_id,
                'published_receipt_remote':marker_remote+'/verified.json',
                'manifest_sha256':manifest_hash, 'manifest_file_count':len(manifest)})
            print('[Drive backup] verified: '+str(reason)+' -> '+self.destination, flush=True)
            return record
        except (DriveBackupError, ValueError, OSError) as error:
            safe_error = ('Local output or publication files could not be read/written; cloud backup is not verified.'
                          if isinstance(error, OSError) else str(error))
            self._record({'operation':'backup', 'status':'failure', 'reason':str(reason),
                          'force':bool(force), 'failed_stage':stage, 'copy_verified':False,
                          'error':safe_error, 'last_successful_utc':
                              self.last_successful.get('timestamp_utc') if self.last_successful else None})
            if isinstance(error, OSError):
                raise DriveBackupError(safe_error) from None
            raise
