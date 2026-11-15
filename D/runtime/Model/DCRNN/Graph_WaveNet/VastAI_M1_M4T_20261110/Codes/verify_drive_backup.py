"""Offline Drive backup contract checks using an injected fake rclone process."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import subprocess
from drive_backup import DEFAULT_REMOTE, EXCLUDES, DriveBackupError, RcloneDriveBackup


SECRET = 'SENSITIVE_TOKEN_MUST_NEVER_APPEAR_IN_RECEIPTS'


class FakeRclone:
    def __init__(self, *, remote=True, fail=None, timeout=None):
        self.calls=[]; self.remote=remote; self.fail=fail; self.timeout=timeout
        self.counts={}

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        command=argv[1]
        self.counts[command]=self.counts.get(command,0)+1
        assert kwargs['shell'] is False and kwargs['timeout']==600
        assert kwargs['capture_output'] is True and kwargs['check'] is False
        assert 'env' not in kwargs  # Existing external rclone environment is inherited.
        assert command not in ('sync','delete','purge','config')
        if command==self.timeout:
            raise subprocess.TimeoutExpired(argv,600,stderr=SECRET)
        failed=command==self.fail or self.fail==(command,self.counts[command])
        return SimpleNamespace(returncode=3 if failed else 0,
            stdout='gdrive:\n' if command=='listremotes' and self.remote else '', stderr=SECRET)


def make(fake, root, mode='full', **overrides):
    config={'enabled':True,'required_for_full':True,'remote':DEFAULT_REMOTE,'binary':'rclone'}
    config.update(overrides)
    return RcloneDriveBackup({'backup':config},root,mode,process_runner=fake,which=lambda _: '/fake/rclone')


def expect_failure(call):
    try:
        call()
    except DriveBackupError as error:
        assert SECRET not in str(error)
    else:
        raise AssertionError('A failed backup/preflight was reported as successful.')


def main():
    test_parent=Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='.drive_backup_test_',dir=test_parent) as temporary:
        assert Path(temporary).resolve().is_relative_to(test_parent)
        root=Path(temporary)/'unique_batch_1';root.mkdir()
        (root/'checkpoint.pt').write_bytes(b'committed checkpoint')
        (root/'training.log').write_text('live log excluded')
        (root/'save.tmp').write_text('uncommitted temp excluded')
        pending=root/'.pending_epoch';pending.mkdir()
        (pending/'uncommitted.pt').write_bytes(b'excluded')
        fake=FakeRclone(); backup=make(fake,root)
        assert backup.preflight()['status']=='success'
        result=backup.backup('epoch_1')
        assert result['status']=='success' and result['copy_verified']
        assert result['destination']==DEFAULT_REMOTE+'/'+root.name
        assert [call[0][1] for call in fake.calls]==['listremotes','lsf','copy','check','copy','check']
        for argv,_ in fake.calls[2:4]:
            patterns=[argv[i+1] for i,value in enumerate(argv[:-1]) if value=='--exclude']
            assert tuple(patterns)==EXCLUDES
        assert '--one-way' in fake.calls[-1][0]
        assert '--exclude' not in fake.calls[-1][0]
        marker_source=Path(fake.calls[-2][0][2])
        marker=json.loads((marker_source/'verified.json').read_text())
        assert marker['main_copy_verified'] and marker['restore_requires_file_manifest_match']
        assert [row['path'] for row in marker['files']]==['checkpoint.pt']
        assert '/Backup_receipts/Published/' in fake.calls[-2][0][3]
        assert (root/'Backup_receipts/last_successful.json').is_file()
        first_success=(root/'Backup_receipts/last_successful.json').read_text()
        resumed=make(FakeRclone(),root)
        assert resumed.last_successful['status']=='success'
        fake.fail='check'
        expect_failure(lambda:backup.backup('epoch_5'))
        assert (root/'Backup_receipts/last_successful.json').read_text()==first_success
        records=sorted((root/'Backup_receipts').glob('*.json'))
        assert len(records)==4  # preflight, successful save, pointer, failed save.
        assert all(SECRET not in item.read_text() for item in records)
        failed=[json.loads(item.read_text()) for item in records
                if json.loads(item.read_text()).get('status')=='failure'][-1]
        assert failed['status']=='failure' and failed['failed_stage']=='check'
        # A failed marker publication/check must not advance the successful pointer.
        publication=FakeRclone(fail=('check',2));pub=make(publication,root)
        expect_failure(lambda:pub.backup('marker_check_failure'))
        assert (root/'Backup_receipts/last_successful.json').read_text()==first_success
        pub_failures=[json.loads(item.read_text()) for item in (root/'Backup_receipts').glob('*.json')
                      if json.loads(item.read_text()).get('failed_stage')=='verify_receipt']
        assert len(pub_failures)==1 and pub_failures[0]['status']=='failure'
        fake.fail=None;fake.timeout='copy'
        expect_failure(lambda:backup.backup('condition_end',force=True))
        assert (root/'Backup_receipts/last_successful.json').read_text()==first_success
        missing=FakeRclone(remote=False)
        expect_failure(make(missing,root/'missing').preflight)
        missing_root=root/'missing';missing_root.mkdir(exist_ok=True)
        expect_failure(make(missing,missing_root).preflight)
        assert [call[0][1] for call in missing.calls]==['listremotes']
        expect_failure(make(FakeRclone(),root,enabled=False).preflight)
        expect_failure(make(FakeRclone(),root,required_for_full=False).preflight)
        pilot=FakeRclone();p=make(pilot,root,mode='pilot')
        assert p.preflight()['status']=='disabled_for_pilot'
        assert p.backup('pilot_end',force=True)['status']=='disabled_for_pilot'
        assert not pilot.calls
        pilot_on=FakeRclone();p=make(pilot_on,root,mode='pilot',backup_in_pilot=True)
        assert p.preflight()['status']=='success'
        assert p.backup('pilot_save')['status']=='success'
        try:
            make(FakeRclone(),root,token=SECRET)
        except DriveBackupError:
            pass
        else:
            raise AssertionError('Inline credentials were accepted.')
    print(json.dumps({'drive_backup_tests':'passed','copy_then_checksum_check':True,
        'failure_closed':True,'failure_preserves_last_success':True,'no_remote_deletes':True,
        'published_marker_copied_and_verified_before_success':True,'marker_failure_closed':True,
        'unique_remote_batch_folder':True,'exclusions_applied':list(EXCLUDES),
        'no_inline_credentials_or_secret_logs':True,'pilot_disabled_without_opt_in':True}))


if __name__=='__main__':
    main()
