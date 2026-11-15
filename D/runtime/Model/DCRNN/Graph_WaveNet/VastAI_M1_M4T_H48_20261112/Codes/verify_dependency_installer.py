from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('h48_dependency_installer', HERE/'install_shared_dependencies.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.base = Path(self.tmp.name)
        self.root = self.base/'TSA'
        self.root.mkdir()
        self.payload = {'Dataset/A/sample.parquet': b'synthetic-observations',
                        'Model/Common/grid.py': b'# synthetic old common\n'}
        self.records = [{'path': k, 'bytes': len(v), 'sha256': hash_bytes(v)} for k, v in self.payload.items()]
        self.code = self.root/installer.CODE_MANIFEST
        self.code.parent.mkdir(parents=True)
        self.code.write_text(json.dumps({'files': [dict(x, role='existing_dependency') for x in self.records]}), encoding='utf-8')
        self.archive = self.base/'dependencies.zip'
        self.write_archive()

    def tearDown(self):
        self.tmp.cleanup()

    def write_archive(self, *, payload=None, records=None, extra=None):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('__main__.py', (HERE/'install_shared_dependencies.py').read_bytes())
            z.writestr('_dependencies_manifest.json', json.dumps({'schema_version': installer.SCHEMA, 'files': records or self.records}))
            for name, data in (payload or self.payload).items():
                z.writestr(name, data)
            if extra:
                z.writestr(*extra)

    def file_state(self):
        return {str(p.relative_to(self.root)): hash_bytes(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}

    def test_first_install_repeat_and_code_preservation(self):
        code_hash = hash_bytes(self.code.read_bytes())
        a = installer.install(self.archive, self.root)
        self.assertEqual(a['installed_files'], 2)
        first = self.file_state()
        b = installer.install(self.archive, self.root)
        self.assertEqual(b['installed_files'], 0)
        self.assertEqual(b['identical_existing_files'], 2)
        self.assertEqual(first, self.file_state())
        self.assertEqual(code_hash, hash_bytes(self.code.read_bytes()))

    def test_plan_writes_nothing(self):
        before = self.file_state()
        result = installer.install(self.archive, self.root, plan=True)
        self.assertEqual(result['missing_files'], 2)
        self.assertEqual(before, self.file_state())

    def test_conflict_preflight_leaves_all_files_unchanged(self):
        conflict = self.root/'Model/Common/grid.py'
        conflict.parent.mkdir(parents=True, exist_ok=True)
        conflict.write_bytes(b'user-modified code')
        before = self.file_state()
        with self.assertRaises(FileExistsError):
            installer.install(self.archive, self.root)
        self.assertEqual(before, self.file_state())
        self.assertFalse((self.root/'Dataset').exists())

    def test_tampered_last_payload_rejected_before_first_write(self):
        tampered = dict(self.payload)
        tampered['Model/Common/grid.py'] = b'bad replacement'
        self.write_archive(payload=tampered)
        before = self.file_state()
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)
        self.assertEqual(before, self.file_state())

    def test_outside_archive_path_rejected(self):
        self.write_archive(extra=('../outside.txt', b'bad'))
        before = self.file_state()
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)
        self.assertEqual(before, self.file_state())
        self.assertFalse((self.base/'outside.txt').exists())

    def test_duplicate_members_rejected(self):
        self.write_archive(extra=('Model/Common/grid.py', b'duplicate'))
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)
        self.assertFalse((self.root/'Dataset').exists())

    def test_disallowed_manifest_file_rejected(self):
        records = [dict(self.records[0], path='Result/checkpoint.pt'), self.records[1]]
        self.write_archive(records=records)
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)

    def test_different_allowlist_rejected(self):
        records = [dict(self.records[0], sha256='0'*64), self.records[1]]
        self.write_archive(records=records)
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)

    def test_missing_code_manifest_rejected(self):
        other = self.base/'empty'
        other.mkdir()
        with self.assertRaises(FileNotFoundError):
            installer.install(self.archive, other)
        self.assertEqual(list(other.iterdir()), [])

    def test_zipapp_cli(self):
        completed = subprocess.run([sys.executable, str(self.archive), '--archive', str(self.archive), '--project-root', str(self.root)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertTrue((self.root/'Dataset/A/sample.parquet').is_file())

    @unittest.skipUnless(hasattr(os, 'symlink'), 'No symlink API')
    def test_symlink_parent_rejected(self):
        outside = self.base/'outside'
        outside.mkdir()
        try:
            os.symlink(outside, self.root/'Dataset', target_is_directory=True)
        except OSError:
            self.skipTest('Symlink creation requires unavailable Windows privilege')
        with self.assertRaises(ValueError):
            installer.install(self.archive, self.root)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
