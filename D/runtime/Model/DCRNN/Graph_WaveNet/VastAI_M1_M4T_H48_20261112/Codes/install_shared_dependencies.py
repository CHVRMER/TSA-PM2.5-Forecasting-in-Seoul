"""Install missing, hash-verified H48 dependencies without overwriting files.

Run: python3 install_shared_dependencies.py --archive /workspace/TSA_DCRNN_H48_Required_Data_20261111.zip --project-root /workspace/TSA
Only Python's standard library is required. No network or credentials are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

SCHEMA = 'dcrnn-h48-required-data-v1'
CODE_MANIFEST = 'Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112/References/upload_manifest.json'
BLOCK = 8 * 1024 * 1024


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('Unsafe dependency path: ' + repr(value))
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts or '.' in path.parts
            or path.as_posix() != value or not path.parts):
        raise ValueError('Unsafe dependency path: ' + repr(value))
    return path


def checked_destination(root, relative):
    relative = safe_relative(relative)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink destination is forbidden: ' + str(current))
    if not candidate.resolve().is_relative_to(root):
        raise ValueError('Dependency escapes the project folder: ' + str(candidate))
    return candidate


def manifest_records(record, *, existing_only=False):
    raw = record.get('files')
    if not isinstance(raw, list):
        raise ValueError('Manifest must contain a files list.')
    records = {}
    for item in raw:
        if existing_only and item.get('role') != 'existing_dependency':
            continue
        name = item.get('path')
        path = safe_relative(name)
        if path.parts[0] not in ('Dataset', 'Model'):
            raise ValueError('Only Dataset/ and declared Model/ dependencies are allowed.')
        size, digest = item.get('bytes'), item.get('sha256')
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid byte count or SHA256: ' + str(name))
        if name in records:
            raise ValueError('Duplicate manifest path: ' + name)
        records[name] = dict(path=name, bytes=size, sha256=digest)
    if not records:
        raise ValueError('No dependencies were declared.')
    return records


def inspect_archive(archive, root):
    # Existing H48 code anchors the exact dependency allowlist and hashes.
    code_manifest = checked_destination(root, CODE_MANIFEST)
    if not code_manifest.is_file():
        raise FileNotFoundError('Extract the H48 code ZIP into this TSA folder first: ' + str(code_manifest))
    allowed = manifest_records(json.loads(code_manifest.read_text(encoding='utf-8-sig')), existing_only=True)
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members are forbidden.')
    for entry in archive.infolist():
        safe_relative(entry.filename)
        if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError('Directory/symlink archive members are forbidden: ' + entry.filename)
    manifest = json.loads(archive.read('_dependencies_manifest.json').decode('utf-8-sig'))
    if manifest.get('schema_version') != SCHEMA:
        raise ValueError('Unexpected dependency archive schema.')
    records = manifest_records(manifest)
    if records != allowed:
        raise ValueError('Dependency archive differs from the installed H48 code manifest.')
    if set(names) != set(records) | {'__main__.py', '_dependencies_manifest.json'}:
        raise ValueError('The archive contains undeclared or missing files.')
    # Validate every source payload BEFORE any destination is written.
    for name, item in records.items():
        info = archive.getinfo(name)
        if info.file_size != item['bytes']:
            raise ValueError('Archive size mismatch: ' + name)
        h = hashlib.sha256()
        count = 0
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(BLOCK), b''):
                h.update(chunk)
                count += len(chunk)
        if count != item['bytes'] or h.hexdigest() != item['sha256']:
            raise ValueError('Archive SHA256 mismatch: ' + name)
    pending, present = [], []
    for name, item in records.items():
        destination = checked_destination(root, name)
        if destination.exists():
            if (not destination.is_file() or destination.stat().st_size != item['bytes']
                    or file_sha256(destination) != item['sha256']):
                raise FileExistsError('Existing file differs; nothing was overwritten: ' + str(destination))
            present.append(item)
        else:
            # A parent must never be a file or other non-directory node.
            for parent in destination.parents:
                if parent == root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise FileExistsError('A dependency parent is not a directory: ' + str(parent))
            pending.append(item)
    return pending, present


def make_parents(root, relative):
    path = safe_relative(relative)
    current = root
    for part in path.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink parent is forbidden: ' + str(current))
        try:
            current.mkdir()
        except FileExistsError:
            if not current.is_dir():
                raise
        if not current.resolve().is_relative_to(root):
            raise ValueError('Parent escapes the project folder.')


def install(archive_path, project_root, *, plan=False):
    project = Path(project_root).expanduser()
    if project.is_symlink() or not project.is_dir():
        raise ValueError('Project root must be an existing, non-symlink directory.')
    root = project.resolve()
    archive_path = Path(archive_path).expanduser().resolve()
    with zipfile.ZipFile(archive_path) as archive:
        pending, present = inspect_archive(archive, root)
        print(json.dumps({'mode': 'plan' if plan else 'install', 'project_root': str(root),
                          'identical_existing_files': len(present), 'missing_files': len(pending),
                          'missing_bytes': sum(x['bytes'] for x in pending)}, ensure_ascii=False), flush=True)
        if plan:
            return {'status': 'planned', 'installed_files': 0, 'missing_files': len(pending),
                    'identical_existing_files': len(present)}
        installed = 0
        for item in pending:
            name = item['path']
            destination = checked_destination(root, name)
            make_parents(root, name)
            # Create a complete temporary file on the destination filesystem.
            # A hard link commits it atomically, refusing an existing name.
            fd, temporary_name = tempfile.mkstemp(prefix='.h48_dependency_', suffix='.pending', dir=destination.parent)
            temporary = Path(temporary_name)
            try:
                digest = hashlib.sha256()
                with os.fdopen(fd, 'wb') as output, archive.open(name) as source:
                    for chunk in iter(lambda: source.read(BLOCK), b''):
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError('Extraction verification failed: ' + name)
                checked_destination(root, name)
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if destination.is_symlink() or not destination.is_file() or file_sha256(destination) != item['sha256']:
                        raise FileExistsError('Destination changed during install; nothing was overwritten: ' + str(destination))
                installed += 1
                print(f'[{installed}/{len(pending)}] verified: {name}', flush=True)
            finally:
                # Remove only this invocation's own temporary file, never a
                # user source, destination, code, checkpoint, or result file.
                temporary.unlink(missing_ok=True)
        for item in pending + present:
            path = checked_destination(root, item['path'])
            if not path.is_file() or path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
                raise ValueError('Final installed-file verification failed: ' + item['path'])
        result = {'status': 'passed', 'installed_files': installed,
                  'identical_existing_files': len(present), 'all_dependencies_verified': len(pending) + len(present),
                  'existing_files_overwritten': False, 'network_used': False}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    try:
        install(args.archive, args.project_root, plan=args.plan)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print('[dependency installation ERROR] ' + str(error), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
