"""Download owned public/shared Drive ZIPs with gdown; install without overwrite.

Downloads only: rclone remains responsible for authenticated result backups.
An explicit mounted-volume TSA path is required. No browser cookies, Google
tokens, model packages, training code, existing datasets, or results are changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.parse import urlsplit, parse_qs
import uuid
import zipfile

DATA_SHA256='25f7f447ad0327ea3803873f39d6aa79db8aa1353fc7b44bd930a9cd1691310f'
CODE_SHA256='4059f4a012ee6eb2bea309ae430180a64a31c631588c39d578526e5b44c5e6f6'
KNOWN_BYTES={DATA_SHA256:477518306,CODE_SHA256:105355}
PACKAGE_NAMES={'data':'TSA_GraphWaveNet_Runpod_M1_M4T_20261110.zip',
               'code':'TSA_GraphWaveNet_VastAI_M1_M4T_20261110.zip'}
CODE_PREFIX='Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/'
REQUIRED={'data':(
    'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet',
    'Dataset/Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet',
    'Dataset/Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet',
    'Dataset/Trajectory_Matching_Expansion_QC_20261108/trajectory_station_neighbors.parquet',
    'Model/Common/score.py',
    'Model/DCRNN/Neighbor_Expansion_QC_20261108/Codes/neighbor_adapter.py'),
    'code':tuple(CODE_PREFIX+'Codes/'+name for name in
                 ('run_experiments.py','checkpoint_store.py','drive_backup.py','restore_verified_backup.py'))}


class DownloadBootstrapError(RuntimeError):
    pass


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def drive_file_id(url):
    value=str(url).strip()
    if re.fullmatch('[A-Za-z0-9_-]{10,}',value):return value
    parsed=urlsplit(value)
    if parsed.scheme!='https' or parsed.hostname not in ('drive.google.com','docs.google.com','drive.usercontent.google.com'):
        raise DownloadBootstrapError('Use a single Google Drive file share link or file ID.')
    if '/folders/' in parsed.path:raise DownloadBootstrapError('Supply a ZIP file link, not a Drive folder link.')
    match=re.search(r'/file/d/([A-Za-z0-9_-]+)(?:/|$)',parsed.path)
    candidate=match.group(1) if match else parse_qs(parsed.query).get('id',[''])[0]
    if not re.fullmatch('[A-Za-z0-9_-]{10,}',candidate):
        raise DownloadBootstrapError('Could not identify a Drive ZIP file ID.')
    return candidate


def _checksum(value):
    value=str(value).lower()
    if not re.fullmatch('[0-9a-f]{64}',value):
        raise DownloadBootstrapError('An expected 64-character SHA256 is mandatory.')
    return value


def _matches(path,expected_hash,expected_bytes):
    return path.is_file() and (expected_bytes is None or path.stat().st_size==expected_bytes) and sha256_file(path)==expected_hash


def download_package(url,cache_dir,kind,expected_sha256,expected_bytes=None,*,redownload=False,download=None):
    """Reuse verified bytes or resume one retained attempt; never delete bad bytes."""
    if kind not in PACKAGE_NAMES:raise ValueError('kind must be data or code')
    expected_sha256=_checksum(expected_sha256)
    expected_bytes=KNOWN_BYTES.get(expected_sha256) if expected_bytes is None else int(expected_bytes)
    if expected_bytes is not None and expected_bytes<1:raise ValueError('Expected ZIP byte count must be positive.')
    file_id=drive_file_id(url)
    identity=hashlib.sha256(file_id.encode()).hexdigest()
    slot=Path(cache_dir).expanduser().resolve()/(kind+'_'+expected_sha256)
    slot.mkdir(parents=True,exist_ok=True)
    attempts=sorted(p for p in slot.glob('attempt_*') if p.is_dir())
    completed_bad=[]
    if not redownload:
        for attempt in reversed(attempts):
            archive=attempt/PACKAGE_NAMES[kind]
            if _matches(archive,expected_sha256,expected_bytes):
                return archive,{'kind':kind,'status':'verified_cache_reused','sha256':expected_sha256,
                                'bytes':archive.stat().st_size,'archive':str(archive)}
        completed_bad=[p for p in attempts if (p/PACKAGE_NAMES[kind]).is_file()]
    attempt=None
    resume_authorized=False
    if not redownload:
        for candidate in reversed(attempts):
            state=candidate/'download_state.json'
            if state.is_file():
                metadata=json.loads(state.read_text(encoding='utf-8'))
                if (metadata.get('drive_file_id_sha256')==identity
                        and not (candidate/PACKAGE_NAMES[kind]).exists()):
                    attempt=candidate;resume_authorized=bool(metadata.get('redownload_authorized'));break
    if completed_bad and not (attempt and resume_authorized and attempt.name>completed_bad[-1].name):
        report=slot/('verification_failure_'+_id()+'.json')
        _write(report,{'status':'checksum_failure','expected_sha256':expected_sha256,
            'existing_archives_preserved':[str(p/PACKAGE_NAMES[kind]) for p in completed_bad],
            'action':'Refetch requires explicit --redownload; existing files are preserved.'})
        raise DownloadBootstrapError('An existing complete ZIP failed SHA256/size verification. Preserved it; use --redownload for a NEW attempt. Report: '+str(report))
    if attempt is None:
        attempt=slot/('attempt_'+_id());attempt.mkdir(exist_ok=False)
    archive=attempt/PACKAGE_NAMES[kind]
    if archive.exists():raise DownloadBootstrapError('Refusing to replace any existing archive.')
    state={'kind':kind,'expected_sha256':expected_sha256,'expected_bytes':expected_bytes,
           'drive_file_id_sha256':identity,'resume_requested':True,'status':'downloading',
           'redownload_authorized':bool(redownload or resume_authorized),
           'archive':str(archive),'cookies_used':False}
    _write(attempt/'download_state.json',state)
    if download is None:
        try:
            import gdown
        except ImportError:
            raise DownloadBootstrapError('Install gdown in the separate transfer environment before running this bootstrap.') from None
        download=gdown.download
        state['gdown_version']=importlib.metadata.version('gdown')
    else:state['gdown_version']='injected-offline-test'
    parameters=inspect.signature(download).parameters
    accepts_kwargs=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    supported=lambda key:key in parameters or accepts_kwargs
    if not all(supported(key) for key in ('id','output','resume','use_cookies')):
        raise DownloadBootstrapError('Installed gdown lacks the required resume/file-ID API; upgrade the separate transfer environment.')
    kwargs={'id':file_id,'output':str(archive),'resume':True,'use_cookies':False}
    if supported('quiet'):kwargs['quiet']=True
    if supported('verify'):kwargs['verify']=True
    if supported('timeout'):kwargs['timeout']=(30,120)
    if supported('retries'):kwargs['retries']=2
    progress_last=[0.]
    if supported('progress'):
        def progress(received,total):
            now=time.monotonic()
            if now-progress_last[0]>=10 or (total is not None and received==total):
                print(f'[{kind}] received {received/1024**2:.1f} MiB'+
                      (f' / {total/1024**2:.1f} MiB' if total else ''),flush=True)
                progress_last[0]=now
        kwargs['progress']=progress
    print(f'[{kind}] gdown {state["gdown_version"]}; resumable transfer; expected SHA256 {expected_sha256}',flush=True)
    try:
        returned=download(**kwargs)
    except Exception:
        state.update(status='transfer_failed',partial_files_preserved=True)
        _write(attempt/'download_state.json',state)
        raise DownloadBootstrapError('gdown transfer failed. Check file sharing/availability; partial files were preserved for rerun. No browser credentials were used.') from None
    if not returned or not _matches(archive,expected_sha256,expected_bytes):
        state.update(status='checksum_failure',received_bytes=archive.stat().st_size if archive.is_file() else None)
        _write(attempt/'download_state.json',state)
        raise DownloadBootstrapError('Downloaded ZIP failed expected SHA256/size verification. Bytes preserved; --redownload starts a separate attempt.')
    state.update(status='download_verified',sha256=expected_sha256,bytes=archive.stat().st_size)
    _write(attempt/'download_state.json',state)
    return archive,state


def _member_path(name):
    raw=name[:-1] if name.endswith('/') else name
    if (not raw or raw.startswith('/') or '\\' in raw or ':' in raw
            or any(p in ('','.','..') for p in raw.split('/'))):
        raise DownloadBootstrapError('Unsafe ZIP member path.')
    return PurePosixPath(raw)


def _safe_destination(root,relative):
    destination=root/relative
    if not destination.resolve().is_relative_to(root):raise DownloadBootstrapError('ZIP target escapes project root.')
    for path in (destination,*destination.parents):
        if path==root.parent:break
        if path.is_symlink():raise DownloadBootstrapError('Extraction refuses existing symlink targets/parents.')
    return destination


def safe_extract_package(archive,project_root,kind,*,required_members=None,inspect_only=False):
    """CRC/path/content preflight, then add NEW files or skip byte-identical ones."""
    if kind not in REQUIRED:raise ValueError('kind must be data or code')
    root=Path(project_root).expanduser().resolve()
    if not root.is_dir():raise DownloadBootstrapError('Create the explicit mounted-volume TSA directory first.')
    expected=REQUIRED[kind] if required_members is None else tuple(required_members)
    additions=[];same=[];records=[];seen={};total=0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative=_member_path(info.filename);name=str(relative)
            key=name.casefold()
            if key in seen:raise DownloadBootstrapError('Duplicate/case-colliding ZIP member path.')
            seen[key]=info.is_dir()
            mode=(info.external_attr>>16)&0xffff
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
                raise DownloadBootstrapError('ZIP symlink/special members are not permitted.')
            if info.flag_bits&1:raise DownloadBootstrapError('Encrypted ZIP members are not supported.')
            if relative.parts[0] not in (('Dataset','Model') if kind=='data' else ('Model',)):
                raise DownloadBootstrapError('ZIP can install only declared Dataset/Model paths, never results.')
            if kind=='code' and not (name.startswith(CODE_PREFIX) or (info.is_dir() and CODE_PREFIX.startswith(name+'/'))):
                raise DownloadBootstrapError('Code ZIP must stay within the existing VastAI project folder.')
            total+=info.file_size
            if total>8*1024**3:raise DownloadBootstrapError('ZIP expanded size exceeds the 8 GiB bootstrap bound.')
            records.append((info,relative))
        files={str(relative) for info,relative in records if not info.is_dir()}
        if not set(expected)<=files:raise DownloadBootstrapError('ZIP lacks required '+kind+' Dataset/Model structure.')
        for key,is_dir in seen.items():
            for parent in PurePosixPath(key).parents:
                if str(parent)!='.' and seen.get(str(parent)) is False:
                    raise DownloadBootstrapError('ZIP contains a file/directory path collision.')
        bad=z.testzip()
        if bad is not None:raise DownloadBootstrapError('ZIP CRC verification failed.')
        if inspect_only:
            return {'kind':kind,'zip_crc_verified':True,'expected_structure_verified':True,
                    'archive':str(Path(archive).resolve()),'files':len(files),'extraction_performed':False}
        # Check every existing destination BEFORE adding any file.
        prepared=[]
        for info,relative in records:
            destination=_safe_destination(root,relative)
            if info.is_dir():
                if destination.exists() and not destination.is_dir():raise DownloadBootstrapError('Existing file conflicts with ZIP directory.')
                continue
            if destination.exists() and not destination.is_file():raise DownloadBootstrapError('Existing directory conflicts with ZIP file.')
            digest=hashlib.sha256()
            with z.open(info) as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            checksum=digest.hexdigest()
            if destination.exists():
                if destination.stat().st_size!=info.file_size or sha256_file(destination)!=checksum:
                    raise DownloadBootstrapError('Existing DIFFERENT file preserved; extraction refused: '+str(relative))
                same.append(str(relative))
            else:prepared.append((info,relative,checksum))
        for info,relative,checksum in prepared:
            destination=_safe_destination(root,relative);destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name('.'+destination.name+'.bootstrap_'+uuid.uuid4().hex+'.tmp')
            try:
                with z.open(info) as source,temporary.open('xb') as output:
                    shutil.copyfileobj(source,output,8*1024*1024);output.flush();os.fsync(output.fileno())
                if sha256_file(temporary)!=checksum:raise DownloadBootstrapError('Staged extraction checksum mismatch.')
                permissions=(info.external_attr>>16)&0o777
                if permissions:temporary.chmod(permissions&0o755)
                try:os.link(temporary,destination)  # Atomic exclusive publication, never replacement.
                except FileExistsError:
                    if sha256_file(destination)!=checksum:raise DownloadBootstrapError('Concurrent DIFFERENT target preserved.')
                except OSError:
                    # Filesystems without hard links still get exclusive no-overwrite creation.
                    with temporary.open('rb') as source,destination.open('xb') as output:
                        shutil.copyfileobj(source,output,8*1024*1024);output.flush();os.fsync(output.fileno())
                additions.append(str(relative))
            finally:
                if temporary.exists():temporary.unlink()  # Only this invocation's checked sibling temp.
    return {'kind':kind,'zip_crc_verified':True,'archive':str(Path(archive).resolve()),
            'added_files':additions,'identical_existing_files_skipped':same,
            'different_existing_files_overwritten':False,'results_paths_untouched':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--data-url',required=True)
    parser.add_argument('--code-url')
    parser.add_argument('--data-sha256',default=DATA_SHA256)
    parser.add_argument('--code-sha256',default=CODE_SHA256)
    parser.add_argument('--data-bytes',type=int)
    parser.add_argument('--code-bytes',type=int)
    parser.add_argument('--redownload',action='store_true',help='Use NEW attempt folders; retain all previous bytes.')
    parser.add_argument('--download-only',action='store_true')
    args=parser.parse_args()
    if not args.project_root.is_absolute() or args.project_root.name!='TSA' or not args.project_root.parent.is_dir():
        raise DownloadBootstrapError('--project-root must explicitly name an existing mounted-volume parent/TSA; no automatic /workspace assumption.')
    project=args.project_root.resolve();project.mkdir(exist_ok=True)
    cache=(args.cache_dir or project.parent/'.tsa_downloads').expanduser().resolve()
    report={'project_root':str(project),'cache_dir':str(cache),'packages':[],
            'training_or_backup_code_changed':False,'network_urls_or_credentials_recorded':False}
    for kind,url,checksum,size in (('data',args.data_url,args.data_sha256,args.data_bytes),
                                  ('code',args.code_url,args.code_sha256,args.code_bytes)):
        if not url:continue
        archive,record=download_package(url,cache,kind,checksum,size,redownload=args.redownload)
        record['archive_validation' if args.download_only else 'extraction']=safe_extract_package(
            archive,project,kind,inspect_only=args.download_only)
        report['packages'].append(record)
    path=cache/('bootstrap_report_'+_id()+'.json');_write(path,report)
    print(json.dumps({'status':'download_verified' if args.download_only else 'verified_and_installed',
        'project_root':str(project),'report':str(path),'packages':[p['kind'] for p in report['packages']]},indent=2))


if __name__=='__main__':main()
