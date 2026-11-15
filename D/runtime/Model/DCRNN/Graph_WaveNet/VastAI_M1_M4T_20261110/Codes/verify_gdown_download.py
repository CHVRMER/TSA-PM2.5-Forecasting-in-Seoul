"""Offline fake-gdown tests: resume, SHA/CRC, ZIP safety, and no overwrite."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import os
import stat
import zipfile
from drive_download_gdown import (CODE_PREFIX,DownloadBootstrapError,
    download_package,safe_extract_package,drive_file_id,sha256_file)


def make_zip(records):
    out=BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as z:
        for name,value in records:z.writestr(name,value)
    return out.getvalue()


class FakeGdown:
    def __init__(self,payload,interrupt=False):
        self.payload=payload;self.interrupt=interrupt;self.calls=[]
    # Deliberately the older minimal signature: no timeout/retries/progress/fuzzy.
    def __call__(self,*,id,output,resume,use_cookies,quiet=True):
        assert resume is True and use_cookies is False and quiet is True
        target=Path(output);self.calls.append(str(target))
        part=target.with_name(target.name+'abcdefgh.part')
        if self.interrupt:
            part.write_bytes(self.payload[:len(self.payload)//2]);self.interrupt=False
            raise ConnectionError('simulated interruption; no actual network')
        existing=part.read_bytes() if part.exists() else b''
        assert self.payload.startswith(existing)
        part.write_bytes(existing+self.payload[len(existing):])
        os.replace(part,target)
        return str(target)


def fails(call):
    try:call()
    except (DownloadBootstrapError,zipfile.BadZipFile):pass
    else:raise AssertionError('Unsafe or corrupted package was accepted.')


def main():
    parent=Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='.gdown_test_',dir=parent) as temporary:
        root=Path(temporary).resolve();assert root.is_relative_to(parent)
        project=root/'TSA';project.mkdir();cache=root/'.tsa_downloads'
        required=('Dataset/a.parquet','Model/Common/score.py')
        payload=make_zip([(required[0],b'observations'),(required[1],b'original scorer')])
        digest=hashlib.sha256(payload).hexdigest();fake=FakeGdown(payload,interrupt=True)
        link='https://drive.google.com/file/d/SYNTHETIC_FILE_12345/view?usp=sharing'
        fails(lambda:download_package(link,cache,'data',digest,len(payload),download=fake))
        archive,record=download_package(link,cache,'data',digest,len(payload),download=fake)
        assert len(fake.calls)==2 and fake.calls[0]==fake.calls[1]
        assert sha256_file(archive)==digest
        reused,_=download_package(link,cache,'data',digest,len(payload),download=fake)
        assert reused==archive and len(fake.calls)==2
        inspected=safe_extract_package(archive,project,'data',required_members=required,inspect_only=True)
        assert inspected['zip_crc_verified'] and not (project/required[0]).exists()
        installed=safe_extract_package(archive,project,'data',required_members=required)
        assert len(installed['added_files'])==2
        installed=safe_extract_package(archive,project,'data',required_members=required)
        assert len(installed['identical_existing_files_skipped'])==2
        # Existing different files are detected in the complete preflight, before additions.
        (project/required[0]).write_bytes(b'user version to preserve')
        collision=root/'collision.zip'
        collision.write_bytes(make_zip([('Dataset/new.parquet',b'must not be installed'),(required[0],b'old')]))
        fails(lambda:safe_extract_package(collision,project,'data',required_members=()))
        assert (project/required[0]).read_bytes()==b'user version to preserve'
        assert not (project/'Dataset/new.parquet').exists()
        archive.write_bytes(b'corrupt retained bytes')
        calls=len(fake.calls)
        fails(lambda:download_package(link,cache,'data',digest,len(payload),download=fake))
        assert len(fake.calls)==calls and archive.read_bytes()==b'corrupt retained bytes'
        fake.interrupt=True
        fails(lambda:download_package(link,cache,'data',digest,len(payload),redownload=True,download=fake))
        fresh,_=download_package(link,cache,'data',digest,len(payload),download=fake)
        assert fresh!=archive and archive.read_bytes()==b'corrupt retained bytes'
        assert fake.calls[-1]==fake.calls[-2]  # Authorized new attempt resumes without a second flag.
        for index,records in enumerate((
            [('../outside',b'bad')], [('/absolute',b'bad')], [('Dataset/../bad',b'bad')],
            [('Dataset/A',b'a'),('Dataset/a',b'b')], [('Result/prior_model.pt',b'bad')],
            [('Dataset/file',b'a'),('Dataset/file/child',b'b')])):
            bad=root/f'bad_{index}.zip';bad.write_bytes(make_zip(records))
            fails(lambda:safe_extract_package(bad,project,'data',required_members=()))
        symlink=root/'symlink.zip'
        with zipfile.ZipFile(symlink,'w') as z:
            item=zipfile.ZipInfo('Dataset/link');item.create_system=3
            item.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(item,'../../outside')
        fails(lambda:safe_extract_package(symlink,project,'data',required_members=()))
        corrupt=bytearray(payload);offset=corrupt.index(b'observations');corrupt[offset]^=1
        crc=root/'crc.zip';crc.write_bytes(corrupt)
        fails(lambda:safe_extract_package(crc,project,'data',required_members=required))
        code=root/'code.zip';code.write_bytes(make_zip([('Model/',b''),(CODE_PREFIX+'Codes/new_helper.py',b'# new')]))
        code_result=safe_extract_package(code,project,'code',required_members=())
        assert code_result['added_files']==[CODE_PREFIX+'Codes/new_helper.py']
        assert drive_file_id(link)=='SYNTHETIC_FILE_12345'
        fails(lambda:drive_file_id('https://drive.google.com/drive/folders/SYNTHETIC_FILE_12345'))
        fails(lambda:download_package(link,cache,'data','',download=fake))
    print({'gdown_bootstrap_tests':'passed','resume_partial':True,'verified_cache_reuse':True,
        'explicit_refetch_preserves_corrupt_archive':True,'authorized_refetch_resumes':True,
        'older_gdown_signature_supported':True,'zip_crc_and_paths_checked':True,
        'existing_different_files_preserved':True,'no_results_overwrites':True,'no_network_executed':True})


if __name__=='__main__':main()
