from pathlib import Path,PurePosixPath
import hashlib,json,zipfile

WORK=Path(__file__).resolve().parent
ARCHIVE=Path('C:/Users/kimjy/Downloads/20261112_003040_429483_85af4013-20261112T020128Z-1-001.zip')
EXTRACTED=WORK/'extracted'

if __name__=='__main__':
    EXTRACTED.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        bad=z.testzip();assert bad is None,bad
        names=z.namelist();assert len(names)==len(set(names))
        kept=[];skipped=[]
        for info in z.infolist():
            rel=PurePosixPath(info.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            if info.is_dir():continue
            if rel.suffix in ('.pt','.pth'):
                skipped.append(info.filename);continue
            target=(EXTRACTED/Path(*rel.parts)).resolve()
            assert target.is_relative_to(EXTRACTED.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            data=z.read(info)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
            kept.append({'path':info.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    with ARCHIVE.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    manifest={'archive':str(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha,
        'zip_crc_passed':True,'archive_members':len(names),'extracted_files':len(kept),'weights_not_deserialized':True,
        'skipped_weight_files':skipped,'files':kept}
    (WORK/'archive_inventory.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','skipped_weight_files')},indent=2))
