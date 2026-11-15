"""Verify only the declared upload files; never download, modify or delete data."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[4])
    args=parser.parse_args()
    root=args.project_root.resolve()
    manifest=Path(__file__).resolve().parent.parent/'References/upload_manifest.json'
    record=json.loads(manifest.read_text(encoding='utf-8-sig'))
    failures=[]
    for item in record['files']:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe relative path in manifest: '+str(relative))
        path=root/relative
        if not path.is_file():failures.append({'path':item['path'],'reason':'missing'})
        elif path.stat().st_size!=item['bytes']:failures.append({'path':item['path'],'reason':'size'})
        elif digest(path)!=item['sha256']:failures.append({'path':item['path'],'reason':'sha256'})
    result={'declared_files':len(record['files']),'failed_files':failures,
            'checked_root':str(root),'all_declared_files_match':not failures}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
