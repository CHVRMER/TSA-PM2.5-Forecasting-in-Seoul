"""Read-only check of an explicitly selected Vast.ai volume mount."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess


def inspect_storage(mount_path,project_root,volume_id=None,*,process_runner=subprocess.run,disk_usage=shutil.disk_usage):
    mount=Path(mount_path).expanduser().resolve()
    project=Path(project_root).expanduser().resolve()
    if not mount.is_dir():raise ValueError('The stated mount directory does not exist: '+str(mount))
    if mount==Path(mount.anchor):raise ValueError('Use the separate volume mount, not the container root filesystem.')
    project.relative_to(mount)
    if project==mount:raise ValueError('Keep the project in a subdirectory such as /workspace/TSA.')
    result=process_runner(['findmnt','--json','--target',str(mount),'--output','TARGET,SOURCE,FSTYPE,OPTIONS'],
                          capture_output=True,text=True,check=False)
    if result.returncode:raise RuntimeError('findmnt failed; check the mount in the Vast instance terminal.')
    records=json.loads(result.stdout).get('filesystems',[])
    if len(records)!=1:raise ValueError('Expected exactly one filesystem for the selected mount.')
    record=records[0]
    actual=Path(record['target']).resolve()
    if actual!=mount:raise ValueError('The path is only a directory inside '+str(actual)+'; it is not the requested volume mount.')
    if record.get('fstype') in {'overlay','tmpfs','ramfs','proc','sysfs'}:
        raise ValueError('Selected storage is a container/RAM/system filesystem, not a separate persistent volume.')
    if 'ro' in record.get('options','').split(','):raise ValueError('The selected mount is read-only.')
    source=str(record.get('source',''))
    if volume_id is not None and ('V.'+str(volume_id)+'/') not in source:
        raise ValueError('The mounted source does not identify the requested Vast volume V.'+str(volume_id))
    usage=disk_usage(mount)
    return {'mount_path':str(mount),'project_root':str(project),'source':source,'fstype':record.get('fstype'),
            'observed_total_bytes':usage.total,'observed_free_bytes':usage.free,
            'expected_volume_id':str(volume_id) if volume_id is not None else None,
            'volume_id_observed_in_source':volume_id is not None,
            'read_only_inspection':True,'cloud_rental_or_mount_changes_performed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mount-path',required=True)
    p.add_argument('--project-root',required=True)
    p.add_argument('--volume-id')
    a=p.parse_args()
    print(json.dumps(inspect_storage(a.mount_path,a.project_root,a.volume_id),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
