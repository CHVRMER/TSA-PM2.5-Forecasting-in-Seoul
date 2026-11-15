"""Run small checks and save each verification invocation independently."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import os
import subprocess
import sys
import uuid

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=HERE.parents[3])
    parser.add_argument('--runtime-device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    root=args.project_root.resolve()
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]
    output=root/'Result/DCRNN/VastAI_H48/Verification'/stamp
    output.mkdir(parents=True,exist_ok=False)
    temporary=output/'temporary'
    temporary.mkdir()
    env=dict(os.environ,PYTHONUNBUFFERED='1',TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary))
    commands=[
        ['verify_upload.py','--project-root',str(root)],
        ['verify_dcrnn_training_contract.py','--output',str(output/'static_contract.json')],
        ['run_experiments.py','--self-test'],
        ['verify_horizon_data.py'],
        ['verify_data_synthetic.py'],
        ['verify_data_dcrnn.py','--project-root',str(root),'--temp-dir',str(temporary)],
        ['verify_combined_score_synthetic.py','--project-root',str(root)],
        ['verify_auto_resume_control.py'],['verify_tmux_launch_control.py'],['verify_interrupt_control.py'],
        ['verify_restore_contract_schema.py'],['verify_restore_verified_backup.py'],
        ['verify_dcrnn_runtime.py','--device',args.runtime_device,'--work-dir',str(temporary),'--output',str(output/'runtime.json')]
    ]
    # The private scorer also checks old-four-horizon regressions and causal
    # 36/48 h seasonal baselines without touching the original shared scorer.
    commands.insert(5,[str(root/'Model/Common/DCRNN_H48_20261111/verify_h48_scoring.py'),
                       '--original-common',str(root/'Model/Common'),'--report',str(output/'h48_scoring.json')])
    records=[]
    for index,command in enumerate(commands,1):
        script=Path(command[0])
        full=[sys.executable,str(script if script.is_absolute() else HERE/script),*command[1:]]
        print(f'[{index}/{len(commands)}] '+command[0],flush=True)
        process=subprocess.run(full,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        (output/(f'{index:02d}_'+Path(command[0]).stem+'.log')).write_text(process.stdout,encoding='utf-8')
        print(process.stdout,flush=True)
        records.append({'script':command[0],'returncode':process.returncode})
        result={'status':'failed' if process.returncode else 'running','checks':records,'output_dir':str(output),'full_training_tested':False}
        (output/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if process.returncode:
            print('Verification failed. Do not start full training. Details: '+str(output),flush=True)
            return process.returncode
    result['status']='passed'
    (output/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Synthetic checks passed. Next: actual-data pilot. Full635-node GPU training was not run by these checks.\n'+str(output),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
