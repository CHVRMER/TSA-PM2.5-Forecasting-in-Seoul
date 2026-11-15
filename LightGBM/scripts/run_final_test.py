"""Final Test entry point. --verify-plan never opens source observations."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
from src.final_test import execute,validate_plan,sha

def main():
    p=argparse.ArgumentParser(description='One final run: exclusive pre-input lock, frozen predictions, one scoring pass')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--verify-plan',action='store_true');group.add_argument('--final',action='store_true')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256')
    a=p.parse_args()
    if a.verify_plan:
        plan=a.plan.resolve();plan.relative_to(ROOT);expected=plan.with_suffix('.sha256').read_text().strip()
        if sha(plan)!=expected:raise ValueError('Plan checksum mismatch')
        result=validate_plan(json.loads(plan.read_text()),ROOT);print(json.dumps({'status':'static_plan_verified','plan_sha256':expected,'production_locks_created':False,**result},ensure_ascii=False,indent=2));return
    result=execute(a.plan,ROOT,final=a.final,expected_sha=a.plan_sha256);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
