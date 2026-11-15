"""Build final manifest from static artifacts and prior source digest; never opens raw data."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.final_test import BASE,MODELS,HS,DATA_DIR,DATA_FILE,CODE,ANCHOR,PROJECT,PRIMARY,SECONDARY,OUTPUT,policy,versions,sha,map_sha,relative,write_json,validate_plan
EXTRA=['configs/adaptive_validation_2023.json','configs/feature_groups.json','src/data_loader.py','src/feature_groups.py','src/common_scoring.py','src/adaptive.py','src/train.py','Models/score.py',DATA_DIR+'/feature_sets.json',DATA_DIR+'/feature_dictionary.csv',DATA_DIR+'/feature_manifest.json']

def registry(root):
    root=Path(root);found={};files=set();base={m:{} for m in BASE};adaptive={m:{} for m in MODELS[6:]}
    for path in (root/'Models').rglob('ensemble_manifest.json'):
        d=json.loads(path.read_text());key=(d['model'],d['horizon'])
        if d['model'] not in BASE:continue
        if key in found:raise ValueError('Duplicate saved model')
        found[key]=path
    for model in BASE:
        for h in HS:
            path=found[model,h];d=json.loads(path.read_text());models=[path.parent/f'{model}_h{h}_seed{s}.txt' for s in [0,1,2]];cfg=root/f'configs/{model.lower()}_{h}h.json';files.update([path,cfg,*models]);base[model][str(h)]={'features':d['feature_names'],'ensemble_manifest':relative(root,path),'config':relative(root,cfg),'model_files':[relative(root,x) for x in models]}
    for model in MODELS[6:]:
        for h in HS:
            path=root/f'Models/{model}/h{h}/weights.json';files.update([path,path.with_name('config.json')]);adaptive[model][str(h)]=relative(root,path)
    files.update(root/x for x in EXTRA);ledger={relative(root,x):sha(x) for x in sorted(files)}
    return base,adaptive,ledger

def make_plan(root,mode,dataset_digest,preserve,validation=None):
    root=Path(root).resolve();base,adaptive,ledger=registry(root);static=dict(ledger)
    for rel in CODE:static[rel]=sha(root/rel)
    metadata=json.loads((root/DATA_DIR/'feature_manifest.json').read_text())
    result={'schema':1,'mode':mode,'project_root':str(root),'status':'frozen_waiting_explicit_final_approval' if mode=='production' else 'synthetic_only','policy':policy(),'versions':versions(),'base_models':base,'adaptive_weights':adaptive,'model_config_sha256':ledger,'model_config_composite_sha256':map_sha(ledger),'static_artifacts_sha256':static,'preserve_sha256':preserve,'dataset':{'file':DATA_FILE,'expected_sha256':dataset_digest,'hash_current_raw_file_opened_during_freeze':False,'metadata':metadata}}
    if mode=='production':
        if root!=PROJECT or len(ledger)!=147 or result['model_config_composite_sha256']!=ANCHOR:raise ValueError('Approved model/config147 commitment changed')
        record='results/validation_2023/Adaptive/preflight.json';result['dataset']['hash_record']=record;static[record]=sha(root/record)
        if validation is None:raise ValueError('Successful synthetic validation required before freeze')
        result['runner_validation']={'file':validation,'sha256':sha(root/validation)};static[validation]=sha(root/validation)
    return result

def main():
    root=ROOT
    if (root/PRIMARY).exists() or (root/SECONDARY).exists() or (root/OUTPUT).exists():raise FileExistsError('Do not alter a claimed final run')
    previous=json.loads((root/'results/validation_2023/Adaptive/preflight.json').read_text());preserve={k:v for k,v in previous['protected_sha256'].items() if not k.startswith(('Models/work/','Models/predictions/'))}
    import pandas as pd
    for filename in ['results/validation_2023/Adaptive/file_inventory.csv','results/shap/validation_2023/file_inventory.csv']:
        p=root/filename;preserve[filename]=sha(p)
        for row in pd.read_csv(p).itertuples():preserve[row.path]=row.sha256
    plan=make_plan(root,'production',previous['original_inputs_sha256'][DATA_FILE],preserve,'results/final_test_runner_validation/verification.json');result=validate_plan(plan,root)
    target=root/'results/test_2024/frozen_plan.json';target.parent.mkdir(parents=True,exist_ok=True);write_json(target,plan);digest=sha(target)
    with target.with_suffix('.sha256').open('x') as f:f.write(digest+'\n')
    print(json.dumps({'status':'frozen_without_raw_data_access','plan':str(target),'plan_sha256':digest,'production_test_started':False,**result},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
