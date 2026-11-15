from pathlib import Path
import zipfile,json,csv,io,collections,hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ZIP_PATHS=[Path('C:/Users/kimjy/Downloads')/f'VastAI-20261111T003043Z-1-00{i}.zip' for i in (1,2)]
handles=[zipfile.ZipFile(p) for p in ZIP_PATHS]
members={}
duplicates=[]
for z in handles:
    for i in z.infolist():
        if i.filename in members:duplicates.append(i.filename)
        members[i.filename]=(z,i)
def read(name):
    z,i=members[name];return z.read(i)
def js(name):return json.loads(read(name))
def records(name):return list(csv.DictReader(io.StringIO(read(name).decode('utf-8-sig'))))
def small(d):return {k:v for k,v in d.items() if not isinstance(v,list) or len(v)<30}

final='VastAI/20261111_014538_406520_eaa72758/'
names=[n for n in members if n.startswith(final)]
(HERE/'final_file_inventory.json').write_text(json.dumps([{'name':n,'bytes':members[n][1].file_size,'archive':members[n][0].filename} for n in names],ensure_ascii=False,indent=2),encoding='utf-8')
report={'final_prefix':final,'total_files':len(names),'duplicate_members':duplicates,
        'second_zip_files':[i.filename for i in handles[1].infolist()],
        'batches':[],'config':js(final+'resolved_configuration.json'), 'conditions':{}}
for n in members:
    if n.endswith('batch_manifest.json'):
        d=js(n);report['batches'].append({'path':n,**d})
for c in ('M1','M2','M3','M4','M3T','M4T'):
    candidate=[n for n in names if f'/Conditions/{c}/' in n]
    d={'files':len(candidate),'json':{},'curves':{}}
    for n in candidate:
        basename=Path(n).name
        if basename in ('run_summary.json','completion.json','resolved_settings.json','resume_copy_manifest.json','stopping_policy_revision.json','training_config.json','model_settings.json','data_contract.json','contract.json'):
            d['json'][n[len(final):]]=js(n)
        if n.endswith('.csv') and any(t in n.lower() for t in ('curve','history')):
            r=records(n);d['curves'][n[len(final):]]=r
    contractname=final+f'Condition_Contracts/{c}.json'
    if contractname in members:d['condition_contract']=js(contractname)
    report['conditions'][c]=d
report['source_contract']=js(final+'source_contract.json') if final+'source_contract.json' in members else None
report['scoring_audits']={n[len(final):]:js(n) for n in names if '/Combined/' in n and n.endswith('scoring_audit.json')}
(HERE/'protocol_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Second ZIP:',report['second_zip_files'])
print('Final file count',len(names))
print('Batches:',[(Path(b['path']).parent.name,b['mode'],b['status']) for b in report['batches']])
print('TRAIN CONFIG',report['config']['training'])
for c,d in report['conditions'].items():
    print('\n',c,'JSON',list(d['json']),'CURVES',[(k,len(v),list(v[0]) if v else []) for k,v in d['curves'].items()])
    for n,j in d['json'].items():
        if n.endswith(('run_summary.json','completion.json','stopping_policy_revision.json')):print(n,small(j))
print('condition contract fields',list(report['conditions']['M1'].get('condition_contract',{})))
print('source fields',list(report['source_contract'] or {}))
