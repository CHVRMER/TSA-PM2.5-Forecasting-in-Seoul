from pathlib import Path
import zipfile,json,io,hashlib
import pandas as pd,numpy as np

p=Path(r'C:/Users/kimjy/Downloads/Regional12_Seed11_20261109-20261110T090404Z-1-001.zip')
out=Path(__file__).resolve().parent
with zipfile.ZipFile(p) as z:
 ns=z.namelist();read=lambda n:json.loads(z.read(n))
 latest=max(n for n in ns if n.endswith('batch_status.json'));b=read(latest)
 records=[];contracts=[];graphs=[];configs=[]
 for item in b['results']:
  condition=item['condition'];base='Regional12_Seed11_20261109/'+item['job_root'].split('/Regional12_Seed11_20261109/')[1]
  plan=read(base+'/job_plan.json');contract=read(base+'/regional_contract.json');contracts.append(contract)
  record={'condition':condition,'status':item['status'],'job_zip_path':base,'allowed_external_series':contract['allowed_external_series'],'allowed_by_country':pd.Series([v.split(':')[0] for v in contract['allowed_station_keys']]).value_counts().to_dict()}
  cfg=plan['training_configuration'];configs.append(cfg)
  record['training_configuration']={k:v for k,v in cfg.items() if k not in ['source_code_hashes','library_versions','input_condition']}
  runs=sorted(n.removesuffix('/experiment_config.json') for n in ns if n.startswith(base+'/Training/') and n.endswith('/experiment_config.json'))
  if not runs:records.append(record);continue
  run=runs[-1];record['run_zip_path']=run
  pre=read(run+'/preprocessing.json');record['preprocessing_keys']=list(pre);record['context_names']=pre.get('context_names')
  summaries=run+'/run_summaries.json'
  if summaries in ns:
   summary=read(summaries)[0];record['summary']=summary
   best=run+'/Models/'+summary['model']+'/seed11/best.pt'
   record['best_sha_matches']=hashlib.sha256(z.read(best)).hexdigest()==summary['checkpoint_sha256']
  else:record['error']=z.read(base+'/error.txt').decode() if base+'/error.txt' in ns else None
  curvefiles=[n for n in ns if n.startswith(run+'/Models/') and n.endswith('/learning_curve.csv')]
  if curvefiles:
   curve=pd.read_csv(io.BytesIO(z.read(curvefiles[-1])))
   record['curve']={'columns':list(curve),'epochs':len(curve),'first_epoch':int(curve.epoch.min()),'last_epoch':int(curve.epoch.max()),'continuous':curve.epoch.tolist()==list(range(1,int(curve.epoch.max())+1)),'best_epoch_recomputed':int(curve.loc[curve.validation_rmse_h6.idxmin(),'epoch']),'best_h6_recomputed':float(curve.validation_rmse_h6.min()),'last_bad_epochs':int(curve.bad_epochs.iloc[-1]),'lr_distinct':sorted(curve.learning_rate_used.unique().tolist()),'first_losses':curve[['epoch','train_loss','validation_rmse_h6']].head(4).to_dict('records'),'last_losses':curve[['epoch','train_loss','validation_rmse_h6']].tail(4).to_dict('records')}
   curve.to_csv(out/(condition+'_learning_curve.csv'),index=False)
  for graphfile in [n for n in ns if n.startswith(run+'/Graphs/') and n.endswith('.npy')]:
   arr=np.load(io.BytesIO(z.read(graphfile)))
   graphs.append({'condition':condition,'shape':list(arr.shape),'nonzero':int(np.count_nonzero(arr)),'sha256_file':hashlib.sha256(z.read(graphfile)).hexdigest(),'sha256_raw':hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()})
  record['completion']=read(run+'/completion.json') if run+'/completion.json' in ns else None
  record['execution_status']=read(run+'/execution_status.json')
  audit=base+'/seed11_resume_audit.json'
  record['seed11_resume_audit']=read(audit) if audit in ns else None
  record['execution_protocol']=read(run+'/execution_protocol.json')
  records.append(record)
 fields=['node_order_sha256','graph_sha256','target_observations_sha256','context_values_sha256','context_names_sha256','origin_grid_sha256','y_raw_sha256','eligible_sha256','origin_positions_sha256','source_universe_signature','target_context_preprocessing_sha256']
 invariant_agreement={field:len({json.dumps(c[field],sort_keys=True) for c in contracts})==1 for field in fields}
 sourcecodeagree=len({json.dumps(c['source_code_hashes'],sort_keys=True) for c in configs})==1
 result={'latest_batch_path':latest,'latest_batch_status':b['status'],'regional_scoring':b.get('regional_scoring'),'regional_comparison_complete':b.get('regional_comparison_complete'),'invariant_agreement':invariant_agreement,'source_code_hashes_all_same':sourcecodeagree,'graphs':graphs,'records':records}
 (out/'protocol_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('LATEST',b['status'],'SCORING',b.get('regional_scoring'))
 print('invariant agreement',invariant_agreement,'sourcecode',sourcecodeagree)
 for r in records:
  c=r.get('curve',{});print(r['condition'],r['status'],'allowed',r['allowed_external_series'],r['allowed_by_country'],'epochs',c.get('epochs'),'best',c.get('best_epoch_recomputed'),'besth6',c.get('best_h6_recomputed'),'lastbad',c.get('last_bad_epochs'),'SHA',r.get('best_sha_matches'),'resume',r['seed11_resume_audit'])
 print('graph summary',graphs[:1], 'allfilesame',len({g['sha256_file'] for g in graphs})==1)
 print('context names DN',records[0].get('context_names'))
