"""Freeze separate QC-neighbor addition and western-Japan reselection inputs.

Reads reviewed data and registries only. The old 26-condition catalog and its
600-series F input remain unchanged. New factor blocks are NOT administrative
regions: the China_East block includes Shandong as well as other provinces.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

VERSION = 'neighbor-factorial-qc-v1.0.0'
STATS = ['now','lag01h','lag03h','lag06h','lag12h','lag24h',
         'mean6h','std6h','coverage6h','mean24h','std24h','coverage24h',
         'change3h','change6h','age_since_valid_h','missing_now']
NEW_RELATIVE = 'Dataset/Neighbors_Expansion_QC_20261108'
SPECS = [
    ('F600_W24',False,False,'old24',600,'primary'),
    ('F616_CNE_W24',True,False,'old24',616,'primary'),
    ('F618_JPE_W24',False,True,'old24',618,'primary'),
    ('F634_BOTH_W24',True,True,'old24',634,'primary'),
    ('F601_W25',False,False,'new25',601,'primary'),
    ('F617_CNE_W25',True,False,'new25',617,'secondary_factorial'),
    ('F619_JPE_W25',False,True,'new25',619,'secondary_factorial'),
    ('F635_BOTH_W25',True,True,'new25',635,'secondary_factorial'),
]
SHANDONG = {'city_linyi','city_taian','city_zaozhuang','city_jining'}
JIANGSU = {'city_lianyungang','city_yancheng','city_huaian','city_taizhou_js','city_wuxi','city_nanjing'}
ANHUI = {'city_huaibei','city_bengbu','city_wuhu','city_bozhou','city_hefei'}
ZHEJIANG = {'city_hangzhou'}


def check(value,message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read_csv(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--new-data-root',type=Path,required=True,
                    help='Staged or installed QC dataset directory; normalized dictionary/registry required.')
    ap.add_argument('--archive',type=Path,default=Path('C:/Users/kimjy/Downloads/Dataset_11_07_regions_v5.zip'))
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args();project=args.project_root.resolve();new=args.new_data_root.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    legacy_dir=project/'Model/Common/Regional_Experiments_20261107'
    old_config=json.loads((legacy_dir/'experiments.json').read_text(encoding='utf-8'))
    old_sets=json.loads((legacy_dir/'feature_sets.json').read_text(encoding='utf-8'))['sets']
    old_regions=json.loads((legacy_dir/'regions.json').read_text(encoding='utf-8'))['regions']
    old_station=read_csv(legacy_dir/'station_registry.csv')
    old_feature=read_csv(legacy_dir/'feature_registry.csv')
    v4=read_csv(project/'Dataset/Trajectory_Matching_National_v4/station_registry.csv')
    v3_dict=read_csv(project/'Dataset/Origin_Features_2021_2024_v3/feature_dictionary.csv')
    new_dict=read_csv(new/'feature_dictionary.csv')
    added=read_csv(new/'added_station_registry.csv')
    required={'name','role','group','dtype','unit','parameters_json','definition_key','time_window','source'}
    check(required<=set(new_dict),'New QC feature dictionary must use normalized schema.')
    check(len(added)==34 and not added.station_key.duplicated().any(),'Expected 34 distinct added series.')
    check(len(new_dict)==544 and new_dict.role.eq('feature').all(),'Expected 544 new feature rows.')
    check(len(old_station)==600 and not old_station.station_key.duplicated().any(),'Old F station registry mismatch.')
    core=old_config['common_features'];old_keys=set(old_config['conditions']['F']['station_keys'])
    old_jp=set(old_station.loc[old_station.country.eq('JP'),'station_key'])
    check(len(core)==70 and len(old_jp)==24,'Expected old common70 and Japanese24.')
    with zipfile.ZipFile(args.archive) as z:
        west=pd.read_csv(z.open('Dataset/Japan/selected_stations_1107.csv'),dtype=str,keep_default_na=False)
    west=west.loc[west.selected.str.lower().eq('true')].copy()
    west_keys=set(west.station_key)
    check(len(west)==len(west_keys)==25,'The full 188-row CSV must be filtered to selected=True25.')
    check(len(old_jp&west_keys)==9,'Expected 9 Japanese stations common to old24/new25.')
    new_west=west_keys-old_jp
    check(len(new_west)==16 and len(old_jp-west_keys)==15,'Japanese reselection composition mismatch.')
    check(new_west<=set(v4.station_key),'New western selection must already exist in old v4 source.')
    west_extra=v4.loc[v4.station_key.isin(new_west)].copy()
    check(west_extra.source_dataset.eq('Japan').all(),'Western history source must retain the old Japan provider.')
    west_extra['experiment_region_id']='JP'
    cne=set(added.loc[added.country.eq('CN'),'station_key'])
    jpe=set(added.loc[added.country.eq('JP'),'station_key'])
    check(len(cne)==16 and len(jpe)==18 and not (set(added.station_key)&old_keys),'New source block mismatch/duplicate.')
    check(set(added.loc[added.country.eq('CN'),'station_id'])==SHANDONG|JIANGSU|ANHUI|ZHEJIANG,'Unexpected Chinese city set.')
    added=added.copy();added['experiment_region_id']=added.country.map({'CN':'CN_OTHER','JP':'JP'})
    added.loc[added.station_id.isin(SHANDONG),'experiment_region_id']='CN_SHANDONG'
    added['china_province']=added.station_id.map({**dict.fromkeys(SHANDONG,'Shandong'),**dict.fromkeys(JIANGSU,'Jiangsu'),
                                                **dict.fromkeys(ANHUI,'Anhui'),**dict.fromkeys(ZHEJIANG,'Zhejiang')}).fillna('')
    added['addition_block']=added.country.map({'CN':'China_East16','JP':'Japan_East18'})
    old_station['addition_block']='legacy_F600';west_extra['addition_block']='WestJapan_newly_selected16'
    stations=pd.concat([old_station,west_extra,added],ignore_index=True).fillna('')
    stations['in_old_F600']=stations.station_key.isin(old_keys)
    stations['in_old_japan24']=stations.station_key.isin(old_jp)
    stations['in_new_japan_west25']=stations.station_key.isin(west_keys)
    stations['metadata_policy']='legacy metadata preserved; new selected provider metadata recorded separately'
    check(len(stations)==650 and not stations.station_key.duplicated().any(),'Expected 650-series union catalog.')
    check('KR:111121' not in set(stations.station_key),'Target duplicated as external series.')
    # Preserve all old station values in every original field, including coordinates/source identifiers.
    compare=stations.set_index('station_key').loc[old_station.station_key,old_station.columns.drop('station_key')]
    for column in old_station.columns.drop('station_key'):
        check(compare[column].tolist()==old_station[column].tolist(),'Legacy metadata changed: '+column)
    stations=stations.sort_values('station_key',kind='stable').reset_index(drop=True)
    records=old_feature.to_dict('records')
    for dictionary,owners,source_id in [(v3_dict,new_west,'v3'),(new_dict,cne|jpe,'new_qc')]:
        found=[]
        for row in dictionary.to_dict('records'):
            if row.get('definition_key')!='station_history': continue
            parameters=json.loads(row['parameters_json'])
            key=parameters.get('station_key')
            if key not in owners: continue
            check(row['role']=='feature' and parameters.get('stat') in STATS,'Invalid station-history role/stat.')
            check('_pm25_' in row['name'],'Non-PM2.5 observation entered the input catalog.')
            records.append({'name':row['name'],'source_id':source_id,'owner_station_key':key,
                            'scope':'station_pm25_history','role':'feature','feature_group':row['group'],
                            'dtype':row['dtype'],'unit':row.get('unit','ug/m3'),'stat':parameters['stat'],
                            'source_dataset':parameters.get('source_dataset',''),
                            'definition_key':'station_history','time_window':row.get('time_window',''),
                            'korean_description':row.get('korean_description',row.get('definition',''))})
            found.append(key)
        check(set(found)==owners and len(found)==16*len(owners),'Feature ownership is incomplete for '+source_id)
    features=pd.DataFrame(records).fillna('')
    check(len(features)==10470 and not features.name.duplicated().any(),'Expected unique 10,470-feature union.')
    own=features.loc[features.scope.eq('station_pm25_history')]
    feature_by={}
    for key,rows in own.groupby('owner_station_key',sort=False):
        check(len(rows)==16 and set(rows.stat)==set(STATS) and not rows.stat.duplicated().any(),'Invalid 16-stat block: '+key)
        feature_by[key]=rows.set_index('stat').name.to_dict()
    index=features.set_index('name')
    check(set(feature_by)==set(stations.station_key),'Station/feature catalog ownership mismatch.')
    conditions={};sets={};summary=[]
    for ident,cn,jp,west_rule,count,priority in SPECS:
        keys=(old_keys if west_rule=='old24' else (old_keys-old_jp)|west_keys)
        if cn: keys=keys|cne
        if jp: keys=keys|jpe
        names=list(core)
        for key in sorted(keys):names.extend(feature_by[key][stat] for stat in STATS)
        check(len(keys)==count and len(names)==70+16*count and len(names)==len(set(names)),'Condition scope/count mismatch: '+ident)
        source_counts={source:int(index.loc[names].source_id.eq(source).sum()) for source in ['v3','national','new_qc']}
        cond={'feature_set':ident,'priority':priority,'factors':{'china_east':cn,'japan_east':jp,'west_selection':west_rule},
              'station_keys':sorted(keys),'expected_external_series':count,'expected_feature_count':len(names),
              'source_feature_counts':source_counts,'core_features_retained':70}
        conditions[ident]=cond;sets[ident]=names
        countries=stations.loc[stations.station_key.isin(keys),'country'].value_counts()
        summary.append({'condition':ident,'priority':priority,'china_east':cn,'japan_east':jp,'west_selection':west_rule,
                        'external_series':count,'including_Junggu_nodes':count+1,'features':len(names),
                        **{source+'_features':source_counts[source] for source in source_counts},
                        **{country+'_series':int(countries.get(country,0)) for country in ['KR','CN','JP','MN']}})
    check(sets['F600_W24']==old_sets['F'],'Baseline F600 feature order/values source must be EXACT old F.')
    check(set(conditions['F600_W24']['station_keys'])==old_keys,'Baseline F600 station set changed.')
    source_files=dict(old_config['source_files']);source_files['new_qc']=NEW_RELATIVE+'/station_history_qc_2021_2024.parquet'
    runtime_actual={key:project/value for key,value in old_config['source_files'].items()}
    runtime_actual['new_qc']=new/'station_history_qc_2021_2024.parquet'
    for source in ['v3','national','new_qc']:
        columns=set(pq.ParquetFile(runtime_actual[source]).schema.names)
        expected=set(features.loc[features.source_id.eq(source),'name'])|{'origin_time_kst'}
        check(expected<=columns,'Requested feature absent in actual '+source+' Parquet.')
    factors={tuple(spec['factors'][name] for name in ['china_east','japan_east','west_selection']):ident for ident,spec in conditions.items()}
    comparisons=[]
    for name,position in [('china_east',0),('japan_east',1),('west_reselection',2)]:
        for current,reference in factors.items():
            is_reference=(current[position] is False) if position<2 else current[position]=='old24'
            if not is_reference: continue
            target=list(current);target[position]=True if position<2 else 'new25';better=factors[tuple(target)]
            primary=current==(False,False,'old24')
            comparisons.append({'id':name+'__'+reference+'__'+better,'factor':name,
                                'priority':'primary' if primary else 'secondary_factorial',
                                'better_condition':better,'reference_condition':reference,
                                'positive_gain_definition':'RMSE(reference)-RMSE(better)'})
    comparisons.append({'id':'joint_additions_old24','factor':'joint_additions','priority':'primary',
                        'better_condition':'F634_BOTH_W24','reference_condition':'F600_W24',
                        'positive_gain_definition':'RMSE(reference)-RMSE(better)'})
    check(len(comparisons)==13,'Expected 12 factorial simple effects plus one joint-addition comparison.')
    configuration={'schema_version':VERSION,'target':old_config['target'],'horizons':[3,6,12,24],'primary_horizon':6,
                   'source_files':source_files,'common_features':core,'conditions':conditions,
                   'baseline_reference':'Model/Common/Regional_Experiments_20261107: F',
                   'baseline_exact_feature_order_preserved':True,'factorial_design':'ChinaEast16 on/off × JapanEast18 on/off × WestJapan old24/new25',
                   'primary_conditions':[ident for ident,spec in conditions.items() if spec['priority']=='primary'],
                   'new_data_QC_only':'legacy F600 data values unchanged; new QC source is separate',
                   'trajectory_or_gas_features_included':False,
                   'matching_candidates_not_equal_model_nodes':True,
                   'union_registry_is_catalog_not_a_single_model_node_set':True,
                   'DCRNN_architecture_modified':False}
    regions={}
    for rid,rows in stations.groupby('experiment_region_id',sort=False):
        regions[rid]={'country':rows.country.iloc[0],'station_keys':rows.station_key.tolist(),
                      'label_ko':old_regions.get(rid,{}).get('label_ko','기타 중국(장쑤·안후이·저장)'),
                      'label_en':rid,'role':'disjoint geographic catalog; not the addition factor'}
    write_json(out/'experiments.json',configuration)
    write_json(out/'feature_sets.json',{'schema_version':VERSION,'sets':sets})
    write_json(out/'regions.json',{'schema_version':VERSION,'regions':regions,
                                'China_East_not_a_disjoint_geographic_region':True})
    write_json(out/'comparisons.json',{'schema_version':VERSION,'primary_horizon':6,'horizons':[3,6,12,24],
                                     'metric':'RMSE','loss_units':'ug/m3','comparisons':comparisons})
    stations.to_csv(out/'station_registry.csv',index=False,encoding='utf-8-sig')
    features.to_csv(out/'feature_registry.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out/'condition_summary.csv',index=False,encoding='utf-8-sig')
    inputs={relative:{'bytes':runtime_actual[source].stat().st_size,'sha256':sha(runtime_actual[source])}
            for source,relative in source_files.items()}
    for relative,actual in [(NEW_RELATIVE+'/feature_dictionary.csv',new/'feature_dictionary.csv'),
                            (NEW_RELATIVE+'/added_station_registry.csv',new/'added_station_registry.csv')]:
        inputs[relative]={'bytes':actual.stat().st_size,'sha256':sha(actual)}
    for name in ['experiments.json','feature_sets.json','feature_registry.csv','station_registry.csv','regions.json']:
        actual=legacy_dir/name;relative=actual.relative_to(project).as_posix()
        inputs[relative]={'bytes':actual.stat().st_size,'sha256':sha(actual)}
    for relative in ['Dataset/Trajectory_Matching_National_v4/station_registry.csv',
                     'Dataset/Origin_Features_2021_2024_v3/feature_dictionary.csv']:
        actual=project/relative;inputs[relative]={'bytes':actual.stat().st_size,'sha256':sha(actual)}
    write_json(out/'source_manifest.json',{'schema_version':VERSION,'status':'passed','built_utc':datetime.now(timezone.utc).isoformat(),
               'files':inputs,'archive':{'path':str(args.archive),'bytes':args.archive.stat().st_size,'sha256':sha(args.archive)},
               'conditions':8,'primary_conditions':5,'secondary_factorial_conditions':3,
               'external_catalog_series':650,'feature_catalog_columns':10470,'baseline_exact_legacy_F':True,
               'japan_old24_new25_common':9,'japan_old24_removed':15,'japan_newwest_selected':16,
               'newChina16':len(cne),'newJapanEast18':len(jpe),'code_sha256':sha(__file__),
               'geographic_and_source_blocks_not_conflated':True,'no_data_or_DCRNN_files_modified':True})
    print(json.dumps({'status':'passed','conditions':8,'external_union':650,'feature_union':10470,'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
