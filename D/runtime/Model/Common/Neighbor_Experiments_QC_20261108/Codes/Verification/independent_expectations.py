"""Independent station and feature ownership for versioned v5 factor experiments.

No producer region/selection/config code is imported. ZIP files are reference
data only. In particular selected_stations_1107.csv contains unselected rows.
"""
from pathlib import Path
from itertools import product
import json
import zipfile
import pandas as pd

STATS={'now','lag01h','lag03h','lag06h','lag12h','lag24h','mean6h','std6h','coverage6h',
       'mean24h','std24h','coverage24h','change3h','change6h','age_since_valid_h','missing_now'}
COUNTS={(False,False,False):600,(True,False,False):616,(False,True,False):618,
        (True,True,False):634,(False,False,True):601,(True,False,True):617,
        (False,True,True):619,(True,True,True):635}


def expected_scopes(project_root,archive_path):
    root=Path(project_root)
    old_config=root/'Model/Common/Regional_Experiments_20261107'
    old_stations=pd.read_csv(old_config/'station_registry.csv',dtype={'station_id':str})
    baseline=set(old_stations.station_key)
    old_sets=json.loads((old_config/'feature_sets.json').read_text(encoding='utf-8'))['sets']
    common=json.loads((old_config/'experiments.json').read_text(encoding='utf-8'))['common_features']
    west24=set(pd.read_csv(root/'Dataset/Origin_Features_2021_2024_v3/japan_balanced_stations.csv').station_key)
    with zipfile.ZipFile(archive_path) as archive:
        west_table=pd.read_csv(archive.open('Dataset/Japan/selected_stations_1107.csv'),dtype={'station_id':str})
        west25=set(west_table.loc[west_table.selected.eq(True),'station_key'])
        added=pd.read_csv(archive.open('Dataset/Regions_1107_Station_History/added_station_registry.csv'),dtype={'station_id':str})
        cne=set(added.loc[added.source_dataset.eq('China_East'),'station_key'])
        jpe=set(added.loc[added.source_dataset.eq('Japan_East'),'station_key'])
        added_dictionary=pd.read_csv(archive.open('Dataset/Regions_1107_Station_History/feature_dictionary.csv'))
    assert len(baseline)==600 and len(west24)==24 and len(west25)==25 and len(west24&west25)==9
    assert len(cne)==16 and len(jpe)==18 and cne.isdisjoint(baseline) and jpe.isdisjoint(baseline)
    assert not cne&jpe and len(common)==70 and 'KR:111121' not in baseline
    old_dict=pd.read_csv(root/'Dataset/Origin_Features_2021_2024_v3/feature_dictionary.csv')
    national_dict=pd.read_csv(root/'Dataset/National_Station_History_2021_2024/feature_dictionary.csv')
    ownership={}
    for dictionary in [old_dict,national_dict]:
        for row in dictionary.itertuples():
            if row.role=='feature' and row.definition_key=='station_history':
                parameter=json.loads(row.parameters_json)
                key=parameter['station_key']
                stat=parameter['stat']
                ownership.setdefault(key,{})[stat]=row.name
    for row in added_dictionary.itertuples():
        if row.role=='feature':
            assert row.station_key in cne|jpe
            assert row.stat in STATS
            ownership.setdefault(row.station_key,{})[row.stat]=row.name
    union=baseline|west25|cne|jpe
    for key in union:
        assert set(ownership[key])==STATS,(key,ownership[key])
        assert len(set(ownership[key].values()))==16
    conditions={}
    for c,e,w in product((False,True),repeat=3):
        keys=(baseline-west24)|west25 if w else baseline.copy()
        if c:keys|=cne
        if e:keys|=jpe
        assert len(keys)==COUNTS[(c,e,w)]
        features=set(common)
        for key in keys:features.update(ownership[key].values())
        assert len(features)==70+16*len(keys)
        conditions[(c,e,w)]={'stations':keys,'features':features}
    assert conditions[(False,False,False)]['features']==set(old_sets['F'])
    return {'conditions':conditions,'common':set(common),'baseline':baseline,'west24':west24,
            'west25':west25,'west_removed':west24-west25,'west_added':west25-west24,
            'china_east':cne,'japan_east':jpe,'ownership':ownership}
