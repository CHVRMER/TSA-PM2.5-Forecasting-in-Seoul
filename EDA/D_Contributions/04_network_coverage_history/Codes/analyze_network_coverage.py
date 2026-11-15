"""Compare external-network coverage on the same 561,024 historical path points."""
from pathlib import Path
import io
import json
import sys
import zipfile

HERE = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, default=HERE.parent/'Results')
args = parser.parse_args()
ARCHIVE = args.archive
CORE = {'KR:831492', 'CN:1662A', 'CN:1008A'}
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

def read(archive, name, columns=None, filters=None):
    return pq.read_table(io.BytesIO(archive.read('Data/Dataset/'+name)), columns=columns, filters=filters).to_pandas()

with zipfile.ZipFile(ARCHIVE) as archive:
    registry = pd.read_csv(io.BytesIO(archive.read('Data/Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv')))
    old_registry = pd.read_csv(io.BytesIO(archive.read('Data/Dataset/Trajectory_Matching_2021_2024/station_registry.csv')))
    balanced = pd.read_csv(io.BytesIO(archive.read('Data/Dataset/Origin_Features_2021_2024_v3/japan_balanced_stations.csv')))
    old_keys = set(old_registry.station_key)
    new_keys = set(registry.station_key)-old_keys
    assert CORE.issubset(old_keys) and len(old_keys)==115 and len(new_keys)==202
    points = read(archive, 'Trajectory_Matching_2021_2024/trajectory_point_coverage.parquet',
        columns=['point_id','arrival_time_kst','lookback_hours','latitude','longitude','height_agl_m',
                 'diagnostic_mixdepth','nearby_station_count','observed_pm25_station_count'])
    v3 = read(archive, 'Trajectory_Matching_2021_2024_v3/trajectory_point_coverage.parquet',
        columns=['point_id','nearby_station_count','observed_pm25_station_count'])
    selected_keys = sorted(CORE | new_keys)
    neighbors = read(archive, 'Trajectory_Matching_2021_2024_v3/trajectory_station_neighbors.parquet',
        columns=['point_id','station_key','pm25_available','pm25','distance_km'],
        filters=[('station_key','in',selected_keys)])
    features = read(archive, 'Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet',
        columns=['origin_time_kst','route_history12h_sector'])
points = points.sort_values(['arrival_time_kst','lookback_hours']).reset_index(drop=True)
assert len(points)==561024 and points.point_id.is_unique
v3 = v3.set_index('point_id').loc[points.point_id]
index = pd.Index(points.point_id)
n = len(points)
arrivals = pd.DatetimeIndex(points.arrival_time_kst.unique()).sort_values()
assert len(arrivals)==11688
origin_codes = arrivals.get_indexer(points.arrival_time_kst)
lookup = registry.set_index('station_key')
sets = {
    'core3': CORE,
    'Japan188': set(registry.loc[registry.country.eq('JP'),'station_key']),
    'Japan24': set(balanced.station_key),
    'Mongolia1': set(registry.loc[registry.country.eq('MN'),'station_key']),
    'CN_Mongolia_border6': set(registry.loc[registry.region_group.eq('Mongolia_border_proxy'),'station_key']),
    'CN_Russia_border7': set(registry.loc[registry.region_group.eq('Russia_border_proxy'),'station_key']),
}
geo, observed, neighbor_counts = {}, {}, {}
for name, keys in sets.items():
    edges = neighbors.loc[neighbors.station_key.isin(keys)]
    positions = index.get_indexer(edges.point_id)
    assert (positions>=0).all() and edges.distance_km.le(75+1e-9).all()
    geo[name] = np.zeros(n,dtype=bool)
    observed[name] = np.zeros(n,dtype=bool)
    geo[name][positions] = True
    valid = edges.pm25_available.to_numpy(bool)
    assert np.array_equal(valid, np.isfinite(edges.pm25.to_numpy(float)))
    observed[name][positions[valid]] = True
    neighbor_counts[name] = len(edges)
geo['old115'] = points.nearby_station_count.to_numpy()>0
observed['old115'] = points.observed_pm25_station_count.to_numpy()>0
new_groups = ['Japan188','Mongolia1','CN_Mongolia_border6','CN_Russia_border7']
new_geo = np.logical_or.reduce([geo[g] for g in new_groups])
new_obs = np.logical_or.reduce([observed[g] for g in new_groups])
geo['full317'] = geo['old115']|new_geo
observed['full317'] = observed['old115']|new_obs
geo['balanced153'] = geo['old115']|np.logical_or.reduce([geo[g] for g in ['Japan24',*new_groups[1:]]])
observed['balanced153'] = observed['old115']|np.logical_or.reduce([observed[g] for g in ['Japan24',*new_groups[1:]]])
assert np.array_equal(geo['full317'], v3.nearby_station_count.to_numpy()>0)
assert np.array_equal(observed['full317'], v3.observed_pm25_station_count.to_numpy()>0)
assert (geo['core3']<=geo['old115']).all() and (observed['core3']<=observed['old115']).all()

def origin_count(mask):
    return int(np.unique(origin_codes[mask]).size)

def origin_counts(mask):
    return np.bincount(origin_codes[mask],minlength=len(arrivals))

network_rows=[]
for name, count in [('core3',3),('old115',115),('balanced153',153),('full317',317)]:
    g,o=geo[name],observed[name]
    per_path=origin_counts(o)
    network_rows.append(dict(network=name,candidates=count,points=n,geographic_points=int(g.sum()),
        geographic_pct=100*g.mean(),observed_points=int(o.sum()),observed_pct=100*o.mean(),
        pm25_given_geographic_pct=100*o.sum()/g.sum(),
        arrivals_with_geographic=origin_count(g),arrivals_with_observed=origin_count(o),
        arrivals_with_observed_pct=100*origin_count(o)/len(arrivals),
        mean_observed_points_per_path=float(per_path.mean()),
        fully_observed_paths=int((per_path==48).sum()),
        geography_but_no_observed_points=int((g&~o).sum())))
networks=pd.DataFrame(network_rows)
networks.to_csv(OUT/'network_coverage.csv',index=False,encoding='utf-8-sig')

increment_rows=[]
for name in new_groups+['Japan24']:
    g,o=geo[name],observed[name]
    gain_g=g&~geo['old115'];gain_o=o&~observed['old115']
    mix_known=np.isfinite(points.diagnostic_mixdepth)&(points.diagnostic_mixdepth>0)&np.isfinite(points.height_agl_m)
    inside=(points.height_agl_m<=points.diagnostic_mixdepth)&(points.height_agl_m>=0)&mix_known
    increment_rows.append(dict(group=name,candidates=len(sets[name]),neighbor_rows=neighbor_counts[name],
        geographic_points=int(g.sum()),observed_points=int(o.sum()),geographic_arrivals=origin_count(g),observed_arrivals=origin_count(o),
        added_geographic_points=int(gain_g.sum()),added_geographic_percentage_points=100*gain_g.mean(),
        added_observed_points=int(gain_o.sum()),added_observed_percentage_points=100*gain_o.mean(),
        added_observed_arrivals=origin_count(gain_o),
        observed_given_geographic_pct=100*o.sum()/g.sum() if g.any() else None,
        observed_inside_mix_points=int((o&inside).sum()),
        observed_known_mix_points=int((o&mix_known).sum()),
        fraction_geo_at_25_48h=100*np.mean(points.lookback_hours[g].to_numpy()>=25) if g.any() else None))
increments=pd.DataFrame(increment_rows)
increments.to_csv(OUT/'new_candidate_coverage.csv',index=False,encoding='utf-8-sig')

bin_rows=[]
for lo,hi in [(1,6),(7,12),(13,24),(25,48)]:
    mask=points.lookback_hours.between(lo,hi).to_numpy()
    for name in ['core3','old115','balanced153','full317',*new_groups]:
        g,o=geo[name]&mask,observed[name]&mask
        bin_rows.append(dict(age_bin=f'{lo:02d}-{hi:02d}',network=name,points=int(mask.sum()),
            geographic_points=int(g.sum()),geographic_pct=100*g.sum()/mask.sum(),
            observed_points=int(o.sum()),observed_pct=100*o.sum()/mask.sum(),
            observed_arrivals=origin_count(o)))
pd.DataFrame(bin_rows).to_csv(OUT/'coverage_by_path_age.csv',index=False,encoding='utf-8-sig')

sectors=features.set_index('origin_time_kst').loc[arrivals,'route_history12h_sector']
point_sector=sectors.to_numpy()[origin_codes]
sector_rows=[]
for sector in sorted(pd.unique(sectors)):
    mask=point_sector==sector
    na=int((sectors==sector).sum())
    for name in ['core3','old115','full317',*new_groups]:
        g,o=geo[name]&mask,observed[name]&mask
        sector_rows.append(dict(sector=sector,network=name,points=int(mask.sum()),arrivals=na,
            geographic_pct=100*g.sum()/mask.sum(),observed_pct=100*o.sum()/mask.sum(),
            geographic_arrivals=origin_count(g),observed_arrivals=origin_count(o),
            observed_arrivals_pct=100*origin_count(o)/na))
pd.DataFrame(sector_rows).to_csv(OUT/'coverage_by_12h_route_sector.csv',index=False,encoding='utf-8-sig')

year_rows=[]
for year in [2021,2022,2023,2024]:
    mask=points.arrival_time_kst.dt.year.eq(year).to_numpy()
    for name in ['core3','old115','full317',*new_groups]:
        year_rows.append(dict(year=year,network=name,observed_pct=100*(observed[name]&mask).sum()/mask.sum(),
            observed_points=int((observed[name]&mask).sum()),observed_arrivals=origin_count(observed[name]&mask)))
pd.DataFrame(year_rows).to_csv(OUT/'coverage_by_year.csv',index=False,encoding='utf-8-sig')

# Rank new stations by how many old-network-uncovered points they connect, not by predictive importance.
station_rows=[]
for key,edges in neighbors.loc[neighbors.station_key.isin(new_keys)].groupby('station_key'):
    positions=index.get_indexer(edges.point_id)
    good=edges.pm25_available.to_numpy(bool)
    station_rows.append(dict(station_key=key,station_name=lookup.loc[key,'station_name'],country=lookup.loc[key,'country'],
        region_group=lookup.loc[key,'region_group'],geographic_points=len(np.unique(positions)),
        added_geographic_points=int((~geo['old115'][np.unique(positions)]).sum()),
        observed_points=int(good.sum()),added_observed_points=int((~observed['old115'][positions[good]]).sum())))
pd.DataFrame(station_rows).sort_values('added_observed_points',ascending=False).to_csv(OUT/'new_station_point_connections.csv',index=False,encoding='utf-8-sig')

report={
    'period':'2021-2024','arrivals':11688,'path_points':561024,'radius_km':75,
    'arrival_height_agl_m':500,'path_ages_hours':[1,48],
    'core3_external_nodes':sorted(CORE),'target_junggu_excluded_from_external_network_count':True,
    'candidate_registry_unchanged':True,'pm25_targets_or_model_performance_used':False,
    'v3_point_coverage_reproduced_from_group_unions':True,
    'new_geo_points_that_already_had_old_geography':int((new_geo&geo['old115']).sum()),
    'new_observed_points_that_already_had_old_observations':int((new_obs&observed['old115']).sum()),
    'new_group_geographic_overlap_points':int((np.sum([geo[g] for g in new_groups],axis=0)>1).sum()),
    'core3_to_old115_added_observed_points':int((observed['old115']&~observed['core3']).sum()),
    'old115_to_full317_added_observed_points':int((observed['full317']&~observed['old115']).sum()),
    'new_geographic_points':int(new_geo.sum()),'new_observed_points':int(new_obs.sum()),
    'additional_paths_with_any_new_observation':origin_count(new_obs&~observed['old115']),
    'new_obs_path_union_count':origin_count(new_obs),
    'networks':network_rows,'new_groups':increment_rows,
    'does_not_estimate_prediction_improvement_transport_mass_or_country_contribution':True,
}
(OUT/'coverage_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(networks.round(4).to_string(index=False))
print(increments.round(4).to_string(index=False))
print(json.dumps({k:v for k,v in report.items() if k not in ['networks','new_groups']},ensure_ascii=False,indent=2))
