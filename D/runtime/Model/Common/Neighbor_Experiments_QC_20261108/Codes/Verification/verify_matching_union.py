"""Independently check old matching preservation and all QC-aligned new matches."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal


class Rows:
    def __init__(self,path):self.iterator=iter(pq.ParquetFile(path).iter_batches(batch_size=500000));self.pending=None
    def take(self,n):
        result=[]
        while n:
            if self.pending is None or not len(self.pending):self.pending=pa.Table.from_batches([next(self.iterator)])
            use=min(n,len(self.pending));result.append(self.pending.slice(0,use));self.pending=self.pending.slice(use);n-=use
        return pa.concat_tables(result)
    def rest(self):
        parts=[self.pending] if self.pending is not None and len(self.pending) else []
        parts.extend(pa.Table.from_batches([batch]) for batch in self.iterator)
        return pa.concat_tables(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--matching-dir',type=Path,required=True)
    parser.add_argument('--qc-data-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();old=args.project_root/'Dataset/Trajectory_Matching_National_v4'
    oldkeys=set(pd.read_csv(old/'station_registry.csv').station_key)
    registry=pd.read_csv(args.matching_dir/'station_registry.csv',dtype={'station_id':str})
    assert len(registry)==856 and registry.station_key.nunique()==856
    newkeys=set(registry.station_key)-oldkeys;assert len(newkeys)==34
    reader=Rows(args.matching_dir/'trajectory_station_neighbors.parquet');kept=0
    for batch in pq.ParquetFile(old/'trajectory_station_neighbors.parquet').iter_batches(batch_size=500000):
        original=pa.Table.from_batches([batch]);actual=reader.take(len(original))
        if not original.equals(actual):assert_frame_equal(original.to_pandas(),actual.to_pandas(),check_dtype=False,check_exact=True)
        kept+=len(original)
    added=reader.rest().to_pandas();assert len(added)==6916
    assert set(added.station_key)<=newkeys and not added.duplicated(['point_id','station_key']).any()
    for col in ('arrival_time_kst','pass_time_kst'):added[col]=pd.DatetimeIndex(added[col]).as_unit('ns')
    assert (added.pass_time_kst==added.arrival_time_kst-pd.to_timedelta(added.lookback_hours,unit='h')).all()
    lookup=registry.set_index('station_key')
    expected=np.full(len(added),np.nan)
    for source,filenames in {
        'China_East_QC':['china_east_hourly_qc_2021_2024_kst.parquet','china_east_context_buffer_qc.parquet'],
        'Japan_East':['japan_east18_hourly_2021_2024_kst.parquet','japan_east18_context_buffer.parquet']}.items():
        raw=pd.concat([pd.read_parquet(args.qc_data_dir/filename,columns=['time_kst','station_id','pm25','pm25_flag']) for filename in filenames],ignore_index=True)
        raw['station_id']=raw.station_id.astype(str);raw['time_kst']=pd.DatetimeIndex(raw.time_kst).as_unit('ns')
        raw['value']=raw.pm25.where(raw.pm25_flag.eq(0)).astype('float32').astype('float64')
        assert not raw.duplicated(['station_id','time_kst']).any()
        values=raw.set_index(['station_id','time_kst']).value
        select=lookup.loc[added.station_key,'source_dataset'].eq(source).to_numpy()
        keys=pd.MultiIndex.from_arrays([lookup.loc[added.loc[select,'station_key'],'station_id'].to_numpy(),added.loc[select,'pass_time_kst']])
        expected[select]=values.reindex(keys).to_numpy()
    assert np.allclose(expected,added.pm25,rtol=0,atol=0,equal_nan=True)
    assert np.array_equal(np.isfinite(expected),added.pm25_available.to_numpy(bool))
    coordinates=pq.read_table(args.project_root/'Dataset/Trajectories/trajectory_points_2021_2024.parquet',
        columns=['arrival_time_kst','lookback_hours','longitude','latitude'],
        filters=[('arrival_height_agl_m','=',500.),('lookback_hours','>=',1.),('lookback_hours','<=',48.)]).to_pandas()
    coordinates['arrival_time_kst']=pd.DatetimeIndex(coordinates.arrival_time_kst).as_unit('ns')
    joined=added.merge(coordinates,on=['arrival_time_kst','lookback_hours'],how='left',validate='many_to_one',indicator=True)
    assert joined['_merge'].eq('both').all()
    ids=joined.arrival_time_kst.dt.tz_convert('UTC').dt.strftime('%Y%m%dT%H%MZ')+'|500m|'+joined.lookback_hours.astype(int).astype(str).str.zfill(2)+'h'
    assert ids.eq(joined.point_id).all()
    lat=np.deg2rad(joined.latitude.to_numpy());lon=np.deg2rad(joined.longitude.to_numpy())
    slat=np.deg2rad(lookup.loc[joined.station_key,'latitude'].to_numpy());slon=np.deg2rad(lookup.loc[joined.station_key,'longitude'].to_numpy())
    hav=np.sin((slat-lat)/2)**2+np.cos(lat)*np.cos(slat)*np.sin((slon-lon)/2)**2
    d=12742*np.arcsin(np.sqrt(np.clip(hav,0,1)))
    assert np.allclose(d,joined.distance_km,rtol=0,atol=1e-6) and np.all(d<=75+1e-9)
    oldcoverage=pd.read_parquet(old/'trajectory_point_coverage.parquet')
    newcoverage=pd.read_parquet(args.matching_dir/'trajectory_point_coverage.parquet')
    assert len(oldcoverage)==len(newcoverage)==561024
    other=[col for col in oldcoverage if col not in ('nearby_station_count','observed_pm25_station_count')]
    assert_frame_equal(oldcoverage[other],newcoverage[other],check_dtype=False,check_exact=True)
    counts=added.groupby('point_id').size().reindex(newcoverage.point_id,fill_value=0).to_numpy()
    available=added.groupby('point_id').pm25_available.sum().reindex(newcoverage.point_id,fill_value=0).to_numpy()
    assert np.array_equal(newcoverage.nearby_station_count,oldcoverage.nearby_station_count+counts)
    assert np.array_equal(newcoverage.observed_pm25_station_count,oldcoverage.observed_pm25_station_count+available)
    # Same chosen old600 nodes, before nearest-node selection, produce exactly the same assignments/graph.
    baseline=pd.read_csv(args.project_root/'Model/Common/Regional_Experiments_20261107/station_registry.csv')
    basekeys=set(baseline.station_key);assert newkeys.isdisjoint(basekeys)
    nodes=pd.concat([pd.DataFrame([{'station_key':'KR:111121','longitude':126.98,'latitude':37.57}]),baseline],ignore_index=True)
    graph_path=args.project_root/'Model/DCRNN/V3_Data_20261107/Codes/original_graphs.py'
    spec=importlib.util.spec_from_file_location('original_graph_algorithm_under_audit',graph_path)
    graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
    start=pd.Timestamp('2021-01-01',tz='Asia/Seoul');end=pd.Timestamp('2021-01-08',tz='Asia/Seoul')
    filters=[('arrival_time_kst','>=',start),('arrival_time_kst','<',end),('station_key','in',sorted(basekeys))]
    columns=['point_id','arrival_time_kst','lookback_hours','station_key','distance_km']
    assignments=[];adjacencies=[]
    arrivals=pd.date_range(start,end,freq='3h',inclusive='left')
    for folder in (old,args.matching_dir):
        edges=pq.read_table(folder/'trajectory_station_neighbors.parquet',columns=columns,filters=filters).to_pandas()
        assignment=graph.nearest_trajectory_assignments(edges,nodes)
        adjacency,info=graph.trajectory_adjacency_from_assignments(assignment,nodes,all_arrivals=arrivals)
        assignments.append(assignment);adjacencies.append(adjacency)
    assert_frame_equal(assignments[0],assignments[1],check_exact=True)
    assert np.array_equal(adjacencies[0],adjacencies[1])
    report={'status':'passed','retained_old_matching_rows_all8columns_exact':kept,
            'new_rows_key_time_distance_PM_availability_verified':len(added),'matching_candidate_union':856,
            'new_rows_disjoint_from_old_baseline600':True,
            'all_old_point_metadata_and_flags_preserved':True,'coverage_counts_recomputed':True,
            'oneweek_baseline_nearest_assignments_exact':True,'oneweek_baseline_trajectory_adjacency_exact':True,
            'baseline_graph_node_order':'same fixed601-node order for old/new source comparison',
            'new_PM_quarantine_hours_matching_rows':0}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
