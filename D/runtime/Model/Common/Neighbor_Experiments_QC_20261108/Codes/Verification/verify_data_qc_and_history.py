"""Independent comparison of frozen causal QC and every rebuilt 34-series feature."""
from pathlib import Path
import argparse,io,json,zipfile
import numpy as np
import pandas as pd
from independent_qc_expectations import expected_pm25_quarantine
from independent_history import expected_history


def readzip(archive,path):
    return pd.read_parquet(io.BytesIO(archive.read(path)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--qc-hourly',type=Path,required=True)
    parser.add_argument('--qc-buffer',type=Path,required=True)
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        old=pd.concat([readzip(archive,'Dataset/China_East/'+name) for name in
                       ('china_east_cities_hourly_2021_2024_kst.parquet','china_east_cities_context_buffer.parquet')],ignore_index=True)
        reference=expected_pm25_quarantine(old)
        expected=old.copy()
        for material,column in [('pm25','quarantine_pm25'),('pm10','quarantine_pm10')]:
            expected.loc[reference[column],material]=np.nan
            expected.loc[reference[column],material+'_flag']=20
        expected['__pm25_bits']=reference.extreme_jump.astype(int)+2*reference.coarse_incoherence.astype(int)
        expected['__pm10_bits']=4*reference.quarantine_pm10.astype(int)
        clean=pd.concat([pd.read_parquet(args.qc_hourly),pd.read_parquet(args.qc_buffer)],ignore_index=True)
        sort=['station_id','time_kst']
        assert not clean.duplicated(sort).any() and len(clean)==len(old)
        clean=clean.sort_values(sort).reset_index(drop=True)
        expected=expected.sort_values(sort).reset_index(drop=True)
        old_sorted=old.sort_values(sort).reset_index(drop=True)
        assert np.array_equal(clean.station_id.to_numpy(),expected.station_id.to_numpy())
        assert pd.DatetimeIndex(clean.time_kst).equals(pd.DatetimeIndex(expected.time_kst))
        for material in ('pm25','pm10'):
            assert np.allclose(clean[material],expected[material],rtol=0,atol=0,equal_nan=True),material
            assert np.array_equal(clean[material+'_flag'],expected[material+'_flag']),material+'_flag'
            assert np.allclose(clean[material+'_raw'],old_sorted[material],rtol=0,atol=0,equal_nan=True)
            assert np.array_equal(clean[material+'_raw_flag'],old_sorted[material+'_flag'])
            assert np.array_equal(clean[material+'_qc_reason_bits'],expected['__'+material+'_bits'])
            assert np.array_equal(clean[material+'_qc_suspect'],expected[material+'_flag'].eq(20))
        for column in ('time_cst','source_id','split'):
            pd.testing.assert_series_equal(clean[column],old_sorted[column],check_dtype=False)
        report={'status':'passed','china_rows':len(clean),'pm25_quarantines':int(reference.quarantine_pm25.sum()),
                'pm10_quarantines':int(reference.quarantine_pm10.sum()),
                'QC_policy_recomputed_independently_from_raw':True,'zip_code_executed':False,
                'unaltered_observations_exact':True,'raw_values_flags_provenance_exact':True,
                'QC_reason_bits_recomputed_independently':True}
        assert report['pm25_quarantines']==3 and report['pm10_quarantines']==1
        changed=old.loc[reference.quarantine_pm25|reference.quarantine_pm10].copy()
        assert changed.time_kst.dt.year.eq(2021).all()
        side=pd.read_parquet(args.history)
        assert len(side)==11688 and not side.origin_time_kst.duplicated().any()
        origins=pd.DatetimeIndex(side.origin_time_kst).as_unit('ns')
        assert origins.equals(pd.date_range('2021-01-01','2025-01-01',freq='3h',inclusive='left',tz='Asia/Seoul').as_unit('ns'))
        dictionary=pd.read_csv(args.dictionary)
        registry=pd.read_csv(archive.open('Dataset/Regions_1107_Station_History/added_station_registry.csv'))
        total=bad=0;maximum=0.;checks=[]
        for source,mainfile,bufferfile in [
            ('China_East',None,None),
            ('Japan_East','japan_east_hourly_2021_2024_kst.parquet','japan_east_context_buffer.parquet')]:
            if source=='China_East':raw=clean
            else:
                selected_ids=registry.loc[registry.source_dataset.eq(source),'station_id'].astype(str)
                raw=pd.concat([readzip(archive,'Dataset/'+source+'/'+filename) for filename in (mainfile,bufferfile)],ignore_index=True)
                raw=raw.loc[raw.station_id.astype(str).isin(selected_ids)].copy()
            for row in registry.loc[registry.source_dataset.eq(source)].itertuples():
                group=raw.loc[raw.station_id.astype(str).eq(str(row.station_id))].sort_values('time_kst')
                valid=group.pm25.where(group.pm25_flag.eq(0))
                wanted=expected_history(group.time_kst,valid,origins)
                definitions=dictionary.loc[dictionary.station_key.eq(row.station_key)] if 'station_key' in dictionary else None
                if definitions is None:
                    parsed=dictionary.parameters_json.map(json.loads)
                    definitions=dictionary.loc[parsed.map(lambda x:x.get('station_key')==row.station_key)].copy()
                    definitions['stat']=parsed.loc[definitions.index].map(lambda x:x['stat'])
                assert len(definitions)==16 and definitions.stat.nunique()==16
                for definition in definitions.itertuples():
                    actual=side[definition.name].to_numpy(float);expected_values=wanted[definition.stat]
                    matches=np.isclose(actual,expected_values,rtol=1e-6,atol=1e-4,equal_nan=True)
                    count=int((~matches).sum());bad+=count;total+=len(actual)
                    finite=np.isfinite(actual)&np.isfinite(expected_values)
                    error=float(np.max(np.abs(actual[finite]-expected_values[finite]))) if finite.any() else 0.
                    maximum=max(maximum,error)
                    checks.append({'feature':definition.name,'station_key':row.station_key,'stat':definition.stat,
                                   'mismatches':count,'finite_max_abs_error':error})
        assert len(checks)==544 and bad==0
        report.update(features=len(checks),origins=len(origins),values_compared=total,mismatched_values=bad,
                      finite_max_abs_error=maximum,rtol=1e-6,atol=1e-4)
        original_side=readzip(archive,'Dataset/Regions_1107_Station_History/regions_1107_station_history.parquet')
        names=[row['feature'] for row in checks]
        change=~np.isclose(original_side[names].to_numpy(float),side[names].to_numpy(float),rtol=0,atol=0,equal_nan=True)
        changed_features={names[n] for n in np.flatnonzero(change.any(axis=0))}
        expected_owners={'CN:city_lianyungang','CN:city_linyi','CN:city_yancheng'}
        name_owners={row['feature']:row['station_key'] for row in checks}
        assert {name_owners[name] for name in changed_features}==expected_owners
        assert int(change.sum())==93 and int(change.any(axis=1).sum())==24
        assert int(change.any(axis=0).sum())==21
        report.update(changed_feature_cells_from_original=int(change.sum()),
                      changed_origins_from_original=int(change.any(axis=1).sum()),
                      changed_feature_columns_from_original=len(changed_features),
                      changes_limited_to_three_QC_cities=True,
                      all_Japan_history_and_other_cells_bitwise_equal=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(checks).to_csv(args.report.with_name('rebuilt_history_feature_checks.csv'),index=False,encoding='utf-8-sig')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
