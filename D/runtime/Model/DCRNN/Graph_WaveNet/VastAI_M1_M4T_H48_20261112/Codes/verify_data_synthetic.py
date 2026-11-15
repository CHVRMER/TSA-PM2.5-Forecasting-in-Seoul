"""Small causal/time/country matching tests; no project files are modified."""
import json
import numpy as np
import pandas as pd
from trajectory_features import aggregate_pm_features, pm_feature_layout


def fixture():
    origins = pd.date_range('2021-01-05', periods=2, freq='3h', tz='Asia/Seoul')
    hours = pd.date_range('2021-01-01', '2021-01-06', freq='h', tz='Asia/Seoul')
    nodes = pd.DataFrame({'station_key':['KR:111121','KR:1','CN:1','CN:2','JP:1','MN:1'],
                          'country':['KR','KR','CN','CN','JP','MN']})
    raw = np.broadcast_to([5.,10.,20.,40.,50.,60.], (len(hours),6)).copy()
    points=[]; edges=[]
    for oi,origin in enumerate(origins):
        for age in range(1,49):
            point=dict(point_id=f'{oi}:{age}',arrival_time_kst=origin,
                       pass_time_kst=origin-pd.Timedelta(hours=age),lookback_hours=age,
                       height_agl_m=100. if age <=24 else 1000.,diagnostic_mixdepth=500.)
            points.append(point)
            for ni,key in enumerate(nodes.station_key):
                edges.append({k:point[k] for k in ('point_id','arrival_time_kst','pass_time_kst','lookback_hours')} |
                             dict(station_key=key,distance_km=50. if key=='CN:2' else 0.,pm25=9999.))
            # Outside the selected fixed node universe: must never enter aggregates.
            edges.append({k:point[k] for k in ('point_id','arrival_time_kst','pass_time_kst','lookback_hours')} |
                         dict(station_key='JP:UNSELECTED',distance_km=0.,pm25=8888.))
    bundle={'nodes':nodes,'hourly_index':hours,'pm_values':raw,
            'origins':pd.DataFrame({'origin_time_kst':origins})}
    return bundle,pd.DataFrame(points),pd.DataFrame(edges),origins


def main():
    bundle,points,edges,origins=fixture()
    names=[x[0] for x in pm_feature_layout()]
    result,diag=aggregate_pm_features(bundle,points,[edges.iloc[:350],edges.iloc[350:]])
    col=lambda name:names.index(name)
    expected=(20+np.exp(-.5)*40)/(1+np.exp(-.5))
    assert result.shape==(2,208)
    assert np.allclose(result[:,col('traj_pm25_cn_b01_06_mean')],expected)
    assert np.all(result[:,col('traj_pm25_kr_b01_06_mean')]==10)
    assert np.all(result[:,col('traj_pm25_cn_b01_06_mean_station_count')]==2)
    assert np.all(result[:,col('traj_pm25_cn_b01_06_unique_stations')]==2)
    assert np.all(result[:,col('traj_pm25_jp_b01_06_coverage')]==1)
    assert np.isnan(result[:,col('traj_pm25_cn_b25_48_inside_mix_mean')]).all()
    assert np.allclose(result[:,col('traj_pm25_cn_b25_48_above_mix_mean')],expected)
    # Saved values deliberately wrong: all donor PM must come from raw pass-time join.
    assert diag['saved_pm_vs_current_raw_mismatches']==len(points)*5
    assert diag['selected_neighbor_rows']==len(points)*5
    future=dict(bundle,pm_values=bundle['pm_values'].copy())
    future['pm_values'][bundle['hourly_index']>origins[0],:]=100000.
    later,_=aggregate_pm_features(future,points,[edges])
    assert np.array_equal(result[0],later[0],equal_nan=True)
    assert not np.array_equal(result[1],later[1],equal_nan=True)
    excluded=dict(bundle,pm_values=bundle['pm_values'].copy())
    excluded['pm_values'][:,4:]=100000.
    other,_=aggregate_pm_features(excluded,points,[edges])
    safe=[i for i,row in enumerate(pm_feature_layout()) if row[1] in ('KR','CN')]
    assert np.array_equal(result[:,safe],other[:,safe],equal_nan=True)
    # Missing raw observations remove coverage without changing geometric denominators.
    missing=dict(bundle,pm_values=bundle['pm_values'].copy())
    pass_row=bundle['hourly_index'].get_loc(origins[0]-pd.Timedelta(hours=1))
    missing['pm_values'][pass_row,2:4]=np.nan
    partial,_=aggregate_pm_features(missing,points,[edges])
    assert np.isclose(partial[0,col('traj_pm25_cn_b01_06_coverage')],5/6)
    assert partial[0,col('traj_pm25_cn_b01_06_valid_points')]==5
    for batches,kwargs in (([edges,edges],{}),([edges],{'radius_km':100.})):
        try:
            aggregate_pm_features(bundle,points,batches,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate matching or unsupported radius was accepted.')
    print(json.dumps({'tests':'passed','weighted_raw_rejoin':True,'future_mutation_invariance':True,
        'excluded_country_invariance':True,'fixed_band_denominators':True,
        'vertical_strata':True,'duplicate_rejection':True,'radius_100_rejection':True}))


if __name__=='__main__':
    main()
