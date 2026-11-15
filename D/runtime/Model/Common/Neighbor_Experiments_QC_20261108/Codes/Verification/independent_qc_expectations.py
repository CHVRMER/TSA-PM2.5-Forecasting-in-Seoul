"""Conservative causal China PM2.5 QC reference, independent of producer code."""
import numpy as np
import pandas as pd


def expected_pm25_quarantine(raw):
    """Return masks for current/past-only frozen rules, aligned to input index."""
    output=pd.DataFrame(index=raw.index,columns=['past6_valid','past6_median','extreme_jump','coarse_incoherence',
                                                'pm10_past6_valid','pm10_past6_median','pm10_extreme_jump'])
    for _,group in raw.groupby('station_id',sort=False):
        group=group.sort_values('time_kst')
        valid=group.pm25.where(group.pm25_flag.eq(0)&np.isfinite(group.pm25)&group.pm25.ge(0))
        indexed=pd.Series(valid.to_numpy(),index=pd.DatetimeIndex(group.time_kst))
        # Exact hourly grid, so rows correspond to clock hours even when values are missing.
        assert indexed.index.is_unique
        assert indexed.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
        previous=indexed.shift(1).rolling(6,min_periods=4)
        median=previous.median().to_numpy()
        count=indexed.shift(1).rolling(6,min_periods=1).count().to_numpy()
        pm=group.pm25.to_numpy(float)
        current=group.pm25_flag.eq(0).to_numpy()&np.isfinite(pm)&(pm>=0)
        coarse=group.pm10.to_numpy(float)
        coarse_valid=group.pm10_flag.eq(0).to_numpy()&np.isfinite(coarse)&(coarse>0)
        jump=current&(count>=4)&(pm>=1000)&(pm>10*np.maximum(median,10))&(pm-median>500)
        incoherent=current&coarse_valid&(pm>=300)&(pm>2*coarse)&(pm-coarse>100)
        coarse_series=pd.Series(np.where(coarse_valid,coarse,np.nan),index=indexed.index)
        coarse_median=coarse_series.shift(1).rolling(6,min_periods=4).median().to_numpy()
        coarse_count=coarse_series.shift(1).rolling(6,min_periods=1).count().to_numpy()
        coarse_jump=coarse_valid&(coarse_count>=4)&(coarse>=5000)&(coarse>10*np.maximum(coarse_median,10))&(coarse-coarse_median>3000)
        output.loc[group.index,'past6_valid']=count
        output.loc[group.index,'past6_median']=median
        output.loc[group.index,'extreme_jump']=jump
        output.loc[group.index,'coarse_incoherence']=incoherent
        output.loc[group.index,'pm10_past6_valid']=coarse_count
        output.loc[group.index,'pm10_past6_median']=coarse_median
        output.loc[group.index,'pm10_extreme_jump']=coarse_jump
    output['extreme_jump']=output.extreme_jump.astype(bool)
    output['coarse_incoherence']=output.coarse_incoherence.astype(bool)
    output['quarantine_pm25']=output.extreme_jump|output.coarse_incoherence
    output['quarantine_pm10']=output.pm10_extreme_jump.astype(bool)
    return output
