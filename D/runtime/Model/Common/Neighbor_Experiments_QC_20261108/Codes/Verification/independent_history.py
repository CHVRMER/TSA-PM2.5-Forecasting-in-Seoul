"""History feature reference computed from exact-hour observations, no producer imports."""
import numpy as np
import pandas as pd


def expected_history(times, values, origins):
    index=pd.DatetimeIndex(times).tz_convert('Asia/Seoul').as_unit('ns')
    assert index.is_unique and index.is_monotonic_increasing
    assert index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    v=pd.Series(np.asarray(values,float),index=index)
    result={'now':v}
    for lag in (1,3,6,12,24):result[f'lag{lag:02d}h']=v.shift(lag)
    for hours in (6,24):
        window=v.rolling(hours,min_periods=1)
        result[f'mean{hours}h']=window.mean()
        result[f'std{hours}h']=window.std(ddof=0)
        result[f'coverage{hours}h']=v.notna().rolling(hours,min_periods=1).sum()/hours
    result['change3h']=v-v.shift(3)
    result['change6h']=v-v.shift(6)
    positions=np.arange(len(v),dtype=float)
    last=pd.Series(np.where(v.notna(),positions,np.nan),index=index).ffill()
    result['age_since_valid_h']=pd.Series(positions,index=index)-last
    result['missing_now']=v.isna().astype(float)
    origins=pd.DatetimeIndex(origins).tz_convert('Asia/Seoul').as_unit('ns')
    return {name:value.reindex(origins).to_numpy(float) for name,value in result.items()}
