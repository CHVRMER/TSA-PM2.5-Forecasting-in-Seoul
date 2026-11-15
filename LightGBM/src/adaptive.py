"""Frozen M1/M3 original-unit mixture. Fit on validation only; no training/test IO."""
import numpy as np
import pandas as pd

SEASONS=('cold','warm')
CONDITIONS={'wind':('west','north','east','south','calm'),'route':('west','north','east','local')}

def condition_frame(frame):
    d=frame.copy()
    d['season']=np.where(d.origin_time_kst.dt.month.isin([11,12,1,2,3]),'cold','warm')
    d['route_group']=d.route_history12h_sector.map({'NW':'west','W':'west','SW':'west','N':'north','NE':'north','E':'east','SE':'east','S':'east','LOCAL':'local'})
    u,v=d.asos108_wind_u_now,d.asos108_wind_v_now
    a=(np.degrees(np.arctan2(-u,-v))+360)%360
    direction=np.select([(a>=225)&(a<315),(a>=315)|(a<45),(a>=45)&(a<135)],['west','north','east'],default='south')
    d['wind_sector']=pd.Series(direction,index=d.index).where(u.notna()&v.notna(),'unknown').where(d.asos108_wind_calm_now.ne(1),'calm')
    return d

def fit_weight(y,a,b,epsilon):
    y,a,b=[np.asarray(x,dtype=np.float64) for x in (y,a,b)]
    if y.ndim!=1 or y.shape!=a.shape or a.shape!=b.shape or not all(np.isfinite(x).all() for x in (y,a,b)) or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Invalid weight estimation input')
    delta=b-a;n=len(delta);den=float(delta@delta);num=float((y-a)@delta);threshold=float(n*epsilon**2)
    reason='empty_pool' if n==0 else 'zero_denominator' if den==0 else 'near_zero_denominator' if den<=threshold else 'none'
    raw=None if reason!='none' else num/den
    return {'n_estimation_pool':n,'numerator':num,'denominator':den,'epsilon':float(epsilon),'threshold_value':threshold,'near_zero_exception_used':reason!='none','exception_reason':reason,'unclipped_weight':raw,'weight_clipped':False if raw is None else bool(raw<0 or raw>1),'final_weight':.5 if raw is None else float(np.clip(raw,0,1))}

def fit_horizon(frame,horizon,epsilon):
    if not frame.origin_time_kst.dt.year.eq(2023).all():raise ValueError('Weights require 2023 validation rows only')
    seasons=[];cells=[]
    def fit(g):return fit_weight(g.y,g.M1_raw,g.M3_raw,epsilon)
    for season in SEASONS:
        sub=frame[frame.season.eq(season)];season_result=fit(sub);pool=f'h{horizon}:season:{season}'
        seasons.append({'horizon':horizon,'season':season,'estimation_pool':pool,**season_result})
        for kind,conditions in CONDITIONS.items():
            col='wind_sector' if kind=='wind' else 'route_group'
            for condition in conditions:
                g=sub[sub[col].eq(condition)];fallback=len(g)<100
                selected=season_result if fallback else fit(g)
                delta=g.M3_raw-g.M1_raw
                cells.append({'horizon':horizon,'kind':kind,'season':season,'condition':condition,'n_cell':len(g),'cell_numerator':float(((g.y-g.M1_raw)*delta).sum()),'cell_denominator':float((delta**2).sum()),'season_fallback_used':fallback,'fallback_reason':'cell_n_less_than_100' if fallback else 'none','estimation_pool':pool if fallback else f'h{horizon}:{kind}:{season}:{condition}',**selected})
    return pd.DataFrame(seasons),pd.DataFrame(cells)

def apply_horizon(frame,seasons,cells,kind):
    """No y required. Unknown/unseen condition uses its season rule."""
    col='wind_sector' if kind=='wind' else 'route_group';records=[]
    lookup={(r['season'],r['condition']):r for r in cells[cells.kind.eq(kind)].to_dict('records')}
    season_rules={r['season']:r for r in seasons.to_dict('records')}
    for s,c in zip(frame.season,frame[col]):
        if s not in season_rules:raise ValueError('Unknown season')
        rule=lookup.get((s,c));unknown=rule is None
        r=season_rules[s] if unknown else rule
        records.append({'condition':'unknown' if pd.isna(c) else c,'weight':r['final_weight'],'estimation_pool':r['estimation_pool'],'n_estimation_pool':r['n_estimation_pool'],'denominator':r['denominator'],'epsilon':r['epsilon'],'threshold_value':r['threshold_value'],'season_fallback_used':True if unknown else r['season_fallback_used'],'fallback_reason':'unknown_condition' if unknown else r['fallback_reason'],'near_zero_exception_used':r['near_zero_exception_used'],'exception_reason':r['exception_reason'],'weight_clipped':r['weight_clipped']})
    d=pd.DataFrame(records,index=frame.index)
    d['y_hat_raw']=(1-d.weight)*frame.M1_raw+d.weight*frame.M3_raw
    return d
