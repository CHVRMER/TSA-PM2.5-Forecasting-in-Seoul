"""Synthetic checks of boundaries, estimation, pooling, and no-label application."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.adaptive import fit_weight,fit_horizon,apply_horizon

def main():
    assert fit_weight([1,2],[1,2],[1,2],1e-4)['final_weight']==.5
    assert fit_weight([1],[1],[1+1e-8],1e-4)['exception_reason']=='near_zero_denominator'
    assert fit_weight([],[],[],1e-4)['final_weight']==.5
    assert fit_weight([.75],[0],[1],.01)['final_weight']==.75
    assert fit_weight([2],[0],[1],.01)['final_weight']==1
    assert fit_weight([-1],[0],[1],.01)['final_weight']==0
    # At the exact threshold use .5; one representable value above it is estimated normally.
    assert fit_weight([.4],[0],[1],1)['final_weight']==.5
    assert fit_weight([.4],[0],[np.nextafter(1.,2.)],1)['near_zero_exception_used'] is False
    n=201;d=pd.DataFrame({'origin_time_kst':pd.date_range('2023-01-01',periods=n,freq='3h',tz='Asia/Seoul'),'season':'cold','wind_sector':['west']*100+['east']*99+['calm','unknown'],'route_group':'west','M1_raw':0.,'M3_raw':1.,'y':[.2]*100+[.8]*101})
    s,c=fit_horizon(d,6,1e-4);west=c[(c.kind=='wind')&c.condition.eq('west')&c.season.eq('cold')].iloc[0];east=c[(c.kind=='wind')&c.condition.eq('east')&c.season.eq('cold')].iloc[0]
    assert west.n_cell==100 and not west.season_fallback_used and np.isclose(west.final_weight,.2)
    assert east.n_cell==99 and east.season_fallback_used and np.isclose(east.final_weight,d.y.mean())
    out=apply_horizon(d.drop(columns='y'),s,c,'wind');assert out.iloc[-1].fallback_reason=='unknown_condition' and np.isclose(out.iloc[-1].weight,d.y.mean())
    tiny=d.copy();tiny['M3_raw']=1e-8;ss,cc=fit_horizon(tiny,6,1e-4);rr=apply_horizon(tiny.drop(columns='y'),ss,cc,'route');assert rr.near_zero_exception_used.all() and rr.weight.eq(.5).all()
    # Closed-form solution must minimize SSE on its own estimation pool over a dense grid.
    rng=np.random.default_rng(7);a=rng.normal(size=150);b=rng.normal(size=150);y=rng.normal(size=150);w=fit_weight(y,a,b,1e-6)['final_weight'];loss=np.sum((y-((1-w)*a+w*b))**2);grid=np.linspace(0,1,10001);assert loss<=np.min(np.sum((y[:,None]-((1-grid)*a[:,None]+grid*b[:,None]))**2,axis=0))+1e-10
    print('PASS: zero/tiny denominator, threshold boundary, clipping, n99/n100, seasonal fallback, unknown condition, label-free apply, SSE optimum')
if __name__=='__main__':main()
