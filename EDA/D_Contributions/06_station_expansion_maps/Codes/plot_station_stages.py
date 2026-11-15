"""Recreate two historical station maps from audited, packaged inputs.

This is a scientific redraw, not raster erasure. The same 822 candidate records,
192 display trajectories and viewport are used in both panels. The second map
adds only historical hand-drawn extension-area outlines and crosshatching.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch

ROOT=Path(__file__).resolve().parents[1]
SEOUL=(126.98,37.57)
COUNTS={'KR_OLD':67,'KR_NEW':505,'CN':48,'CN_BORDER':13,'JP':188,'MN':1}
GROUP_STYLES=[
    ('KR_OLD','South Korea, existing: 67','^','#d07a2d',26),
    ('KR_NEW','South Korea, national additions: 505','^','#f2b880',14),
    ('CN','China stations: 48','o','#a23b8f',26),
    ('CN_BORDER','China city means near Russia/Mongolia: 13','s','#8b4a3c',34),
    ('JP','Japan west, before reselection: 188','o','#2e7d5b',26),
    ('MN','Mongolia: 1','D','#5b3f8f',50),
]
NAMES=['01_national_expansion_before_international_additions',
       '02_national_expansion_with_planned_regions']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(folder):
    registry=pd.read_csv(folder/'Results/plotted_candidates_822.csv',dtype={'station_id':str})
    paths=pd.read_parquet(folder/'Results/display_trajectory_sample.parquet')
    outlines=json.loads((folder/'Results/planned_region_boundaries.geojson').read_text(encoding='utf-8'))
    land=json.loads((folder/'Codes/assets/countries.geojson').read_text(encoding='utf-8'))
    assert len(registry)==822 and registry.station_key.is_unique
    assert not registry.station_key.eq('KR:111121').any()
    assert registry.plot_group.value_counts().to_dict()==COUNTS
    assert registry.country.value_counts().to_dict()=={'KR':572,'JP':188,'CN':61,'MN':1}
    assert not registry.country.eq('RU').any()
    kr=registry.loc[registry.country.eq('KR')]
    assert not kr.network.str.contains('도로',na=False).any()
    assert np.isfinite(registry[['longitude','latitude']]).all().all()
    assert registry.longitude.between(100,145).all() and registry.latitude.between(25,55).all()
    assert paths.arrival_time_kst.nunique()==192
    assert len(paths)==192*49 and paths.arrival_height_agl_m.eq(500).all()
    counts=paths.groupby('arrival_time_kst').lookback_hours.agg(['count','nunique','min','max'])
    assert counts['count'].eq(49).all() and counts['nunique'].eq(49).all()
    assert counts['min'].eq(0).all() and counts['max'].eq(48).all()
    bymonth=pd.DatetimeIndex(paths.arrival_time_kst.unique()).strftime('%Y-%m')
    assert pd.Series(bymonth).value_counts().eq(4).all()
    assert len(outlines['features'])==2
    return registry,paths,outlines,land


def draw(registry,paths,outlines,land,planned,output):
    # Fixed viewport and axes rectangle preserve the attached image template.
    plt.rcParams.update({'font.family':'DejaVu Sans','hatch.linewidth':0.55,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(11,11),dpi=150)
    ax=fig.add_axes([0.057,0.160,0.917,0.764])
    for feature in land['features']:
        geom=feature['geometry']
        if geom['type'] not in ('Polygon','MultiPolygon'):continue
        pieces=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
        for poly in pieces:
            xy=np.asarray(poly[0],dtype=float)
            if xy[:,0].max()<100 or xy[:,0].min()>145 or xy[:,1].max()<25 or xy[:,1].min()>55:continue
            ax.fill(xy[:,0],xy[:,1],color='#f4f4f1',ec='#666666',lw=.5,zorder=0)
    ax.set_facecolor('#e9eff4')
    for _,part in paths.groupby('arrival_time_kst',sort=True):
        part=part.sort_values('lookback_hours')
        ax.plot(part.longitude,part.latitude,c='#8a9aa8',lw=.4,alpha=.45,zorder=1)
    if planned:
        for feature in outlines['features']:
            vertices=feature['geometry']['coordinates'][0]
            color=feature['properties']['color']
            ax.add_patch(Polygon(vertices,closed=True,facecolor='none',edgecolor=color,
                                 hatch='xx',linewidth=0,alpha=.32,zorder=2.5))
            xy=np.asarray(vertices,float)
            ax.plot(xy[:,0],xy[:,1],ls='--',c=color,lw=1.6,zorder=3)
    handles=[]
    drawn=[]
    for key,label,marker,color,size in GROUP_STYLES:
        group=registry.loc[registry.plot_group.eq(key)]
        assert len(group)==COUNTS[key]
        ax.scatter(group.longitude,group.latitude,marker=marker,s=size,c=color,
                   ec='white',lw=.4,zorder=4)
        handles.append(Line2D([],[],ls='',marker=marker,markerfacecolor=color,
                              markeredgecolor='white',markeredgewidth=.4,markersize=max(5,np.sqrt(size)),label=label))
        drawn.extend(group.station_key.tolist())
    assert len(drawn)==822 and len(set(drawn))==822
    ax.scatter(*SEOUL,marker='*',s=260,c='#b81d2c',ec='white',lw=.6,zorder=6)
    handles.append(Line2D([],[],ls='',marker='*',markerfacecolor='#b81d2c',
                          markeredgecolor='white',markersize=13,label='Seoul Jung-gu target (separate)'))
    if planned:
        handles.extend(Patch(facecolor='none',edgecolor=f['properties']['color'],hatch='xx',
                            linestyle='--',label=f['properties']['legend']) for f in outlines['features'])
    ax.set(xlim=(100,145),ylim=(25,55),xlabel='Longitude (°E)',ylabel='Latitude (°N)',aspect=1.25)
    ax.set_xticks(np.arange(100,146,5));ax.set_yticks(np.arange(25,56,5))
    ax.grid(alpha=.3)
    title=('National expansion stage: 822 candidates + planned extension areas' if planned else
           'After national Korea expansion: 822 candidates (before international additions)')
    ax.set_title(title,fontsize=12.2,pad=6)
    legend=ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.045),
                     ncol=2,fontsize=8.2,frameon=False,handlelength=1.5,columnspacing=1.8)
    foot=('Hatched areas mark later China East / Japan East additions; no new station points are included.' if planned else
          'Historical candidate set: Korea 572 + China 61 + Japan 188 + Mongolia 1; not the final model-input list.')
    fig.text(.5,.023,foot,ha='center',fontsize=7.25,color='#4b5963')
    fig.text(.5,.010,'Grey paths: 192 arrivals in 2021–2024; 500 m AGL; 0–48 h lookback. Outlines are selection areas, not administrative borders.',
             ha='center',fontsize=7.1,color='#4b5963')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();lb=legend.get_window_extent(renderer)
    assert lb.x0>=0 and lb.y0>=0 and lb.x1<=fig.bbox.x1 and lb.y1<=fig.bbox.y1,'Legend outside canvas.'
    assert lb.y0>fig.bbox.height*.037,f'Legend overlaps footnotes: bottom={lb.y0/fig.bbox.height:.4f}.'
    name=NAMES[int(planned)]
    for ext in ['png','pdf','svg']:
        target=output/(name+'.'+ext)
        if target.exists():raise FileExistsError('Refusing to overwrite '+str(target))
        fig.savefig(target,dpi=150)
    positions=ax.get_position().bounds
    plt.close(fig)
    return {'file_stem':name,'candidate_count':len(drawn),'candidate_keys_sha256':hashlib.sha256('\n'.join(sorted(drawn)).encode()).hexdigest(),
            'planned_area_count':2 if planned else 0,'new_international_points_drawn':0,'axes_bounds':positions,
            'trajectory_count':192,'png_pixels':[1650,1650]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root',type=Path,default=ROOT)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    folder=args.input_root.resolve()
    out=args.output_dir or folder/'Results'/('Replot_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=True)
    reg,paths,regions,land=verify_inputs(folder)
    records=[draw(reg,paths,regions,land,planned,out) for planned in [False,True]]
    assert records[0]['candidate_keys_sha256']==records[1]['candidate_keys_sha256']
    assert records[0]['axes_bounds']==records[1]['axes_bounds']
    manifest={'created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'historical_stage':'After nationwide Korean expansion, before China_East16/Japan_East18 and Japan-west reselection',
      'same_candidates_and_trajectories_in_both_maps':True,'no_raw_data_or_models_modified':True,
      'candidate_group_counts':COUNTS,'maps':records,
      'input_sha256':{str(p.relative_to(folder)):sha(p) for p in [folder/'Results/plotted_candidates_822.csv',folder/'Results/display_trajectory_sample.parquet',folder/'Results/planned_region_boundaries.geojson',folder/'Codes/assets/countries.geojson']}}
    (out/'visualization_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'verified_candidates':822,'display_paths':192,'maps':2},ensure_ascii=False))


if __name__=='__main__':main()
