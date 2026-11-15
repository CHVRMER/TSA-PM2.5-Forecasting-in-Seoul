"""Map the exact executed DCRNN F634 universe (634 external series + target).

Inputs are packaged node coordinates and the identical historical-map display
trajectories. No model fitting, HYSPLIT rerun or source-data changes.
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
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
STYLES=[
 ('KR_OLD','South Korea, existing: 67','^','#d07a2d',26,67),
 ('KR_NEW','South Korea, national additions: 447','^','#f2b880',14,447),
 ('CN','China stations: 48','o','#a23b8f',26,48),
 ('CN_BORDER','China city means near Russia/Mongolia: 13','s','#8b4a3c',34,13),
 ('CN_EAST','China East city means: 16','D','#1f77b4',40,16),
 ('JP_WEST','Japan west, model selection: 24','o','#2e7d5b',26,24),
 ('JP_EAST','Japan East stations: 18','P','#00a0b0',46,18),
 ('MN','Mongolia: 1','D','#5b3f8f',50,1),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root',type=Path,default=ROOT)
    ap.add_argument('--output-dir',type=Path)
    a=ap.parse_args();base=a.input_root.resolve()
    output=a.output_dir or base/'Results'/('Replot_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=True)
    stem='dcrnn_final_network_634_external_635_nodes'
    if any((output/(stem+'.'+ext)).exists() for ext in ['png','pdf','svg']):raise FileExistsError('Refusing to overwrite an existing map.')
    nodes=pd.read_csv(base/'Results/final_nodes_635.csv',dtype={'station_id':str})
    assert len(nodes)==635 and nodes.station_key.is_unique and nodes.node_index.is_unique
    assert set(nodes.node_index)==set(range(635))
    target=nodes.loc[nodes.plot_group.eq('TARGET')]
    assert len(target)==1 and target.station_key.item()=='KR:111121' and target.node_index.item()==0
    external=nodes.loc[nodes.plot_group.ne('TARGET')]
    assert len(external)==634
    assert external.country.value_counts().to_dict()=={'KR':514,'CN':77,'JP':42,'MN':1}
    assert not external.country.eq('RU').any()
    assert not external.loc[external.country.eq('KR'),'network'].str.contains('도로',na=False).any()
    assert np.isfinite(nodes[['longitude','latitude']]).all().all()
    assert nodes.longitude.between(100,145).all() and nodes.latitude.between(25,55).all()
    paths=pd.read_parquet(base/'Results/display_trajectory_sample.parquet')
    assert len(paths)==192*49 and paths.arrival_time_kst.nunique()==192
    assert paths.arrival_height_agl_m.eq(500).all() and paths.lookback_hours.between(0,48).all()
    borders=json.loads((base/'Results/selected_region_boundaries.geojson').read_text(encoding='utf-8'))
    land=json.loads((base/'Codes/assets/countries.geojson').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','hatch.linewidth':.55,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(11,11),dpi=150)
    ax=fig.add_axes([.057,.160,.917,.764])
    for feature in land['features']:
        geom=feature['geometry']
        if geom['type'] not in ('Polygon','MultiPolygon'):continue
        pieces=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
        for p in pieces:
            xy=np.asarray(p[0],float)
            if xy[:,0].max()<100 or xy[:,0].min()>145 or xy[:,1].max()<25 or xy[:,1].min()>55:continue
            ax.fill(xy[:,0],xy[:,1],color='#f4f4f1',ec='#666666',lw=.5,zorder=0)
    ax.set_facecolor('#e9eff4')
    for _,p in paths.groupby('arrival_time_kst',sort=True):
        p=p.sort_values('lookback_hours')
        ax.plot(p.longitude,p.latitude,c='#8a9aa8',lw=.4,alpha=.45,zorder=1)
    for feature in borders['features']:
        vertices=feature['geometry']['coordinates'][0];color=feature['properties']['color']
        ax.add_patch(Polygon(vertices,closed=True,facecolor='none',edgecolor=color,hatch='xx',linewidth=0,alpha=.32,zorder=2.5))
        xy=np.asarray(vertices,float);ax.plot(xy[:,0],xy[:,1],ls='--',c=color,lw=1.6,zorder=3)
    handles=[];drawn=[];counts={}
    for key,label,marker,color,size,expected in STYLES:
        p=nodes.loc[nodes.plot_group.eq(key)]
        assert len(p)==expected,(key,len(p),expected)
        ax.scatter(p.longitude,p.latitude,marker=marker,s=size,c=color,ec='white',lw=.4,zorder=4)
        handles.append(Line2D([],[],ls='',marker=marker,markerfacecolor=color,markeredgecolor='white',
                              markeredgewidth=.4,markersize=max(5,np.sqrt(size)),label=label))
        drawn.extend(p.station_key.tolist());counts[key]=len(p)
    assert len(drawn)==634 and set(drawn)==set(external.station_key)
    ax.scatter(target.longitude,target.latitude,marker='*',s=260,c='#b81d2c',ec='white',lw=.6,zorder=6)
    handles.append(Line2D([],[],ls='',marker='*',markerfacecolor='#b81d2c',markeredgecolor='white',markersize=13,
                          label='Seoul Jung-gu target: 1'))
    ax.set(xlim=(100,145),ylim=(25,55),xlabel='Longitude (°E)',ylabel='Latitude (°N)',aspect=1.25)
    ax.set_xticks(np.arange(100,146,5));ax.set_yticks(np.arange(25,56,5));ax.grid(alpha=.3)
    ax.set_title('DCRNN final network: 634 external series + Seoul Jung-gu (635 nodes)',fontsize=12.2,pad=6)
    legend=ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.045),ncol=2,fontsize=8.2,
                     frameon=False,handlelength=1.5,columnspacing=1.8)
    notes=[fig.text(.5,.023,'External series: Korea 514 + China 77 + Japan 42 + Mongolia 1. Dashed areas: China East / Japan East.',
                    ha='center',fontsize=7.25,color='#4b5963'),
           fig.text(.5,.010,'Grey paths: 192 arrivals (2021–2024), 500 m AGL, 0–48 h. City means use approximate coordinates; paths are not source contributions.',
                    ha='center',fontsize=7.1,color='#4b5963')]
    fig.canvas.draw();renderer=fig.canvas.get_renderer();lb=legend.get_window_extent(renderer)
    assert lb.x0>=0 and lb.x1<=fig.bbox.x1 and lb.y0>fig.bbox.height*.037
    for note in notes:
        box=note.get_window_extent(renderer);assert box.x0>=0 and box.x1<=fig.bbox.x1 and box.y0>=0
    for ext in ['png','pdf','svg']:fig.savefig(output/(stem+'.'+ext),dpi=150)
    axes=ax.get_position().bounds;plt.close(fig)
    record={'created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'model_universe':'F634_BOTH_W24',
        'executed_batch_reference':'20261112_003040_429483_85af4013','external_series':634,'target_nodes':1,'total_nodes':635,
        'country_external_counts':external.country.value_counts().to_dict(),'plot_group_counts':counts,
        'external_keys_sha256':hashlib.sha256('\n'.join(sorted(drawn)).encode()).hexdigest(),
        'no_excluded_background_stations':True,'png_pixels':[1650,1650],'axes_bounds':axes,
        'trajectory_sample_count':192,'display_path_sha256':sha(base/'Results/display_trajectory_sample.parquet'),
        'node_file_sha256':sha(base/'Results/final_nodes_635.csv'),'original_data_or_model_modified':False}
    (output/'visualization_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'external':634,'target':1,'total':635},ensure_ascii=False))


if __name__=='__main__':main()
