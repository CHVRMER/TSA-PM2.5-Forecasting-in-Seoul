"""Preserve the attached legacy map layouts; change observation grouping only.

The heatmap and trajectory sample are separate figures. Original map extent,
coordinate presentation, heatmap formula/palette and sampled arrivals are reused.
Use current experiment observations, not the old 822 preliminary candidates.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,importlib.util,json,uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PROJECT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
HERE=Path(__file__).resolve().parent
TARGET='KR:111121'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_reference(path):
    spec=importlib.util.spec_from_file_location('preserved_matching_map_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def markers(ax,r,order,colors,reference):
    count=0
    for rid in order:
        for city,marker in [(False,'o'),(True,'D')]:
            part=r.loc[r.region_id.eq(rid)&r.is_city_average.eq(city)]
            if part.empty:continue
            ax.scatter(part.longitude,part.latitude,s=27 if rid.startswith('KR_') else (48 if city else 38),
                       marker=marker,c=colors[rid],edgecolors='white',linewidths=.60,alpha=.97,zorder=6)
            count+=len(part)
    ax.scatter(*reference.SEOUL,marker='*',s=230,color='#b81d2c',edgecolors='white',linewidths=1.2,zorder=8)
    assert count==634

def legend(fig,r,order,colors,labels):
    handles=[]
    for rid in order:
        part=r.loc[r.region_id.eq(rid)]
        marker='D' if part.is_city_average.all() else 'o'
        handles.append(Line2D([],[],linestyle='',marker=marker,markersize=7.2,
                              markerfacecolor=colors[rid],markeredgecolor='white',
                              label=labels[rid]+f': {len(part)}'))
    handles.append(Line2D([],[],linestyle='',marker='*',markersize=11.5,markerfacecolor='#b81d2c',
                          markeredgecolor='white',label='서울 도착점(중구 타깃 별도)'))
    obj=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.159),ncol=4,
                   frameon=False,fontsize=9.2,columnspacing=1.9,handletextpad=.5,labelspacing=.50)
    return obj

def footer(fig,heatmap):
    line1=('2021–2024 | 전체 11,688개 · 3시간 간격 서울 도착 · 500m AGL · 과거 1–48시간 위치 · 도착 0시간 제외'
           if heatmap else '회색 선: 기존과 동일한 192개 표본 궤적(월별 4개 × 48개월) · 전체 11,688개 중 표시 · 과거 1–48시간')
    fig.text(.5,.047,line1,ha='center',fontsize=8.7,color='#425562')
    fig.text(.5,.028,'점 색상=현재 17개 권역 / 원=개별 관측소 / 마름모=중국 도시평균 대표좌표. 수송 빈도·경로는 PM2.5 농도나 오염 기여율이 아닙니다.',
             ha='center',fontsize=8.35,color='#425562')
    fig.text(.5,.010,'현재 실험 634계열(국내 514·중국 77·일본 42·몽골 1)을 표시합니다. 기존 822개는 과거 전체 후보군입니다. Basemap: Natural Earth.',
             ha='center',fontsize=8.15,color='#425562')

def common_annotations(ax,r,ref,heatmap):
    ref.annotation(ax,'Seoul',ref.SEOUL,(9,-12),weight='bold')
    ref.annotation(ax,'Ulaanbaatar',(106.929,47.928),(-23,12))
    if heatmap:
        for text,xy in [('MONGOLIA',(111.8,49.2)),('CHINA',(113.5,35.2)),('RUSSIA',(136.2,52.4)),('JAPAN',(137.7,34.4))]:
            ax.text(*xy,text,fontsize=11,color='#425868',alpha=.82,ha='center',zorder=3)
        heihe=r.loc[r.station_key.eq('CN:city_heihe')].iloc[0]
        ref.annotation(ax,'Heihe (Chinese city mean)',(heihe.longitude,heihe.latitude),(-111,13),fontsize=8.5)
        for extent in [ref.KOREA_EXTENT,ref.JAPAN_EXTENT]:
            ax.add_patch(Rectangle((extent[0],extent[2]),extent[1]-extent[0],extent[3]-extent[2],
                                   fill=False,edgecolor='#4b5563',linewidth=.8,linestyle='--',zorder=3))

def create_map(r,order,colors,labels,ref,shapes,frequency,frame,chosen,out,heatmap,dpi):
    fig=plt.figure(figsize=(14,12))
    ax=fig.add_axes([.09,.265,.80,.62] if heatmap else [.09,.22,.80,.665])
    mesh=ref.base_map(ax,shapes,ref.MAIN_EXTENT,frequency if heatmap else None,grid_step=5)
    lines=0
    if not heatmap:
        for arrival,path in frame.loc[frame.arrival_time_kst.isin(chosen)].groupby('arrival_time_kst',sort=True):
            path=path.sort_values('lookback_hours',ascending=False)
            ax.plot(path.longitude,path.latitude,color='#6c8fa4',alpha=.30,linewidth=.7,zorder=2.5)
            lines+=1
        assert lines==192
    else:
        assert lines==0
    markers(ax,r,order,colors,ref)
    common_annotations(ax,r,ref,heatmap)
    title=('634 observation series along Seoul air-mass pathways' if heatmap else 'Observation regions and sampled air-mass paths')
    subtitle=('Current experiment | Korea 514 + China 77 + Japan 42 + Mongolia 1 | 17 fixed regions'
              if heatmap else '634 observation series in 17 regions | 2021–2024 arrivals | 500 m AGL | same 192 display paths')
    fig.text(.5,.956,title,ha='center',fontsize=20,weight='bold',color='#182c39')
    fig.text(.5,.925,subtitle,ha='center',fontsize=11.5,color='#425868')
    if heatmap:ref.colorbar(fig,mesh,[.20,.196,.60,.016])
    obj=legend(fig,r,order,colors,labels)
    footer(fig,heatmap)
    fig.canvas.draw()
    bbox=obj.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert bbox.x0>=0 and bbox.x1<=1 and bbox.y0>.06,tuple(bbox.bounds)
    stem='station_regions_17_heatmap' if heatmap else 'station_regions_17_trajectory_sample'
    for ext in ('png','svg'):fig.savefig(out/f'{stem}.{ext}',dpi=dpi,facecolor='white')
    plt.close(fig)
    return {'stem':stem,'heatmap_present':heatmap,'sample_trajectory_lines':lines,
            'external_series':634,'regions':17,'target_marker_separate':1,
            'legend_bbox_figure':list(map(float,bbox.bounds)),'extent':list(ref.MAIN_EXTENT),
            'coordinate_display':'same lon/lat layout and aspect as attached original'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=PROJECT)
    ap.add_argument('--output-root',type=Path,default=HERE.parent/'Results')
    ap.add_argument('--dpi',type=int,default=220)
    args=ap.parse_args();project=args.project_root.resolve()
    config=project/'Model/Common/Regional12_Seed11_20261109'
    colors_path=HERE/'region_color_scheme.csv'
    reference_path=HERE/'matching_map_reference.py'
    basemap_path=HERE/'assets/ne_110m_admin_0_countries.geojson'
    trajectory_path=project/'Dataset/Trajectories/trajectory_points_2021_2024.parquet'
    old_manifest=project/'EDA/Matching_Candidates_National_Neighbors_2021_2024/visualization_manifest.json'
    inputs=[config/'regions.json',config/'station_registry.csv',config/'experiments.json',colors_path,reference_path,basemap_path,trajectory_path,old_manifest]
    before={str(p):sha(p) for p in inputs}
    r=pd.read_csv(config/'station_registry.csv',dtype=str,keep_default_na=False)
    geo=json.loads((config/'regions.json').read_text(encoding='utf-8-sig'))
    experiments=json.loads((config/'experiments.json').read_text(encoding='utf-8-sig'))
    scheme=pd.read_csv(colors_path,dtype=str,keep_default_na=False)
    colors=scheme.set_index('region_id').color_hex.to_dict();labels=scheme.set_index('region_id').label_ko.to_dict()
    order=geo['region_order'];owners={}
    for rid in order:
        keys=geo['regions'][rid]['station_keys']
        assert len(keys)==int(scheme.set_index('region_id').loc[rid,'series_count'])
        for k in keys:
            assert k not in owners
            owners[k]=rid
    assert len(order)==17 and len(r)==r.station_key.nunique()==len(owners)==634
    assert set(r.station_key)==set(owners)==set(experiments['conditions']['F']['station_keys'])
    assert TARGET not in owners and set(order)==set(colors)
    assert r.groupby('country').size().to_dict()=={'CN':77,'JP':42,'KR':514,'MN':1}
    r['region_id']=r.station_key.map(owners);assert r.region_id.eq(r.experiment_region_id).all()
    r['longitude']=pd.to_numeric(r.longitude);r['latitude']=pd.to_numeric(r.latitude)
    assert np.isfinite(r[['longitude','latitude']]).all().all()
    r['is_city_average']=r.source_dataset.isin(['China_Border_Proxy','China_East_QC'])|r.station_id.str.startswith('city_')
    assert r.is_city_average.sum()==29 and r.loc[r.is_city_average,'country'].eq('CN').all()
    ref=load_reference(reference_path)
    assert list(ref.MAIN_EXTENT)==[100,141,28,55]
    assert r.longitude.between(100,141).all() and r.latitude.between(28,55).all()
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.is_file():
        font_manager.fontManager.addfont(str(font));font_name=font_manager.FontProperties(fname=str(font)).get_name()
    else:
        available={f.name for f in font_manager.fontManager.ttflist}
        font_name=next((n for n in ['Noto Sans CJK KR','Noto Sans CJK JP','NanumGothic','AppleGothic'] if n in available),None)
        if not font_name:raise RuntimeError('A Korean font such as Noto Sans CJK KR is required.')
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'svg.fonttype':'path'})
    frame=ref.read_full_period_paths(trajectory_path);counts,frequency=ref.visit_frequency(frame)
    arrivals=pd.DatetimeIndex(frame.arrival_time_kst.unique()).sort_values();months=arrivals.strftime('%Y-%m');chosen=[]
    for month in pd.unique(months):
        candidates=arrivals[months==month];positions=((np.arange(4)+.5)*len(candidates)/4).astype(int)
        chosen.extend(candidates[positions])
    assert len(chosen)==len(set(chosen))==192
    old=json.loads(old_manifest.read_text(encoding='utf-8-sig'))
    assert old['trajectory_source_sha256']==sha(trajectory_path)
    assert old['arrivals']==11688 and old['trajectory_points']==len(frame)==561024
    assert pd.DatetimeIndex(pd.to_datetime(old['display_sample_arrivals'],utc=True)).equals(pd.DatetimeIndex(chosen).tz_convert('UTC'))
    assert old['max_frequency_pct']==float(frequency.max())
    assert old['nonzero_global_grid_cells']==int(np.count_nonzero(counts))
    shapes=ref.country_shapes(basemap_path)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]
    out=args.output_root.resolve()/('Run_'+stamp);out.mkdir(parents=True,exist_ok=False)
    figures=[create_map(r,order,colors,labels,ref,shapes,frequency,frame,chosen,out,True,args.dpi),
             create_map(r,order,colors,labels,ref,shapes,frequency,frame,chosen,out,False,args.dpi)]
    assert figures[0]['heatmap_present'] and figures[0]['sample_trajectory_lines']==0
    assert not figures[1]['heatmap_present'] and figures[1]['sample_trajectory_lines']==192
    scheme.to_csv(out/'region_counts_and_colors_17.csv',index=False,encoding='utf-8-sig')
    r[['station_key','station_name','city','country','region_id','longitude','latitude','source_dataset','is_city_average']].to_csv(out/'plotted_observations_634.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'arrival_time_kst':[t.isoformat() for t in chosen]}).to_csv(out/'sampled_arrivals_192.csv',index=False,encoding='utf-8-sig')
    for name,digest in before.items():assert sha(name)==digest,name
    report={'status':'verified','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'external_observations':634,'old_preliminary_candidates':822,'regions':17,
            'country_counts':{'KR':514,'CN':77,'JP':42,'MN':1},'city_means':29,'point_stations':605,
            'figures':figures,'separated_layers_verified':True,'reference_manifest_matched':True,
            'hysplit':{'arrivals':11688,'height_agl_m':500,'ages_h':[1,48],'grid_degrees':1,
                       'fixed_denominator':11688,'same_path_revisits_counted_once':True,'sample_paths':192,
                       'sample_per_month':4,'colour_map':'YlGnBu','colour_limits_percent':[.01,100]},
            'seoul_marker':{'coordinate':list(ref.SEOUL),'meaning':'unchanged HYSPLIT receptor representative coordinate; target excluded from634'},
            'inputs_unchanged':True,'input_sha256':before,'model_training':False}
    dump(out/'visualization_verification.json',report)
    print(json.dumps({'status':'verified','output':str(out),'files':[p.name for p in out.iterdir()]},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
