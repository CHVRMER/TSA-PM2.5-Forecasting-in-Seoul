"""Classify 2021–2022 historical Seoul air paths using geographic polygons.

Primary passage = intersection of modeled hourly-position line segments with
polygons. Point visits are a separate sensitivity. No PM or station data select
regions; no target-year beyond 2022 is read here.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import shape,mapping,box,Point,LineString,MultiLineString,Polygon

REGIONS=[
 ('KR_CAPITAL','Capital region','수도권','KR','KR'),
 ('KR_GANGWON','Gangwon','강원권','KR','KR'),
 ('KR_CHUNGCHEONG','Chungcheong','충청권','KR','KR'),
 ('KR_HONAM','Honam','호남권','KR','KR'),
 ('KR_DAEGYEONG','Daegyeong','대경권','KR','KR'),
 ('KR_DONGNAM','Dongnam','동남권','KR','KR'),
 ('KR_JEJU','Jeju','제주권','KR','KR'),
 ('CN_SHANDONG','Shandong','산둥','CN','CN'),
 ('CN_BTH','Beijing / Tianjin / Hebei','베이징·톈진·허베이','CN','CN'),
 ('CN_LIAONING','Liaoning','랴오닝','CN','CN'),
 ('CN_JILIN','Jilin','지린','CN','CN'),
 ('CN_HEILONGJIANG','Heilongjiang','헤이룽장','CN','CN'),
 ('CN_INNER_MONGOLIA','Inner Mongolia','내몽골','CN','CN'),
 ('CN_OTHER','Other mainland China','기타 중국','CN','CN'),
 ('JP','Japan','일본','OTHER','JP'),
 ('MN','Mongolia','몽골','OTHER','MN'),
 ('RU','Russia','러시아','OTHER','RU'),
 ('KP','North Korea','북한','OTHER','KP'),
 ('OTHER_LAND','Other land','기타 육지','OTHER','MULTI'),
 ('SEA_OR_UNCLASSIFIED','Sea / geographically unclassified','해상·지리적 미분류','OTHER','SEA_OR_UNCLASSIFIED'),
]
KR_MAP={
 'Gangwon':'KR_GANGWON','Gyeonggi':'KR_CAPITAL','Incheon':'KR_CAPITAL','Seoul':'KR_CAPITAL',
 'South Chungcheong':'KR_CHUNGCHEONG','North Chungcheong':'KR_CHUNGCHEONG','Daejeon':'KR_CHUNGCHEONG','Sejong':'KR_CHUNGCHEONG',
 'North Jeolla':'KR_HONAM','South Jeolla':'KR_HONAM','Gwangju':'KR_HONAM',
 'South Gyeongsang':'KR_DONGNAM','Busan':'KR_DONGNAM','Ulsan':'KR_DONGNAM',
 'North Gyeongsang':'KR_DAEGYEONG','Daegu':'KR_DAEGYEONG','Jeju':'KR_JEJU',
}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def polygonal(geom):
 geom=shapely.make_valid(geom)
 if geom.geom_type in ['Polygon','MultiPolygon']:return geom
 pieces=[g for g in shapely.get_parts(geom) if g.geom_type in ['Polygon','MultiPolygon']]
 return shapely.union_all(pieces) if pieces else Polygon()

def build_regions(assets):
 kor=json.loads((assets/'KOR_ADM1.geojson').read_text(encoding='utf-8'))
 china=json.loads((assets/'CHN_ADM1.geojson').read_text(encoding='utf-8'))
 japan=json.loads((assets/'JPN_ADM0.geojson').read_text(encoding='utf-8'))
 world=json.loads((assets/'ne_110m_admin_0_countries.geojson').read_text(encoding='utf-8'))
 parts={r[0]:[] for r in REGIONS};members={r[0]:[] for r in REGIONS}
 assert len(kor['features'])==17 and len(china['features'])==34
 for f in kor['features']:
  name=f['properties']['shapeName'];rid=KR_MAP[name]
  parts[rid].append(polygonal(shape(f['geometry'])));members[rid].append(name)
 excluded=[]
 for f in china['features']:
  name=f['properties']['shapeName']
  if name in ['Taiwan Province','Hong Kong Special Administrative Region','Macau Special Administrative Region']:
   excluded.append(name);continue
  if name=='Shandong Province':rid='CN_SHANDONG'
  elif name in ['Beijing Municipality','Tianjin Municipality','Hebei Province']:rid='CN_BTH'
  elif name=='Liaoning Province':rid='CN_LIAONING'
  elif name=='Jilin Province':rid='CN_JILIN'
  elif name=='Heilongjiang Province':rid='CN_HEILONGJIANG'
  elif name=='Inner Mongolia Autonomous Region':rid='CN_INNER_MONGOLIA'
  else:rid='CN_OTHER'
  parts[rid].append(polygonal(shape(f['geometry'])));members[rid].append(name)
 parts['JP']=[polygonal(shape(f['geometry'])) for f in japan['features']];members['JP']=['Japan MLIT national polygon']
 world_geoms=[];other_country_geoms=[]
 for f in world['features']:
  geom=polygonal(shape(f['geometry']));world_geoms.append(geom)
  code=f['properties']['ADM0_A3']
  if code not in {'KOR','CHN','JPN','MNG','RUS','PRK'}:other_country_geoms.append(geom)
  if code in {'MNG','RUS','PRK'}:
   rid={'MNG':'MN','RUS':'RU','PRK':'KP'}[code];parts[rid].append(geom);members[rid].append(f['properties']['NAME'])
 raw={rid:polygonal(shapely.union_all(gs)) for rid,gs in parts.items() if gs}
 land=polygonal(shapely.union_all(world_geoms))
 claimed=Polygon();geoms={};areas=[]
 # Explicit fixed priority resolves mixed-source boundary slivers. Area is a
 # geometric degree-squared diagnostic, never an area or emissions estimate.
 for rid,*_ in REGIONS[:-2]:
  if rid not in raw:raise ValueError('No geometry for '+rid)
  overlap=raw[rid].intersection(claimed)
  g=polygonal(raw[rid].difference(claimed));geoms[rid]=g
  areas.append({'region_id':rid,'source_overlap_degree2':overlap.area,'final_degree2':g.area})
  claimed=polygonal(shapely.union_all([claimed,g]))
 # Never interpret coarse-country minus detailed-administration shoreline
 # slivers as another country. Only actual non-focus country features enter
 # OTHER_LAND; unresolved source gaps go to the explicit unclassified class.
 geoms['OTHER_LAND']=polygonal(shapely.union_all(other_country_geoms).difference(claimed))
 union_land=polygonal(shapely.union_all([claimed,geoms['OTHER_LAND']]))
 geoms['SEA_OR_UNCLASSIFIED']=polygonal(box(-180,-90,180,90).difference(union_land))
 members['OTHER_LAND']=['Natural Earth countries excluding KOR, CHN, JPN, MNG, RUS, PRK; source shoreline slivers excluded']
 members['SEA_OR_UNCLASSIFIED']=['Complement of classified land in lon/lat world; may include shoreline/source discrepancies']
 return geoms,members,areas,excluded,land

def crossing_line(a,b):
 """Short longitude branch; never draw an antimeridian crossing across Asia."""
 x1,y1=a;x2,y2=b
 if abs(x2-x1)<=180:return LineString([a,b])
 x2u=x2+360 if x2<x1 else x2-360
 limit=180 if x2u>180 else -180
 frac=(limit-x1)/(x2u-x1);mid=y1+frac*(y2-y1)
 return MultiLineString([[(x1,y1),(limit,mid)],[(-limit,mid),(x2,y2)]])

def tests(geoms):
 places={'Seoul':(126.98,37.57,'KR_CAPITAL'),'Gangwon':(128.0,37.8,'KR_GANGWON'),
 'Daejeon':(127.38,36.35,'KR_CHUNGCHEONG'),'Gwangju':(126.85,35.15,'KR_HONAM'),
 'Daegu':(128.6,35.87,'KR_DAEGYEONG'),'Busan':(129.07,35.18,'KR_DONGNAM'),
 'Hallasan':(126.53,33.36,'KR_JEJU'),'Jinan':(117.1,36.65,'CN_SHANDONG'),
 'Beijing':(116.4,39.9,'CN_BTH'),'Shenyang':(123.43,41.8,'CN_LIAONING'),
 'Changchun':(125.3,43.9,'CN_JILIN'),'Harbin':(126.6,45.75,'CN_HEILONGJIANG'),
 'Hohhot':(111.7,40.8,'CN_INNER_MONGOLIA'),'Nanjing':(118.78,32.06,'CN_OTHER'),
 'Tokyo':(139.7,35.7,'JP'),'Ulaanbaatar':(106.92,47.92,'MN'),
 'Khabarovsk':(135.06,48.48,'RU'),'Pyongyang':(125.75,39.04,'KP'),
 'Delhi':(77.2,28.6,'OTHER_LAND'),'Yellow_Sea':(124,35,'SEA_OR_UNCLASSIFIED'),
 'Weihai':(122.12,37.42,'CN_SHANDONG'),'Fukuoka':(130.4,33.59,'JP'),
 'Vladivostok':(131.89,43.12,'RU')}
 for name,(x,y,rid) in places.items():
  assert geoms[rid].covers(Point(x,y)),(name,rid)
 hole=Polygon([(0,0),(4,0),(4,4),(0,4)],holes=[[(1,1),(3,1),(3,3),(1,3)]])
 assert not hole.covers(Point(2,2)) and hole.covers(Point(0,2))
 assert box(1,1,2,2).intersects(LineString([(0,1.5),(3,1.5)]))
 assert not box(115,59,125,61).intersects(crossing_line((170,60),(-170,60)))
 return {'status':'passed','place_checks':len(places),'holes_preserved':True,
         'boundary_touch_included':True,'segment_without_endpoint_in_region_detected':True,
         'antimeridian_short_branch':True}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--trajectories',type=Path,required=True)
 ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
 root=args.out;assets=Path(__file__).resolve().parent/'assets';out=root/'Results';out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();geoms,members,overlap,excluded,ne_land=build_regions(assets)
 test_result=tests(geoms)
 definition=[]
 for order,(rid,en,ko,group,country) in enumerate(REGIONS):
  definition.append({'region_id':rid,'label_en':en,'label_ko':ko,'group':group,'country':country,
   'sort_order':order,'members':'|'.join(members[rid]),'point_hours_definition':'number of hourly modeled positions; not continuous residence time'})
 pd.DataFrame(definition).to_csv(out/'region_definitions.csv',index=False,encoding='utf-8-sig')
 geo={'type':'FeatureCollection','features':[{'type':'Feature','properties':d,'geometry':mapping(geoms[d['region_id']])} for d in definition]}
 (assets/'analysis_regions.geojson').write_text(json.dumps(geo,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 filters=[('arrival_time_utc','>=',pd.Timestamp('2021-01-01',tz='Asia/Seoul').tz_convert('UTC').to_pydatetime()),
          ('arrival_time_utc','<',pd.Timestamp('2023-01-01',tz='Asia/Seoul').tz_convert('UTC').to_pydatetime()),
          ('arrival_height_agl_m','=',500.),('lookback_hours','>=',1.),('lookback_hours','<=',48.)]
 columns=['arrival_time_kst','arrival_time_utc','lookback_hours','latitude','longitude','height_agl_m','diagnostic_mixdepth']
 p=pq.read_table(args.trajectories,columns=columns,filters=filters).to_pandas().sort_values(['arrival_time_kst','lookback_hours']).reset_index(drop=True)
 arrivals=pd.DatetimeIndex(p.arrival_time_kst.unique()).sort_values()
 assert len(arrivals)==5840 and len(p)==280320
 assert p.groupby('arrival_time_kst').lookback_hours.apply(lambda a:np.array_equal(a.to_numpy(),np.arange(1,49))).all()
 assert np.isfinite(p[['longitude','latitude']]).all().all()
 assert p.longitude.between(-180,180).all() and p.latitude.between(-90,90).all()
 coords=p[['longitude','latitude']].to_numpy().reshape(5840,48,2)
 point_geoms=shapely.points(coords.reshape(-1,2))
 assigned=np.full(len(p),-1,np.int16);point_multi=np.zeros(len(p),np.int16)
 point_masks={}
 for index,(rid,*_) in enumerate(REGIONS):
  shapely.prepare(geoms[rid]);mask=shapely.covers(geoms[rid],point_geoms)
  point_multi+=mask;assigned[(assigned<0)&mask]=index
  point_masks[rid]=mask.reshape(5840,48)
 assert (assigned>=0).all(),'A modeled point was omitted from geographic categories'
 p['region_id']=np.array([r[0] for r in REGIONS],dtype=object)[assigned]
 p.to_parquet(out/'classified_trajectory_points_train.parquet',index=False,compression='zstd')
 rows=[];window_stats={}
 for lo,hi,window in [(1,48,'01-48'),(7,48,'07-48'),(13,48,'13-48')]:
  a=coords[:,lo-1:hi-1,:];b=coords[:,lo:hi,:]
  pair=np.stack([a,b],axis=-2).reshape(-1,2,2)
  lines=shapely.linestrings(pair)
  crosses=np.abs(pair[:,1,0]-pair[:,0,0])>180
  for ix in np.flatnonzero(crosses):lines[ix]=crossing_line(pair[ix,0],pair[ix,1])
  nseg=hi-lo;window_stats[window]={'hourly_positions':hi-lo+1,'one_hour_segments':nseg,'antimeridian_segments_split':int(crosses.sum())}
  for rid,en,ko,group,country in REGIONS:
   hit=shapely.intersects(geoms[rid],lines).reshape(5840,nseg)
   point_counts=point_masks[rid][:,lo-1:hi].sum(axis=1)
   visited_points=point_counts>0;visited_segments=hit.any(axis=1)
   assert (visited_points<=visited_segments).all(),(window,rid)
   rows.append(pd.DataFrame({'origin_time_kst':arrivals,'region_id':rid,'label_en':en,'label_ko':ko,
      'group':group,'country':country,'window':window,'visited_segments':visited_segments,
      'visited_points':visited_points,'point_hours':point_counts,'intersecting_segments':hit.sum(axis=1)}))
  print(f'Geographic window {window} completed in {time.monotonic()-start:.1f}s',flush=True)
 visits=pd.concat(rows,ignore_index=True)
 assert len(visits)==5840*len(REGIONS)*3 and not visits.duplicated(['origin_time_kst','region_id','window']).any()
 visits.to_parquet(out/'origin_visits.parquet',index=False,compression='zstd')
 passage=visits.groupby(['region_id','label_en','label_ko','window'],sort=False).agg(
    arrivals=('origin_time_kst','size'),segment_visit_arrivals=('visited_segments','sum'),point_visit_arrivals=('visited_points','sum'),hourly_point_count=('point_hours','sum')).reset_index()
 passage['segment_visit_pct']=100*passage.segment_visit_arrivals/passage.arrivals
 passage['point_visit_pct']=100*passage.point_visit_arrivals/passage.arrivals
 passage.to_csv(out/'geographic_passage_rates.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(overlap).to_csv(out/'source_boundary_overlap_diagnostics.csv',index=False,encoding='utf-8-sig')
 sources=[]
 for country,level in [('KOR','ADM1'),('CHN','ADM1'),('JPN','ADM0')]:
  meta=json.loads((assets/f'{country}_{level}_metadata.json').read_text(encoding='utf-8'))
  sources.append({'asset':f'{country}_{level}.geojson','metadata_api':f'https://www.geoboundaries.org/api/current/gbOpen/{country}/{level}/',
    'download_url':meta['gjDownloadURL'],'actual_media_url':meta['gjDownloadURL'].replace('https://github.com/','https://media.githubusercontent.com/media/').replace('/raw/','/'),
    'source':meta['boundarySource'],'year_represented':meta['boundaryYearRepresented'],'license':meta['boundaryLicense'],
    'sha256':digest(assets/f'{country}_{level}.geojson')})
 sources.append({'asset':'ne_110m_admin_0_countries.geojson','source':'Natural Earth 1:110m country polygons, existing audited local copy',
  'url':'https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson','license':'Public Domain',
  'year_represented':'not available in reused local asset; fixed geometry pinned by sha256',
  'sha256':digest(assets/'ne_110m_admin_0_countries.geojson')})
 summary={'status':'passed','period':'2021-2022 training origins only','arrivals':5840,'hourly_points':280320,
  'regions':len(REGIONS),'windows':window_stats,'arrival_height_agl_m':500,'age_zero_excluded':True,
  'segment_interpolation':'straight lon/lat line between modeled hourly positions; antimeridian shortest branch; not resolved subhourly true trajectory',
  'passage_boundary_policy':'intersects includes boundary touches; multiple regions may be visited; rates need not sum100',
  'point_assignment':'fixed REGION order on shared boundaries; all modeled points assigned once',
  'point_boundary_multiple_covers':int((point_multi>1).sum()),'point_assignment_count':int((assigned>=0).sum()),
  'station_radius_matching_used':False,'country_border_city13_is_geographic_region':False,
  'country_metadata_excluded_from_mainland_group':excluded,
  'other_land_policy':'Only non-focus Natural Earth country features; never coarse-minus-detailed coastline slivers',
  'sea_unclassified_note':'land-complement category, not guaranteed actual sea: mixed-source coastlines and small unresolved geography included',
  'boundary_epoch_note':'KOR2021,CHN2019,JPN2022 fixed static geography; not a reconstruction of every boundary in2021/22',
  'test_result':test_result,'sources':sources,'trajectory_source':str(args.trajectories),'trajectory_sha256':digest(args.trajectories),
  'elapsed_seconds':time.monotonic()-start}
 (out/'geographic_classification_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:summary[k] for k in ['status','arrivals','hourly_points','regions','elapsed_seconds']},ensure_ascii=False))

if __name__=='__main__':main()
