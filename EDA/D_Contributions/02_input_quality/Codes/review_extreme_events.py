"""Read-only case review; keep raw and QC values separate, never change inputs."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT = Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, default=PROJECT)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.project.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


inputs = {
    'station': root/'Dataset/China/china_48stations_hourly_2021_2024_kst.parquet',
    'city': root/'Dataset/Neighbors_Expansion_QC_20261108/china_east_hourly_qc_2021_2024_kst.parquet',
    'metadata': root/'Model/Common/Regional12_Seed11_20261109/station_registry.csv',
    'qc_changes': root/'Dataset/Neighbors_Expansion_QC_20261108/qc_cell_changes.csv',
}
before = {str(p): sha(p) for p in inputs.values()}
frames = {k: pd.read_parquet(inputs[k]) for k in ('station', 'city')}
registry = pd.read_csv(inputs['metadata'])
cases = [
    dict(id='tangshan_9985',kind='station',station_id='3578A',country='CN',
         place='허베이성 탕산시 정부서비스센터(政府服务中心)',english='Tangshan 3578A',
         time_cst='2021-03-15 15:00',pollutant='pm25',value=9985.,lat=39.66941,lon=118.24498,
         judgment='strong_unverified_incoherence',status='retained_in_analysis',
         actual_event='대규모 황사 배경은 공식 확인. 이 지점의 PM2.5=9985 자체는 독립 검증하지 못함.'),
    dict(id='lianyungang_1500',kind='city',station_id='city_lianyungang',country='CN',
         place='장쑤성 롄윈강(连云港) 도시평균',english='Lianyungang city mean',
         time_cst='2021-03-15 19:00',pollutant='pm25',value=1500.,lat=34.60,lon=119.22,
         judgment='strong_unverified_incoherence',status='already_quarantined_flag20',
         actual_event='인접 날짜 황사와 이 도시평균의 PM2.5/PM10 불일치를 구분. 해당 1500을 뒷받침할 공식 시간별 검증자료 미확보.'),
    dict(id='linyi_1451',kind='city',station_id='city_linyi',country='CN',
         place='산둥성 린이(临沂) 도시평균',english='Linyi city mean',
         time_cst='2021-03-17 16:00',pollutant='pm25',value=1451.,lat=35.10,lon=118.36,
         judgment='strong_unverified_incoherence',status='already_quarantined_flag20',
         actual_event='황사 잔류 가능성과 별개로 고립된 PM2.5 급등·PM10 불일치. 해당값의 공식 독립 검증자료 미확보.'),
    dict(id='yancheng_1689',kind='city',station_id='city_yancheng',country='CN',
         place='장쑤성 옌청(盐城) 도시평균',english='Yancheng city mean',
         time_cst='2021-03-23 16:00',pollutant='pm25',value=1689.,lat=33.35,lon=120.16,
         judgment='ambiguous_joint_spike',status='already_quarantined_flag20',
         actual_event='PM2.5와 PM10이 함께 상승해 실제 국지적 연소 플룸 가능성을 남겨야 함. 정확한 사건 원인/원측정소 평균 재현 미확인.'),
    dict(id='nanjing_pm10_9211',kind='city',station_id='city_nanjing',country='CN',
         place='장쑤성 난징(南京) 도시평균',english='Nanjing city mean',
         time_cst='2021-12-07 19:00',pollutant='pm10',value=9211.,lat=32.06,lon=118.80,
         judgment='unverified_isolated_coarse_peak',status='already_quarantined_flag20',
         actual_event='PM10 사례이며 PM2.5가 아님. 도시평균으로 1시간만 높고 전후 결측/급락. 해당시각 극심한 먼지 사건 근거 미확보.'),
    dict(id='anshan_cny_1976',kind='station',station_id='1751A',country='CN',
         place='랴오닝성 안산(鞍山) 铁西三道街',english='Anshan 1751A; CNY',
         time_cst='2024-02-10 01:00',pollutant='pm25',value=1976.,lat=41.0971,lon=122.9642,
         judgment='event_consistent_not_instrument_certified',status='retained_in_analysis',
         actual_event='설날 자정 부근에 여러 시간에 걸쳐 PM2.5·PM10 동반 상승. 집중 연소/불꽃놀이와 부합하나 개별 관측 원인은 미확정.'),
]
contexts = []
rows = []
for c in cases:
    frame = frames[c['kind']]
    t = pd.Timestamp(c['time_cst'], tz='Asia/Shanghai').tz_convert('Asia/Seoul')
    part = frame.loc[frame.station_id.astype(str).eq(c['station_id']) & frame.time_kst.between(t-pd.Timedelta(hours=12),t+pd.Timedelta(hours=12))].copy()
    assert len(part) == 25, (c['id'],len(part))
    part = part.sort_values('time_kst').reset_index(drop=True)
    for p in ('pm25','pm10'):
        part[p+'_review_value'] = part[p+'_raw'] if p+'_raw' in part else part[p]
    hit = part.loc[part.time_kst.eq(t)].iloc[0]
    assert hit[c['pollutant']+'_review_value'] == c['value'], c
    meta = registry.loc[registry.station_key.eq('CN:'+c['station_id'])]
    if c['kind']=='station':
        assert len(meta)==1
        assert abs(float(meta.iloc[0].latitude)-c['lat'])<.0001
        assert abs(float(meta.iloc[0].longitude)-c['lon'])<.0001
    record = dict(c,time_kst=t.isoformat(),time_cst=pd.Timestamp(c['time_cst'],tz='Asia/Shanghai').isoformat(),
                  pm25=float(hit.pm25_review_value),pm10=float(hit.pm10_review_value),
                  coordinate_kind='city_seat_approximation_not_monitor_location' if c['kind']=='city' else 'recorded_station_metadata_historical_relocation_not_certified',
                  analysis_pm25=None if pd.isna(hit.pm25) else float(hit.pm25),
                  analysis_pm10=None if pd.isna(hit.pm10) else float(hit.pm10))
    part.insert(0,'case_id',c['id']);part.insert(1,'hours_relative',(part.time_kst-t)/pd.Timedelta(hours=1))
    part.to_csv(out/(c['id']+'_context.csv'),index=False,encoding='utf-8-sig')
    contexts.append(part);rows.append(record)

pd.DataFrame(rows).to_csv(out/'case_inventory.csv',index=False,encoding='utf-8-sig')
fig,axes=plt.subplots(3,2,figsize=(15,12),constrained_layout=True)
for ax,c,d in zip(axes.flat,cases,contexts):
    times=d.time_kst.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    ax.plot(times,d.pm25_review_value,'o-',ms=3,color='#0072B2',label='PM2.5 (preserved raw)')
    ax.plot(times,d.pm10_review_value,'s-',ms=3,color='#D55E00',label='PM10 (preserved raw)')
    t=pd.Timestamp(c['time_cst'])
    ax.axvline(t,color='#555555',linestyle=':',lw=1)
    ax.set_title(c['english']+' | '+t.strftime('%Y-%m-%d %H:%M CST'),fontsize=11)
    ax.set_yscale('log');ax.set_ylabel('Concentration (ug/m3; log scale)')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
fig.suptitle('Extreme-value review: isolated inconsistency vs sustained joint episode\nRecorded concentrations are evidence for review, not instrument certification.',fontsize=15)
fig.savefig(out/'extreme_value_contexts.png',dpi=170)
plt.close(fig)
sources=[
    {'title':'중국 생태환경부 2021-03-15 황사 실황','url':'https://www.mee.gov.cn/xxgk2018/xxgk/xxgk15/202103/t20210315_824656.html','supports':'실제 황사; 베이징09CST PM10=6450, PM2.5=462. 탕산 개별값 검증이 아님.'},
    {'title':'ACP 2023 황사 관측 연구','url':'https://acp.copernicus.org/articles/23/8241/2023/','supports':'거친 먼지 지배 황사. 탕산 개별값 검증이 아님.'},
    {'title':'이권호 외2024 울진 산불 관측','url':'https://www.jekosae.or.kr/xml/40960/40960.pdf','supports':'2022-03-07 09KST 울진 PM2.5=803, PM10=910과 산불 스모크 해석을 정확히 보고.'},
    {'title':'생태환경부2020 설날 불꽃놀이/기상 해설','url':'https://www.mee.gov.cn/xxgk2018/xxgk/xxgk15/202001/t20200128_761019.html','supports':'설날 집중 연소와 PM2.5 상승의 일반 메커니즘. 2024 안산 개별값 검증이 아님.'},
]
report={'status':'read_only_case_review','inputs_modified':False,'cases':rows,'primary_sources':sources,
        'positive_control':{'station_id':'KR:437421','name':'울진군','time_kst':'2022-03-07T09:00:00+09:00',
                            'pm25':803,'pm10_confirmed_by_paper':910,'lat':36.99723,'lon':129.40292,
                            'address':'경상북도 울진군 울진읍 읍내10길19 울진읍사무소 옥상',
                            'judgment':'exact_time_and_value_supported_by_primary_wildfire_study'},
        'limitations':['도시평균 좌표는 측정소 좌표가 아님.','원시 기기 상태/검교정 로그 미확보.',
                       '측정소 역사적 이전 이력은 독립 검증하지 못함.','사건 발생 사실과 개별 농도값 유효성은 구분.',
                       '인접 시각 값은 사후 진단용이며 실시간 입력 전처리 규칙이 아님.'],
        'input_sha256':before}
for p,digest in before.items():
    assert sha(Path(p))==digest,p
(out/'event_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'output':str(out),'cases':len(rows),'inputs_unchanged':True},ensure_ascii=False))
