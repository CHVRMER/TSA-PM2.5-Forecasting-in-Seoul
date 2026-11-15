"""Plot audited nationwide domestic and neighboring matching candidates over full-period descriptive HYSPLIT visit frequency.

No model fitting, concentration weighting, HYSPLIT rerun, or dataset alteration.
Dependencies: numpy, pandas, pyarrow, matplotlib. Basemap is bundled locally.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REGISTRY_MEMBER = 'Data/Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv'
BALANCED_MEMBER = 'Data/Dataset/Origin_Features_2021_2024_v3/japan_balanced_stations.csv'
SEOUL = (126.98, 37.57)
MAIN_EXTENT = (100, 141, 28, 55)
KOREA_EXTENT = (123.6, 132.4, 32.5, 39.7)
JAPAN_EXTENT = (128, 135, 30.5, 36.5)
STYLES = {
    'KR_OLD': dict(label='Korea existing', color='#df7529', marker='^', size=36),
    'KR_NEW': dict(label='Korea newly added', color='#6b3fa0', marker='o', size=20),
    'CN': dict(label='China: 48 stations', color='#b3248a', marker='o', size=42),
    'JP': dict(label='Japan: 188 stations', color='#167d60', marker='o', size=29),
    'MN': dict(label='Mongolia: 1 station', color='#5c3096', marker='D', size=115),
    'CN_MN': dict(label='China city means near Mongolia: 6', color='#efb52e', marker='s', size=73),
    'CN_RU': dict(label='China city means near Russia: 7', color='#9b3b32', marker='s', size=73),
}
SOURCE_COUNTS = {'KR': 67, 'CN': 48, 'JP': 188, 'MN': 1, 'CN_MN': 6, 'CN_RU': 7}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry(registry_path, existing_registry=None):
    raw=Path(registry_path).read_bytes()
    registry=pd.read_csv(io.BytesIO(raw),dtype={'station_id':str,'station_key':str})
    assert registry.station_key.is_unique and len(registry)>0
    assert not registry.station_key.eq('KR:111121').any(), 'Seoul target must be separate'
    assert np.isfinite(registry[['latitude','longitude']]).all().all()
    assert registry.latitude.between(-90,90).all() and registry.longitude.between(-180,180).all()
    if existing_registry is not None:
        old=pd.read_csv(existing_registry,dtype=str)
        old_kr=set(old.loc[old.country.eq('KR'),'station_key'])
    elif 'is_new_domestic' in registry:
        flag=registry.is_new_domestic.astype(str).str.lower().isin(['true','1','yes'])
        old_kr=set(registry.loc[registry.country.eq('KR')&~flag,'station_key'])
    else:
        raise ValueError('Pass --existing-registry (original115/317 CSV) or is_new_domestic in registry')
    registry['map_group']=registry.country
    registry.loc[registry.country.eq('KR')&registry.station_key.isin(old_kr),'map_group']='KR_OLD'
    registry.loc[registry.country.eq('KR')&~registry.station_key.isin(old_kr),'map_group']='KR_NEW'
    registry.loc[registry.region_group.eq('Mongolia_border_proxy'),'map_group']='CN_MN'
    registry.loc[registry.region_group.eq('Russia_border_proxy'),'map_group']='CN_RU'
    counts=registry.map_group.value_counts().to_dict()
    assert set(counts).issubset(STYLES), counts
    assert not registry.country.eq('RU').any(), 'No Russian observation source was supplied'
    for key,style in STYLES.items():
        style['label']=style['label'].split(':')[0]+': '+str(counts.get(key,0))
    assert ((registry.longitude>=MAIN_EXTENT[0])&(registry.longitude<=MAIN_EXTENT[1])
            &(registry.latitude>=MAIN_EXTENT[2])&(registry.latitude<=MAIN_EXTENT[3])).all()
    return registry,hashlib.sha256(raw).hexdigest()


def read_full_period_paths(path):
    begin = pd.Timestamp('2021-01-01', tz='Asia/Seoul').tz_convert('UTC')
    end = pd.Timestamp('2025-01-01', tz='Asia/Seoul').tz_convert('UTC')
    columns = ['arrival_time_kst', 'arrival_height_agl_m', 'lookback_hours', 'latitude', 'longitude']
    filters = [('arrival_time_utc', '>=', begin.to_pydatetime()),
               ('arrival_time_utc', '<', end.to_pydatetime()),
               ('arrival_height_agl_m', '=', 500.0),
               ('lookback_hours', '>=', 1.0), ('lookback_hours', '<=', 48.0)]
    frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    frame = frame.sort_values(['arrival_time_kst', 'lookback_hours']).reset_index(drop=True)
    assert frame.arrival_time_kst.dt.tz is not None
    assert frame.arrival_time_kst.dt.year.isin([2021, 2022, 2023, 2024]).all()
    assert frame.arrival_height_agl_m.eq(500).all()
    assert np.isin(frame.lookback_hours, np.arange(1, 49)).all()
    assert not frame.duplicated(['arrival_time_kst', 'lookback_hours']).any()
    count = frame.groupby('arrival_time_kst').lookback_hours.agg(['count', 'nunique'])
    assert len(count) == 11688 and len(frame) == 561024
    assert count.eq(48).all().all()
    expected = pd.date_range('2021-01-01', '2025-01-01', freq='3h', inclusive='left', tz='Asia/Seoul')
    assert pd.DatetimeIndex(count.index).equals(expected), 'Incomplete or off-grid arrivals'
    assert count.groupby(count.index.year).size().to_dict() == {2021:2920, 2022:2920, 2023:2920, 2024:2928}
    assert np.isfinite(frame[['latitude', 'longitude']]).all().all()
    return frame


def visit_frequency(frame):
    """Each path counts once per 1-degree cell visited at hourly ages 1..48."""
    arrivals = pd.DatetimeIndex(frame.arrival_time_kst.unique()).sort_values()
    codes = arrivals.get_indexer(frame.arrival_time_kst)
    ix = np.floor(frame.longitude.to_numpy() + 180).astype(int)
    iy = np.floor(frame.latitude.to_numpy() + 90).astype(int)
    assert ((0 <= ix) & (ix < 360) & (0 <= iy) & (iy < 180)).all()
    cell = iy*360 + ix
    unique_pairs = np.unique(codes*(180*360) + cell)
    counts = np.bincount(unique_pairs % (180*360), minlength=180*360).reshape(180, 360)
    frequency = counts*100.0/len(arrivals)
    assert frequency.min() >= 0 and frequency.max() <= 100
    # Independent path-by-path set calculation guards against residence-time overcounting.
    independent = np.zeros_like(counts)
    for _, path in frame.groupby('arrival_time_kst', sort=False):
        cells = set(zip(np.floor(path.latitude+90).astype(int), np.floor(path.longitude+180).astype(int)))
        for cy, cx in cells:
            independent[cy, cx] += 1
    assert np.array_equal(independent, counts)
    assert len(unique_pairs) == counts.sum()
    return counts, frequency


def country_shapes(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    polygons, borders = [], []
    for feature in doc['features']:
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            parts = [geom['coordinates']]
        elif geom['type'] == 'MultiPolygon':
            parts = geom['coordinates']
        else:
            continue
        for part in parts:
            ring = np.asarray(part[0], dtype=float)
            if ring[:, 0].max() < 70 or ring[:, 0].min() > 155 or ring[:, 1].max() < 15 or ring[:, 1].min() > 70:
                continue
            polygons.append(ring)
            # Avoid artificial cross-map line segments at the antimeridian.
            split = np.flatnonzero(np.abs(np.diff(ring[:, 0])) > 180)+1
            borders.extend(segment for segment in np.split(ring, split) if len(segment) > 1)
    return polygons, borders


def base_map(ax, shapes, extent, frequency=None, grid_step=5, label_size=10):
    polygons, borders = shapes
    ax.set_facecolor('#eef3f6')
    ax.add_collection(PolyCollection(polygons, facecolors='#f8f8f5', edgecolors='none', zorder=0))
    mesh = None
    if frequency is not None:
        mesh = ax.pcolormesh(np.arange(-180, 181), np.arange(-90, 91),
                             np.ma.masked_where(frequency == 0, frequency),
                             cmap='YlGnBu', norm=LogNorm(vmin=0.01, vmax=100),
                             shading='flat', zorder=1, rasterized=True)
    ax.add_collection(LineCollection(borders, colors='#687781', linewidths=0.55, alpha=0.82, zorder=2))
    x0, x1, y0, y1 = extent
    ax.set(xlim=(x0, x1), ylim=(y0, y1))
    ax.set_aspect(1/np.cos(np.deg2rad((y0+y1)/2)))
    ax.set_xticks(np.arange(np.ceil(x0/grid_step)*grid_step, x1+0.001, grid_step))
    ax.set_yticks(np.arange(np.ceil(y0/grid_step)*grid_step, y1+0.001, grid_step))
    ax.grid(color='#73838c', alpha=0.18, linewidth=0.6, zorder=2.2)
    ax.set_xlabel('Longitude (degrees east)', fontsize=label_size+1)
    ax.set_ylabel('Latitude (degrees north)', fontsize=label_size+1)
    ax.tick_params(labelsize=label_size)
    for spine in ax.spines.values():
        spine.set_color('#7c8a92')
        spine.set_linewidth(0.8)
    return mesh


def points(ax, registry, scale=1):
    for group, style in STYLES.items():
        part = registry.loc[registry.map_group.eq(group)]
        ax.scatter(part.longitude, part.latitude, marker=style['marker'], s=style['size']*scale,
                   c=style['color'], edgecolors='white' if group not in ['CN_MN', 'CN_RU'] else '#242d36',
                   linewidths=0.65, alpha=0.95, zorder=5 if group not in ['CN_MN', 'CN_RU', 'MN'] else 6)
    ax.scatter(*SEOUL, marker='*', s=230*scale, color='#b81d2c', edgecolors='white', linewidths=1.2, zorder=8)


def annotation(ax, text, xy, offset=(7, 7), fontsize=9.5, weight='normal'):
    return ax.annotate(text, xy, xytext=offset, textcoords='offset points', fontsize=fontsize,
                       color='#23323c', weight=weight, zorder=9,
                       bbox=dict(boxstyle='round,pad=0.13', fc='white', ec='none', alpha=0.83))


def legend(fig, y=0.118, ncol=4):
    handles = [Line2D([], [], linestyle='', marker=style['marker'], markersize=8,
                      markerfacecolor=style['color'], markeredgecolor='white' if group not in ['CN_MN', 'CN_RU'] else '#242d36',
                      label=style['label']) for group, style in STYLES.items()]
    handles.append(Line2D([], [], linestyle='', marker='*', markersize=13,
                          markerfacecolor='#b81d2c', markeredgecolor='white', label='Seoul Junggu target (separate)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=9.8, columnspacing=1.7, handletextpad=0.5)


def colorbar(fig, mesh, rect):
    axis = fig.add_axes(rect)
    bar = fig.colorbar(mesh, cax=axis, orientation='horizontal', extend='min', ticks=[0.01, 0.1, 1, 10, 100])
    bar.ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
    bar.ax.tick_params(labelsize=9)
    bar.set_label('Trajectories visiting each 1-degree cell (%; logarithmic colour scale)', fontsize=10, labelpad=5)


def footnotes(fig, heatmap=True):
    if heatmap:
        fig.text(0.5, 0.040, '2021-2024 | 11,688 arrivals, every 3 h | 500 m AGL on arrival | hourly positions at ages 1-48 h; age 0 excluded.',
                 ha='center', fontsize=8.5, color='#425562')
        fig.text(0.5, 0.022, 'Darker cells mean more frequent air-mass passage, not higher PM2.5 or pollution-source contributions. Display crops do not change the denominator.',
                 ha='center', fontsize=8.3, color='#425562')
    else:
        fig.text(0.5, 0.040, 'Grey lines: 192 monthly-balanced display paths (4 per month), from all 11,688 arrivals; each line shows ages 1-48 h.',
                 ha='center', fontsize=8.5, color='#425562')
        fig.text(0.5, 0.022, 'Paths describe modeled air-mass motion, not PM2.5 concentrations, emission sources or country contributions.',
                 ha='center', fontsize=8.3, color='#425562')
    fig.text(0.5, 0.004, 'No Russian monitoring data are included. Border squares are Chinese city means at approximate city-centre coordinates. Basemap: Natural Earth.',
             ha='center', fontsize=8.2, color='#425562')


def save(fig, output, stem):
    for ext in ('png', 'svg'):
        fig.savefig(output/f'{stem}.{ext}', dpi=220, facecolor='white')
    plt.close(fig)


def overview(registry, frequency, shapes, output):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.09, 0.265, 0.80, 0.62])
    mesh = base_map(ax, shapes, MAIN_EXTENT, frequency)
    points(ax, registry)
    for text, xy in [('MONGOLIA', (111.8, 49.2)), ('CHINA', (113.5, 35.2)), ('RUSSIA', (136.2, 52.4)),
                      ('JAPAN', (137.7, 34.4))]:
        ax.text(*xy, text, fontsize=11, color='#425868', alpha=0.82, ha='center', zorder=3)
    annotation(ax, 'Seoul', SEOUL, (9, -12), weight='bold')
    annotation(ax, 'Ulaanbaatar', (106.929, 47.928), (-23, 12))
    heihe = registry.loc[registry.station_key.eq('CN:city_heihe')].iloc[0]
    annotation(ax, 'Heihe (Chinese city mean)', (heihe.longitude, heihe.latitude), (-111, 13), fontsize=8.5)
    ax.add_patch(Rectangle((KOREA_EXTENT[0], KOREA_EXTENT[2]), KOREA_EXTENT[1]-KOREA_EXTENT[0], KOREA_EXTENT[3]-KOREA_EXTENT[2],
                           fill=False, edgecolor='#4b5563', linewidth=0.8, linestyle='--', zorder=3))
    ax.add_patch(Rectangle((JAPAN_EXTENT[0], JAPAN_EXTENT[2]), JAPAN_EXTENT[1]-JAPAN_EXTENT[0], JAPAN_EXTENT[3]-JAPAN_EXTENT[2],
                           fill=False, edgecolor='#4b5563', linewidth=0.8, linestyle='--', zorder=3))
    fig.text(0.5, 0.956, f'{len(registry):,} matching candidates along Seoul air-mass pathways', ha='center', fontsize=20, weight='bold', color='#182c39')
    fig.text(0.5, 0.925, f"All audited candidates shown | Korea {int(registry.country.eq('KR').sum())} + China {int(registry.country.eq('CN').sum())} + Japan {int(registry.country.eq('JP').sum())} + Mongolia {int(registry.country.eq('MN').sum())}", ha='center', fontsize=11.5, color='#425868')
    colorbar(fig, mesh, [0.20, 0.196, 0.60, 0.016])
    legend(fig, 0.147)
    footnotes(fig)
    save(fig, output, 'matching_candidates_national_neighbors_heatmap')


def detail(registry, frequency, shapes, output):
    fig = plt.figure(figsize=(15, 10.5))
    axk = fig.add_axes([0.065, 0.285, 0.42, 0.575])
    axj = fig.add_axes([0.555, 0.285, 0.39, 0.575])
    mesh = base_map(axk, shapes, KOREA_EXTENT, frequency, grid_step=1)
    base_map(axj, shapes, JAPAN_EXTENT, frequency, grid_step=1)
    points(axk, registry.loc[registry.country.eq('KR')], scale=1.25)
    points(axj, registry.loc[registry.country.eq('JP')], scale=1.20)
    axk.set_title(f"South Korea | {int(registry.country.eq('KR').sum())} candidates + Seoul target", fontsize=12.5, pad=12, color='#243c4b')
    axj.set_title(f"Western Japan | all {int(registry.country.eq('JP').sum())} matching stations", fontsize=12.5, pad=12, color='#243c4b')
    annotation(axk, 'Seoul', SEOUL, (10, -13), fontsize=10, weight='bold')
    baengnyeong = registry.loc[registry.station_key.eq('KR:831492')].iloc[0]
    for text, xy, offset in [('Baengnyeong', (baengnyeong.longitude, baengnyeong.latitude), (-17, 9)),
                              ('East / NE candidates', (128.12, 38.5), (0, 12))]:
        annotation(axk, text, xy, offset, fontsize=9)
    for text, xy, offset in [('Kyushu', (130.75, 32.2), (-10, -30)),
                              ('Chugoku', (132.6, 35.0), (-10, 24)),
                              ('Tsushima', (129.32, 34.35), (-37, 14))]:
        annotation(axj, text, xy, offset, fontsize=9.5)
    fig.text(0.5, 0.955, 'Regional detail of the expanded matching network', ha='center', fontsize=20, weight='bold', color='#182c39')
    fig.text(0.5, 0.922, 'Same 1-degree visit-frequency grid and colour limits as the overview; no smoothing or local renormalization',
             ha='center', fontsize=10.8, color='#425868')
    colorbar(fig, mesh, [0.20, 0.205, 0.60, 0.018])
    legend(fig, 0.147)
    footnotes(fig)
    save(fig, output, 'matching_candidates_national_neighbors_regional_detail')


def trajectory_sample(registry, frame, shapes, output):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.09, 0.22, 0.80, 0.665])
    base_map(ax, shapes, MAIN_EXTENT, grid_step=5)
    arrivals = pd.DatetimeIndex(frame.arrival_time_kst.unique()).sort_values()
    chosen = []
    months = arrivals.strftime('%Y-%m')
    for month in pd.unique(months):
        candidates = arrivals[months == month]
        positions = ((np.arange(4)+0.5)*len(candidates)/4).astype(int)
        chosen.extend(candidates[positions])
    assert len(chosen) == 192 and len(set(chosen)) == 192
    for arrival, path in frame.loc[frame.arrival_time_kst.isin(chosen)].groupby('arrival_time_kst'):
        path = path.sort_values('lookback_hours', ascending=False)
        ax.plot(path.longitude, path.latitude, color='#6c8fa4', alpha=0.30, linewidth=0.7, zorder=2.5)
    points(ax, registry)
    annotation(ax, 'Seoul', SEOUL, (9, -12), weight='bold')
    annotation(ax, 'Ulaanbaatar', (106.929, 47.928), (-23, 12))
    fig.text(0.5, 0.956, 'Matching candidates and sampled air-mass paths', ha='center', fontsize=20, weight='bold', color='#182c39')
    fig.text(0.5, 0.925, f'{len(registry):,} candidates | 2021-2024 arrivals, 500 m AGL | paths change from one arrival to the next', ha='center', fontsize=11.5, color='#425868')
    legend(fig, 0.147)
    footnotes(fig, heatmap=False)
    save(fig, output, 'matching_candidates_national_neighbors_trajectory_sample')
    return [arrival.isoformat() for arrival in chosen]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--existing-registry',type=Path)
    parser.add_argument('--trajectories',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    output=args.out/'Results';output.mkdir(parents=True,exist_ok=True)
    basemap=Path(__file__).resolve().parent/'assets/ne_110m_admin_0_countries.geojson'
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.titleweight':'normal'})
    registry,registry_hash=read_registry(args.registry,args.existing_registry)
    frame=read_full_period_paths(args.trajectories)
    counts,frequency=visit_frequency(frame)
    shapes=country_shapes(basemap)
    overview(registry,frequency,shapes,output)
    detail(registry,frequency,shapes,output)
    sampled=trajectory_sample(registry,frame,shapes,output)
    registry.to_csv(output/'plotted_candidate_registry.csv',index=False,encoding='utf-8-sig')
    manifest={'created_date_kst':'2026-11-07','registry':str(args.registry),'registry_sha256':registry_hash,
        'candidate_count':len(registry),'group_counts':registry.map_group.value_counts().to_dict(),
        'actual_russian_observations':0,'seoul_target_separate_from_candidate_count':True,
        'trajectory_source':str(args.trajectories),'trajectory_source_sha256':sha256(args.trajectories),
        'arrivals':11688,'trajectory_points':len(frame),'arrival_height_agl_m':500,
        'first_arrival_kst':frame.arrival_time_kst.min().isoformat(),'last_arrival_kst':frame.arrival_time_kst.max().isoformat(),
        'arrival_interval_hours':3,'ages_hours':[1,48],'age_zero_excluded':True,
        'grid_degrees':1,'formula':'100 * unique trajectories visiting a cell / all 11688 arrivals',
        'max_frequency_pct':float(frequency.max()),'nonzero_global_grid_cells':int(np.count_nonzero(counts)),
        'frequency_independently_recomputed_per_path_sets':True,
        'display_extent':list(MAIN_EXTENT),'display_crop_does_not_change_denominator':True,
        'all_candidates_inside_overview':True,'display_sample_arrivals':sampled,
        'source_datasets_models_and_previous_figures_modified':False,
        'purpose':'full-period descriptive geography; not feature selection or forecast performance',
        'colour_scale_percent':[.01,100],
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.iterdir()) if p.is_file()}}
    (args.out/'visualization_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['candidate_count','group_counts','arrivals','trajectory_points']},ensure_ascii=False))


if __name__=='__main__':
    main()
