"""Render geographic route EDA tables without re-estimating statistical results.

Usage: python Codes/plot_results.py --eda-root <analysis directory>
Inputs are produced by run_geographic_eda.py and classify_geographic_routes.py.
All row orders are geographic, never an adjusted-effect ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HORIZONS = [3, 6, 12, 24]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
PRIMARY_WINDOW = "07-48"
COLORS = {"01-48": "#9aacbd", "07-48": "#007d86", "13-48": "#d18430"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
    "savefig.facecolor": "white", "axes.titlepad": 10,
})


def note(fig, text, y=0.025, x=0.018, size=9):
    fig.text(x, y, text, ha="left", va="bottom", color="#475563", fontsize=size, linespacing=1.6)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required EDA table is missing: {path}")
    return pd.read_csv(path)


def ordered(frame, ids, **filters):
    for col, val in filters.items():
        frame = frame.loc[frame[col] == val]
    if frame["region_id"].duplicated().any():
        raise ValueError(f"Duplicate region rows after filters {filters}")
    return frame.set_index("region_id").reindex(ids)


def separators(ax, n=20):
    for y in (6.5, 13.5, 18.5):
        if y < n - 0.5:
            ax.axhline(y, color="#bcc5cd", lw=0.85, zorder=1)


def save(fig, result_dir, stem, files):
    for ext in ("png", "svg"):
        path = result_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
        files.append(path)
    plt.close(fig)


def passage_plot(r, defs, files):
    d = read_csv(r / "passage_rates.csv")
    ids = defs.region_id.tolist()
    labels = defs.label_en.tolist()
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(13.8, 10.9))
    fig.subplots_adjust(left=.22, right=.93, top=.855, bottom=.14)
    for offset, window in [(-.18, "01-48"), (.18, "07-48")]:
        s = ordered(d, ids, window=window, season="ALL")
        ax.barh(y + offset, s.segment_pass_pct, .32, color=COLORS[window],
                label=f"{window.replace('-', '–')} h before arrival", zorder=2)
    s13 = ordered(d, ids, window="13-48", season="ALL")
    ax.scatter(s13.segment_pass_pct, y + .18, marker="|", color=COLORS["13-48"], s=80,
               label="13–48 h sensitivity", zorder=4, linewidths=1.6)
    s = ordered(d, ids, window="07-48", season="ALL")
    for i, row in enumerate(s.itertuples()):
        value = row.segment_pass_pct
        ax.text(min(value + .8, 101.0), i + .18, f"{value:.1f}%", va="center", fontsize=9, color="#005d64")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 108), ylim=(len(ids)-.45, -.6),
           xlabel="Arrivals with a modeled path intersecting the region (%)")
    ax.xaxis.grid(True, color="#e3e8ec", zorder=0)
    ax.set_axisbelow(True)
    separators(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.22,.902),
               ncol=3, frameon=False, fontsize=9, borderaxespad=0)
    fig.suptitle("Where did Seoul-bound air paths pass?", x=.22, y=.968, ha="left", fontsize=20, weight="bold")
    fig.text(.22, .926, "2021–2022 training period | 5,840 arrivals | 500 m AGL arrival height", fontsize=11, color="#4d5c69")
    note(fig, "Passage is intersection with geographic polygons along adjacent hourly modeled path segments; age 0 is excluded.\n"
         "A path can visit multiple regions. Percentages do not sum to 100%. Boundaries may count in both adjacent regions.\n"
         "Sea / unclassified includes mixed-source coastline gaps; it is not guaranteed actual sea.\n"
         "These are modeled air-path passage rates, not pollution mass, source contributions, or station data availability.")
    save(fig, r, "regional_passage_rates", files)


def heatmap(ax, values, labels, columns, title, cmap, vmin=None, vmax=None, fmt=".1f", counts=None, small_n=30):
    vals = np.asarray(values, dtype=float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e8eaed")
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels,
           xticks=np.arange(len(columns)), xticklabels=columns, title=title)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    lo, hi = im.get_clim()
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            if np.isfinite(val):
                suffix = "†" if counts is not None and counts[i, j] < small_n else ""
                text = f"{val:{fmt}}{suffix}"
                color = "white" if (val - lo) / max(hi-lo, 1e-12) > .60 else "#152b3b"
            else:
                text, color = "NA", "#65707b"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    return im


def raw_outcome_plot(r, defs, files):
    d = read_csv(r / "raw_region_future_outcomes.csv")
    d = d.loc[d.window == PRIMARY_WINDOW]
    ids = defs.region_id.tolist()
    main = ordered(d, ids, horizon=6)
    labels = [f"{name}  [n={int(n):,}]" for name,n in zip(defs.label_en, main.passed_target_n)]
    valmean = d.pivot(index="region_id", columns="horizon", values="passed_future_mean").reindex(index=ids, columns=HORIZONS).to_numpy()
    valhigh = d.pivot(index="region_id", columns="horizon", values="passed_high_pct").reindex(index=ids, columns=HORIZONS).to_numpy()
    counts = d.pivot(index="region_id", columns="horizon", values="passed_target_n").reindex(index=ids, columns=HORIZONS).to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(15.0, 11), gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(left=.25, right=.96, top=.86, bottom=.15, wspace=.28)
    finite = valmean[np.isfinite(valmean)]
    vmax = max(30, np.ceil(finite.max()/5)*5)
    im1 = heatmap(axs[0], valmean, labels, [f"+{h} h" for h in HORIZONS],
                  "Future mean PM2.5 (µg/m³)", "YlGnBu", vmin=0, vmax=vmax, counts=counts)
    im2 = heatmap(axs[1], valhigh, [""]*len(ids), [f"+{h} h" for h in HORIZONS],
                  "Future concentration ≥35 µg/m³ (%)", "YlOrRd", vmin=0, vmax=100, counts=counts)
    for ax,im in zip(axs,(im1,im2)):
        separators(ax)
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.04, pad=.045)
    fig.suptitle("Future Junggu PM2.5 after a path passed each region", x=.25, y=.974, ha="left", fontsize=18, weight="bold")
    fig.text(.25,.932,"Raw descriptive associations | 2021–2022 training | passage 7–48 h before arrival",fontsize=11,color="#4d5c69")
    note(fig, "Row brackets show valid +6 h target counts among passed arrivals; exact target availability differs by horizon. † denotes fewer than 30 passed targets.\n"
         "No passage at this window gives NA. Raw means combine season and current-PM differences; they do not isolate transport effects.\n"
         "The hourly ≥35 µg/m³ research category is not an official air-quality alert. Passage regions overlap; no source or causal contributions are estimated.", y=.02)
    save(fig, r, "raw_future_outcomes_by_region", files)


def contrast_plot(r, defs, files):
    ids = defs.region_id.tolist()
    a = ordered(read_csv(r/"adjusted_region_contrasts.csv"), ids, window=PRIMARY_WINDOW, horizon=6)
    raw = ordered(read_csv(r/"raw_region_future_outcomes.csv"), ids, window=PRIMARY_WINDOW, horizon=6)
    fig, axs = plt.subplots(1,3,figsize=(17,11.8), gridspec_kw={"width_ratios":[1,1,.88]})
    fig.subplots_adjust(left=.16,right=.985,top=.795,bottom=.15,wspace=.20)
    y = np.arange(len(ids))
    for ax in axs:
        ax.set(ylim=(len(ids)-.4,-.7), yticks=y)
        separators(ax)
        ax.tick_params(axis="y",length=0)
    axs[0].set_yticklabels(defs.label_en)
    axs[1].set_yticklabels([""]*len(ids))
    axs[2].set_yticklabels([""]*len(ids))
    labels=[("raw_mean_difference","adjusted_mean_difference","adjusted_mean_diff_ci95_low","adjusted_mean_diff_ci95_high","Mean concentration difference (µg/m³)"),
            ("raw_high_difference_pp","adjusted_high_difference_pp","adjusted_high_diff_ci95_low","adjusted_high_diff_ci95_high","≥35 proportion difference (percentage points)")]
    for ax,(rk,ak,lk,hk,xlabel) in zip(axs[:2],labels):
        ax.axvline(0,color="#798995",lw=1)
        ax.xaxis.grid(True,color="#e4e8ec",zorder=0)
        raw_v=raw[rk].to_numpy(float)
        ax.scatter(raw_v,y-.14,color="#8795a0",s=32,marker="x",zorder=3)
        for i,(_,row) in enumerate(a.iterrows()):
            v=row.get(ak,np.nan)
            lowcov=row.target_population_coverage_pct < 50
            unstable=row.get("CI_status") != "stable_enough"
            color="#c37827" if lowcov or unstable else "#007d86"
            if not np.isfinite(v):
                ax.text(.97,i+.16,"NA",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=9,color="#697782")
                continue
            lo,hi=row.get(lk,np.nan),row.get(hk,np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo,hi],[i+.16]*2,color=color,lw=1.6,zorder=3)
                ax.plot([lo,lo],[i+.10,i+.22],color=color,lw=1)
                ax.plot([hi,hi],[i+.10,i+.22],color=color,lw=1)
            ax.scatter([v],[i+.16],s=39,facecolors="white" if lowcov or unstable else color,edgecolors=color,zorder=4)
        ax.set_xlabel(xlabel)
    axs[0].set_title("+6 h PM2.5: passed minus non-passed")
    axs[1].set_title("+6 h high proportion: passed minus non-passed")
    ax=axs[2]
    ax.set_title("Standardization support")
    ax.set_xlim(0,1)
    ax.set_xticks([])
    ax.spines[["left","bottom"]].set_visible(False)
    ax.text(.02,-.66,"Pass / non-pass n",weight="bold",fontsize=9)
    ax.text(.77,-.66,"Coverage",weight="bold",fontsize=9,ha="center")
    for i,(_,row) in enumerate(a.iterrows()):
        cov=row.target_population_coverage_pct
        status=row.status=="estimable"
        flag="*" if cov<50 and status else ""
        txt=f"{int(row.pass_n_on_support):,} / {int(row.nonpass_n_on_support):,}" if status else "No common strata"
        ax.text(.02,i+.12,txt,va="center",fontsize=9,color="#465561" if status else "#89939b")
        ax.text(.77,i+.12,f"{cov:.1f}%{flag}" if status else "NA",va="center",ha="center",fontsize=9,color="#ad6a22" if cov<50 else "#264c58")
    fig.suptitle("Does the association persist after season and current PM are considered?", x=.16,y=.973,ha="left",fontsize=18,weight="bold")
    fig.text(.16,.931,"2021–2022 training | path passage 7–48 h before arrival | future target +6 h",fontsize=11,color="#4d5c69")
    handles=[Line2D([0],[0],color="#8795a0",marker="x",ls="none",label="Raw contrast"),
             Line2D([0],[0],color="#007d86",marker="o",ls="-",label="Season/current-bin standardized, 95% block-bootstrap interval"),
             Line2D([0],[0],color="#c37827",marker="o",markerfacecolor="white",ls="none",label="Limited support (<50%) or unstable interval")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.16,.905),ncol=1,frameon=False,fontsize=9)
    note(fig,"Standardized within season × current-PM bins using common strata with ≥10 pass and ≥10 non-pass arrivals; original support stays fixed in resampling.\n"
         "Coverage is the fraction of eligible current-observed training arrivals represented by those strata. * <50%: restricted population; NA: no common support.\n"
         "Intervals use 500 calendar 7-day block-bootstrap replicates; unstable intervals are omitted. Different rows can represent different populations—do not rank them.\n"
         "Residual confounding and within-bin current-PM differences remain. The ≥35 hourly category is for research; these contrasts are not causal transport contributions.",y=.018,size=9)
    save(fig,r,"adjusted_h6_region_contrasts",files)


def seasonal_plot(r,defs,files):
    d=read_csv(r/"seasonal_region_future_outcomes.csv")
    d=d.loc[(d.window==PRIMARY_WINDOW)&(d.horizon==6)]
    ids=defs.region_id.tolist()
    vals=[d.pivot(index="region_id",columns="season",values=key).reindex(index=ids,columns=SEASONS).to_numpy(float)
          for key in ("passed_future_mean","passed_high_pct","passed_target_n")]
    fig,axs=plt.subplots(1,2,figsize=(15,11))
    fig.subplots_adjust(left=.21,right=.96,top=.86,bottom=.15,wspace=.28)
    finite=vals[0][np.isfinite(vals[0])]
    im1=heatmap(axs[0],vals[0],defs.label_en.tolist(),SEASONS,"+6 h mean PM2.5 (µg/m³)","YlGnBu",0,max(40,np.ceil(finite.max()/5)*5),counts=vals[2])
    im2=heatmap(axs[1],vals[1],[""]*len(ids),SEASONS,"+6 h concentration ≥35 µg/m³ (%)","YlOrRd",0,100,counts=vals[2])
    for ax,im in zip(axs,(im1,im2)):
        separators(ax)
        fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.04,pad=.045)
    fig.suptitle("Future PM2.5 following regional passage differs across seasons",x=.21,y=.974,ha="left",fontsize=18,weight="bold")
    fig.text(.21,.932,"Raw within-season outcomes | 2021–2022 training | path passage 7–48 h before arrival",fontsize=11,color="#4d5c69")
    note(fig,"DJF = Dec–Feb; MAM = Mar–May; JJA = Jun–Aug; SON = Sep–Nov. † fewer than 30 passed arrivals with a valid +6 h target.\n"
         "NA means no eligible passage arrivals in that season. These raw cells are not adjusted for current PM, circulation, rainfall, or other co-occurring regions.\n"
         "The hourly ≥35 µg/m³ category is a research threshold, not an official alert. Associations do not estimate pollution sources or causal contributions.",y=.025)
    save(fig,r,"seasonal_raw_h6_outcomes",files)


def polygon_parts(geometry):
    if not geometry:return
    if geometry["type"]=="Polygon":
        yield np.asarray(geometry["coordinates"][0])
    elif geometry["type"]=="MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield np.asarray(polygon[0])
    elif geometry["type"]=="GeometryCollection":
        for geo in geometry["geometries"]:
            yield from polygon_parts(geo)


def geography_plot(r,defs,files):
    assets=r.parent/"Codes"/"assets"
    regions=assets/"analysis_regions.geojson"
    pointsfile=r/"classified_trajectory_points_train.parquet"
    if not regions.exists() or not pointsfile.exists():
        raise FileNotFoundError("Geographic overview needs region polygons and classified training positions")
    features=json.loads(regions.read_text(encoding="utf-8"))["features"]
    world=json.loads((assets/"ne_110m_admin_0_countries.geojson").read_text(encoding="utf-8"))["features"]
    pts=pd.read_parquet(pointsfile)
    dates=pts.arrival_time_kst.drop_duplicates().sort_values()
    # Two deterministic, time-spread arrivals per month: 48 paths in 24 months.
    calendar=pd.DataFrame({"t":dates})
    calendar["month"]=calendar.t.dt.strftime("%Y-%m")
    selected=[]
    for _,g in calendar.groupby("month",sort=True):
        selected.extend(g.t.iloc[[len(g)//3,2*len(g)//3]].tolist())
    pts=pts.loc[pts.arrival_time_kst.isin(selected)]
    fig,axs=plt.subplots(1,2,figsize=(17,10.2),gridspec_kw={"width_ratios":[1.65,1]})
    fig.subplots_adjust(left=.045,right=.985,top=.85,bottom=.15,wspace=.12)
    palettes={"KR":plt.get_cmap("Blues"),"CN":plt.get_cmap("YlOrBr")}
    kr_cn_n={"KR":0,"CN":0}
    regioncolors={}
    for feature in features:
        prop=feature["properties"];rid=prop["region_id"];group=prop.get("group","")
        if group in palettes:
            kr_cn_n[group]+=1
            c=palettes[group](.2+kr_cn_n[group]*.08)
        else:c={"JP":"#dac2e0","MN":"#c7dfc7","RU":"#c8d4e5","KP":"#d2d2d2","OTHER_LAND":"#eeeeea"}.get(rid,"#eeeeea")
        regioncolors[rid]=c
    bounds=[(95,145,20,58),(123.3,131.2,32.5,39.7)]
    for ax,(west,east,south,north) in zip(axs,bounds):
        ax.set_facecolor("#eff7fa")
        base=[p for feature in world for p in polygon_parts(feature["geometry"])]
        ax.add_collection(PolyCollection(base,facecolors="#e9e9e3",edgecolors="#a3afb4",linewidths=.3,zorder=0))
        for feature in features:
            parts=list(polygon_parts(feature["geometry"]))
            if parts:ax.add_collection(PolyCollection(parts,facecolors=regioncolors[feature["properties"]["region_id"]],edgecolors="#748995",linewidths=.4,alpha=.85,zorder=1))
        for _,g in pts.groupby("arrival_time_kst",sort=True):
            g=g.sort_values("lookback_hours")
            ax.plot(g.longitude,g.latitude,color="#2d536b",alpha=.23,lw=.65,zorder=2)
        ax.scatter(126.98,37.57,marker="*",s=165,color="#b52b3a",edgecolor="white",lw=.8,zorder=6)
        ax.set(xlim=(west,east),ylim=(south,north),xlabel="Longitude (°E)",ylabel="Latitude (°N)")
        ax.set_aspect(1/np.cos(np.deg2rad(40)))
        ax.grid(color="#c7d2d7",alpha=.45,lw=.6)
    axs[0].set_title("Regional context: 48 sampled training paths")
    axs[1].set_title("Domestic geographic groups")
    for label,x,y in [("Russia",132,55),("Mongolia",104,48),("Inner Mongolia",112,44),("Heilongjiang",129,49),("Jilin",127.2,44.1),("Liaoning",121,41),("BTH",116,38.8),("Shandong",117.1,35),("Japan",138,34.5),("North Korea",127,40.1)]:
        axs[0].text(x,y,label,fontsize=9,ha="center",va="center",color="#2a414c",bbox=dict(facecolor="white",alpha=.72,edgecolor="none",pad=.8),zorder=4)
    for label,x,y in [("Capital",126.3,37.4),("Gangwon",128.45,38),("Chungcheong",127.0,36.4),("Honam",126.5,34.8),("Daegyeong",128.9,36.5),("Dongnam",129.3,35.4),("Jeju",126.55,33.3)]:
        axs[1].text(x,y,label,fontsize=9,ha="center",va="center",color="#22435a",bbox=dict(facecolor="white",alpha=.8,edgecolor="none",pad=.8),zorder=4)
    fig.suptitle("Geographic regions used to summarize Seoul-bound air paths",x=.045,y=.973,ha="left",fontsize=20,weight="bold")
    fig.text(.045,.93,"Training arrivals 2021–2022 | 500 m AGL arrival height | ages 1–48 h | two arrivals per month sampled for display",fontsize=11,color="#4d5c69")
    note(fig,"Background colors identify geographic regions, not passage intensity or PM2.5. Passage statistics use every validated training path, not just this display sample.\n"
         "Curves are HYSPLIT modeled air paths, not observed particle tracks. Region intersections do not establish emissions, transport mass, or country contributions.\n"
         "Boundaries: geoBoundaries / Natural Earth; Japan: MLIT National Land Numerical Information, CC BY 4.0 (2022, grouped for this display).\n"
         "Pinned source URLs, licenses, geometry versions, and boundary/coastline limitations are listed in geographic_classification_manifest.json.",y=.025)
    save(fig,r,"geographic_regions_and_training_paths",files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-root",type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args()
    r=args.eda_root.resolve()/"Results"
    defs=read_csv(r/"region_definitions.csv").sort_values("sort_order")
    if len(defs)!=20 or defs.region_id.duplicated().any():raise ValueError("Expected exactly 20 distinct geographic regions")
    files=[]
    for plot in (passage_plot,raw_outcome_plot,contrast_plot,seasonal_plot,geography_plot):
        plot(r,defs,files)
        print(f"Rendered {plot.__name__}",flush=True)
    manifest={"primary_window":PRIMARY_WINDOW,"training_years":[2021,2022],"geographic_order":defs.region_id.tolist(),
              "figures":[{"file":str(f.name),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
              "notes":["Raw descriptive and standardized associational results are kept distinct.","Adjusted populations may differ by region; figure rows are not an effect ranking.","No source mass or causal country contribution is estimated."]}
    (r/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved {len(files)} figure files to {r}")


if __name__=="__main__":main()
