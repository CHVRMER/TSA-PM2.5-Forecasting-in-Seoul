"""Plot the root-produced conditional tables without changing any calculation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent
OUT=WORK/'Figures'
OUT.mkdir(exist_ok=True)
H=[12,24,36,48]
COLORS={2023:'#296e9a',2024:'#bf6327'}
MARKERS={2023:'o',2024:'s'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.spines.left':False,
    'axes.edgecolor':'#8999a3','figure.facecolor':'white','axes.facecolor':'white',
    'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})

def draw(data,kind,order,labels,title,filename,xlim,xticks):
    fig=plt.figure(figsize=(16.5,10.4))
    gs=fig.add_gridspec(2,5,width_ratios=[1,.28,.23,1,.28],
                      left=.14,right=.985,top=.79,bottom=.18,wspace=.10,hspace=.47)
    for panel,h in enumerate(H):
        row,col=divmod(panel,2)
        ax=fig.add_subplot(gs[row,col*3]); counts=fig.add_subplot(gs[row,col*3+1])
        y=np.arange(4)[::-1]
        ax.set_xlim(*xlim);ax.set_ylim(-.55,3.65);ax.set_xticks(xticks)
        ax.set_yticks(y,labels)
        ax.set_title(f'{h}-hour forecast',loc='left',fontweight='bold',pad=12)
        ax.axvline(0,color='#465c6b',linewidth=1.25,zorder=1)
        ax.grid(axis='x',color='#e5eaee',linewidth=.8)
        ax.tick_params(axis='y',length=0,pad=10)
        ax.set_xlabel('RMSE improvement: M3 vs M1 (%)',labelpad=8)
        counts.set(xlim=(0,1),ylim=ax.get_ylim());counts.axis('off')
        counts.text(.04,3.60,'n (days)',fontsize=9.5,color='#536772',va='bottom')
        for yr,offset in [(2023,.135),(2024,-.135)]:
            part=data[data.year.eq(yr)&data.condition_type.eq(kind)&data.horizon.eq(h)].set_index('condition_value').loc[order]
            assert len(part)==4
            pos=y+offset
            low=part.improvement_ci_low_7d.to_numpy(float)
            high=part.improvement_ci_high_7d.to_numpy(float)
            point=part.improvement_pct.to_numpy(float)
            assert np.isfinite(np.r_[low,high,point]).all() and np.all(low<=high)
            ax.hlines(pos,low,high,color=COLORS[yr],linewidth=1.7,alpha=.9,zorder=2)
            ax.vlines(low,pos-.045,pos+.045,color=COLORS[yr],linewidth=1.25,zorder=2)
            ax.vlines(high,pos-.045,pos+.045,color=COLORS[yr],linewidth=1.25,zorder=2)
            ax.scatter(point,pos,color=COLORS[yr],s=48,marker=MARKERS[yr],
                       edgecolors='white',linewidth=.6,zorder=3)
            for yy,(_,r) in zip(pos,part.iterrows()):
                counts.text(.04,yy,f'{int(r.n):,} ({int(r.unique_origin_days)})',
                            color=COLORS[yr],fontsize=9.4,va='center')
    fig.suptitle('DCRNN · M3 vs M1 · seed 11\n'+title,
                 y=.985,fontsize=19,fontweight='bold')
    fig.text(.5,.873,'Positive improvement = lower RMSE after adding all China observations to the domestic baseline.',
             ha='center',fontsize=11,color='#415969')
    handles=[Line2D([0],[0],marker=MARKERS[yr],color=COLORS[yr],linewidth=1.8,
                    label=f'{yr} '+('validation' if yr==2023 else 'test'),markersize=6) for yr in (2023,2024)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),
               ncol=2,frameon=False,fontsize=11,columnspacing=3)
    fig.text(.5,.066,'Exploratory, single seed. Bars: paired 7-day calendar-block bootstrap 95% pointwise confidence intervals.',
             ha='center',fontsize=10.4,color='#415969')
    fig.text(.5,.039,'No comparison passes the default joint bootstrap + Holm-64 criterion (0 / 64). Pointwise intervals are not simultaneous intervals.',
             ha='center',fontsize=10.1,color='#415969')
    note=('n counts eligible 3-hour forecast origins; days counts distinct KST origin dates. These are predictive, not physical contribution estimates.'
          if kind=='season' else
          '† Local has only 29–66 origins on 24–38 days: treat large point estimates cautiously. Routes describe past positions, not countries or emission sources.')
    fig.text(.5,.012,note,ha='center',fontsize=9.5,color='#536772')
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{filename}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    data=pd.read_csv(WORK/'Tables/conditional_primary.csv')
    assert len(data)==64 and set(data.year)=={2023,2024}
    assert not data.exploratory_improved.any() and not data.exploratory_worsened.any()
    draw(data,'season',['DJF','MAM','JJA','SON'],
         ['Winter (DJF)','Spring (MAM)','Summer (JJA)','Autumn (SON)'],
         'Season-specific improvements at 12–48 hours',
         'dcrnn_h48_season_M3_vs_M1_forest',(-8.0,11.5),[-5,0,5,10])
    draw(data,'route_group',['west','north','east','local'],
         ['W / NW / SW','N / NE','East / SE / S','Local (<25 km) †'],
         'Past-trajectory route-specific improvements at 12–48 hours',
         'dcrnn_h48_route_M3_vs_M1_forest',(-18.0,20.0),[-15,-10,-5,0,5,10,15,20])

if __name__=='__main__':main()
