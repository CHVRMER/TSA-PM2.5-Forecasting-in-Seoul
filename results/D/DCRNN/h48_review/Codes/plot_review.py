"""Shareable plots from the frozen, already-scored DCRNN H48 result tables.

The original 84-test Holm family is never recalculated or reduced here.
Source result files and model inputs are not modified.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
OUT = WORK / 'Figures'
OUT.mkdir(exist_ok=True)
H = [3, 6, 12, 24, 36, 48]
ORDER = ['M1', 'M2', 'M3', 'M4', 'M3T', 'M4T']
COLORS = {'M1':'#1b3b5a', 'M2':'#d48225', 'M3':'#067a75',
          'M4':'#8c4c99', 'M3T':'#3b7eb9', 'M4T':'#c24e5b'}
MARKERS = {'M1':'o', 'M2':'s', 'M3':'^', 'M4':'D', 'M3T':'P', 'M4T':'X'}
LABELS = {'M1':'M1: domestic', 'M2':'M2: + nearby China',
          'M3':'M3: + all China', 'M4':'M4: + Japan / Mongolia',
          'M3T':'M3T: M3 + trajectory features',
          'M4T':'M4T: M4 + trajectory features'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.titlesize':13, 'axes.labelsize':11, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.edgecolor':'#83929b',
    'axes.titleweight':'semibold', 'figure.facecolor':'white',
    'axes.facecolor':'white', 'savefig.facecolor':'white',
    'pdf.fonttype':42, 'ps.fonttype':42})


def grid(ax):
    ax.grid(axis='y', color='#e1e7eb', linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_xticks(H)
    ax.set_xlabel('Forecast horizon (hours)')


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    metrics = pd.read_csv(WORK / 'Tables/test_condition_metrics.csv')
    base = pd.read_csv(WORK / 'Tables/test_baseline_metrics.csv')
    comp = pd.read_csv(WORK / 'Tables/test_predeclared_condition_comparisons.csv')
    assert set(metrics.input_condition) == set(ORDER)
    assert len(comp) == 42
    assert comp.improved.eq((comp.ci95_high < 0) & (comp.p_holm < .05)).all()
    significant = comp.improved.to_numpy(bool)
    print(f'Original Holm-84 input contrasts passing improvement criterion: {significant.sum()}/42')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    for condition in ORDER:
        part = metrics[metrics.input_condition.eq(condition)].set_index('horizon').loc[H]
        for ax in axes:
            ax.plot(H, part.RMSE, color=COLORS[condition], marker=MARKERS[condition],
                    linewidth=2, markersize=6, label=LABELS[condition],
                    linestyle='--' if condition.endswith('T') else '-')
    persistence = base[base.model.eq('persistence')].set_index('horizon').loc[H]
    axes[0].plot(H, persistence.RMSE, color='#737e84', marker='v',
                 linestyle=':', linewidth=2.4, label='Persistence')
    axes[0].set(title='A  |  Models and persistence', ylabel='RMSE (µg/m³)', ylim=(4.9, 17.3))
    axes[1].set(title='B  |  Six input conditions (expanded scale)', ylabel='RMSE (µg/m³)', ylim=(5.1, 12.9))
    for ax in axes:
        grid(ax)
    fig.suptitle('DCRNN · 2024 test · seed 11\nForecast error across 3–48-hour horizons',
                 fontsize=17, fontweight='bold', y=.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5,.06), fontsize=9.5)
    fig.text(.5,.012, 'Same fixed evaluation rows across conditions at each horizon. '
             'Lines are descriptive; the plot does not establish differences between conditions.',
             ha='center', fontsize=9, color='#536772')
    fig.subplots_adjust(top=.79, bottom=.29, left=.07, right=.985, wspace=.19)
    save(fig, 'dcrnn_h48_2024_rmse_horizon_curves')

    pairs = [('M2','M1'), ('M3','M2'), ('M3','M1'), ('M4','M3'),
             ('M4','M1'), ('M3T','M3'), ('M4T','M4')]
    matrix = np.empty((7,6)); stars = np.zeros((7,6),bool)
    for i,(candidate,reference) in enumerate(pairs):
        part = comp[comp.candidate_condition.eq(candidate) & comp.reference_condition.eq(reference)].set_index('horizon').loc[H]
        matrix[i] = part.improvement_pct
        stars[i] = (part.ci95_high < 0) & (part.p_holm < .05)
    lim = max(5.0, float(np.ceil(np.max(np.abs(matrix)))))
    fig, ax = plt.subplots(figsize=(11.7,7.4))
    im=ax.imshow(matrix, cmap='RdBu', norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim), aspect='auto')
    for i in range(7):
        for j in range(6):
            suffix=' *' if stars[i,j] else ''
            ax.text(j,i,f'{matrix[i,j]:+.2f}%{suffix}',ha='center',va='center',
                    color='white' if abs(matrix[i,j])>lim*.62 else '#172c3b',fontsize=11)
    ax.set_xticks(range(6),[f'{h} h' for h in H])
    ax.set_yticks(range(7),[f'{a} vs {b}' for a,b in pairs])
    ax.set_xlabel('Forecast horizon')
    ax.set_xticks(np.arange(-.5,6,1),minor=True)
    ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2)
    ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.92)
    cb.set_label('RMSE improvement relative to reference (%)\nPositive = lower RMSE; negative = higher RMSE')
    fig.suptitle('DCRNN · 2024 test · seed 11\nPredeclared input-condition comparisons',
                 fontsize=17,fontweight='bold',y=.98)
    ax.set_title('No pair passes the original Holm-84 + bootstrap improvement criterion (0 / 42).',
                 fontsize=11,pad=16,color='#52636e')
    fig.text(.075,.084,'Cell = 100 × (RMSE_reference − RMSE_candidate) / RMSE_reference.',fontsize=10,color='#415966')
    fig.text(.075,.052,'* requires both bootstrap 95% CI upper bound < 0 and original Holm-adjusted p < 0.05.',fontsize=10,color='#415966')
    fig.text(.075,.02,'The original 84-test family is unchanged. Predictive error differences are not pollution-source contribution rates.',fontsize=9.2,color='#536772')
    fig.subplots_adjust(top=.80,bottom=.17,left=.13,right=.91)
    save(fig,'dcrnn_h48_2024_predeclared_improvement_heatmap')

    m1=metrics[metrics.input_condition.eq('M1')].set_index('horizon').loc[H]
    fig, axes=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[1,1.15]})
    ax=axes[0]
    gain=ax.plot(H,m1.RMSE_improvement_pct,color='#067a75',marker='o',linewidth=2.4,label='M1: RMSE gain over persistence')
    ax.set(ylabel='RMSE improvement over persistence (%)',ylim=(12,32),title='A  |  Relative gain and explained variation')
    twin=ax.twinx()
    skill=twin.plot(H,m1.R2,color='#9a4d29',marker='s',linewidth=2.4,linestyle='--',label='M1: R²')
    twin.set_ylabel('R²',color='#9a4d29'); twin.set_ylim(0,1)
    twin.spines['right'].set_visible(True)
    twin.spines['right'].set_color('#9a4d29')
    twin.tick_params(axis='y',colors='#9a4d29')
    for h in (3,24,48):
        ax.annotate(f'{m1.loc[h,"RMSE_improvement_pct"]:.1f}%',(h,m1.loc[h,'RMSE_improvement_pct']),
                    xytext=(0,9),textcoords='offset points',ha='center',color='#067a75',fontsize=9.5)
        twin.annotate(f'{m1.loc[h,"R2"]:.2f}',(h,m1.loc[h,'R2']),xytext=(0,-17),
                      textcoords='offset points',ha='center',color='#9a4d29',fontsize=9.5)
    grid(ax)
    ax.legend(gain+skill,[x.get_label() for x in gain+skill],loc='lower left',frameon=False,fontsize=9.2)
    for condition in ORDER:
        part=metrics[metrics.input_condition.eq(condition)].set_index('horizon').loc[H]
        axes[1].plot(H,part.MBE,color=COLORS[condition],marker=MARKERS[condition],linewidth=2,
                     linestyle='--' if condition.endswith('T') else '-',label=condition)
    axes[1].axhline(0,color='#62717b',linewidth=1.1)
    axes[1].set(title='B  |  Mean bias across input conditions',ylabel='MBE = prediction − observation (µg/m³)',ylim=(-3.9,1.7))
    grid(axes[1])
    axes[1].text(.015,.025,'Below zero = average underprediction',transform=axes[1].transAxes,fontsize=9,color='#536772')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.31),ncol=6,frameon=False,fontsize=10,columnspacing=1.2)
    fig.suptitle('DCRNN · 2024 test · seed 11\nLong-horizon gains require careful interpretation',fontsize=17,fontweight='bold',y=.99)
    fig.text(.5,.068,'M1: RMSE rises from 5.59 (3 h) to 11.89 (48 h), even as its percentage gain over persistence grows.',
             ha='center',fontsize=10,color='#415966')
    fig.text(.5,.03,'All values describe forecast errors, not physical transport or national pollution contributions.',
             ha='center',fontsize=9.5,color='#536772')
    fig.subplots_adjust(top=.79,bottom=.27,left=.07,right=.98,wspace=.40)
    save(fig,'dcrnn_h48_2024_long_horizon_skill_bias')


if __name__=='__main__':
    main()
