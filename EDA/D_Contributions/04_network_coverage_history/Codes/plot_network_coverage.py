"""Plot audited historical trajectory-to-station coverage, not forecast accuracy."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'Results'
net = pd.read_csv(OUT / 'network_coverage.csv').set_index('network')
ages = pd.read_csv(OUT / 'coverage_by_path_age.csv')
groups = pd.read_csv(OUT / 'new_candidate_coverage.csv').set_index('group')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), gridspec_kw={'width_ratios': [1, 1.2, 1.1]})
colors = ['#8492a6', '#347a9c', '#bb5727']
names = ['core3', 'old115', 'full317']
labels = ['Original 3', 'Previous 115', 'Expanded 317']
values = net.loc[names, 'observed_pct'].to_numpy()
axes[0].bar(labels, values, color=colors, width=.64)
for x, value in enumerate(values):
    axes[0].text(x, value + 1.2, f'{value:.2f}%', ha='center', fontweight='bold')
axes[0].set_ylim(0, 58)
axes[0].set_ylabel('Historical path points with valid PM2.5 (%)')
axes[0].set_title('Most coverage gained at 3 → 115', loc='left', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', labelsize=10)
axes[0].grid(axis='y', alpha=.17)
axes[0].set_axisbelow(True)

age_labels = ['01-06', '07-12', '13-24', '25-48']
x = np.arange(4)
for j, (name, label, color) in enumerate(zip(names, labels, colors)):
    series = ages.loc[ages.network.eq(name)].set_index('age_bin').loc[age_labels, 'observed_pct']
    axes[1].bar(x + (j-1)*.24, series, width=.23, color=color, label=label)
axes[1].set_xticks(x, ['1–6', '7–12', '13–24', '25–48'])
axes[1].set_xlabel('Hours BEFORE arrival t (historical path age)')
axes[1].set_ylabel('Valid PM2.5 point coverage (%)')
axes[1].set_ylim(0, 101)
axes[1].grid(axis='y', alpha=.17)
axes[1].set_axisbelow(True)
axes[1].legend(frameon=False, fontsize=9)
axes[1].set_title('Distant path sections remain sparse', loc='left', fontsize=12, fontweight='bold')

gnames = ['Japan188', 'CN_Russia_border7', 'CN_Mongolia_border6', 'Mongolia1']
glabels = ['Japan (188)', 'China / Russia-border (7)', 'China / Mongolia-border (6)', 'Ulaanbaatar (1)']
gv = groups.loc[gnames, 'added_observed_percentage_points'].to_numpy()
axes[2].barh(np.arange(4), gv, color=['#3967a2','#7a5a9e','#b38d39','#4b9175'], height=.6)
for y, value in enumerate(gv):
    axes[2].text(value + .015, y, f'+{value:.3f} pp', va='center', fontsize=10)
axes[2].set_yticks(np.arange(4), glabels, fontsize=9)
axes[2].invert_yaxis()
axes[2].set_xlim(0, 1.11)
axes[2].set_xlabel('Added coverage (percentage points)')
axes[2].grid(axis='x', alpha=.17)
axes[2].set_axisbelow(True)
axes[2].set_title('317 mainly fills uncommon corridors', loc='left', fontsize=12, fontweight='bold')

fig.suptitle('Observation-network expansion | Seoul arrivals, 2021–2024', x=.045, ha='left', fontsize=18, fontweight='bold', y=.98)
fig.text(.045, .89, 'Same 11,688 arrivals × 48 hourly historical positions = 561,024 points | 75 km radius | 500 m AGL arrival', fontsize=11, color='#465567')
fig.text(.045, .075, 'A point is counted once when at least one nearby candidate has valid PM2.5 at its historical passing time; no imputation. Junggu target excluded.', fontsize=9, color='#465567')
fig.text(.045, .035, 'Coverage is not forecast improvement or source attribution. Border candidates are Chinese city averages; no Russian observation station is included.', fontsize=9, color='#465567')
fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.78, wspace=.72)
fig.savefig(OUT / 'network_expansion_coverage.png', dpi=200, facecolor='white')
fig.savefig(OUT / 'network_expansion_coverage.svg', facecolor='white')
plt.close(fig)
