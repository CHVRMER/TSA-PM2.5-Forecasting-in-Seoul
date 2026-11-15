"""Plot descriptive national-data QC; no forecast targets or model scores."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'Results'
data=pd.read_csv(OUT/'region_year_quality.csv')
regions=['Capital','Gangwon','Chungcheong','Honam','Yeongnam','Jeju']
counts=data.pivot(index='region_group',columns='year_kst',values='stations').loc[regions]
missing=data.pivot(index='region_group',columns='year_kst',values='missing_fraction_inside_record_spans').loc[regions]*100
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.8,5.6),gridspec_kw={'width_ratios':[1,1.1]})
colors=['#4b668b','#54a3a6','#b08351','#9f6388','#66865e','#7d79ae']
for color,region in zip(colors,regions):
    axes[0].plot(counts.columns,counts.loc[region],'-o',color=color,label=region,lw=2,ms=5)
axes[0].set_xticks([2021,2022,2023,2024]);axes[0].set_ylabel('Stations with source rows in each year')
axes[0].grid(axis='y',alpha=.2);axes[0].set_title('Recorded station coverage',loc='left',fontweight='bold')
axes[0].legend(ncol=2,frameon=False,fontsize=9,loc='lower right')
im=axes[1].imshow(missing,cmap='YlOrRd',vmin=0,vmax=12,aspect='auto')
axes[1].set_yticks(range(len(regions)),regions);axes[1].set_xticks(range(4),[2021,2022,2023,2024])
axes[1].set_title('Missing PM2.5 inside source-record spans',loc='left',fontweight='bold')
for i in range(6):
    for j in range(4):axes[1].text(j,i,f'{missing.iloc[i,j]:.1f}%',ha='center',va='center',color='white' if missing.iloc[i,j]>8.5 else '#263241')
fig.colorbar(im,ax=axes[1],label='Missing (%)',fraction=.045,pad=.03)
fig.suptitle('National PM2.5 data quality | urban, suburban and national-background networks',x=.06,ha='left',fontsize=15,fontweight='bold',y=.97)
fig.text(.06,.045,'Sparse original records; no imputation. Record spans are not independently confirmed commissioning/closure periods.',fontsize=9,color='#536174')
fig.subplots_adjust(left=.08,right=.97,bottom=.17,top=.80,wspace=.43)
fig.savefig(OUT/'national_quality_overview.png',dpi=200,facecolor='white',bbox_inches='tight')
plt.close(fig)
