import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import sys
from utilsN.FPKM_functions import *

output_dir = '/path_to_output_folder/'

### CELL ###
GS_filepath = 'Cell_LM39_25FOVs_GuidestarGenes_training_extendedmisid.csv'
merfish_filepath = 'Cell_LM39_25FOVs_MerfishData_extendedmisid.csv'
fpkm_path = '../DataForFigures/cell_fpkm_data.tsv'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
colocalization_counts_filepath = '../DataForFigures/Cell_LM39_25FOVs_Colocalisation_counts.csv'
genelist = ['Acly', 'Gpam','Hnf4a','Ube2z']
negative_control_filepath = '../DataForFigures/Cell_LM39_25FOVs_negativecontrolresults.csv'

# Fig 1d
colocalization_counts = pd.read_csv(colocalization_counts_filepath)

GS_colocal = colocalization_counts.loc[(colocalization_counts['type']=='Merfish') & (colocalization_counts['gene'].isin(genelist))]
GS_colocal['sum'] = GS_colocal['1'] + GS_colocal['0']
GS_colocal['colocal_percent'] = GS_colocal['1'] / GS_colocal['sum'] * 100

fig,ax=plt.subplots(figsize=(4,5))
ax.bar(GS_colocal['gene'], GS_colocal['colocal_percent'])
ax.set_ylim(0,100)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'cell_colocalisation_MERFISH.svg'))

# Colocalization percentage and negative control for MERFISH
colocalization_counts = pd.read_csv(colocalization_counts_filepath)
neg_control = pd.read_csv(negative_control_filepath)

neg_control = pd.read_csv(negative_control_filepath)

colocal_values = neg_control['colocalisation counts'].tolist()
total_values = []
genelist_temp = []

for i,g in enumerate(genelist):
    genelist_temp.append(g)
    data_subset = colocalization_counts.loc[(colocalization_counts['gene']==g)&(colocalization_counts['type']=='Merfish')]
    total_values.append(data_subset[['0','1']].values[0].sum())

total_df = pd.DataFrame({'merfish gene': genelist_temp, 'total': total_values})

neg_control['total'] = [0] * len(neg_control)
for row in range(len(neg_control)):
    merfish_gene = neg_control['merfish gene'].iloc[row]
    neg_control['total'].iloc[row] = total_df['total'].loc[total_df['merfish gene'] == merfish_gene]
neg_control['colocal_percent'] = neg_control['colocalisation counts']/neg_control['total']
groups = [neg_control['merfish gene'].iloc[i] + '_' + neg_control['guidestar gene'].iloc[i] for i in range(len(neg_control))]

fig,ax=plt.subplots(figsize=(4,5))
ax.barh(groups,neg_control['colocal_percent'])
ax.barh(groups,1-neg_control['colocal_percent'],left=neg_control['colocal_percent'])

fig.legend(title = 'Colocalized:',labels = ['Yes','No'], loc='upper center',ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'cell_negcontrol_bar_horizontal.svg'))

# Colocalization percentage and negative control for Guidestar
colocalization_counts = pd.read_csv(colocalization_counts_filepath)

GS_colocal = colocalization_counts.loc[(colocalization_counts['type']=='Guidestar') & (colocalization_counts['gene'].isin(genelist))]
GS_colocal['sum'] = GS_colocal['1'] + GS_colocal['0']
GS_colocal['colocal_percent'] = GS_colocal['1'] / GS_colocal['sum'] * 100

fig,ax=plt.subplots(figsize=(4,5))
ax.bar(GS_colocal['gene'], GS_colocal['colocal_percent'])
ax.set_ylim(0,60)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'cell_negcontrol_bar_Guidestar_horizontal.svg'))

### LIVER ###

GS_filepath = 'Liver_LM39_49FOVs_GuidestarGenes_training_extendedmisid.csv'
merfish_filepath = 'Liver_LM39_49FOVs_MerfishData_extendedmisid.csv'
fpkm_path = '../DataForFigures/fpkm_data.tsv'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
colocalization_counts_filepath = '../DataForFigures//Liver_LM39_49FOVs_Colocalisation_counts.csv'
genelist = ['Hnf4a','Pigr','Gpam','Acly']
negative_control_filepath = '../DataForFigures/Liver_LM39_49FOVs_negativecontrolresults.csv'

# Colocalization percentage and negative control for MERFISH
colocalization_counts = pd.read_csv(colocalization_counts_filepath)
neg_control = pd.read_csv(negative_control_filepath)

neg_control = pd.read_csv(negative_control_filepath)

colocal_values = neg_control['colocalisation counts'].tolist()
total_values = []
genelist_temp = []

for i,g in enumerate(genelist):
    genelist_temp.append(g)
    data_subset = colocalization_counts.loc[(colocalization_counts['gene']==g)&(colocalization_counts['type']=='Merfish')]
    total_values.append(data_subset[['0','1']].values[0].sum())

total_df = pd.DataFrame({'merfish gene': genelist_temp, 'total': total_values})

neg_control['total'] = [0] * len(neg_control)
for row in range(len(neg_control)):
    merfish_gene = neg_control['merfish gene'].iloc[row]
    neg_control['total'].iloc[row] = total_df['total'].loc[total_df['merfish gene'] == merfish_gene]
neg_control['colocal_percent'] = neg_control['colocalisation counts']/neg_control['total']
groups = [neg_control['merfish gene'].iloc[i] + '_' + neg_control['guidestar gene'].iloc[i] for i in range(len(neg_control))]

fig,ax=plt.subplots(figsize=(4,5))
ax.barh(groups,neg_control['colocal_percent'])
ax.barh(groups,1-neg_control['colocal_percent'],left=neg_control['colocal_percent'])

fig.legend(title = 'Colocalized:',labels = ['Yes','No'], loc='upper center',ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'tissue_negcontrol_bar_horizontal_v2.svg'))

# Colocalization percentage and negative control for Guidestar
colocalization_counts = pd.read_csv(colocalization_counts_filepath)

GS_colocal = colocalization_counts.loc[(colocalization_counts['type']=='Guidestar') & (colocalization_counts['gene'].isin(genelist))]
GS_colocal['sum'] = GS_colocal['1'] + GS_colocal['0']
GS_colocal['colocal_percent'] = GS_colocal['1'] / GS_colocal['sum'] * 100

fig,ax=plt.subplots(figsize=(4,5))
ax.bar(GS_colocal['gene'], GS_colocal['colocal_percent'])
ax.set_ylim(0,30)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'tissue_negcontrol_bar_Guidestar_horizontal.svg'))
