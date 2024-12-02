### TO CLEAN

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
colocalization_counts_filepath = '../DataForFigures/Cell_LM39_25FOVs_Colocalisation_counts.csv'
genelist = ['Acly', 'Gpam','Hnf4a','Ube2z']
distopt_filepath = '../DataForFigures/Cell_LM39_25FOVs_distopt_results.csv'

distopt_df = pd.read_csv(distopt_filepath)
colocalization_counts_df = pd.read_csv(colocalization_counts_filepath)

fig,axes = plt.subplots(2,2,figsize=(16,10))
axes = axes.ravel()

for i,merfish_gene in enumerate(genelist):
    temp_df = distopt_df.loc[distopt_df['merfish gene'] == merfish_gene]
    total_merfish_counts = colocalization_counts_df[['1','0']].loc[(colocalization_counts_df['gene']==merfish_gene) & (colocalization_counts_df['type']=='Merfish')].sum(axis=1).values[0]
    temp_df['colocal_percent'] = (temp_df.loc[:, 'colocalisation counts'] / total_merfish_counts)*100
    sns.scatterplot(data=temp_df, x = 'colocal distance',y ='colocal_percent',hue='guidestar gene', ax=axes[i])

    axes[i].set_title(merfish_gene)
    axes[i].set_yticks(np.arange(0, 100+1, 20))
    if i < 3:
        axes[i].legend([])

fig.suptitle('Percentge MERFISH callouts colocalized with Guidestar spots')
fig.supylabel('% colocalized')
fig.supxlabel('distance')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'cell_distance_optimization.svg'))

### LIVER ###
colocalization_counts_filepath = '../DataForFigures/Liver_LM39_49FOVs_Colocalisation_counts.csv'
genelist = ['Acly','Gpam','Hnf4a','Pigr']
distopt_filepath = '../DataForFigures/Liver_LM39_49FOVs_distopt_results.csv'

distopt_df = pd.read_csv(distopt_filepath)
colocalization_counts_df = pd.read_csv(colocalization_counts_filepath)

fig,axes = plt.subplots(2,2,figsize=(16,10))
axes = axes.ravel()

for i,merfish_gene in enumerate(genelist):
    temp_df = distopt_df.loc[distopt_df['merfish gene'] == merfish_gene]
    total_merfish_counts = colocalization_counts_df[['1','0']].loc[(colocalization_counts_df['gene']==merfish_gene) & (colocalization_counts_df['type']=='Merfish')].sum(axis=1).values[0]
    temp_df['colocal_percent'] = (temp_df.loc[:, 'colocalisation counts'] / total_merfish_counts)*100
    sns.scatterplot(data=temp_df, x = 'colocal distance',y ='colocal_percent',hue='guidestar gene', ax=axes[i])

    axes[i].set_title(merfish_gene)
    axes[i].set_yticks(np.arange(0, 100+1, 20))
    if i < 3:
        axes[i].legend([])

fig.suptitle('Percentge MERFISH callouts colocalized with Guidestar spots')
fig.supylabel('% colocalized')
fig.supxlabel('distance')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'liver_distance_optimization.svg'))
