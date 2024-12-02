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
misid_filepath = '../DataForFigures/BFT_misid_cell_extended.csv'
fpkm_path = '../DataForFigures/cell_fpkm_data.tsv'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
genelist = ['Acly', 'Gpam','Hnf4a','Ube2z']

# Misid validation
GS_genes_data = pd.read_csv(GS_filepath)
merfish_data = pd.read_csv(merfish_filepath)
misid_data = pd.read_csv(misid_filepath)

results_dict_gene = {key: {} for key in genelist}  # initialize results dict
for key in results_dict_gene.keys():
    results_dict_gene[key] = {'TP': [], 'FP': [], 'FN': [],
                              'precision': [], 'recall': [], 'F1': []}

for bft in misid_data['bft']:
    for g in genelist:
        TP = len(GS_genes_data.loc[(GS_genes_data['gene'] == g) & (GS_genes_data['Label'] == 1) &
                                   (GS_genes_data['bfs'] <= bft)])
        FP = len(GS_genes_data.loc[(GS_genes_data['gene']  == g) & (GS_genes_data['Label'] == 0) &
                                   (GS_genes_data['bfs'] <= bft)])
        FN = len(GS_genes_data.loc[(GS_genes_data['gene']  == g) & (GS_genes_data['Label'] == 1) &
                                   (GS_genes_data['bfs'] > bft)])
        try:
            precision = TP / (TP+FP)
            recall = TP / (TP+FN)
            F1 = 2 * precision * recall / (precision + recall)
            results_dict_gene[g]['TP'].append(TP)
            results_dict_gene[g]['FP'].append(FP)
            results_dict_gene[g]['FN'].append(FN)
        except:
            results_dict_gene[g]['TP'].append(np.nan)
            results_dict_gene[g]['FP'].append(np.nan)
            results_dict_gene[g]['FN'].append(np.nan)

        results_dict_gene[g]['precision'].append(precision)
        results_dict_gene[g]['recall'].append(recall)
        results_dict_gene[g]['F1'].append(F1)

BFT_results_bygene = pd.concat({k: pd.DataFrame(v) for k, v in results_dict_gene.items()}, axis=1)
BFT_results_bygene.to_csv(os.path.join(output_dir,'cell_extendedmisid_bygene_metrics.csv'))

BFT_results_bygene['misid'] = misid_data['misid']

fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,6))
axes = axes.ravel()
for i,g in enumerate(genelist):
    axes[i].plot(BFT_results_bygene['misid'], BFT_results_bygene[g,'precision'], '-o', markersize=4, label='precision',c='royalblue')
    axes[i].plot(BFT_results_bygene['misid'], BFT_results_bygene[g,'recall'], '-o', markersize=4, label='recall',c='tomato')
    axes[i].plot(BFT_results_bygene['misid'], BFT_results_bygene[g,'F1'], '-o', markersize=4, label='F1',c='dimgrey')
    axes[i].plot(BFT_results_bygene['misid'].iloc[BFT_results_bygene[g,'F1'].idxmax()],BFT_results_bygene[g,'F1'].iloc[BFT_results_bygene[g,'F1'].idxmax()],'*',c='yellow',markersize=8)
    axes[i].set(xlabel=None,ylabel=None)
    axes[i].set_title(g)
    # axes[i].set_yticks([0.4,0.6,0.8,1.0],fontsize=12)
    axes[i].plot([0, BFT_results_bygene['misid'].max()], [0,BFT_results_bygene['misid'].max()], ls='--', c='darkgrey')
    axes[i].set_xlim(-0.05,BFT_results_bygene['misid'].max()+0.05)
    axes[i].set_ylim(0.4,1.05)
    axes[i].tick_params(labelbottom=True, labelleft=True)
    axes[i].legend().remove()
   
fig.supylabel('precision / Recall / F1')
fig.supxlabel('Misidentification rate')
fig.legend(labels = ['precision','recall','F1'], loc='lower center',ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(output_dir,'cell_bygenemetricsplot_scaled.svg'))