import pandas as pd
import numpy as np
import os
from FPKM_functions import *

# LIVER
GS_filepath = '../GuidestarGenes_training.csv'
merfish_filepath = '../_MerfishData.csv'
fpkm_path = '../'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
output_dir = '../'

GS_genes_data = pd.read_csv(GS_filepath)
merfish_data = pd.read_csv(merfish_filepath)

fpkm_result = []
misid_result = []
counts_result = []
bft_list = []

for bft in np.linspace(0.03,1,200, endpoint=True):
    print('bft', bft)
    bft_list.append(bft)
    merfish_data['bft_result'] = 0
    merfish_data.loc[merfish_data['bfs'] <= bft, 'bft_result'] = 1
    
    model_callouts = list(merfish_data['gene'].loc[merfish_data['bft_result']==1])
    counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, model_callouts)
    fpkm_val, _ = calcLogCorrelation(np.array(counts_arr), np.array(fpkm_list))
    mis_id_rate = calc_misid(counts_arr, model_callouts)
    print('merfish FPKM:', fpkm_val)
    print('merfish counts:', np.sum(counts_arr))
    print('merfish misid:',mis_id_rate)
    
    fpkm_result.append(fpkm_val)
    misid_result.append(mis_id_rate)
    counts_result.append(np.sum(counts_arr))
    
result_df = pd.DataFrame({'bft':bft_list, 'fpkm': fpkm_result, 'counts': counts_result, 'misid': misid_result})
result_df.to_csv(os.path.join(output_dir, 'BFT_misid_Liver_extended.csv'))