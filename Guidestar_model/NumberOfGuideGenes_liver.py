"""
Find what combination of guide genes achieves best F1 score using RF
"""

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from itertools import combinations

GS_filepath = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/tissue/colocaldist4_20240723_try2/Liver_LM39_49FOVs_GuidestarGenes_training.csv'
GS_genes_data = pd.read_csv(GS_filepath)
output_dir = '/Volumes/Seagate Backup Plus Drive/1 year RA data/20240723/'
output_name = 'NumGenes_liver.csv'
GS_gene_list = ['Acly','Hnf4a','Gpam','Pigr']
blank_ratio = 0
merfish_ratio = 1
blank_no = 7
blank_list = ['Blank' + str(i) for i in range(1,blank_no+1)]
GS_genes_data = GS_genes_data.loc[GS_genes_data['gene'].isin(GS_gene_list+blank_list)]
cv_fold = 5
features_to_use = ['meaninten1_frob','mindist1', 'size'] # conventional features for optimization

print('training file used:', GS_filepath)

# add types to allow stratification
type_list = []
for row in range(len(GS_genes_data)):
    if GS_genes_data['Label'].iloc[row] == 0:
        if GS_genes_data['gene'].iloc[row] in blank_list:
            type_list.append('blank')
        elif GS_genes_data['gene'].iloc[row] in GS_gene_list:
            type_list.append('merfishonly')
    elif GS_genes_data['Label'].iloc[row] == 1:
        type_list.append('colocalgene')
        
GS_genes_data['type'] = type_list

# test set
X_test = GS_genes_data[features_to_use + ['type', 'Label']].loc[GS_genes_data['fov'].isin([i for i in range(0,9)])]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin([i for i in range(0,9)])]
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin([i for i in range(9,49)])]
print('test fovs:', [i for i in range(0,9)])

pd.options.mode.chained_assignment = None
precision_list = []
recall_list = []
f1_list = []
comb_list = []
combid_list = []
numgenes_list = []

random.seed(0)
custom_cv = []
training_fovs = [i for i in range(49) if i not in range(0,9)]
cv_fovs = [i for i in range(49) if i not in range(0,9)]
for iter in range(cv_fold):
    val_index = random.sample(cv_fovs, 8)
    for fov in val_index:
        cv_fovs.remove(fov)
    train_index = [i for i in training_fovs if i not in val_index]
    custom_cv.append((train_index,val_index))

for num_gene in range(1,len(GS_gene_list)+1):
    
    combs = list(combinations(GS_gene_list, num_gene))
    for id, i in enumerate(combs):
        GS_genes_totrain = list(i)
        
        # resample training data
        resampled_GS_training = pd.DataFrame()
        for train_fov in training_fovs:
            print('resampling fov', train_fov)
            fov_subset = GS_training.loc[GS_training['fov']== train_fov]

            GS_df = fov_subset.loc[(fov_subset['type']=='colocalgene') & (fov_subset['gene'].isin(GS_genes_totrain))]
            merfish_df = fov_subset.loc[(fov_subset['type']=='merfishonly') & (fov_subset['gene'].isin(GS_genes_totrain))]
            blank_df = fov_subset.loc[fov_subset['type']=='blank']

            print('Number of blanks (0)s: ', len(blank_df))
            print('Number of merfishonly (0)s: ', len(merfish_df))
            print('Number of colocalized genes (1)s: ', len(GS_df))

            if blank_ratio is not None and len(blank_df) > int(len(merfish_df)*blank_ratio):
                blank_df_resampled =  resample(blank_df, replace = False, n_samples = int(len(merfish_df)*blank_ratio), random_state=0)
            else:
                blank_df_resampled = blank_df
                
            if merfish_ratio is not None and len(merfish_df) > int(len(GS_df)*merfish_ratio):
                merfish_df_resampled =  resample(merfish_df, replace = False, n_samples = int(len(GS_df)*merfish_ratio), random_state=0)
            else:
                merfish_df_resampled = merfish_df
                
            print('Number of blanks (0)s: ', len(blank_df_resampled))
            print('Number of merfishonly (0)s: ', len(merfish_df_resampled))
            print('Number of colocalized genes (1)s: ', len(GS_df))
            
            resampled_fov = pd.concat([GS_df, blank_df_resampled, merfish_df_resampled])
            resampled_GS_training = pd.concat([resampled_GS_training,resampled_fov])
            
        resampled_GS_training = resampled_GS_training.reset_index()
            
        for fold in range(cv_fold):
            train_df = resampled_GS_training.loc[resampled_GS_training['fov'].isin(custom_cv[fold][0])]
            val_df = GS_training.loc[(GS_training['fov'].isin(custom_cv[fold][1])) & (GS_training['type'].isin(['colocalgene','merfishonly']))]

            if 'blank' in np.unique(val_df['type']):
                print('validation has blanks!')
        
            RF = RandomForestClassifier(random_state=0, verbose=0,n_jobs=-3, min_samples_split=0.001, min_impurity_decrease=0.01, max_features=1,
                                     max_depth=12,criterion='entropy',min_weight_fraction_leaf=0.175,n_estimators=300)
            RF.fit(train_df[features_to_use], train_df['Label'])
            val_pred = RF.predict(val_df[features_to_use])
            
            precision = precision_score(val_df['Label'], val_pred)
            recall = recall_score(val_df['Label'], val_pred)
            f1 = f1_score(val_df['Label'], val_pred)
            print('f1',f1)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            comb_list.append(i)
            combid_list.append(id)
            numgenes_list.append(num_gene)
        
        
results_df = pd.DataFrame({'combination': comb_list, 'combid': combid_list,'number': numgenes_list, 'precision': precision_list, 'recall':recall_list, 'F1':f1_list})
results_df.to_csv(os.path.join(output_dir,output_name))