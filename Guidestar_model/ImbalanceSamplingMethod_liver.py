"""
Check upsampling vs downsampling for class imbalance
        
### liver
1. No sampling
        blank_ratio = 0, merfish_ratio = None
2. Upsampling of colocalised to match MERFISH only, no sampling of MERFISH only. Where 1s < 0s
        blank_ratio = 0, merfish_ratio = None, colocal_ratio = 1
3. Downsampling of MERFISH only to match colocalised, no sampling of colocalised. Where 1s < 0s
        blank_ratio = 0, merfish_ratio = 1, colocal_ratio = None

"""
import time
import itertools
import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from DiscretisationFunctions import *

GS_filepath = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/tissue/colocaldist4_20240723_try2/Liver_LM39_49FOVs_GuidestarGenes_training.csv'
GS_genes_data = pd.read_csv(GS_filepath)
output_dir = '/Volumes/Seagate Backup Plus Drive/1 year RA data/20240723/'

# LIVER
model_type = 'RF'
GS_gene_list = ['Acly','Hnf4a','Pigr','Gpam']
blank_no = 7
blank_list = ['Blank' + str(i) for i in range(1,blank_no+1)]
cv_fold = 5
features_to_use = ['meaninten1_frob', 'mindist1', 'size'] # use conventional features for optimizing sampling method
bft = 0.21
nfovs = 49
output_name = 'Imbalance_sampling_liver_conv3feat_v3.csv'
sampling_dict = {'opt1':{'blank_ratio':0, 'merfish_ratio':None, 'colocal_ratio':None},
                 'opt2':{'blank_ratio':0, 'merfish_ratio':None, 'colocal_ratio':1},
                 'opt3':{'blank_ratio':0, 'merfish_ratio':1, 'colocal_ratio':None}}

print('training file used:', GS_filepath)
GS_genes_data = GS_genes_data.loc[GS_genes_data['gene'].isin(GS_gene_list+blank_list)]

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

random.seed(0)
custom_cv = []
training_fovs = [i for i in range(49) if i not in range(0,9)]
cv_fovs = [i for i in range(49) if i not in range(0,9)]
random.seed(0)
for iter in range(cv_fold):
    val_index = random.sample(cv_fovs, 8)
    for fov in val_index:
        cv_fovs.remove(fov)
    train_index = [i for i in training_fovs if i not in val_index]
    custom_cv.append((train_index,val_index))

# resample training data
result_sampling_method = []
result_precision = []
result_recall = []
result_f1 = []
result_auroc = []

for opt in sampling_dict.keys():
    print('sampling method ', opt)
    blank_ratio = sampling_dict[opt]['blank_ratio']
    merfish_ratio = sampling_dict[opt]['merfish_ratio']
    colocal_ratio = sampling_dict[opt]['colocal_ratio']
        
    resampled_GS_training = pd.DataFrame()
    for train_fov in training_fovs:
        print('resampling fov', train_fov)
        fov_subset = GS_training.loc[GS_training['fov']== train_fov]

        GS_df = fov_subset.loc[fov_subset['type']=='colocalgene']
        merfish_df = fov_subset.loc[fov_subset['type']=='merfishonly']
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
        elif merfish_ratio is not None and len(merfish_df) < int(len(GS_df)*merfish_ratio):
            merfish_df_resampled =  resample(merfish_df, replace = True, n_samples = int(len(GS_df)*merfish_ratio), random_state=0)
        else:
            merfish_df_resampled = merfish_df
            
        if colocal_ratio is not None and len(GS_df) > int(len(merfish_df_resampled)*colocal_ratio):
            GS_df_resampled =  resample(GS_df, replace = False, n_samples = int(len(merfish_df_resampled)*colocal_ratio), random_state=0)
        elif colocal_ratio is not None and len(GS_df) < int(len(merfish_df_resampled)*colocal_ratio):
            GS_df_resampled =  resample(GS_df, replace = True, n_samples = int(len(merfish_df_resampled)*colocal_ratio), random_state=0)
        else:
            GS_df_resampled = GS_df
            
        print('Number of blanks (0)s: ', len(blank_df_resampled))
        print('Number of merfishonly (0)s: ', len(merfish_df_resampled))
        print('Number of colocalized genes (1)s: ', len(GS_df_resampled))
        
        resampled_fov = pd.concat([GS_df_resampled, blank_df_resampled, merfish_df_resampled])
        resampled_GS_training = pd.concat([resampled_GS_training,resampled_fov])

    for fold in range(cv_fold):
        train_df = resampled_GS_training.loc[resampled_GS_training['fov'].isin(custom_cv[fold][0])]
        val_df = GS_training.loc[(GS_training['fov'].isin(custom_cv[fold][1])) & (GS_training['type'].isin(['colocalgene','merfishonly']))]

        if 'blank' in np.unique(val_df['type']): # make sure no blanks in validation set
            print('validation has blanks!')
    
        # basic model parameters
        RF = RandomForestClassifier(random_state=0, verbose=0,n_jobs=-3, min_samples_split=0.001, min_impurity_decrease=0.01, max_features=1,
                                    max_depth=12,criterion='entropy',min_weight_fraction_leaf=0.175,n_estimators=300)
        RF.fit(train_df[features_to_use], train_df['Label'])
        val_pred = RF.predict(val_df[features_to_use])
        
        precision = precision_score(val_df['Label'], val_pred)
        recall = recall_score(val_df['Label'], val_pred)
        f1 = f1_score(val_df['Label'], val_pred)
        print('f1',f1)

        result_precision.append(precision)
        result_recall.append(recall)
        result_f1.append(f1)
        result_sampling_method.append(opt)

result_df = pd.DataFrame({'method': result_sampling_method, 
                          'precision': result_precision, 
                          'recall': result_recall, 
                          'f1': result_f1})

result_df.to_csv(os.path.join(output_dir,output_name))