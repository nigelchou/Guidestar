### Feature selection


import time
import itertools
import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from DiscretisationFunctions import *
import sys
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

GS_filepath = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/cells/colocaldist4_20240610/Cell_LM39_25FOVs_GuidestarGenes_training.csv'
merfish_filepath = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/cells/colocaldist4_20240610/Cell_LM39_25FOVs_MerfishData.csv'
fpkm_path = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/cells/100uM_data/codebook/fpkm_data.tsv'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])

GS_genes_data = pd.read_csv(GS_filepath)
merfish_data = pd.read_csv(merfish_filepath)
output_dir = '/Volumes/Seagate Backup Plus Drive/1 year RA data/20240610/'

# CELL
GS_gene_list = ['Acly','Hnf4a','Ube2z','Gpam']
blank_no = 7
blank_list = ['Blank' + str(i) for i in range(1,blank_no+1)]
cv_fold = 5
blank_ratio = 1
merfish_ratio = None
bft = 0.29
nfovs = 25
output_name = 'FeatureSelection_cell_v2.csv'

features_to_use = ['meaninten1_frob', 'offinten1_frob', 'meaninten2_frob', 'offinten2_frob', 'meaninten3_frob', 'offinten3_frob',
                    'meaninten1_2_ratio_frob_clipped','meaninten1_3_ratio_frob_clipped','onoff_ratio1_frob_clipped',
                    'mindist1','mindist2_clipped','mindist3_clipped',
                    'size','edge_ratio','size_conn']

print('training file used:', GS_filepath)
GS_genes_data = GS_genes_data.loc[GS_genes_data['gene'].isin(GS_gene_list+blank_list)]

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

# Process features 
temp = GS_genes_data['mindist2'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['mindist2_clipped'] = np.clip(GS_genes_data['mindist2'],a_min=np.min(GS_genes_data['mindist2']),a_max=GS_genes_data['mindist2'].iloc[np.nanargmax(temp)])

temp = GS_genes_data['mindist3'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['mindist3_clipped'] = np.clip(GS_genes_data['mindist3'],a_min=np.min(GS_genes_data['mindist3']),a_max=GS_genes_data['mindist3'].iloc[np.nanargmax(temp)])

GS_genes_data['total_area'] = GS_genes_data['size_x'] * GS_genes_data['size_y']
GS_genes_data['edge_ratio'] = np.max(GS_genes_data[['size_x','size_y']].values,axis=1) / np.min(GS_genes_data[['size_x','size_y']].values,axis=1) # long edge / short edge

GS_genes_data['meaninten1_2_ratio_frob'] = GS_genes_data['meaninten1_frob'] / GS_genes_data['meaninten2_frob']
temp = GS_genes_data['meaninten1_2_ratio_frob'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['meaninten1_2_ratio_frob_clipped'] = np.clip(GS_genes_data['meaninten1_2_ratio_frob'],a_min=np.min(GS_genes_data['meaninten1_2_ratio_frob']),a_max=14)
GS_genes_data['meaninten1_3_ratio_frob'] = GS_genes_data['meaninten1_frob'] / GS_genes_data['meaninten3_frob']
temp = GS_genes_data['meaninten1_3_ratio_frob'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['meaninten1_3_ratio_frob_clipped'] = np.clip(GS_genes_data['meaninten1_3_ratio_frob'],a_min=np.min(GS_genes_data['meaninten1_3_ratio_frob']),a_max=14)
GS_genes_data['onoff_ratio1_frob'] = GS_genes_data['meaninten1_frob'] / GS_genes_data['offinten1_frob']
temp = GS_genes_data['onoff_ratio1_frob'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['onoff_ratio1_frob_clipped'] = np.clip(GS_genes_data['onoff_ratio1_frob'],a_min=np.min(GS_genes_data['onoff_ratio1_frob']),a_max=10)

# test set
testing_fovs = [i for i in range(20,25)]
training_fovs = [i for i in range(20)]
X_test = GS_genes_data[features_to_use + ['type', 'Label', 'fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test['bft_result'] = 0
Y_test.loc[Y_test['bfs'] <= bft, 'bft_result'] = 1
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin(training_fovs)]
print('train fovs:', training_fovs)
print('test fovs:', testing_fovs)

GS_training = GS_training.reset_index()

# CV splits
random.seed(0)
custom_cv = []
cv_fovlist = [i for i in range(0,20)]
for iter in range(cv_fold):
    val_index = random.sample(cv_fovlist, 4)
    for fov in val_index:
        cv_fovlist.remove(fov)
    train_index = [i for i in range(0,20) if i not in val_index]
    print("val fovs: ", val_index)
    print("train fovs: ", train_index)
    
    resampled_GS_training = pd.DataFrame()
    for train_fov in train_index:
        print('fov', train_fov)
        fov_subset = GS_training.loc[GS_training['fov']==train_fov]

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
        else:
            merfish_df_resampled = merfish_df
            
        print('Number of blanks (0)s: ', len(blank_df_resampled))
        print('Number of merfishonly (0)s: ', len(merfish_df_resampled))
        print('Number of colocalized genes (1)s: ', len(GS_df))
        
        resampled_fov = pd.concat([GS_df, blank_df_resampled, merfish_df_resampled])
        resampled_GS_training = pd.concat([resampled_GS_training,resampled_fov])
    
    custom_cv.append((GS_training.loc[resampled_GS_training.index.values].index.values, GS_training.loc[(GS_training['fov'].isin(val_index)) & (GS_training['type'].isin(['colocalgene','merfishonly']))].index.values))

# define model
RF = RandomForestClassifier(random_state=0, verbose=0,n_jobs=-3, min_samples_split=0.001, min_impurity_decrease=0.01, max_features=1,
                                    max_depth=12,criterion='entropy',min_weight_fraction_leaf=0.175,n_estimators=300)

sfs1 = SFS(estimator=RF, 
           k_features=(1,12),
           forward=True,
           floating=True, 
           scoring='f1',
           cv=custom_cv,
           verbose=2,
           n_jobs=-3)


pipe = Pipeline([
    ('discretizer', ChiMergeTransformer(attr_list=features_to_use,min_intervals=2,hot_start_num_intervals=1000,allow_early_stop=True)),
    ('remove_labels', columnDropperTransformer(columns=['Label'])),
    ('sfs', sfs1),
])

print('features', features_to_use)

pipe.fit(GS_training[features_to_use+['Label']], GS_training['Label'])
pd.DataFrame(sfs1.subsets_).to_csv(os.path.join(output_dir, output_name + '.csv'))

print('results file:', os.path.join(output_dir, output_name + '.csv'))