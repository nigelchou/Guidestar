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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from DiscretisationFunctions import *
import sys
from utilsN.FPKM_functions import *

GS_filepath = '/Volumes/Seagate Backup Plus Drive/1 year RA data/Guidestar files/cells/colocaldist4_20230312/Cell_LM39_25FOVs_GuidestarGenes_training_dist4.csv'
GS_genes_data = pd.read_csv(GS_filepath)
print('fovs used:', np.unique(GS_genes_data['fov']))

output_dir = '/Volumes/Seagate Backup Plus Drive/1 year RA data/20230513/'
output_name = 'hyperparam_optimization_cell_RF'

GS_gene_list = ['Acly','Ube2z','Gpam','Hnf4a']
blank_no = 7
blank_list = ['Blank' + str(i) for i in range(1,blank_no+1)]
cv_fold = 5
features_to_use = ['meaninten1_frob', 'mindist2_clipped', 'mindist3_clipped', 'size']
blank_ratio = 1
merfish_ratio = None

print('genes',GS_gene_list)
print('features used', features_to_use)
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

# test set
X_test = GS_genes_data[features_to_use + ['type', 'Label']].loc[GS_genes_data['fov'].isin([i for i in range(20,25)])]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin([i for i in range(20,25)])]
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin([i for i in range(20)])]
print('test fovs:', np.unique(Y_test['fov']))

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

# parameter space
max_depth_list = [12]
n_estimators_list = [100,200,300,400,500]
criterion_list = ['gini','entropy']
min_impurity_decrease_list = [0.001,0.01,0.1]
min_weight_fraction_leaf_list = [0.01,0.1,0.15,0.175]
max_samples_list = [0.5,0.65,0.8,1.0]

result_max_depth = []
result_n_estimators = []
result_criterion = []
result_max_samples = []
result_min_impurity_decrease = []
result_min_weight_fraction_leaf = []

result_c = []
result_gamma = []
result_degree = []

result_f1 = []
result_precision = []
result_recall = []
result_auroc = []

counter = 1
for max_depth in max_depth_list:
    for n_estimators in n_estimators_list:
        for criterion in criterion_list:
            for max_samples in max_samples_list:
                for min_impurity_decrease in min_impurity_decrease_list:
                    for min_weight_fraction_leaf in min_weight_fraction_leaf_list:
                        
                        pipe = Pipeline([
                            ('discretizer', ChiMergeTransformer(attr_list=features_to_use,min_intervals=2,hot_start_num_intervals=1000,allow_early_stop=True)),
                            ('remove_labels', columnDropperTransformer(columns=['Label'])),
                            ('model', RandomForestClassifier(random_state=0, verbose=0,n_jobs=-3, min_samples_split=0.001, 
                                                            min_impurity_decrease=min_impurity_decrease, max_features=1,
                                                            max_depth=max_depth,criterion=criterion,max_samples=max_samples,
                                                            min_weight_fraction_leaf=min_weight_fraction_leaf,n_estimators=n_estimators))
                        ])
                        
                        scores = cross_validate(pipe, GS_training[features_to_use+['Label']], 
                                                GS_training['Label'], cv=custom_cv, 
                                                n_jobs=5, scoring=['precision', 'recall', 'f1','roc_auc'])
                        
                        print(counter)
                        
                        result_max_depth.append(max_depth)
                        result_n_estimators.append(n_estimators)
                        result_criterion.append(criterion)
                        result_max_samples.append(max_samples)
                        result_min_impurity_decrease.append(min_impurity_decrease)
                        result_min_weight_fraction_leaf.append(min_weight_fraction_leaf)
                        result_precision.append(scores['test_precision'])
                        result_recall.append(scores['test_recall'])
                        result_f1.append(scores['test_f1'])
                        result_auroc.append(scores['test_roc_auc'])
                        
                        counter += 1


result_df = pd.concat([pd.DataFrame(result_precision,columns=['precision_'+str(i) for i in range(5)]), 
                    pd.DataFrame(result_recall,columns=['recall_'+str(i) for i in range(5)]),
                    pd.DataFrame(result_f1,columns=['f1_'+str(i) for i in range(5)]),
                    pd.DataFrame(result_auroc,columns=['auroc_'+str(i) for i in range(5)])],
                    axis=1)
result_df['max_depth'] = result_max_depth
result_df['n_estimators'] = result_n_estimators
result_df['criterion'] = result_criterion
result_df['max_samples'] = result_max_samples
result_df['min_impurity_decrease'] = result_min_impurity_decrease
result_df['min_weight_fraction_leaf'] = result_min_weight_fraction_leaf

result_df.to_csv(os.path.join(output_dir,output_name + '.csv'))