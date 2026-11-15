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
import copy
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


base_dir = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver')
GS_filepath = os.path.join(base_dir, 'Tissue_LM50GS_86FOVs_GuidestarGenes_training_set1.csv')
merfish_filepath = os.path.join(base_dir, 'Tissue_LM50GS_86FOVs_MerfishData_set1.csv')
fpkm_path = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/codebook/fpkm_data.txt')
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])

GS_genes_data = pd.read_csv(GS_filepath)
merfish_data = pd.read_csv(merfish_filepath)
output_dir = os.path.join(base_dir, 'GS_model_result')
os.makedirs(output_dir, exist_ok=True)

# MBRAIN
GS_gene_list = ['Cacna1a', 'Slc1a2', 'Cacna1i']
blank_no = 24
blank_list = [i for i in GS_genes_data['gene'].unique() if i.startswith('Blank')]
cv_fold = 5
blank_ratio = 0
merfish_ratio = None
colocal_ratio = None
bft = 0.252
nfovs = 86
output_name = 'FeatureSelection_mBRAIN_v2'

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
testing_fovs = ['000_006','000_007','000_008','000_009','001_006','001_007','001_008','001_009',
                '002_006','002_007','002_008','002_009','003_006','003_007','003_008','003_009']
training_fovs = [i for i in GS_genes_data['fov'].unique() if i not in testing_fovs]
X_test = GS_genes_data[features_to_use + ['type', 'Label']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test['bft_result'] = 0
Y_test.loc[Y_test['bfs'] <= bft, 'bft_result'] = 1
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin(training_fovs)]
print('train fovs:', training_fovs)
print('test fovs:', testing_fovs)

GS_training = GS_training.reset_index()

# CV splits
val_size = len(training_fovs) // cv_fold # adaptive validation size determinant (account for potential large drop in No. FOVs)
# 13 in this case
random.seed(0)
custom_cv = []
cv_fovs = copy.deepcopy(training_fovs)
for iter in range(cv_fold):
    n = min(val_size, len(cv_fovs))
    val_index = random.sample(cv_fovs, n)
    for fov in val_index:
        cv_fovs.remove(fov)
    train_index = [i for i in training_fovs if i not in val_index]
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

        if blank_ratio == 'match':
            colocal_diff = len(GS_df) - len(merfish_df)
            if colocal_diff < 0:
                blank_df_resampled = pd.DataFrame()
            elif colocal_diff > 0:
                blank_df_resampled = resample(blank_df, replace=True, n_samples=int(abs(colocal_diff)), random_state=0)
            else:
                blank_df_resampled = pd.DataFrame()
        elif blank_ratio is not None and isinstance(blank_ratio, (int, np.integer)) == True and len(blank_df) > int(
                len(merfish_df) * blank_ratio):  # match blanks to merfishonly
            blank_df_resampled = resample(blank_df, replace=False, n_samples=int(len(merfish_df) * blank_ratio),
                                          random_state=0)
        else:
            blank_df_resampled = pd.DataFrame()

        if merfish_ratio is not None and len(merfish_df) > int(len(GS_df) * merfish_ratio):
            merfish_df_resampled = resample(merfish_df, replace=False, n_samples=int(len(GS_df) * merfish_ratio),
                                            random_state=0)
        elif merfish_ratio is not None and len(merfish_df) < int(len(GS_df) * merfish_ratio):
            merfish_df_resampled = resample(merfish_df, replace=True, n_samples=int(len(GS_df) * merfish_ratio),
                                            random_state=0)
        else:
            merfish_df_resampled = merfish_df

        if colocal_ratio is not None and len(GS_df) > int(len(merfish_df_resampled) * colocal_ratio):
            GS_df_resampled = resample(GS_df, replace=False, n_samples=int(len(merfish_df_resampled) * colocal_ratio),
                                       random_state=0)
        elif colocal_ratio is not None and len(GS_df) < int(len(merfish_df_resampled) * colocal_ratio):
            GS_df_resampled = resample(GS_df, replace=True, n_samples=int(len(merfish_df_resampled) * colocal_ratio),
                                       random_state=0)
        else:
            GS_df_resampled = GS_df
            
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
#
# perm_records = []      # one row per (seed, size) with the selected set
# perm_subsets = []      # full subsets_ table per seed, for later inspection
#
# for seed in range(5):
#     rng = np.random.default_rng(seed)
#     perm = list(rng.permutation(features_to_use))
#     print(f"\n=== seed {seed} | order: {perm}")
#
#     RF = RandomForestClassifier(random_state=0, verbose=0, n_jobs=-3,
#                                 min_samples_split=0.001, min_impurity_decrease=0.01,
#                                 max_features=1, max_depth=12, criterion='entropy',
#                                 min_weight_fraction_leaf=0.175, n_estimators=300)
#
#     sfs_i = SFS(estimator=RF, k_features=(1, 12), forward=True, floating=True,
#                 scoring='f1', cv=custom_cv, verbose=0, n_jobs=-3)
#
#     pipe_i = Pipeline([
#         ('discretizer', ChiMergeTransformer(attr_list=perm, min_intervals=2,
#                                             hot_start_num_intervals=1000,
#                                             allow_early_stop=True)),
#         ('remove_labels', columnDropperTransformer(columns=['Label'])),
#         ('sfs', sfs_i),
#     ])
#
#     pipe_i.fit(GS_training[perm + ['Label']], GS_training['Label'])
#
#     for k, v in sfs_i.subsets_.items():
#         names = tuple(sorted(perm[i] for i in v['feature_idx']))   # map back to NAMES
#         perm_records.append({
#             'seed': seed,
#             'order': '|'.join(perm),
#             'size': k,
#             'feature_idx': str(v['feature_idx']),
#             'feature_names': '|'.join(names),
#             'avg_F1': v['avg_score'],
#             'std_F1': float(np.std(v['cv_scores'], ddof=1)),
#             'min_F1': float(np.min(v['cv_scores'])),
#             'cv_scores': '|'.join(f"{x:.6f}" for x in v['cv_scores']),
#         })
#
# perm_df = pd.DataFrame(perm_records)
# perm_df.to_csv(os.path.join(output_dir, 'sfs_permutation_runs.csv'), index=False)

chosen_idx = [0, 10, 12]
chosen_feats = [features_to_use[i] for i in chosen_idx]
print('chosen features:', chosen_feats)

prec, rec, f1s = [], [], []
for train_idx, val_idx in custom_cv:
    train_df = GS_training.loc[train_idx]
    val_df   = GS_training.loc[val_idx]

    RF = RandomForestClassifier(random_state=0, n_jobs=-3, min_samples_split=0.001,
                                min_impurity_decrease=0.01, max_features=1, max_depth=12,
                                criterion='entropy', min_weight_fraction_leaf=0.175, n_estimators=300)
    RF.fit(train_df[chosen_feats], train_df['Label'])
    pred = RF.predict(val_df[chosen_feats])

    prec.append(precision_score(val_df['Label'], pred))
    rec.append(recall_score(val_df['Label'], pred))
    f1s.append(f1_score(val_df['Label'], pred))

print(f"precision {np.mean(prec):.4f} ± {np.std(prec, ddof=1):.4f}")
print(f"recall    {np.mean(rec):.4f} ± {np.std(rec, ddof=1):.4f}")
print(f"F1        {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")