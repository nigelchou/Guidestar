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
import copy

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


base_dir = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver')
GS_filepath = f'{base_dir}/Tissue_LM50GS_86FOVs_GuidestarGenes_training_set1.csv'
GS_genes_data = pd.read_csv(GS_filepath)
output_dir = f'{base_dir}/GS_model_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_name = 'NumGenes_LM50GS_mbrain.csv'
GS_gene_list = ['Cacna1a', 'Slc1a2', 'Cacna1i'] # drop <15% colocalization, drop Itpr1 bad quality spots (contain a lot of noise)
blank_ratio = 0
merfish_ratio = None
blank_no = 24
blank_list = [i for i in GS_genes_data['gene'].unique() if i.startswith('Blank')]
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
print(GS_genes_data['type'].value_counts().to_frame(name='Count'))

# test set
testing_fovs = ['000_006','000_007','000_008','000_009','001_006','001_007','001_008','001_009',
                '002_006','002_007','002_008','002_009','003_006','003_007','003_008','003_009']
training_fovs = [i for i in GS_genes_data['fov'].unique() if i not in testing_fovs]
X_test = GS_genes_data[features_to_use + ['type', 'Label']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin(training_fovs)]
print('test fovs:', testing_fovs)
print(f'number of test fovs: {len(testing_fovs)}')
print(f'number of train fovs: {len(training_fovs)}')

pd.options.mode.chained_assignment = None
precision_list = []
recall_list = []
f1_list = []
comb_list = []
combid_list = []
numgenes_list = []

random.seed(0)
val_size = len(training_fovs) // cv_fold # adaptive validation size determinant (account for potential large drop in No. FOVs)
print(val_size)# val_size = 14
custom_cv = []
cv_fovs = copy.deepcopy(training_fovs)
for iter in range(cv_fold):
    n = min(val_size, len(cv_fovs))
    val_index = random.sample(cv_fovs, n)
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

            if blank_ratio is not None and len(blank_df) > int(len(merfish_df) * blank_ratio):
                blank_df_resampled = resample(blank_df, replace=False, n_samples=int(len(merfish_df) * blank_ratio),
                                              random_state=0)
            else:
                blank_df_resampled = blank_df

            if merfish_ratio is not None and len(merfish_df) > int(len(GS_df) * merfish_ratio):
                merfish_df_resampled = resample(merfish_df, replace=False, n_samples=int(len(GS_df) * merfish_ratio),
                                                random_state=0)
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
        
        
results_df = pd.DataFrame({'combination': comb_list, 'combid': combid_list,
                           'number': numgenes_list, 'precision': precision_list,
                           'recall':recall_list, 'F1':f1_list})
results_df.to_csv(os.path.join(output_dir,output_name))
# aggregate across folds per combination
summary = (results_df
           .groupby(['combid', 'number'])
           .agg(mean_precision=('precision', 'mean'),
                mean_recall=('recall', 'mean'),
                mean_F1=('F1', 'mean'),
                std_F1=('F1', 'std'),
                min_F1=('F1', 'min'))
           .reset_index())

# attach the actual gene tuple back (comb_list is per-row; take first per combid)
combid_to_genes = results_df.groupby(['combid', 'number'])['combination'].first()
summary = summary.merge(
    combid_to_genes.reset_index(),
    on=['combid', 'number'],
    how='left'
)

# coefficient of variation: penalizes unstable combos
summary['cv_F1'] = summary['std_F1'] / summary['mean_F1']

# rank: highest mean F1 first, break ties by lower std
summary = summary.sort_values(['mean_F1', 'std_F1'],
                              ascending=[False, True]).reset_index(drop=True)

summary.to_csv(os.path.join(output_dir, 'NumGenes_summary.csv'), index=False)
print(summary.head(10))

import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots(figsize=(6,3))
sns.boxplot(data=results_df, x='number', y='F1', color='white', showfliers=False, ax=ax)
sns.stripplot(data=results_df, x='number', y='F1', color='black', alpha=0.7, ax=ax)
# ax.set_title('F1')
ax.set_xlabel("")
ax.set_ylabel("")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# fig.suptitle('LM50 brain set 1 3gene no resample')
# fig.supxlabel('Number of Guidestar genes', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'NumGenes_brain_set1_3gene_noresample.svg'))

print(results_df[['combination', 'F1']].groupby(by='combination').mean().sort_values(by='F1', ascending=False))