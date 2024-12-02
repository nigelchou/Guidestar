import sys
sys.path.append("..")

import time
import itertools
import pandas as pd
import os
import numpy as np
from typing import Union
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from Guidestar_model.DiscretisationFunctions import *
from utilsN.FPKM_functions import *

GS_filepath = 'Liver_LM39_49FOVs_GuidestarGenes_training.csv'
merfish_filepath = 'Liver_LM39_49FOVs_MerfishData.csv'
fpkm_path = '../DataForFigures/liver_fpkm_data.tsv'
df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])

GS_genes_data = pd.read_csv(GS_filepath)
merfish_data = pd.read_csv(merfish_filepath)
output_dir = '../DataForFigures/'

# LIVER
GS_gene_list = ['Acly','Hnf4a','Pigr','Gpam']
blank_no = 7
blank_list = ['Blank' + str(i) for i in range(1,blank_no+1)]
cv_fold = 5
features_to_use = ['meaninten1_frob', 'mindist2_clipped', 'mindist3_clipped', 'size']
blank_ratio = 0
merfish_ratio = 1
bft = 0.21
nfovs = 49

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

# Process features 
temp = GS_genes_data['mindist2'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['mindist2_clipped'] = np.clip(GS_genes_data['mindist2'],a_min=np.min(GS_genes_data['mindist2']),a_max=GS_genes_data['mindist2'].iloc[np.nanargmax(temp)])

temp = GS_genes_data['mindist3'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
GS_genes_data['mindist3_clipped'] = np.clip(GS_genes_data['mindist3'],a_min=np.min(GS_genes_data['mindist3']),a_max=GS_genes_data['mindist3'].iloc[np.nanargmax(temp)])

temp = merfish_data['mindist2'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
merfish_data['mindist2_clipped'] = np.clip(merfish_data['mindist2'],a_min=np.min(merfish_data['mindist2']),a_max=merfish_data['mindist2'].iloc[np.nanargmax(temp)])

temp = merfish_data['mindist3'].copy().to_numpy()
temp[~np.isfinite(temp)] = -np.inf
merfish_data['mindist3_clipped'] = np.clip(merfish_data['mindist3'],a_min=np.min(merfish_data['mindist3']),a_max=merfish_data['mindist3'].iloc[np.nanargmax(temp)])

# define test set
testing_fovs = [i for i in range(0,9)]
training_fovs = [i for i in range(9,49)]
X_test = GS_genes_data[features_to_use + ['type', 'Label', 'fov', 'gene']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test['bft_result'] = 0
Y_test.loc[Y_test['bfs'] <= bft, 'bft_result'] = 1
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin(training_fovs)]
print('train fovs:', training_fovs)
print('test fovs:', testing_fovs)

# resample training data
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
    else:
        merfish_df_resampled = merfish_df
        
    print('Number of blanks (0)s: ', len(blank_df_resampled))
    print('Number of merfishonly (0)s: ', len(merfish_df_resampled))
    print('Number of colocalized genes (1)s: ', len(GS_df))
    
    resampled_fov = pd.concat([GS_df, blank_df_resampled, merfish_df_resampled])
    resampled_GS_training = pd.concat([resampled_GS_training,resampled_fov])
    
resampled_GS_training = resampled_GS_training.reset_index()

# define model
print('features used:', features_to_use)
RF = RandomForestClassifier(random_state=0, verbose=0,n_jobs=-3, min_samples_split=0.001, min_impurity_decrease=0.01, max_features=1,
                                max_depth=12,criterion='gini',min_weight_fraction_leaf=0.01,n_estimators=100, max_samples= 0.8)
print('random forest model:', RF)
cm = ChiMergeTransformer(attr_list=features_to_use,min_intervals=2,hot_start_num_intervals=1000,allow_early_stop=True)

pipe = Pipeline([
    ('discretizer', cm),
    ('remove_labels', columnDropperTransformer(columns=['Label'])),
    ('model', RF)
])

pipe.fit(resampled_GS_training[features_to_use+['Label']], resampled_GS_training['Label'])

model_precision = []
model_recall = []
model_f1 = []
model_auroc = []
bft_precision = []
bft_recall = []
bft_f1 = []
bft_auroc = []
for test_fov in testing_fovs:
    fov_subset = X_test.loc[X_test['fov'] == test_fov]
    fov_subset_transformed = cm.transform(fov_subset[features_to_use])
    for col in fov_subset_transformed.columns:
        fov_subset_transformed[col] = fov_subset_transformed[col].fillna(fov_subset_transformed[col].max())
    Y_fov_subset = Y_test.loc[Y_test['fov'] == test_fov]
    Y_fov_subset['pred'] = RF.predict(fov_subset_transformed[features_to_use])
    Y_fov_subset['pred_score'] = RF.predict_proba(fov_subset_transformed[features_to_use])[:, 1]
    
    model_precision.append(precision_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                           Y_fov_subset['pred'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    model_recall.append(recall_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                     Y_fov_subset['pred'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    model_f1.append(f1_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                             Y_fov_subset['pred'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    model_auroc.append(roc_auc_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                     Y_fov_subset['pred_score'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    
    bft_precision.append(precision_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                         Y_fov_subset['bft_result'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    bft_recall.append(recall_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                   Y_fov_subset['bft_result'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    bft_f1.append(f1_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                           Y_fov_subset['bft_result'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    bft_auroc.append(roc_auc_score(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],
                                   1 - Y_fov_subset['bfs'].loc[Y_fov_subset['gene'].isin(GS_gene_list)]))
    
results_df = pd.DataFrame({'fov': testing_fovs, 'model_precision': model_precision, 'model_recall': model_recall,
                           'model_f1': model_f1, 'model_auroc': model_auroc,
                           'bft_precision': bft_precision, 'bft_recall': bft_recall,
                           'bft_f1': bft_f1, 'bft_auroc': bft_auroc})
results_df.to_csv(os.path.join(output_dir, 'testset_results_liver.csv'))

# merfish by fov
model_fpkm_result = []
model_counts_result = []
model_misid_result = []

bft_fpkm_result = []
bft_counts_result = []
bft_misid_result = []

merfish_data['bft_result'] = 0
merfish_data.loc[merfish_data['bfs'] <= bft, 'bft_result'] = 1
for merfish_fov in range(nfovs):
    merfish_subset = merfish_data[features_to_use + ['bfs', 'bft_result', 'gene']].loc[merfish_data['fov'] == merfish_fov]
    merfish_subset_transformed = cm.transform(merfish_subset[features_to_use])
    for col in merfish_subset_transformed.columns:
        merfish_subset_transformed[col] = merfish_subset_transformed[col].fillna(merfish_subset_transformed[col].max())
    merfish_subset_transformed['pred'] = RF.predict(merfish_subset_transformed[features_to_use])
    model_callouts = list(merfish_subset['gene'].loc[merfish_subset_transformed['pred']==1])
    model_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, model_callouts)
    model_fpkm_val, _ = calcLogCorrelation(np.array(model_counts_arr), np.array(fpkm_list))
    model_mis_id_rate = calc_misid(model_counts_arr, model_callouts)
    
    bft_callouts = list(merfish_subset['gene'].loc[merfish_subset['bft_result']==1])
    bft_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, bft_callouts)
    bft_fpkm_val, _ = calcLogCorrelation(np.array(bft_counts_arr), np.array(fpkm_list))
    bft_mis_id_rate = calc_misid(bft_counts_arr, bft_callouts)
    
    model_fpkm_result.append(model_fpkm_val)
    model_counts_result.append(np.sum(model_counts_arr))
    model_misid_result.append(model_mis_id_rate)
    
    bft_fpkm_result.append(bft_fpkm_val)
    bft_counts_result.append(np.sum(bft_counts_arr))
    bft_misid_result.append(bft_mis_id_rate)
    
merfish_results_df = pd.DataFrame({'fov':[i for i in range(nfovs)], 
                                   'model_fpkm': model_fpkm_result, 'model_counts':model_counts_result, 'model_misid': model_misid_result,
                                   'bft_fpkm': bft_fpkm_result, 'bft_counts': bft_counts_result, 'bft_misid': bft_misid_result})
merfish_results_df.to_csv(os.path.join(output_dir, 'merfish_results_liver.csv'))

# MERFISH whole
merfish_data['bft_result'] = 0
merfish_data.loc[merfish_data['bfs'] <= bft, 'bft_result'] = 1

merfish_subset = merfish_data[features_to_use + ['bfs', 'bft_result', 'gene']]
merfish_subset_transformed = cm.transform(merfish_subset[features_to_use])
for col in merfish_subset_transformed.columns:
    merfish_subset_transformed[col] = merfish_subset_transformed[col].fillna(merfish_subset_transformed[col].max())
merfish_subset_transformed['pred'] = RF.predict(merfish_subset_transformed[features_to_use])
model_callouts = list(merfish_subset['gene'].loc[merfish_subset_transformed['pred']==1])
model_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, model_callouts)
model_fpkm_val, _ = calcLogCorrelation(np.array(model_counts_arr), np.array(fpkm_list))
model_mis_id_rate = calc_misid(model_counts_arr, model_callouts)

bft_callouts = list(merfish_subset['gene'].loc[merfish_subset['bft_result']==1])
bft_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, bft_callouts)
bft_fpkm_val, _ = calcLogCorrelation(np.array(bft_counts_arr), np.array(fpkm_list))
bft_mis_id_rate = calc_misid(bft_counts_arr, bft_callouts)

print('model fpkm', model_fpkm_val)
print('model counts', np.sum(model_counts_arr))
print('model misid', model_mis_id_rate)

print('BFT fpkm', bft_fpkm_val)
print('BFT counts',np.sum(bft_counts_arr))
print('BFT misid',bft_mis_id_rate)

### SuppFig9 ###

# Precision, recall, F1 on GS genes
results_df = pd.read_csv(os.path.join(output_dir,'testset_results_liver.csv'))
merfish_results_df = pd.read_csv(os.path.join(output_dir, 'merfish_results_liver.csv'))

fig,axes = plt.subplots(1,3,figsize=(8,6))
axes = axes.ravel()
results_df_long = pd.melt(results_df,id_vars=['fov'])

sns.boxplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_precision','bft_precision'])], x = 'variable',y='value', color='white',showfliers=False,ax=axes[0])
sns.swarmplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_precision','bft_precision'])], x = 'variable',y='value',ax=axes[0])

sns.boxplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_recall','bft_recall'])], x = 'variable',y='value', color='white',showfliers=False,ax=axes[1])
sns.swarmplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_recall','bft_recall'])], x = 'variable',y='value',ax=axes[1])

sns.boxplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_f1','bft_f1'])], x = 'variable',y='value', color='white',showfliers=False,ax=axes[2])
sns.swarmplot(data=results_df_long.loc[results_df_long['variable'].isin(['model_f1','bft_f1'])], x = 'variable',y='value',ax=axes[2])

axes[0].set_title('precision')
axes[1].set_title('recall')
axes[2].set_title('f1')

axes[0].set_ylim(0,1)
axes[1].set_ylim(0,1)
axes[2].set_ylim(0,1)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'testset_results_liver.svg'))

# FPKM, counts on all genes
fig,axes = plt.subplots(1,2)
axes = axes.ravel()
merfish_results_df_long = pd.melt(merfish_results_df,id_vars=['fov'])

sns.boxplot(data=merfish_results_df_long.loc[merfish_results_df_long['variable'].isin(['model_fpkm','bft_fpkm'])], x = 'variable',y='value', color='white',showfliers=False,ax=axes[0])
sns.swarmplot(data=merfish_results_df_long.loc[merfish_results_df_long['variable'].isin(['model_fpkm','bft_fpkm'])], x = 'variable',y='value',ax=axes[0])
axes[0].plot([0,1], [model_fpkm_val,bft_fpkm_val],'^',color='black',markersize=7)

sns.boxplot(data=merfish_results_df_long.loc[merfish_results_df_long['variable'].isin(['model_counts','bft_counts'])], x = 'variable',y='value', color='white',showfliers=False,ax=axes[1])
sns.swarmplot(data=merfish_results_df_long.loc[merfish_results_df_long['variable'].isin(['model_counts','bft_counts'])], x = 'variable',y='value',ax=axes[1])
axes[1].plot([0,1], [np.sum(model_counts_arr)/nfovs,np.sum(bft_counts_arr)/nfovs],'^',color='black',markersize=7)

axes[0].set_ylim(0,1)
axes[1].set_ylim(0,18000) # liver

axes[0].set_title('fpkm')
axes[1].set_title('counts')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'merfish_results_liver_withwhole.svg'))

# ROC, AUC
### by fovs with misid, no error bars
tprs_model = []
fprs_model = []
aucs_model = []
tprs_bft = []
fprs_bft = []
aucs_bft = []
mean_fpr = np.linspace(0,1,100)

fig1,ax1=plt.subplots(1,1,figsize=(6,6))

for j,test_fov in enumerate(testing_fovs):
    fov_subset = X_test.loc[X_test['fov'] == test_fov]
    fov_subset_transformed = cm.transform(fov_subset[features_to_use])
    for col in fov_subset_transformed.columns:
        fov_subset_transformed[col] = fov_subset_transformed[col].fillna(fov_subset_transformed[col].max())
    Y_fov_subset = Y_test.loc[Y_test['fov'] == test_fov]
    Y_fov_subset['pred'] = RF.predict(fov_subset_transformed[features_to_use])
    Y_fov_subset['pred_score'] = RF.predict_proba(fov_subset_transformed[features_to_use])[:, 1]
    print(Y_fov_subset['pred_score'].min(),Y_fov_subset['pred_score'].max())
    
    roc_model=RocCurveDisplay.from_predictions(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],Y_fov_subset['pred_score'].loc[Y_fov_subset['gene'].isin(GS_gene_list)])
    roc_bft=RocCurveDisplay.from_predictions(Y_fov_subset['Label'].loc[Y_fov_subset['gene'].isin(GS_gene_list)],1-Y_fov_subset['bfs'].loc[Y_fov_subset['gene'].isin(GS_gene_list)])
    
    interp_tpr_model = np.interp(mean_fpr,roc_model.fpr,roc_model.tpr)
    interp_tpr_model[0]=0
    interp_fpr_model = np.interp(mean_fpr,roc_model.tpr,roc_model.fpr)
    interp_fpr_model[0]=0
    
    tprs_model.append(interp_tpr_model)
    fprs_model.append(interp_fpr_model)
    
    interp_tpr_bft = np.interp(mean_fpr,roc_bft.fpr,roc_bft.tpr)
    interp_tpr_bft[0]=0
    interp_fpr_bft = np.interp(mean_fpr,roc_bft.tpr,roc_bft.fpr)
    interp_fpr_bft[0]=0
    
    tprs_bft.append(interp_tpr_bft)
    fprs_bft.append(interp_fpr_bft)
    
    ax1.plot(roc_model.fpr, roc_model.tpr, linestyle = 'dashed', alpha=0.3, color='tab:blue')
    ax1.plot(roc_bft.fpr, roc_bft.tpr, linestyle='dashed', alpha=0.3, color='tab:orange')
    
    TP = len(Y_fov_subset.loc[(Y_fov_subset['Label']==1) & (Y_fov_subset['bft_result']==1)& (Y_fov_subset['gene'].isin(GS_gene_list))])
    FP = len(Y_fov_subset.loc[(Y_fov_subset['Label']==0) & (Y_fov_subset['bft_result']==1)& (Y_fov_subset['gene'].isin(GS_gene_list))])
    FN = len(Y_fov_subset.loc[(Y_fov_subset['Label']==1) & (Y_fov_subset['bft_result']==0)& (Y_fov_subset['gene'].isin(GS_gene_list))])
    TN = len(Y_fov_subset.loc[(Y_fov_subset['Label']==0) & (Y_fov_subset['bft_result']==0)& (Y_fov_subset['gene'].isin(GS_gene_list))])

    misid_fpr = FP / (FP+TN)
    misid_tpr = TP / (TP+FN)

    ax1.plot(misid_fpr, misid_tpr, 'x', color='black')
    print(misid_fpr, misid_tpr)
    
mean_tpr_model = np.mean(tprs_model,axis=0)
mean_tpr_model[-1]=1
mean_tpr_bft = np.mean(tprs_bft,axis=0)
mean_tpr_bft[-1]=1
ax1.plot(mean_fpr,mean_tpr_model,color='tab:blue', linewidth=3, label='Guidestar')
ax1.plot(mean_fpr,mean_tpr_bft,color='tab:orange', linewidth=3, label='bft')
ax1.legend(loc='lower right')
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
fig1.savefig(os.path.join(output_dir,'ROC_liver_withmisid.svg'))

# on full test set
X_test_transformed = cm.transform(X_test[features_to_use])
for col in X_test_transformed.columns:
    X_test_transformed[col] = X_test_transformed[col].fillna(X_test_transformed[col].max())
Y_test['pred'] = RF.predict(X_test_transformed[features_to_use])
Y_test['pred_score'] = RF.predict_proba(X_test_transformed[features_to_use])[:, 1]

with open(os.path.join(output_dir,'full_testset_liver.txt'), "w") as f:
    print('features', features_to_use, file=f)
    print('full test set metrics', file=f)
    print('bft precision', precision_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['bft_result'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('bft recall', recall_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['bft_result'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('bft f1', f1_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['bft_result'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('bft auroc', roc_auc_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], 1 - Y_test['bfs'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)

    print('model precision', precision_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['pred'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('model recall', recall_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['pred'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('model f1', f1_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['pred'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    print('model auroc', roc_auc_score(Y_test['Label'].loc[Y_test['gene'].isin(GS_gene_list)], Y_test['pred_score'].loc[Y_test['gene'].isin(GS_gene_list)]), file=f)
    
f.close()

# Correlation with FPKM
merfish_data['bft_result'] = 0
merfish_data.loc[merfish_data['bfs'] <= bft, 'bft_result'] = 1

merfish_subset = merfish_data[features_to_use + ['bfs', 'bft_result', 'gene']]
merfish_subset_transformed = cm.transform(merfish_subset[features_to_use])
for col in merfish_subset_transformed.columns:
    merfish_subset_transformed[col] = merfish_subset_transformed[col].fillna(merfish_subset_transformed[col].max())
merfish_subset_transformed['pred'] = RF.predict(merfish_subset_transformed[features_to_use])
model_callouts = list(merfish_subset['gene'].loc[merfish_subset_transformed['pred']==1])
model_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, model_callouts)

bft_callouts = list(merfish_subset['gene'].loc[merfish_subset['bft_result']==1])
bft_counts_arr, _, fpkm_list = plot_fpkm_corr(df_fpkm, bft_callouts)

model_fpkm_val, _ = calcLogCorrelation(np.array(model_counts_arr), np.array(fpkm_list))
bft_fpkm_val, _ = calcLogCorrelation(np.array(bft_counts_arr), np.array(fpkm_list))

correlation_df = pd.DataFrame({'fpkm': fpkm_list, 'Guidestar': model_counts_arr, 'BFT': bft_counts_arr})

def plotScatter(ax: Figure.axes,
                df: pd.DataFrame,
                x_column: str,
                y_column: str,
                y_column2: str,
                fontprops: fm.FontProperties,
                spot_size: float = 30,
                alpha: float = 0.6,
                is_inset: bool = False,
                background_alpha: float = 0.8,
                xlim_offset: Union[None, float] = None,
                ylim_offset: Union[None, float] = 1,
                ) -> None:
    """
    plot a scatterplot

    """

    def _findLogLowerBound(array: np.ndarray,
                           ) -> float:
        """
        calculate the lower bound on a log plot for the given array of values
        """
        # remove 0s from array (log of 0 undefined), also reject negative values
        array = array[array > 0.]
        min_val = np.amin(array)

        return 10 ** (np.floor(np.log10(min_val)))

    if is_inset:
        # within another plot
        sns.set_style("dark")
        label_color = "white"
    else:
        # standalone plot
        sns.set_style("darkgrid")
        label_color = "black"

    # counts for each axis
    count_values1 = df[y_column].values
    count_values2 = df[y_column2].values
    fpkm_values = df[x_column].values

    scatterplot = ax.scatter(
        x=fpkm_values, y=count_values1,
        s=spot_size, alpha=alpha,
        edgecolors="none",
        label=y_column
    )
    
    scatterplot = ax.scatter(
        x=fpkm_values, y=count_values2,
        s=spot_size, alpha=alpha,
        edgecolors="none",
        label=y_column2
    )

    ax.set_ylabel(
        "count", labelpad=0, font_properties=fontprops,
        color=label_color,
    )

    # ax_scatter.set_yscale("log")
    count_values = np.concatenate((count_values1, count_values2))
    linthreshy = _findLogLowerBound(count_values)
    print(f"linthreshy (counts axis) : {linthreshy}\n")

    ax.set_yscale(
        "symlog", linthresh=linthreshy, linscale=0.2
    )

    ax.set_xlabel(
        "FPKM value", labelpad=0, font_properties=fontprops,
        color=label_color,
    )

    # find value of linthreshx that is closest to
    # the lowest nonzero FPKM value

    linthreshx = _findLogLowerBound(fpkm_values)
    print(f"linthreshx (FPKM axis): {linthreshx}\n")

    ax.set_xscale(
        "symlog", linthresh=linthreshx, linscale=0.2
    )

    # set default axes limits
    ax.set_xlim((linthreshx, None))
    ax.set_ylim((linthreshy, None))

    if xlim_offset is not None:
        nonzero_fpkm_counts = fpkm_values[fpkm_values>0]
        ax.set_xlim((np.nanmin(nonzero_fpkm_counts) - xlim_offset, None))

    if ylim_offset is not None:
        nonzero_counts = count_values[count_values > 0]
        ax.set_ylim((np.nanmin(nonzero_counts) - ylim_offset, None))

    if is_inset:
        ax.patch.set_alpha(background_alpha)
        for side in ['top',
                     'bottom', 'left',
                     'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(
            axis='both', which='major',
            labelsize=5,
            labelcolor=label_color,
            pad=0,
        )
        
    ax.legend()

    return scatterplot

fig,ax=plt.subplots(1,1)
results_fontprops = fm.FontProperties(
                size=12, family="Arial", weight="bold"
            )
correlation_plot = plotScatter(ax=ax, df=correlation_df, x_column='fpkm', y_column='Guidestar', y_column2='BFT', fontprops=results_fontprops)
ax.set_title('model corr: ' + str(np.round(model_fpkm_val,3)) + ' bft corr: ' + str(np.round(bft_fpkm_val,3)))
fig.savefig(os.path.join(output_dir, 'Tissue_correlationwithfpkm.svg'))