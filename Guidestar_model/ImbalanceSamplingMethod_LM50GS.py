"""
Check upsampling vs downsampling for class imbalance
1. No sampling
        blank_ratio = 0, merfish_ratio = None, colocal_ratio = None
2. No sampling of colocalised and MERFISH only, augment MERFISH only (0)s with blanks (0)s. no matter whether 1s > 0s
        blank_ratio = 1, merfish_ratio = None, colocal_ratio = None
3. Downsampling of colocalised to match MERFISH only, no sampling of MERFISH only. Where 1s > 0s
        blank_ratio = 0, merfish_ratio = None, colocal_ratio = 1
4. Upsampling of MERFISH only to match colocalised, no sampling of colocalised. Where 1s > 0s
        blank_ratio = 0, merfish_ratio = 1, colocal_ratio = None
5. No sampling of colocalised and MERFISH only, augment MERFISH only (0)s with blanks (0)s to match colocalised (1)s. Where 1s > 0s
        blank_ratio = 'match', merfish_ratio = None, colocal_ratio = None
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
import copy
from matplotlib.ticker import FormatStrFormatter

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


base_dir = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver')
GS_filepath = f'{base_dir}/Tissue_LM50GS_86FOVs_GuidestarGenes_training_set1.csv'
GS_genes_data = pd.read_csv(GS_filepath)
output_dir = f'{base_dir}/GS_model_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# MBRAIN
model_type = 'RF'
GS_gene_list = ['Cacna1a', 'Slc1a2', 'Cacna1i']
blank_no = 24
blank_list = [i for i in GS_genes_data['gene'].unique() if i.startswith('Blank')]
cv_fold = 5
features_to_use = ['meaninten1_frob', 'mindist1', 'size'] # use conventional features for optimizing sampling method
nfovs = 86
output_name = 'Imbalance_sampling_mBrain_conv3feat_v3.csv'
sampling_notes = 'Imbalance_sampling_brain_conv3feat_v3.txt'

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

print(GS_genes_data['type'].value_counts().to_frame(name='Count'))

# test set
testing_fovs = ['000_006','000_007','000_008','000_009','001_006','001_007','001_008','001_009',
                '002_006','002_007','002_008','002_009','003_006','003_007','003_008','003_009']
training_fovs = [i for i in GS_genes_data['fov'].unique() if i not in testing_fovs]
X_test = GS_genes_data[features_to_use + ['type', 'Label']].loc[GS_genes_data['fov'].isin(testing_fovs)]
Y_test = GS_genes_data[['Label','gene','bfs','type','fov']].loc[GS_genes_data['fov'].isin(testing_fovs)]
GS_training = GS_genes_data.loc[GS_genes_data['fov'].isin(training_fovs)]
print('test fovs:', testing_fovs)

val_size = len(training_fovs) // cv_fold # adaptive validation size determinant (account for potential large drop in No. FOVs)
# 13 in this case
random.seed(0)
custom_cv = []
cv_fovs = copy.deepcopy(training_fovs)
for it in range(cv_fold):
    n = min(val_size, len(cv_fovs))
    val_index = random.sample(cv_fovs, n)
    for fov in val_index:
        cv_fovs.remove(fov)
    train_index = [i for i in training_fovs if i not in val_index]
    custom_cv.append((train_index, val_index))

# resample training data
pd.options.mode.chained_assignment = None
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
        fov_subset = GS_training.loc[GS_training['fov'] == train_fov]

        GS_df = fov_subset.loc[fov_subset['type'] == 'colocalgene']
        merfish_df = fov_subset.loc[fov_subset['type'] == 'merfishonly']
        blank_df = fov_subset.loc[fov_subset['type'] == 'blank']

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
        print('Number of colocalized genes (1)s: ', len(GS_df_resampled))

        resampled_fov = pd.concat([GS_df_resampled, blank_df_resampled, merfish_df_resampled])
        resampled_GS_training = pd.concat([resampled_GS_training, resampled_fov])

    with open(os.path.join(output_dir, sampling_notes), 'a') as f:
        f.write(f'Sampling method: {opt}\n')
        f.write(np.unique(resampled_GS_training['type'], return_counts=True).__str__() + '\n\n')
        f.write(np.unique(resampled_GS_training['Label'], return_counts=True).__str__() + '\n\n')

    for fold in range(cv_fold):
        train_df = resampled_GS_training.loc[resampled_GS_training['fov'].isin(custom_cv[fold][0])]
        val_df = GS_training.loc[
            (GS_training['fov'].isin(custom_cv[fold][1])) & (GS_training['type'].isin(['colocalgene', 'merfishonly']))]

        if 'blank' in np.unique(val_df['type']):  # make sure no blanks in validation set
            print('validation has blanks!')

        # basic model parameters
        RF = RandomForestClassifier(random_state=0, verbose=0, n_jobs=-3, min_samples_split=0.001,
                                    min_impurity_decrease=0.01, max_features=1,
                                    max_depth=12, criterion='entropy', min_weight_fraction_leaf=0.175, n_estimators=300)
        RF.fit(train_df[features_to_use], train_df['Label'])
        val_pred = RF.predict(val_df[features_to_use])

        precision = precision_score(val_df['Label'], val_pred)
        recall = recall_score(val_df['Label'], val_pred)
        f1 = f1_score(val_df['Label'], val_pred)
        print('f1', f1)

        result_precision.append(precision)
        result_recall.append(recall)
        result_f1.append(f1)
        result_sampling_method.append(opt)

result_df = pd.DataFrame({'method': result_sampling_method,
                          'precision': result_precision,
                          'recall': result_recall,
                          'f1': result_f1})

result_df.to_csv(os.path.join(output_dir, output_name))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(data=result_df, x='method', y='f1', color='white', showfliers=False, ax=ax[0])
sns.swarmplot(data=result_df, x='method', y='f1', color='black', ax=ax[0])
sns.boxplot(data=result_df, x='method', y='f1', color='white', showfliers=False, ax=ax[1])
sns.swarmplot(data=result_df, x='method', y='f1', color='black', ax=ax[1])
# Increase tick label size
for a in ax:
    a.tick_params(axis='both', labelsize=14)

    # Increase x/y axis label size
    a.set_xlabel('Method', fontsize=16)
    a.set_ylabel('F1', fontsize=16)

    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'SamplingMethod_brain_set1_v3.svg'))

summary = (result_df
           .groupby(['method'])
           .agg(mean_precision=('precision', 'mean'),
                mean_recall=('recall', 'mean'),
                mean_F1=('f1', 'mean'),
                std_F1=('f1', 'std'),
                min_F1=('f1', 'min'))
           .reset_index())

import re
import string

def parse_sampling_notes(path):
    with open(path) as f:
        text = f.read()

    comp = {}
    blocks = re.split(r'Sampling method:\s*(\w+)', text)
    # blocks = ['', 'opt1', '<body1>', 'opt2', '<body2>', ...]
    for i in range(1, len(blocks), 2):
        opt  = blocks[i]
        body = blocks[i + 1]

        # first tuple in the body is the TYPE line:
        # (array(['colocalgene', 'merfishonly'], ...), array([56505, 61979], ...))
        m = re.search(r"array\(\[(.*?)\].*?array\(\[(.*?)\]", body, re.S)
        if not m:
            continue
        labels = [s.strip().strip("'\"") for s in m.group(1).split(',') if s.strip()]
        counts = [int(x) for x in re.findall(r'\d+', m.group(2))]
        d = dict(zip(labels, counts))

        comp[opt] = {'colocal': d.get('colocalgene', 0),
                     'merfish': d.get('merfishonly', 0),
                     'blank':   d.get('blank', 0)}
    return comp

comp = parse_sampling_notes(os.path.join(output_dir, sampling_notes))
comp = {'opt1': {'colocal': 59461, 'merfish': 55689, 'blank': 0},
        'opt2': {'colocal': 59461, 'merfish': 59461, 'blank': 0},
        'opt3': {'colocal': 59461, 'merfish': 59461, 'blank': 0}}

print(comp)
# order: colocalized (blue), merfish only (red), blanks (grey)

colors = {'colocal': '#5b8fd6', 'merfish': '#e64a3b', 'blank': '#b8b8b8'}
sublabels = ['no sampling',
             'downsampled\ncolocalized', 'upsampled\nmerfish only']

x = np.arange(len(comp.keys()))

fig, (ax_box, ax_bar) = plt.subplots(
    2, 1, figsize=(10, 14),
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35}
)

# ---- top: boxplot + swarm ----
sns.boxplot(data=result_df, x='method', y='f1', color='white',
            showfliers=False, width=0.5, ax=ax_box)
sns.swarmplot(data=result_df, x='method', y='f1', color='black',
              size=4, ax=ax_box)
ax_box.tick_params(axis='both', labelsize=20)
ax_box.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax_box.set_xlabel('')

# mean labels above each box
means = result_df.groupby('method')['f1'].mean().reindex(comp.keys())
for i, m in enumerate(means):
    # weight = 'bold' if m == means.max() else 'normal' # bold max F1
    weight = 'bold' if i == 0 else 'normal' # bold the 1st opt (no sampling)
    ax_box.text(i, ax_box.get_ylim()[1], f'{m:.3f}',
                ha='center', va='bottom', fontweight=weight, fontsize=20)

labels = list(string.ascii_uppercase[:len(means)])   # ['A', 'B', 'C', ...]
ax_box.set_xticks(range(len(means)))
ax_box.set_xticklabels(labels, fontsize=16, fontweight='bold')
# ---- bottom: stacked bars ----
bar_w = 0.09
for i, meth in enumerate(comp.keys()):
    c = comp[meth]
    total = c['colocal'] + c['merfish'] + c['blank']
    # stack order from bottom: merfish (red), blank (grey), colocal (blue)
    # matched to image: red at bottom, grey middle, blue top
    bottom = 0
    for key in ['merfish', 'blank', 'colocal']:
        h = c[key] / total * 100   # as percentage; drop the /total for raw counts
        if h > 0:
            ax_bar.bar(x[i], h, bar_w, bottom=bottom,
                       color=colors[key], edgecolor='none')
            bottom += h
    ax_bar.bar(x[i], bottom, bar_w, bottom=0,
                   fill=False, edgecolor='black', linewidth=0.8)

# --- remove the x-axis line and tick marks on the bar panel only ---
ax_bar.spines['bottom'].set_visible(False)
ax_bar.tick_params(axis='x', length=0)   # length=0 hides tick marks; pad pushes labels down
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(sublabels, fontsize=8)
ax_bar.set_yticks([])
for spine in ['top', 'right', 'left']:
    ax_bar.spines[spine].set_visible(False)
ax_bar.set_xlim(ax_box.get_xlim())   # <-- align bars under boxes

# ---- legend (colored text, top-left of bar panel) ----
legend_items = [('colocalized', colors['colocal']),
                ('merfish only', colors['merfish']),
                ('blanks', colors['blank'])]
for j, (lab, col) in enumerate(legend_items):
    ax_bar.text(-0.12, 1.0 - j * 0.14, lab, color=col,
                fontsize=7, fontweight='bold', va='top', ha='left',
                transform=ax_bar.transAxes, clip_on=False)

fig.text(0.5, 0.30, 'Training data sampling method', ha='center', fontsize=10, fontweight='bold')
ax_box.set_ylabel('')
fig.savefig(os.path.join(output_dir, 'sampling_with_composition.svg'), bbox_inches='tight')
plt.show()