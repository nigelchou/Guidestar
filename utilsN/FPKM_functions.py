import numpy as np
from scipy.stats import pearsonr

def plot_fpkm_corr(df_FPKM, all_gene_list):
    counts = []
    fpkm = []
    gene_list = []
    for gene in np.unique(all_gene_list):
        counts.append(all_gene_list.count(gene))
        row = df_FPKM.loc[df_FPKM['genes'] == gene]
        fpkm.append(float(row['FPKM']))
        gene_list.append(gene)

    FPKM = np.array(fpkm)
    RNA_counts = np.array(counts)
    correlation, p_val = calcLogCorrelation(np.array(counts), np.array(fpkm))

    #     plt.figure(figsize=(5,5))
    #     plt.scatter(FPKM, RNA_counts)
    #     for i, txt in enumerate(gene_list):
    #         plt.annotate(txt, (FPKM[i], RNA_counts[i]))
    #     plt.xscale("symlog")
    #     plt.yscale("symlog")

    #     plt.xlim(left=0)
    #     plt.ylim(bottom=min(RNA_counts))
    #     plt.xlabel("FPKM")
    #     plt.ylabel("RNA Count")
    #     plt.title("FPKM Corr:" + str(np.round(correlation,3)) + " Callouts:" + str(sum(RNA_counts)))
    #     print("Spots Detected:", sum(RNA_counts))

    return counts, gene_list, fpkm


def calcLogCorrelation(array1: np.ndarray,
                       array2: np.ndarray, ):
    """
    calculate log-correlation between 2 arrays using scipy's pearsonr
     - usually a FPKM value array and some kind of count
    returns (correlation, p_value) same as scipy's pearsonr
    """
    # print(f"array1 type = {array1.dtype}"
    # f"array2 type = {array2.dtype}")

    # mask out 0 and non-finite values of array
    combined_mask = np.logical_and(np.logical_and(np.isfinite(array1), array1 > 0),
                                   np.logical_and(np.isfinite(array2), array2 > 0))

    return pearsonr(np.log10(array1[combined_mask]), np.log10(array2[combined_mask]))

def calc_misid(counts_arr, model_callouts):
    counts_arr = np.array(counts_arr)
    fpkm_gene_list = list(np.unique(model_callouts))
    blank_ids = [i for i,j in enumerate(fpkm_gene_list) if 'Blank' in j]
    blank_counts = counts_arr[blank_ids]
    mean_blank_count = blank_counts.mean()
    gene_ids = [i for i in range(len(fpkm_gene_list)) if i not in blank_ids]
    gene_counts = counts_arr[gene_ids]
    mean_gene_count = gene_counts.mean()
    mis_id_rate = mean_blank_count / mean_gene_count
    return mis_id_rate

def find_confusion_metrics_overall(Y_test, results_dict):
    """ finds overall confusion metrics """
    TP = len(Y_test.loc[(Y_test['Labels'] == 1) & (Y_test['predicted'] == 1)])
    FP = len(Y_test.loc[(Y_test['Labels'] == 0) & (Y_test['predicted'] == 1)])
    TN = len(Y_test.loc[(Y_test['Labels'] == 0) & (Y_test['predicted'] == 0)])
    FN = len(Y_test.loc[(Y_test['Labels'] == 1) & (Y_test['predicted'] == 0)])
    results_dict['TP'].append(TP)
    results_dict['FP'].append(FP)
    results_dict['TN'].append(TN)
    results_dict['FN'].append(FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    results_dict['precision'].append(precision)
    results_dict['recall'].append(recall)
    results_dict['F1'].append(F1)
    results_dict['counts'].append(TP + FP)

def find_confusion_metrics_overall_v2(Y_test, results_dict,genelist):
    """ finds overall confusion metrics """
    
    geneonly_subset = Y_test.loc[Y_test['Genes'].isin(genelist)]
    
    TP = len(geneonly_subset.loc[(geneonly_subset['Labels'] == 1) & (geneonly_subset['predicted'] == 1)])
    FP = len(geneonly_subset.loc[(geneonly_subset['Labels'] == 0) & (geneonly_subset['predicted'] == 1)])
    TN = len(geneonly_subset.loc[(geneonly_subset['Labels'] == 0) & (geneonly_subset['predicted'] == 0)])
    FN = len(geneonly_subset.loc[(geneonly_subset['Labels'] == 1) & (geneonly_subset['predicted'] == 0)])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    results_dict['TP'].append(TP)
    results_dict['FP'].append(FP)
    results_dict['TN'].append(TN)
    results_dict['FN'].append(FN)
    results_dict['precision'].append(precision)
    results_dict['recall'].append(recall)
    results_dict['F1'].append(F1)
    results_dict['counts'].append(TP + FP)

def find_confusion_metrics_gene(Y_test, genelist, results_dict):
    """ finds confusion metrics for each guide gene """

    for gene in genelist:
        subset_data = Y_test.loc[Y_test['Genes']==gene]
        TP = len(subset_data.loc[(subset_data['Labels'] == 1) & (subset_data['predicted'] == 1)])
        FP = len(subset_data.loc[(subset_data['Labels'] == 0) & (subset_data['predicted'] == 1)])
        TN = len(subset_data.loc[(subset_data['Labels'] == 0) & (subset_data['predicted'] == 0)])
        FN = len(subset_data.loc[(subset_data['Labels'] == 1) & (subset_data['predicted'] == 0)])
        results_dict[gene]['TP'].append(TP)
        results_dict[gene]['FP'].append(FP)
        results_dict[gene]['TN'].append(TN)
        results_dict[gene]['FN'].append(FN)
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        results_dict[gene]['precision'].append(precision)
        results_dict[gene]['recall'].append(recall)
        results_dict[gene]['F1'].append(F1)
        results_dict[gene]['counts'].append(TP + FP)


def find_confusion_metrics_gene_v2(Y_test, genelist, results_dict):
    """ finds confusion metrics for each guide gene """

    for gene in genelist:
        subset_data = Y_test.loc[Y_test['Genes'] == gene]
        TP = len(subset_data.loc[(subset_data['Labels'] == 1) & (subset_data['predicted'] == 1)])
        FP = len(subset_data.loc[(subset_data['Labels'] == 0) & (subset_data['predicted'] == 1)])
        TN = len(subset_data.loc[(subset_data['Labels'] == 0) & (subset_data['predicted'] == 0)])
        FN = len(subset_data.loc[(subset_data['Labels'] == 1) & (subset_data['predicted'] == 0)])

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

        results_dict[gene]['TP'].append(TP)
        results_dict[gene]['FP'].append(FP)
        results_dict[gene]['TN'].append(TN)
        results_dict[gene]['FN'].append(FN)
        results_dict[gene]['precision'].append(precision)
        results_dict[gene]['recall'].append(recall)
        results_dict[gene]['F1'].append(F1)
        results_dict[gene]['counts'].append(TP + FP)