# -*- coding: utf-8 -*-
"""
Merge smFISH and MERFISH coords files.

18 Jun 2020

@author: Mike Huang
"""
import h5py
import shutil
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import pandas as pd    

mer_coords_path = None
sm_coords_path = None
fpkm_path = '../Data/100uM_data/codebook/fpkm_data.tsv'

if not mer_coords_path:
        root = tk.Tk()
        root.withdraw()
        mer_coords_path= filedialog.askopenfilename(
            title="Open MERFISH coords file"
        )
        root.destroy()
        
if not sm_coords_path:
        root = tk.Tk()
        root.withdraw()
        sm_coords_path= filedialog.askopenfilename(
            title="Open smFISH coords file"
        )
        root.destroy()



output_path = os.path.dirname(sm_coords_path)

output_file = "coords_smfishmerfish_merged.hdf5"
mer_coords_path_copy = os.path.join(output_path,output_file)
shutil.copy(mer_coords_path, mer_coords_path_copy)

with h5py.File(sm_coords_path,'r') as sm_file:
    with h5py.File(mer_coords_path,'r') as mer_file:
        with h5py.File(mer_coords_path_copy,'w') as mer_filew:
            for gene in mer_file:
                mer_filew.create_dataset(gene, data = mer_file[gene])
            for gene in sm_file:
                mer_filew.create_dataset(gene, data = sm_file[gene])
                print(mer_file.keys())
            merfish_counts = [mer_file[gene].shape[0] for gene in mer_file]
            merfish_genes = list(mer_file.keys())
            smfish_counts = [sm_file[gene].shape[0] for gene in sm_file]
            smfish_genes = list(sm_file.keys())
            
print(f"Merged file saved to {mer_coords_path_copy}")


def FPKM_correlation(RNA_counts, FPKM, genes, smfish_mask, blanks_threshold=0, fig_path = None, verbose = True):
    '''
    FPKM correlation plot
    RNA_counts: vector of integers with the counts for each RNA species
    FPKM: vector of floats with the FPKM values for each RNA species
    Prints FPKM correlation plot with pearson regression score and callout number.
    '''
    smfish_mask = np.array(smfish_mask).astype(np.bool)
    FPKM = np.array(FPKM)
    RNA_counts = np.array(RNA_counts)
    blankinds = FPKM <= blanks_threshold
    geneinds = np.logical_not(blankinds)*np.logical_not(smfish_mask)
    logtotal_codecounts = np.log10(RNA_counts[(RNA_counts>0) & (FPKM>0)])
    logfpkm = np.log10(FPKM[(RNA_counts>0) & (FPKM>0)])
    num_genes = FPKM[FPKM>blanks_threshold].shape[0]
    max_blank_count = max(RNA_counts[blankinds])
    num_genes_greater_maxblank = sum([1 for count in RNA_counts if count>max_blank_count])
    confidence_score = (num_genes_greater_maxblank/num_genes)*100
    misid_rt = np.round(RNA_counts[blankinds].mean()/RNA_counts.mean(),3)
    plt.style.use('seaborn')
    if verbose:
        plt.rcParams['figure.figsize'] = [10,10]
        fig, ax = plt.subplots(2)
        ax[0].scatter(FPKM[geneinds], RNA_counts[geneinds], alpha=0.5)
        ax[0].scatter(FPKM[smfish_mask], RNA_counts[smfish_mask], color='r', alpha=0.5)
        ax[0].scatter(FPKM[blankinds], RNA_counts[blankinds], color='g', alpha=0.5)
        smfish_inds = np.arange(FPKM.shape[0])[smfish_mask]
        for i, txt in enumerate(genes[smfish_mask]):
            ax[0].annotate(txt, (FPKM[smfish_inds[i]],RNA_counts[smfish_inds[i]]))
        ax[0].legend(["MERFISH","smFISH","blanks"])
        ax[0].axhline(y=max(RNA_counts[blankinds]), color='r', linestyle='--')
        ax[0].set_xscale("symlog")
        ax[0].set_yscale("symlog")
        ax[0].set_xlim(left=0)
        ax[0].set_ylim(bottom=min(RNA_counts))
        ax[0].set_xlabel("FPKM")
        ax[0].set_ylabel("RNA Count")
        print("FPKM log10 Correlation with blanks:", pearsonr(logtotal_codecounts, logfpkm))
        print("Spots Detected:", sum(RNA_counts))
        ax[1].set_title(f"FPKM Corr: {np.round(pearsonr(logtotal_codecounts, logfpkm)[0],3)} Callouts: {sum(RNA_counts)} Misid rate: {misid_rt}")
        barinds = np.arange(FPKM.shape[0])
        barinds_sorted = np.argsort(-RNA_counts)
        ax[1].bar(barinds[geneinds[barinds_sorted]], -np.sort(-RNA_counts[geneinds]))
        ax[1].bar(barinds[smfish_mask[barinds_sorted]], -np.sort(-RNA_counts[smfish_mask]), color='r')
        ax[1].bar(barinds[blankinds[barinds_sorted]], -np.sort(-RNA_counts[blankinds]), color='g')
        ax[1].set_yscale("symlog")
        #if num_blanks>0:
        #    ax[1].axhline(y=max(RNA_counts[-num_blanks:]), color='r', linestyle='--')
        ax[1].set_xlabel("RNA")
        ax[1].set_ylabel("RNA Count")
        if fig_path:
            plt.savefig(os.path.join(fig_path,'qc_plots','smFISH+MERFISH_FPKM_Correlation.png'),dpi=300)
        else:
            plt.show()
        print("Confidence %",confidence_score)
        print("Misidentification rate", misid_rt)
    return pearsonr(logtotal_codecounts, logfpkm)[0], sum(RNA_counts), confidence_score


df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
blankinds = df_fpkm['FPKM'] == 0
blanks_str = df_fpkm['genes'].loc[df_fpkm.FPKM==0]
if len(df_fpkm.loc[blankinds]) == 0:
    blankinds = df_fpkm['FPKM'] < 1
    blanks_str = df_fpkm['genes'].loc[df_fpkm.FPKM<1]
    
fpkm_vals = [df_fpkm['FPKM'].loc[df_fpkm.genes==gene].values[0] for gene in merfish_genes+smfish_genes]
gene_vals = np.array(merfish_genes+smfish_genes)
smfish_mask = np.array([0]*len(merfish_counts)+[1]*len(smfish_counts)).astype(np.bool)
FPKM_correlation(merfish_counts+smfish_counts,fpkm_vals,gene_vals, smfish_mask,fig_path=output_path)

