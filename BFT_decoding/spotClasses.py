# -*- coding: utf-8 -*-
"""
Spot Classes
Utilities for post processing spots, validation, and QC.
Readjust thresholding parameters in accordance to validation plots. 
Post processing with adaptive thresholding, and/or with magnitude, distance, and size cutoffs
See function "run_adaptive_thresholding_analysis" for workflow
@author: Mike Huang
    
"""

import re
import os
import shutil
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import tkinter as tk
from tkinter import filedialog
from scipy.signal import convolve2d
from BFT_decoding.geneData import GeneData
from mpl_toolkits.axes_grid1 import ImageGrid
        
class FpkmData:
    def __init__(self, fpkm_path, blank_fpkm_threshold = 0):
        assert blank_fpkm_threshold >= 0, "blank_fpkm_threshold cannot be negative"
        self.df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
        self.genes_names = self.df_fpkm.genes
        self.genes_names_sorted = self.df_fpkm.sort_values(by=['FPKM'], ascending=False).genes
        self.fpkm_sorted = self.df_fpkm.sort_values(by=['FPKM'], ascending=False).FPKM
        self.blank_inds = self.df_fpkm['FPKM'] == 0
        self.blank_names = self.df_fpkm['genes'].loc[self.df_fpkm.FPKM==0]
        if len(self.df_fpkm.loc[self.blank_inds]) == 0 or blank_fpkm_threshold != 0:
            self.blank_inds = self.df_fpkm['FPKM'] < 1
            self.blank_names = self.df_fpkm['genes'].loc[self.df_fpkm.FPKM<1]
        self.num_blanks = len(self.blank_names)
        self.num_total = len(self.df_fpkm)
        self.num_genes = self.num_total - self.num_blanks
        
class SpotsData:
    def __init__(self, 
                 fpkmData: FpkmData,
                 processed_path: str, 
                 microscope_type: str, # "'Triton' or 'Dory'"
                 ):
        self.fpkmData = fpkmData
        self.processed_path = processed_path
        self.microscope_type = microscope_type
        self.zcoords = []
        self.xcoords = []
        self.ycoords = []
        self.fovs = []
        self.min_dists = []
        self.mean_intensities = []
        self.spot_sizes = []
        self.genes = []
        self.blank_fraction_scores = []
        assert microscope_type in ['Dory','Triton'], ("microscope_type must either by 'Dory' or 'Triton'.\
                                                      Enter 'Dory' for file schemas with single FOV numerical values. \
                                                      Enter 'Triton' for X and Y FOV numerical values.");
    def get_coords_filelist(self):
        coord_data_pattern = re.compile(
        r"fov_(\d+|\d+_\d+)_coord_iter(\d+).hdf5", flags=re.IGNORECASE)
        # Copy files to backup folder
        coords_filelist = []
        for file in os.listdir(self.processed_path):
            match = re.match(coord_data_pattern, file)
            if match:
                coords_filelist.append(file)
        return coords_filelist
    
    def cache_coords_files(self):
        coords_cache_path = 'coords_cache'
        coords_filelist = self.get_coords_filelist()

        if not os.path.exists(os.path.join(self.processed_path, coords_cache_path)):
            os.mkdir(os.path.join(self.processed_path, coords_cache_path))
            for file in coords_filelist:
                coords_file_path = os.path.join(self.processed_path,file)
                coords_cache = os.path.join(os.path.join(self.processed_path, coords_cache_path),file)
                if os.path.exists(coords_file_path):
                    shutil.copy(coords_file_path, coords_cache)
    
    def get_fov_from_filename(self, file: str):
        if self.microscope_type == "Triton":
            fov = '_'.join(file.split('_')[1:3])
        elif self.microscope_type == "Dory":
            fov = file.split('_')[1]
        return fov
    
    def load_spots_fishwatcher(self):
        filepath = os.path.join(self.processed_path, 'cellData.h5')
        with h5py.File(filepath, 'r') as f:
            for gene in f.keys():
                if gene!='preset':
                    spots_count = f['meanIntensity'].shape[-1]
                    self.ycoords.append(f['centroid3'][0])
                    self.xcoords.append(f['centroid3'][1])
                    self.spot_sizes.append(f['area'][0])
                    self.min_dists.append(f['pixelDist'][0])
                    self.mean_intensities.append(f['meanIntensity'][0])
                    self.genes.append([gene]*spots_count)
        self.ycoords = np.concatenate(self.ycoords)
        self.xcoords = np.concatenate(self.xcoords)
        self.spot_sizes = np.concatenate(self.spot_sizes)
        self.min_dists = np.concatenate(self.min_dists)
        self.mean_intensities = np.concatenate(self.mean_intensities)
        self.genes = np.concatenate(self.genes)
                    
        
    
    def load_spots_from_hdf5(self):
        # iterate through files and get spot metrics
        coords_filelist = self.get_coords_filelist()
        for file in coords_filelist:
            fov = self.get_fov_from_filename(file)
            print(f"Loading spot metrics from FOV {fov}")
            coords_cache = os.path.join(os.path.join(self.processed_path, "coords_cache"),file)
            with h5py.File(coords_cache,'r') as f:
                # try:
                #     assert f[list(f.keys())[0]].attrs["mean_intensity_ind"] == 5, (
                #     "mean_intensity_ind mismatch. Check index on coord output corresponding to mean_intensity."
                #     ) 
                # except:
                #     raise NameError("The 'mean_intensity' metric not in h5 coord file. Generate coord h5 files with a decodeFunctions script compatible with blank fraction thresholding.")
                for gene in f.keys():
                    for spot in f[gene]:
                        self.zcoords.append(spot[0])
                        self.ycoords.append(spot[1])
                        self.xcoords.append(spot[2])
                        self.fovs.append(fov)
                        self.spot_sizes.append(spot[3])
                        self.min_dists.append(spot[4])
                        self.mean_intensities.append(spot[5])
                        self.genes.append(gene)
        
        # Convert lists to np.array for conditional subsetting
        self.zcoords = np.array(self.zcoords)
        self.ycoords = np.array(self.ycoords)
        self.xcoords = np.array(self.xcoords)
        self.fovs = np.array(self.fovs)
        self.spot_sizes = np.array(np.clip(self.spot_sizes, 1, 2**8-1), dtype=np.uint8)
        self.min_dists = np.array(self.min_dists)
        self.mean_intensities = np.array(self.mean_intensities)
        self.genes = np.array(self.genes)

    def threshold_mask(self,
                        blank_fraction_threshold: float = None,
                        distance_threshold: float = None,
                        intensity_threshold: float = None,
                        size_threshold: int = None,                        
                        ):
        
        if not blank_fraction_threshold:
            blank_fraction_threshold = 1
        if not distance_threshold:
            distance_threshold = 1
        if not intensity_threshold:
            intensity_threshold = 0
        if not size_threshold:
            size_threshold = 1
            
        return (self.blank_fraction_scores <= blank_fraction_threshold) & \
               (self.min_dists <= distance_threshold) & \
               (self.mean_intensities >= intensity_threshold) & \
               (self.spot_sizes >= size_threshold)
               
    def save_to_thresholded_spots_hdf5(self,
                                       blank_fraction_threshold: float,
                                       distance_threshold: float,
                                       intensity_threshold: float,
                                       size_threshold: int,
                                       ):
        mask = self.threshold_mask(blank_fraction_threshold,
                                   distance_threshold,
                                   intensity_threshold,
                                   size_threshold)
        coords_filelist = self.get_coords_filelist()
        
        for file in coords_filelist:
            fov = self.get_fov_from_filename(file)
            print(f"Outputting blank fraction thresholded coords at {file}")
            coords_file_path = os.path.join(self.processed_path, file)
            
            with h5py.File(coords_file_path, 'w') as coords_file:
                fov_cond = np.logical_and(mask,self.fovs == fov)
                genes_fov = self.genes[fov_cond]
                zcoords_fov = self.zcoords[fov_cond]
                xcoords_fov = self.xcoords[fov_cond]
                ycoords_fov = self.ycoords[fov_cond]
                spot_sizes_fov = self.spot_sizes[fov_cond]
                bf_fov = self.blank_fraction_scores[fov_cond]
                min_dists_fov = self.min_dists[fov_cond]
                mean_intensities_fov = self.mean_intensities[fov_cond]
                
                for i, gene in enumerate(self.fpkmData.genes_names):
                    gene_cond = genes_fov == gene
                    zcoords_gene = zcoords_fov[gene_cond]
                    xcoords_gene = xcoords_fov[gene_cond]
                    ycoords_gene = ycoords_fov[gene_cond]
                    spot_sizes_gene = spot_sizes_fov[gene_cond]
                    bf_gene = bf_fov[gene_cond]
                    mean_intensities_gene = mean_intensities_fov[gene_cond]
                    min_dists_gene = min_dists_fov[gene_cond]
                    num_spots = xcoords_gene.shape[0]
                    list_of_spot_params = [[zcoords_gene[j],
                                            ycoords_gene[j],
                                            xcoords_gene[j],
                                            spot_sizes_gene[j],
                                            min_dists_gene[j],
                                            mean_intensities_gene[j],
                                            bf_gene[j]] for j in range(num_spots)]
                    if list_of_spot_params:
                        gene_spots_data = np.vstack(list_of_spot_params)
                        gene_dataset = coords_file.create_dataset(
                            gene, data=gene_spots_data
                        )
                    else:  # no spots found
                        gene_dataset = coords_file.create_dataset(
                            gene, shape=(0, 7)
                        )
                    dataset_attrs = {
                                    "gene_index": i,
                                    "FPKM_data": self.fpkmData.df_fpkm.FPKM[i],
                                    }
                    for attr in dataset_attrs:
                        gene_dataset.attrs[attr] = dataset_attrs[attr]
    
    def as_pandas_dataframe(self):
        df_spots = pd.DataFrame({"gene":self.genes,
                                 "FOV":self.fovs,
                                 "mean intensities":self.mean_intensities,  
                                 "min distance":self.min_dists,
                                 "centroid-z":self.zcoords,
                                 "centroid-x":self.xcoords,
                                 "centroid-y":self.ycoords,
                                 "size":self.spot_sizes})
        if self.blank_fraction_scores:
            df_spots['blank_fraction_score'] = self.blank_fraction_scores
        return df_spots

class SpotsHistogram:
    def __init__(self,
                 spotsData:SpotsData,
                 fpkmData:FpkmData):
        self.spotsData = spotsData
        self.fpkmData = fpkmData

  
    def generate_3d_histogram(self,
                           subset_blanks: bool,
                           subset_genes: bool,
                           num_bins: int,
                           ) -> np.array:

            self.distance_bins = np.linspace(0,self.spotsData.min_dists.max()+1e-2, num_bins)

            self.intensity_bins = np.logspace(-2, 1, int(num_bins))

            self.intensity_bins_log = np.log10(self.intensity_bins)
            min_size = self.spotsData.spot_sizes.min()
            self.size_bins = np.array(list(range(min_size,6))+[int(1e4)])
            if subset_genes and subset_blanks: # subset all barcodes
                mask = np.ones(self.spotsData.genes.shape[0], dtype = np.bool)
            else:
                mask = np.isin(self.spotsData.genes, 
                               self.fpkmData.blank_names,
                               invert = subset_genes)
            return np.histogramdd(np.moveaxis(
                                    np.vstack(
                                    [self.spotsData.min_dists[mask],
                                     self.spotsData.mean_intensities[mask],
                                     self.spotsData.spot_sizes[mask]]
                                    )
                                    ,0,1),
                                    bins=(self.distance_bins,self.intensity_bins,self.size_bins),
                                    density=False)[0]
                        
    def generate_blank_fraction_heatmap(self,
                                        num_bins: int,
                                        kde_sigma: float,
                                        eps: float,
                                        plot_heatmaps = True):
        
        def gaussian_kernel(amp, sigma):
            def twoD_Gaussian(xy, amplitude, xo, yo, sigma):
                (x,y) = xy
                xo = float(xo)
                yo = float(yo)
                a = 1/(2*sigma**2)
                g = amplitude*np.exp( - (a*((x-xo)**2) + a*(y-yo)**2))
                return g.ravel()
            cropdim = int(sigma*4)+1
            s = (cropdim-1)/2 #padding
            xpts = np.arange(0,cropdim)
            ypts = np.arange(0,cropdim)
            xpts, ypts = np.meshgrid(xpts, ypts)
            return twoD_Gaussian((xpts,ypts), amp, s, s, sigma).reshape(cropdim,cropdim)
    
        blank_hist = self.generate_3d_histogram(subset_blanks = True,
                                                subset_genes = False,
                                                num_bins = num_bins)
        gene_hist = self.generate_3d_histogram(subset_blanks = False,
                                                subset_genes = True,
                                                num_bins = num_bins)
        total_hist = blank_hist + gene_hist

        if kde_sigma:
            kernel = gaussian_kernel(1,kde_sigma)
            for i in range(self.size_bins.shape[0]-1):
                gene_hist[:,:,i] = convolve2d(gene_hist[:,:,i], kernel, mode='same',boundary='symmetric')
                blank_hist[:,:,i] = convolve2d(blank_hist[:,:,i], kernel, mode='same',boundary='symmetric')
                total_hist[:,:,i] = convolve2d(total_hist[:,:,i], kernel, mode='same',boundary='symmetric')
        
        blank_hist = (blank_hist + eps)/self.fpkmData.num_blanks
        total_hist = (total_hist + 2*eps)/self.fpkmData.num_total
        gene_hist = (gene_hist + eps)/self.fpkmData.num_genes
        
        blank_fraction_heatmap = blank_hist / total_hist
        
        if plot_heatmaps:
            self.plot_heatmaps(gene_hist, blank_hist, blank_fraction_heatmap,
                               self.distance_bins, self.intensity_bins,
                               savepath = os.path.join(self.spotsData.processed_path, 
                               f"qc_plots/Signal_noise_blankfraction_heatmaps_{num_bins}bins.png"))
        return np.clip(blank_fraction_heatmap,0,1)
     
    def assign_blank_fraction_scores(self, blank_fraction_heatmap):
        print("Assigning blank fraction scores to spots")

        for i in range(self.spotsData.min_dists.shape[0]):
            distance_bin = np.digitize(self.spotsData.min_dists[i],self.distance_bins)-1
            intensity_bin = np.digitize(np.clip(self.spotsData.mean_intensities[i],0,1e1-1e-5), self.intensity_bins)-1
            size_bin = np.digitize(self.spotsData.spot_sizes[i], self.size_bins)-1
            self.spotsData.blank_fraction_scores.append(blank_fraction_heatmap[distance_bin][intensity_bin][size_bin])
        self.spotsData.blank_fraction_scores = np.array(self.spotsData.blank_fraction_scores)
        self.spotsData.blank_fraction_scores[np.isnan(self.spotsData.blank_fraction_scores)] = 1
        print("Spot assignment complete")


    def plot_heatmaps(self,
                      gene_pdf,
                      blank_pdf,
                      blank_fraction,
                      xbins,
                      ybins,
                      savepath=None):
        plt.style.use('seaborn')
        num_size_bins = self.size_bins.shape[0]
        plt.rcParams['figure.figsize'] = [4*(num_size_bins-1),8]
        fig, ax = plt.subplots(3, num_size_bins-1)
        vmax = gene_pdf.max()
        
        blank_fraction_vmax = 1

        for i in range(num_size_bins-1):
            im = ax[0,i].pcolormesh(xbins, ybins, gene_pdf[:,:,i].T, norm=mpl.colors.LogNorm(vmax=vmax),
                                    cmap='jet')
            cb = plt.colorbar(im, ax=ax[0,i], fraction=0.046, pad=0.04)
            if i == num_size_bins-2:
                cb.set_label(label="gene barcode count")
            if blank_pdf[:,:,i].sum() != 0: #check if there is a blank in those size bins
                im = ax[1,i].pcolormesh(xbins, ybins, blank_pdf[:,:,i].T, norm=mpl.colors.LogNorm(vmax=vmax),
                                       cmap='jet')
                cb = plt.colorbar(im, ax=ax[1,i], fraction=0.046, pad=0.04)
            else:
                im = ax[1,i].pcolormesh(xbins, ybins, blank_pdf[:,:,i].T, cmap='jet')
                cb = plt.colorbar(im, ax=ax[1,i], fraction=0.046, pad=0.04)
                
            
            if i == num_size_bins-2:
                cb.set_label(label="blank barcode count")
            im = ax[2,i].pcolormesh(xbins, ybins, blank_fraction[:,:,i].T, vmin=0, vmax=blank_fraction_vmax, cmap='jet')
            cb = plt.colorbar(im, ax=ax[2,i], fraction=0.046, pad=0.04)
            if i == num_size_bins-2:
                cb.set_label(label="blank fraction")
    
            ax[0,i].set_xlim([xbins[0],xbins[-1]])
            ax[0,i].set_ylim([1e-2,1e1])
            ax[0,i].set_yscale('log')
            ax[1,i].set_xlim([xbins[0],xbins[-1]])
            ax[1,i].set_ylim([1e-2,1e1])
            ax[1,i].set_yscale('log')
            ax[2,i].set_xlim([xbins[0],xbins[-1]])
            ax[2,i].set_ylim([1e-2,1e1])
            ax[2,i].set_yscale('log')
            ax[2,i].set_xlabel("Min distance")
            if i == num_size_bins-2:
                ax[0,i].set_title(f"spot size >= {self.size_bins[i]}")
            else:
                ax[0,i].set_title(f"spot size = {self.size_bins[i]}")
            if i == 0:
                ax[0,0].set_ylabel('Mean intensity (log10)')
                ax[1,0].set_ylabel('Mean intensity (log10)')
                ax[2,0].set_ylabel('Mean intensity (log10)')
            plt.tight_layout()
        if not savepath:
            plt.show()
        else:
            plt.savefig(savepath)
        plt.close()
 
class SpotInspector:
    def __init__(self,
                 spotsData: SpotsData,
                 geneData: GeneData,
                 ):
        self.spotsData = spotsData
        self.geneData = geneData
    
    def load_fov(self, fov):
        def zero_borders(imgs, shifts, pad=10):
            left_border = np.clip(shifts[:,1].max(),0,None)
            right_border = np.clip(shifts[:,1].min(),None,0)
            bottom_border = np.clip(shifts[:,0].max(),0,None)
            top_border = np.clip(shifts[:,0].min(),None,0)

            if top_border != 0:
                imgs[:,top_border:] = 0
            if bottom_border != 0:
                imgs[:,:bottom_border] = 0
            if left_border != 0:
                imgs[:,:,:left_border] = 0
            if right_border != 0:
                imgs[:,:,right_border:] = 0
            return imgs
        file_path = os.path.join(self.spotsData.processed_path, f"FOV_{fov}_imagedata_iter0.hdf5")
        with h5py.File(file_path, 'r') as f:
            shifts = f.attrs['registration_shifts'].astype(np.int16)
            imgs = zero_borders(np.moveaxis(f['filtered_clipped'][0],-1,0),shifts)

        for bit in range(self.geneData.num_bits):
            imgs[bit] /= np.percentile(imgs[bit], 99.9)
        
        return imgs

    def visualize_spot(self, images, oninds, y, x, crop_pad=5, plot_type='all'):
        #plt.style.use('ggplot')
        images = np.moveaxis(images,0,-1)
        codelen = images.shape[-1]
        s = int(crop_pad)
        y = int(y)
        x = int(x)
        img_crop = images[y-s+1:y+s,x-s+1:x+s]
        max_intensity = np.percentile(img_crop,99)
        plt.rcParams["axes.grid"] = False
        fig = plt.figure(1,(10,10))
        if plot_type == 'all':
            grid = ImageGrid(fig, 111, nrows_ncols = (4,4))
            for i in range(codelen):
                grid[i].imshow(img_crop[:,:,i], vmin=0, vmax=max_intensity,cmap='viridis',interpolation='none')
                if i in oninds:
                    grid[i].set_title(f"ON {i+1}", color='r')
                else:
                    grid[i].set_title(f"{i+1}", color='r')
        if plot_type == 'onbits':
            grid = ImageGrid(fig, 111, nrows_ncols = (1, len(oninds)))
            ind=0
            for i in range(codelen):
                if i in oninds:
                    grid[ind].imshow(img_crop[:,:,i], vmin=0, vmax=max_intensity,cmap='viridis',interpolation='none')
                    grid[ind].set_title(f"{i}", color='r')
                    ind+=1
        plt.show()
        
    def oninds(self,
               gene):
        code = self.geneData.codebook_data.loc[gene].values[:self.geneData.num_bits].astype(int)
        return [i for i in range(self.geneData.num_bits) if code[i]==1]

    def plot_gene(self, 
                  gene, 
                  fov, 
                  num_spots,
                  blank_fraction_min = 0,
                  plot_type = 'onbits',
                  ):
        
        imgs = self.load_fov(fov)
        subset_cond = (self.spotsData.genes==gene) & (self.spotsData.fovs==fov)
        
        blank_fraction_scores = self.spotsData.blank_fraction_scores[subset_cond]
        bf_argsort = blank_fraction_scores.argsort()
        xcoords = self.spotsData.xcoords[subset_cond][bf_argsort]
        ycoords = self.spotsData.ycoords[subset_cond][bf_argsort]
        blank_fraction_scores = blank_fraction_scores[bf_argsort]
        xcoords = xcoords[blank_fraction_scores >= blank_fraction_min]
        ycoords = ycoords[blank_fraction_scores >= blank_fraction_min]
        blank_fraction_scores = blank_fraction_scores[blank_fraction_scores >= blank_fraction_min]
        codebook = self.geneData.codebook_data

        for i in range(num_spots):
            print(ycoords[i], xcoords[i])
            self.visualize_spot(imgs,
                           self.oninds(gene),
                           ycoords[i],
                           xcoords[i],
                           plot_type=plot_type
                           )       

class ValidationMetrics:
    def __init__(self,
                 spotsData: SpotsData,
                 fpkmData: FpkmData,
                 ):
        self.spotsData = spotsData
        self.fpkmData = fpkmData
        #self.qc_path = spotsData.processed_path
        self.qc_path = os.path.join(spotsData.processed_path, 'qc_plots')
        
    def get_rna_counts(self, genesarray: np.array):
        '''
        Get RNA Counts from array of called out gene identities in the order 
        of the genes in the codebook
        '''
        genes_codebook = self.fpkmData.df_fpkm.genes.values
        rna_counts_df = pd.Series(genesarray).value_counts()
        rna_counts = []
        for rna in genes_codebook:
            if rna in rna_counts_df.index:
                rna_counts.append(rna_counts_df[rna])
            else:
                rna_counts.append(0)
        return np.array(rna_counts)
    
    def fpkm_log_correlation(self,
                             rna_counts: np.array,
                             blanks_threshold=0, 
                             log_method = 'mask', #options 'mask', 'symlog', 'eps'
                             ):
        '''
        FPKM correlation plot
        RNA_counts: vector of integers with the counts for each RNA species
        FPKM: vector of floats with the FPKM values for each RNA species
        Prints FPKM correlation plot with pearson regression score and callout number.
        '''
        assert log_method in ['mask', 'symlog', 'eps'], "Method must be 'mask',  'symlog', or 'eps'"
        FPKM = self.fpkmData.df_fpkm.FPKM.values
        
        if log_method == 'mask':
            rna_counts_log = np.log10(rna_counts[(rna_counts>0) & (FPKM>0)])
            fpkm_log = np.log10(FPKM[(rna_counts>0) & (FPKM>0)])
            
        elif log_method == 'symlog':
            rna_counts_log = rna_counts.copy()
            rna_counts_log[rna_counts>0] = np.log10(rna_counts[rna_counts>0])
            fpkm_log = FPKM.copy()
            fpkm_log[fpkm_log>1] = np.log10(fpkm_log[fpkm_log>1])
            
        elif log_method == 'eps':
            rna_counts_log = np.log10(rna_counts+1)
            fpkm_log = np.log10(FPKM+0.01)
        return pearsonr(rna_counts_log, fpkm_log)[0]
    
    def misidentification_rate(self, rna_counts):
        mean_blank_barcode_count = rna_counts[self.fpkmData.blank_inds].mean()
        mean_barcode_count = rna_counts.mean()
        return np.round(mean_blank_barcode_count/mean_barcode_count,3)

    def roc_data(self,
                 roc_param_range: np.array,
                 roc_param_type: str = 'blank_fraction',
                 blank_fraction_threshold: float = 1,
                 distance_threshold: float = 1,
                 intensity_threshold: float = 0,
                 size_threshold: int = 1,
                 ):
        '''
        Receiver operating characteristic curve for FPKM vs callout,
        and misidentification rate vs callouts
        Input threshold params and single value to vary
        Get masked genes array
        Get FPKM and callouts with masked genes array
        
        Inputs
        -------
        roc_param_range: np.array - specify param range 
        roc_param_type: str - param to vary. Choices, 'blank_fraction', 'distance',
                        'intensity', 'size'
        blank_fraction_threshold: float - select fixed threshold if param is not roc_param_type
        distance_threshold: float - select fixed threshold if param is not roc_param_type
        intensity_threshold: float- select fixed threshold if param is not roc_param_type
        size_threshold: int - select fixed threshold if param is not roc_param_type
        
        Returns
        -------
        fpkm_corr: list -> fpkm correlation at each threshold
        misid_rate: list -> misidentification rate at each threshold
        spot_counts: list -> number of callouts at each threshold
        '''
        assert roc_param_type in ['blank_fraction', 'distance', 'intensity', 'size'], \
            "Error: roc_param_type must be either 'blank_raction', 'distance', \
                'intensity', 'size'"
                
        print(f"Calculating FPKM correlation, misidentification rate, and callouts for {roc_param_type}")
        fpkm_corr = []
        spot_counts = []
        misid_rate = []
        
        for threshold in roc_param_range:
            if roc_param_type == 'blank_fraction':
                blank_fraction_threshold = threshold
            elif roc_param_type == 'distance':
                distance_threshold = threshold
            elif roc_param_type == 'intensity':
                intensity_threshold = threshold
            elif roc_param_type == 'size':
                size_threshold = threshold
            threshold_mask = self.spotsData.threshold_mask(blank_fraction_threshold,
                                          distance_threshold,
                                          intensity_threshold,
                                          size_threshold)
            genes_selected = self.spotsData.genes[threshold_mask]
            rna_counts = self.get_rna_counts(genes_selected)
            fpkm_corr.append(self.fpkm_log_correlation(rna_counts))
            misid_rate.append(self.misidentification_rate(rna_counts))
            spot_counts.append(genes_selected.shape[0])
        return fpkm_corr, misid_rate, spot_counts
        
        
    def default_roc_series(self, 
                           blank_fraction_threshold_range: np.array,
                           intensity_threshold_range: np.array,
                           hamming_weight: int = 4,
                           size_threshold: int = 2,
                           ):
        
        assert hamming_weight in [2,4], "Hamming weight of 2 or 4 is supported"
            
        roc_data_set = []
        roc_labels = []
        # blank fraction ROC
        roc_data_set.append(self.roc_data(blank_fraction_threshold_range))
        roc_labels.append("Blank fraction threshold")
        
        # intensity ROC at distance thresholds
        def normalized_euclidean_dist(a,b):
            a_norm = np.array(a)/np.linalg.norm(a)
            b_norm = np.array(b)/np.linalg.norm(b)
            return np.linalg.norm(a_norm - b_norm)
       
        if hamming_weight == 2:
            zero_to_one_flip = normalized_euclidean_dist([1,1,1],[1,1,0])
            one_to_zero_flip = normalized_euclidean_dist([1,1],[1,0])
            distance_thresholds = [zero_to_one_flip, one_to_zero_flip]
        elif hamming_weight == 4:
            zero_to_one_flip = normalized_euclidean_dist([1,1,1,1,1],[1,1,1,1,0])
            one_to_zero_flip = normalized_euclidean_dist([1,1,1,1],[1,1,1,0])
            zero_to_one_flip_x2 = normalized_euclidean_dist([1,1,1,1,1,1],[1,1,1,1,0,0])
            distance_thresholds = [zero_to_one_flip, one_to_zero_flip, zero_to_one_flip_x2]
        
        for distance_threshold in distance_thresholds:
            roc_data_set.append(self.roc_data(roc_param_range = intensity_threshold_range,
                                              roc_param_type = 'intensity',
                                              distance_threshold = distance_threshold,
                                              size_threshold = size_threshold))
            roc_labels.append(f"Intensity thresh, distance thresh = {np.round(distance_threshold,3)}")
        return roc_data_set, roc_labels
    
    def plot_roc_series(self, 
                        roc_data_set: list,
                        roc_labels: list,
                        show_plot = True,
                        ):
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [5, 10]
        fig, ax = plt.subplots(3)
        
        for data, label in zip(roc_data_set, roc_labels):
            fpkm_corr = data[0]
            misid_rate = data[1]
            spot_counts = data[2]
            
            ax[0].plot(spot_counts, fpkm_corr, marker='.', label = label)
            ax[0].set_xlabel("Callouts")
            ax[0].set_ylabel("FPKM Correlation")
            ax[1].plot(misid_rate, fpkm_corr, marker='.')
            ax[1].set_xlabel("Misidentification Rate")
            ax[1].set_ylabel("FPKM Correlation")
            ax[2].plot(spot_counts, misid_rate, marker='.')
            ax[2].set_ylabel("Callouts")
            ax[2].set_ylabel("Misidentification Rate")
        ax[0].legend()
        
        plt.show(block=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.qc_path, "roc_plots.png"), dpi=100)
        plt.close()

        
    def plot_threshold_vs_validation_metrics(self,
                                     threshold_range,
                                     roc_data,
                                     threshold_label,
                                     show_plot = False):
        
        fpkm_cor, misid_rate, callouts = roc_data
        plt.rcParams['figure.figsize'] = [5,10]
        fig, ax = plt.subplots(3)
        ax[0].plot(threshold_range, fpkm_cor, marker='.')
        ax[0].set_xlabel(threshold_label)
        ax[0].set_ylabel("FPKM Correlation")
        ax[1].plot(threshold_range, misid_rate, marker='.')
        ax[1].set_xlabel(threshold_label)
        ax[1].set_ylabel("Misidentification rate")
        ax[2].plot(threshold_range, callouts, marker='.')
        ax[2].set_xlabel(threshold_label)
        ax[2].set_ylabel("Callouts")
        
        plt.show(block=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.qc_path, f"{threshold_label}_vs_validation_metrics.png"), dpi=100)
        plt.close()

        
    def fpkm_correlation_plot(self,
                              rna_counts, 
                              fig_path = None):
        fpkm = self.fpkmData.df_fpkm.FPKM.values
        fpkm_correlation = self.fpkm_log_correlation(rna_counts)
        misid_rt = self.misidentification_rate(rna_counts)
        blankinds = self.fpkmData.blank_inds
        geneinds = ~blankinds
        plt.rcParams['figure.figsize'] = [5,10]
        fig, ax = plt.subplots(2)
        ax[0].scatter(fpkm[geneinds], rna_counts[geneinds])
        ax[0].scatter(fpkm[blankinds], rna_counts[blankinds], color='r')
        ax[0].set_xscale("symlog")
        ax[0].set_yscale("symlog")
        ax[0].set_xlim(left=0)
        ax[0].set_ylim(bottom=min(rna_counts))
        ax[0].set_xlabel("FPKM")
        ax[0].set_ylabel("RNA Count")
        ax[1].set_title(f"FPKM Corr: {fpkm_correlation} Callouts: {sum(rna_counts)} Misid rate: {misid_rt}")
        ax[1].bar(np.arange(fpkm[geneinds].shape[0]), -np.sort(-rna_counts[geneinds]))
        ax[1].bar(np.arange(fpkm[geneinds].shape[0],fpkm.shape[0],1), -np.sort(-rna_counts[blankinds]))
        ax[1].set_yscale("symlog")
        ax[1].set_xlabel("RNA")
        ax[1].set_ylabel("RNA Count")
        if fig_path:
            plt.savefig(fig_path)
        else:
            plt.show()


    def blank_fraction_at_misid_rate(self,
                                     misid_target,
                                     blank_fraction_threshold_range,
                                     roc_data_,
                                     ):
        fpkm_cor, misid_rate, callouts = roc_data_
        misid_rate = np.array(misid_rate)
        ind_at_threshold = misid_rate[misid_rate < misid_target].shape[0]      
        blank_fraction_threshold = blank_fraction_threshold_range[ind_at_threshold-1]
        print(f"At misidentification rate target = {misid_target},")
        print(f"Blank fraction threshold = {np.round(blank_fraction_threshold,2)}")
        print(f"FPKM Correlation = {np.round(fpkm_cor[ind_at_threshold-1],2)}")
        print(f"Callouts = {callouts[ind_at_threshold-1]}")
        return blank_fraction_threshold
        
    def blank_fraction_pdf_by_gene(self,
                                   upper_boundary = 0.4):
        print("Computing blank_fraction probability densities by gene")
        print(f"PDF upper boundary = {upper_boundary}")
        gene_blank_fraction_pdf = []
        pdf_bins = np.arange(0, upper_boundary, 0.01)
        blank_fraction_subset = self.spotsData.blank_fraction_scores[self.spotsData.blank_fraction_scores < upper_boundary]
        genes_subset = self.spotsData.genes[self.spotsData.blank_fraction_scores < upper_boundary]
        
        for gene in self.fpkmData.genes_names_sorted:
            genes_mask = genes_subset == gene
            dat = blank_fraction_subset[genes_mask]
            pdf_plot = plt.hist(dat, bins=pdf_bins)
            gene_blank_fraction_pdf.append(pdf_plot[0]/pdf_plot[0].sum())
        return gene_blank_fraction_pdf
    
    def pearson_skewness(self, dat):
        median = np.percentile(dat, 50)
        return 3*(dat.mean()-median)/dat.std()
    
    def pearson_skewness_list(self,
                              upper_boundary = 0.4):
        skewness_list = []
        blank_fraction_subset = self.spotsData.blank_fraction_scores[self.spotsData.blank_fraction_scores < upper_boundary]
        genes_subset = self.spotsData.genes[self.spotsData.blank_fraction_scores < upper_boundary]
        
        for gene in self.fpkmData.genes_names_sorted:
            genes_mask = genes_subset == gene
            dat = blank_fraction_subset[genes_mask]
            if dat.shape[0] > 0:
                skewness_list.append(self.pearson_skewness(dat))
            else:
                skewness_list.append(0)
        return skewness_list
    
    def plot_fpkm_skewness(self,
                           skewness_list,
                           color_label = None,
                           color_legend = None,
                           show_plot = False):
        color_order = ['b','r']
        skewness_list = np.array(skewness_list)
        plt.rcParams['figure.figsize'] = [4,4]
        if color_label is not None:
            for i, label in enumerate(set(color_label)):
                plt.scatter(self.fpkmData.fpkm_sorted[color_label==label], 
                            skewness_list[color_label==label], 
                            alpha=0.3,
                            c=color_order[i],
                            label=color_legend[i])
            plt.legend()
        else:
            plt.scatter(self.fpkmData.fpkm_sorted, skewness_list, alpha=0.3)
        plt.xscale('symlog')
        plt.xlabel('FPKM Value')
        plt.ylabel('Pearson Skewness')
        
        plt.show()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.qc_path, "fpkm_vs_skewness.png"), dpi=100)
        plt.close()
        
    def plot_rna_counts_skewness(self,
                                 skewness_list,
                                 blank_fraction_threshold = 0.4,
                                 color_label = None,
                                 color_legend = None,
                                 show_plot = False,
                                 ):
        color_order = ['b','r']
        skewness_list = np.array(skewness_list)
        rna_counts = self.get_rna_counts(self.spotsData.genes[self.spotsData.threshold_mask(blank_fraction_threshold)])
        rna_counts_sorted = rna_counts[self.fpkmData.fpkm_sorted.index]
        plt.figure()
        plt.rcParams['figure.figsize'] = [4,4]
        if color_label is not None:
            for i, label in enumerate(set(color_label)):
                plt.scatter(rna_counts_sorted[color_label==label],
                            skewness_list[color_label==label],
                            alpha=0.3,
                            c=color_order[i],
                            label=color_legend[i])
            plt.legend()
        else:
            plt.scatter(rna_counts_sorted, skewness_list, alpha=0.3)
        plt.xscale('symlog')
        plt.xlabel('RNA Counts')
        plt.ylabel('Pearson Skewness')        
        plt.show()
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.qc_path, "RNA_counts_vs_skewness.png"), dpi=100)
        plt.close()
    
    def js_divergence_distance_matrix(self,
                                      pdf_list: list):
        print("Calculating pairwise JS Divergence distance matrix")
        from scipy.spatial.distance import jensenshannon
        n_genes = self.fpkmData.num_total
        dist_matrix = np.ndarray((n_genes,n_genes))
        for i in range(n_genes):
            for j in range(n_genes):
                dist_matrix[i,j] = jensenshannon(pdf_list[i],pdf_list[j])
        return dist_matrix
          
    def cluster_distance_matrix(self,
                                dist_matrix,
                                linkage="complete"
                                ):
        from sklearn.cluster import AgglomerativeClustering
        num_blanks = self.fpkmData.num_blanks
        clustering = AgglomerativeClustering(linkage=linkage, affinity="precomputed").fit(dist_matrix[:-num_blanks,:-num_blanks])
        cluster_labels = np.concatenate([clustering.labels_,[clustering.labels_.max()+1]*num_blanks])
        print("Cluster labels")
        print(cluster_labels)
        return cluster_labels
        
        
    def cluster_genes_by_blank_fraction_pdf_(self):
        gene_blank_fraction_pdf = self.blank_fraction_pdf_by_gene()
        dist_matrix = self.js_divergence_distance_matrix(gene_blank_fraction_pdf)
        cluster_labels = self.cluster_distance_matrix(dist_matrix)
        return cluster_labels
        
        
    def plot_blank_fraction_violin(self,
                                   cluster_labels: np.array = None,
                                   genes_per_row = 20,
                                   upper_boundary = 0.4,
                                   show_plot = False,
                                   sort_by = 'rna_counts', # 'rna_counts' or 'fpkm'
                                   ):
        import seaborn as sns
        num_total = self.fpkmData.num_total
        idx = np.arange(0,num_total+genes_per_row, genes_per_row)
        num_subplots = idx.shape[0]-1
        if cluster_labels is not None:
            colors = mpl.cm.get_cmap('Set1')
            palette = np.array([colors.colors[0]]*num_total)
            for cluster in range(1, cluster_labels.max()+1):
                palette[cluster_labels==cluster] = colors.colors[cluster]
        if sort_by != 'fpkm':
            rna_counts = self.get_rna_counts(self.spotsData.genes[self.spotsData.threshold_mask(upper_boundary)])
        plt.rcParams['figure.figsize'] = [20,3*num_subplots]
        fig, ax = plt.subplots(num_subplots)
        for i in range(idx.shape[0]-1):
            if sort_by == 'fpkm':
                geneslist = self.fpkmData.genes_names_sorted[idx[i]:idx[i+1]]
            else:
                geneslist = self.fpkmData.genes_names[rna_counts.argsort()[::-1]][idx[i]:idx[i+1]]
            if cluster_labels is not None:  
                palette_ = palette[idx[i]:idx[i+1]]
            else:
                palette_ = None
            geneidx = np.in1d(self.spotsData.genes, geneslist)
            df_spots_selectgenes = pd.DataFrame({'gene': self.spotsData.genes[geneidx],
                                                 'blank fraction score': self.spotsData.blank_fraction_scores[geneidx]})
            
            sns.violinplot(ax=ax[i], x="gene", y="blank fraction score",
                           data=df_spots_selectgenes.loc[df_spots_selectgenes['blank fraction score'] < upper_boundary],
                           order = geneslist, palette = palette_)
            ax[i].set_ylim([0,upper_boundary])
        if show_plot:
            plt.show()
        else:
            fig.tight_layout()
            fig.savefig(os.path.join(self.qc_path, "blank_fraction_distributions_by_gene.png"), dpi=100)
            #plt.close()

    def blank_fraction_at_misid_rate(self,
                                     misid_target,
                                     blank_fraction_threshold_range,
                                     roc_data_,
                                     ):
        fpkm_cor, misid_rate, callouts = roc_data_
        misid_rate = np.array(misid_rate)
        ind_at_threshold = misid_rate[misid_rate < misid_target].shape[0]      
        blank_fraction_threshold = blank_fraction_threshold_range[ind_at_threshold-1]
        print(f"At misidentification rate target = {misid_target},")
        print(f"Blank fraction threshold = {np.round(blank_fraction_threshold,2)}")
        print(f"FPKM Correlation = {np.round(fpkm_cor[ind_at_threshold-1],2)}")
        print(f"Callouts = {callouts[ind_at_threshold-1]}")
        return blank_fraction_threshold
        
    def blank_fraction_pdf_by_gene(self,
                                   upper_boundary = 0.4):
        print("Computing blank_fraction probability densities by gene")
        print(f"PDF upper boundary = {upper_boundary}")
        gene_blank_fraction_pdf = []
        pdf_bins = np.arange(0, upper_boundary, 0.01)
        blank_fraction_subset = self.spotsData.blank_fraction_scores[self.spotsData.blank_fraction_scores < upper_boundary]
        genes_subset = self.spotsData.genes[self.spotsData.blank_fraction_scores < upper_boundary]
        
        for gene in self.fpkmData.genes_names_sorted:
            genes_mask = genes_subset == gene
            dat = blank_fraction_subset[genes_mask]
            pdf_plot = plt.hist(dat, bins=pdf_bins)
            gene_blank_fraction_pdf.append(pdf_plot[0]/pdf_plot[0].sum())
        return gene_blank_fraction_pdf
    
    def pearson_skewness(self, dat):
        median = np.percentile(dat, 50)
        return 3*(dat.mean()-median)/dat.std()
    
    def pearson_skewness_list(self,
                              upper_boundary = 0.4):
        skewness_list = []
        blank_fraction_subset = self.spotsData.blank_fraction_scores[self.spotsData.blank_fraction_scores < upper_boundary]
        genes_subset = self.spotsData.genes[self.spotsData.blank_fraction_scores < upper_boundary]
        
        for gene in self.fpkmData.genes_names_sorted:
            genes_mask = genes_subset == gene
            dat = blank_fraction_subset[genes_mask]
            if dat.shape[0] > 0:
                skewness_list.append(self.pearson_skewness(dat))
            else:
                skewness_list.append(0)
        return skewness_list
    
    def plot_fpkm_skewness(self, 
                           skewness_list,
                           show_plot = False):
        plt.rcParams['figure.figsize'] = [4,4]
        plt.scatter(self.fpkmData.fpkm_sorted, skewness_list, alpha=0.3)
        plt.xscale('symlog')
        plt.xlabel('FPKM Value')
        plt.ylabel('Pearson Skewness')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.qc_path, "fpkm_vs_skewness.png"), dpi=100)
        plt.close()
        
    def plot_rna_counts_skewness(self,
                                 skewness_list,
                                 blank_fraction_threshold = 0.4,
                                 show_plot = False,
                                 ):
        rna_counts = self.get_rna_counts(self.spotsData.genes[self.spotsData.threshold_mask(blank_fraction_threshold)])
        rna_counts_sorted = rna_counts[self.fpkmData.fpkm_sorted.index]
        plt.rcParams['figure.figsize'] = [4,4]
        plt.scatter(rna_counts_sorted, skewness_list, alpha=0.3)
        plt.xscale('symlog')
        plt.xlabel('RNA Counts')
        plt.ylabel('Pearson Skewness')
       
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(self.qc_path, "RNA_counts_vs_skewness.png"), dpi=100)
        plt.close()
    
    def js_divergence_distance_matrix(self,
                                      pdf_list: list):
        print("Calculating pairwise JS Divergence distance matrix")
        from scipy.spatial.distance import jensenshannon
        n_genes = self.fpkmData.num_total
        dist_matrix = np.ndarray((n_genes,n_genes))
        for i in range(n_genes):
            for j in range(n_genes):
                dist_matrix[i,j] = jensenshannon(pdf_list[i],pdf_list[j])
        return dist_matrix
          
    def cluster_distance_matrix(self,
                                dist_matrix,
                                linkage="complete"
                                ):
        from sklearn.cluster import AgglomerativeClustering
        num_blanks = self.fpkmData.num_blanks
        clustering = AgglomerativeClustering(linkage=linkage, affinity="precomputed").fit(dist_matrix[:-num_blanks,:-num_blanks])
        cluster_labels = np.concatenate([clustering.labels_,[clustering.labels_.max()+1]*num_blanks])
        print("Cluster labels")
        print(cluster_labels)
        return cluster_labels
        
        
    def cluster_genes_by_blank_fraction_pdf_(self):
        gene_blank_fraction_pdf = self.blank_fraction_pdf_by_gene()
        dist_matrix = self.js_divergence_distance_matrix(gene_blank_fraction_pdf)
        cluster_labels = self.cluster_distance_matrix(dist_matrix)
        return cluster_labels
        
        
    def plot_blank_fraction_violin(self,
                                   cluster_labels: np.array = None,
                                   genes_per_row = 20,
                                   upper_boundary = 0.4,
                                   show_plot = False,
                                   ):
        import seaborn as sns
        num_total = self.fpkmData.num_total
        idx = np.arange(0,num_total+genes_per_row, genes_per_row)
        num_subplots = idx.shape[0]-1
        if cluster_labels is not None:
            colors = mpl.cm.get_cmap('Set1')
            palette = np.array([colors.colors[0]]*num_total)
            for cluster in range(1, cluster_labels.max()+1):
                palette[cluster_labels==cluster] = colors.colors[cluster]
        plt.rcParams['figure.figsize'] = [20,3*num_subplots]
        fig, ax = plt.subplots(num_subplots)
        for i in range(idx.shape[0]-1):
            geneslist = self.fpkmData.genes_names_sorted[idx[i]:idx[i+1]]
            if cluster_labels is not None:  
                palette_ = palette[idx[i]:idx[i+1]]
            else:
                palette_ = None
            geneidx = np.in1d(self.spotsData.genes, geneslist)
            df_spots_selectgenes = pd.DataFrame({'gene': self.spotsData.genes[geneidx],
                                                 'blank fraction score': self.spotsData.blank_fraction_scores[geneidx]})
            
            sns.violinplot(ax=ax[i], x="gene", y="blank fraction score",
                           data=df_spots_selectgenes.loc[df_spots_selectgenes['blank fraction score'] < upper_boundary],
                           order = geneslist, palette = palette_)
            ax[i].set_ylim([0,upper_boundary])
        if show_plot:
            plt.show()
        else:
            fig.tight_layout()
            fig.savefig(os.path.join(self.qc_path, "blank_fraction_distributions_by_gene.png"), dpi=100)
            plt.close()
        

def run_blank_fraction_analysis(processed_path, 
                                fpkm_path,
                                microscope_type,
                                hamming_weight,
                                num_bins,
                                eps,
                                kde_sigma,
                                misid_target,
                                blank_fraction_threshold,
                                distance_threshold,
                                intensity_threshold,
                                size_threshold):
    # Load FPKM Data
    fpkmData = FpkmData(fpkm_path)
    
    # Load spots data from coord hdf5 files
    spotsData = SpotsData(fpkmData, processed_path, microscope_type = microscope_type)
    spotsData.cache_coords_files()
    spotsData.load_spots_from_hdf5()

    
    # Generate blank fraction heatmap
    spotsHist = SpotsHistogram(spotsData, fpkmData)
    blank_fraction_heatmap = spotsHist.generate_blank_fraction_heatmap(num_bins = num_bins,
                                                                       kde_sigma = kde_sigma,
                                                                       eps = eps)

    # Assign blank fraction scores to spots
    spotsHist.assign_blank_fraction_scores(blank_fraction_heatmap)

    # Validation with ROC Curves
    valData = ValidationMetrics(spotsData, fpkmData)
    blank_fraction_threshold_range = np.arange(spotsData.blank_fraction_scores.min()+0.01, 1, 0.01)
    intensity_threshold_range = intensity_threshold_range = np.arange(2e-1, 1.2, 2e-2)
    roc_data_set, roc_labels = valData.default_roc_series(blank_fraction_threshold_range,
                                                          intensity_threshold_range,
                                                          hamming_weight)
    valData.plot_roc_series(roc_data_set, roc_labels)
    valData.plot_threshold_vs_validation_metrics(blank_fraction_threshold_range,
                                             roc_data_set[0],
                                             "blank_fraction_threshold")

    # Get blank_fraction_threshold at misidentification rate target if not defined
    if not blank_fraction_threshold:
        blank_fraction_threshold = valData.blank_fraction_at_misid_rate(misid_target,
                                                                        blank_fraction_threshold_range,
                                                                        roc_data_set[0])

    # Filter spots by specified thresholds and save
    spotsData.save_to_thresholded_spots_hdf5(blank_fraction_threshold,
                                             distance_threshold,
                                             intensity_threshold,
                                             size_threshold)

    # Blank Fraction QC
    valData.plot_blank_fraction_violin()
    skewness_list = valData.pearson_skewness_list()
    valData.plot_fpkm_skewness(skewness_list)
    valData.plot_rna_counts_skewness(skewness_list)

    return spotsData, valData, roc_labels, roc_data_set, skewness_list

if __name__ == "__main__":
    processed_path = None
    hamming_weight = 4
    num_bins = 60
    eps = 0
    kde_sigma = 0
    microscope_type = 'Dory' #Triton or Dory
    misid_target = 0.05
    blank_fraction_threshold = None
    distance_threshold = None
    intensity_threshold = None
    size_threshold = None
    
    
    if not processed_path:
        root = tk.Tk()
        root.withdraw()
        processed_path = filedialog.askdirectory(
            title="Choose directory containing coords h5 files"
        )
        root.destroy()
        
    timestamp = '_'.join(processed_path.split('_')[-2:])
    params_path = os.path.join(processed_path,f'qc_plots/parameters__{timestamp}.txt')
    if os.path.exists(params_path):
        with open(params_path,'r') as params:
            for line in params.readlines():
                if line.split('\t')[0] == 'fpkm_filepath':
                    fpkm_path = line.split('\t')[-1].split('\n')[0]
                    break
    
    # if not os.path.exists(fpkm_path):
    #     root = tk.Tk()
    #     root.withdraw()
    #     fpkm_path = filedialog.askopenfilename(
    #         title="Open fpkm_data.tsv file"
    #     )
    #     root.destroy()

    fpkm_path =  '//172.20.29.25/Jeeranan/MIKE/dataset/20191108_JM_L39_AML_2cD/100uM_data/codebook/fpkm_data.tsv' #r'\\10.217.24.76\Jinyue\Jeeranan\MLO_FPKM.tsv'

    spotdata, valdata, roc_label, roc_data,skewness_list = run_blank_fraction_analysis(processed_path,
                                fpkm_path,
                                microscope_type,
                                hamming_weight,
                                num_bins,
                                eps,
                                kde_sigma,
                                misid_target,
                                blank_fraction_threshold,
                                distance_threshold,
                                intensity_threshold,
                                size_threshold)

    #
    # svm_file = pd.read_csv('//10.217.24.76/Jeeranan/Guidestar_stuff/SVM/SVM_results.csv')
    # c = svm_file['C'][:7]
    # counts = list(svm_file['count'][:7])
    # fpkm = list(svm_file['fpkm'][:7])
    # misid = list(svm_file['misid rate'][:7])
    #
    #
    # plt.style.use('seaborn')
    # plt.rcParams['figure.figsize'] = [5, 10]
    # fig, ax = plt.subplots(3)
    #
    # for data, label in zip(roc_data_set, roc_labels):
    #     fpkm_corr = data[0]
    #     misid_rate = data[1]
    #     spot_counts = data[2]
    #
    #     ax[0].plot(spot_counts, fpkm_corr, marker='.', label = label)
    #     ax[0].set_xlabel("Callouts")
    #     ax[0].set_ylabel("FPKM Correlation")
    #     ax[1].plot(misid_rate, fpkm_corr, marker='.')
    #     ax[1].set_xlabel("Misidentification Rate")
    #     ax[1].set_ylabel("FPKM Correlation")
    #     ax[2].plot(spot_counts, misid_rate, marker='.')
    #     ax[2].set_ylabel("Callouts")
    #     ax[2].set_ylabel("Misidentification Rate")
    # ax[0].plot(counts, fpkm, marker='.', label='SVM')
    # ax[1].plot(misid, fpkm, marker='.', label='SVM')
    # ax[2].plot(counts, misid, marker='.', label='SVM')
    # ax[0].legend()



    