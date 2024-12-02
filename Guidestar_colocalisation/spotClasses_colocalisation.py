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
from scipy.signal import convolve2d
        
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
                 microscope_type: str, # "'Triton' or 'Dory'" or 'Confocal'
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
        self.onoff_ratio1 = []
        self.mindist2 = []
        self.meaninten2 = []
        self.onoff_ratio2 = []
        self.mindist3 = []
        self.meaninten3 = []
        self.onoff_ratio3 = []
        self.mindist1_2_ratio = []
        self.mindist1_3_ratio = []
        self.size_x = []
        self.size_y = []
        self.size_conn = []
        self.genes = []
        self.blank_fraction_scores = []
        assert microscope_type in ['Dory','Triton', 'Confocal'], ("microscope_type must either by 'Dory' or 'Triton'.\
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
    
    def get_fov_from_filename(self, file: str):
        if self.microscope_type == "Triton":
            fov = '_'.join(file.split('_')[1:3])
        elif self.microscope_type == "Dory" or 'Confocal':
            fov = file.split('_')[1]
        return fov
    
    def load_spots_from_hdf5(self):
        # iterate through files and get spot metrics
        coords_filelist = self.get_coords_filelist()
        for file in coords_filelist:
            fov = self.get_fov_from_filename(file)
            print(f"Loading spot metrics from FOV {fov}")
            coords_cache = os.path.join(os.path.join(self.processed_path, 'coords_cache'),file)
            with h5py.File(coords_cache,'r') as f:
                for gene in f.keys():
                    for spot in f[gene]:
                        self.zcoords.append(spot[0])
                        self.ycoords.append(spot[1])
                        self.xcoords.append(spot[2])
                        self.fovs.append(fov)
                        self.spot_sizes.append(spot[3])
                        self.min_dists.append(spot[4])
                        self.mean_intensities.append(spot[5])
                        self.onoff_ratio1.append(spot[6])
                        self.mindist2.append(spot[7])
                        self.meaninten2.append(spot[8])
                        self.onoff_ratio2.append(spot[9])
                        self.mindist3.append(spot[10])
                        self.meaninten3.append(spot[11])
                        self.onoff_ratio3.append(spot[12])
                        self.mindist1_2_ratio.append(spot[13])
                        self.mindist1_3_ratio.append(spot[14])
                        self.size_x.append(spot[15])
                        self.size_y.append(spot[16])
                        self.size_conn.append(spot[17])
                        self.genes.append(gene)

        # Convert lists to np.array for conditional subsetting
        self.zcoords = np.array(self.zcoords)
        self.ycoords = np.array(self.ycoords)
        self.xcoords = np.array(self.xcoords)
        self.fovs = np.array(self.fovs)
        self.spot_sizes = np.array(np.clip(self.spot_sizes, 1, 2**8-1), dtype=np.uint8)
        self.min_dists = np.array(self.min_dists)
        self.mean_intensities = np.array(self.mean_intensities)
        self.onoff_ratio1 = np.array(self.onoff_ratio1)
        self.mindist2 = np.array(self.mindist2)
        self.meaninten2 = np.array(self.meaninten2)
        self.onoff_ratio2 = np.array(self.onoff_ratio2)
        self.mindist3 = np.array(self.mindist3)
        self.meaninten3 = np.array(self.meaninten3)
        self.onoff_ratio3 = np.array(self.onoff_ratio3)
        self.mindist1_2_ratio = np.array(self.mindist1_2_ratio)
        self.mindist1_3_ratio = np.array(self.mindist1_3_ratio)
        self.size_x = np.array(self.size_x)
        self.size_y = np.array(self.size_y)
        self.size_conn = np.array(self.size_conn)
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
               
    def save_to_bfs_spots_hdf5(self, save_path= None): # save spot with bfs to new path

        coords_filelist = self.get_coords_filelist()
        print(coords_filelist)

        for file in coords_filelist:
            fov = self.get_fov_from_filename(file)
            print(f"Outputting blank fraction thresholded coords at {file}")
            coords_file_path = os.path.join(save_path, file)

            with h5py.File(coords_file_path, 'w') as coords_file:
                fov_cond = [self.fovs == fov]
                print(self.fovs == fov)
                genes_fov = self.genes[fov_cond]
                zcoords_fov = self.zcoords[fov_cond]
                xcoords_fov = self.xcoords[fov_cond]
                ycoords_fov = self.ycoords[fov_cond]
                spot_sizes_fov = self.spot_sizes[fov_cond]
                bf_fov = self.blank_fraction_scores[fov_cond]
                min_dists_fov = self.min_dists[fov_cond]
                mean_intensities_fov = self.mean_intensities[fov_cond]
                
                onoff_ratio1_fov = self.onoff_ratio1[fov_cond]
                mindist2_fov = self.mindist2[fov_cond]
                meaninten2_fov = self.meaninten2[fov_cond]
                onoff_ratio2_fov = self.onoff_ratio2[fov_cond]
                mindist3_fov = self.mindist3[fov_cond]
                meaninten3_fov = self.meaninten3[fov_cond]
                onoff_ratio3_fov = self.onoff_ratio3[fov_cond]
                mindist1_2_ratio_fov = self.mindist1_2_ratio[fov_cond]
                mindist1_3_ratio_fov = self.mindist1_3_ratio[fov_cond]
                
                size_x_fov = self.size_x[fov_cond]
                size_y_fov = self.size_y[fov_cond]
                size_conn_fov = self.size_conn[fov_cond]

                for i, gene in enumerate(self.fpkmData.genes_names):
                    gene_cond = genes_fov == gene
                    zcoords_gene = zcoords_fov[gene_cond]
                    xcoords_gene = xcoords_fov[gene_cond]
                    ycoords_gene = ycoords_fov[gene_cond]
                    spot_sizes_gene = spot_sizes_fov[gene_cond]
                    bf_gene = bf_fov[gene_cond]
                    mean_intensities_gene = mean_intensities_fov[gene_cond]
                    min_dists_gene = min_dists_fov[gene_cond]
                    
                    onoff_ratio1_gene = onoff_ratio1_fov[gene_cond]
                    mindist2_gene = mindist2_fov[gene_cond]
                    meaninten2_gene = meaninten2_fov[gene_cond]
                    onoff_ratio2_gene = onoff_ratio2_fov[gene_cond]
                    mindist3_gene = mindist3_fov[gene_cond]
                    meaninten3_gene = meaninten3_fov[gene_cond]
                    onoff_ratio3_gene = onoff_ratio3_fov[gene_cond]
                    mindist1_2_ratio_gene = mindist1_2_ratio_fov[gene_cond]
                    mindist1_3_ratio_gene = mindist1_3_ratio_fov[gene_cond]
                    
                    size_x_gene = size_x_fov[gene_cond]
                    size_y_gene = size_y_fov[gene_cond]
                    size_conn_gene = size_conn_fov[gene_cond]
                    
                    num_spots = xcoords_gene.shape[0]
                    list_of_spot_params = [[zcoords_gene[j],
                                            ycoords_gene[j],
                                            xcoords_gene[j],
                                            spot_sizes_gene[j],
                                            min_dists_gene[j],
                                            mean_intensities_gene[j],
                                            onoff_ratio1_gene[j],
                                            mindist2_gene[j],
                                            meaninten2_gene[j],
                                            onoff_ratio2_gene[j],
                                            mindist3_gene[j],
                                            meaninten3_gene[j],
                                            onoff_ratio3_gene[j],
                                            mindist1_2_ratio_gene[j],
                                            mindist1_3_ratio_gene[j],
                                            size_x_gene[j],
                                            size_y_gene[j],
                                            size_conn_gene[j],
                                            bf_gene[j]] for j in range(num_spots)]
                    if list_of_spot_params:
                        gene_spots_data = np.vstack(list_of_spot_params)
                        gene_dataset = coords_file.create_dataset(
                            gene, data=gene_spots_data
                        )
                    else:  # no spots found
                        gene_dataset = coords_file.create_dataset(
                            gene, shape=(0, 16)
                        )
                    dataset_attrs = {
                        "gene_index": i,
                        "FPKM_data": self.fpkmData.df_fpkm.FPKM[i],
                    }
                    for attr in dataset_attrs:
                        gene_dataset.attrs[attr] = dataset_attrs[attr]
    

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
    
    def generate_blank_fraction_heatmap_extendedmisid(self,
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
        return np.clip(blank_fraction_heatmap,0,5)
     
    def assign_blank_fraction_scores(self, blank_fraction_heatmap):
        print("Assigning blank fraction scores to spots")
        self.spotsData.blank_fraction_scores = []
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
            im = ax[1,i].pcolormesh(xbins, ybins, blank_pdf[:,:,i].T, norm=mpl.colors.LogNorm(vmax=vmax),
                                    cmap='jet')
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
        