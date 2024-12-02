### Imports ###
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import glob
import os
import copy
import h5py
from readClasses import readDoryImg
from frequencyFilter import butter2d
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.spatial import cKDTree
from utilsN.registrationFunctions import register_translation
from spotClasses_colocalisation import FpkmData, SpotsData, SpotsHistogram

### Spot class ###
class Spot:
    def __init__(self, x, y, fov, gene, type, **kwargs):
        self.x = x
        self.y = y
        self.fov = fov
        self.gene = gene
        self.type = type
        self.colocalisation_status = 0
        for keys, values in kwargs.items():
            setattr(self, keys, values)
            
    def __repr__(self):
        return "Spot"

    def shift_xy_coord(self, shifts):
        shift_spot = copy.deepcopy(self) # copy to prevent changing the original list too
        shift_spot.x = self.x+shifts[0]
        shift_spot.y = self.y+shifts[1]
        return shift_spot

    def is_overlap(self, spot, distance_pixel):
        return abs(self.x - spot.x) <= distance_pixel and abs(self.y - spot.y) <= distance_pixel
    
### Spots class (group of spot) ###
class Spots:
    def __init__(self, list_of_spots):
        self.num_spots = len(list_of_spots)
        self.all_spots = list_of_spots
        self.all_fovs = self._get_number_fov()
        self.all_genes = self._get_gene_name()
        self._sort_by_xy()

    def __repr__(self):
        return "Spots"
    
    def check_colocalisation_status(self):
        colocalisation_counts = {'0':0, '1':0}
        for spot in self.all_spots:
            if spot.colocalisation_status == 0:
                colocalisation_counts['0'] += 1
            if spot.colocalisation_status == 1:
                colocalisation_counts['1'] += 1
        return colocalisation_counts
    
    def reset_colocalisation_status(self):
        spot_new = []
        for spot in self.all_spots:
            spot.colocalisation_status = 0
            spot_new.append(spot)
        return Spots(spot_new)

    def get_by_gene(self, gene):
        spot_by_gene = []
        for spot in self.all_spots:
            if spot.gene == gene:
                spot_by_gene.append(spot)
        return Spots(spot_by_gene)

    def get_by_fov(self, fov):
        spot_by_fov = []
        for spot in self.all_spots:
            if spot.fov == fov:
                spot_by_fov.append(spot)
        return Spots(spot_by_fov)
    
    def get_by_colocalisation(self, status):
        spot_by_colocalisation = []
        for spot in self.all_spots:
            if spot.colocalisation_status == status:
                spot_by_colocalisation.append(spot)
        return Spots(spot_by_colocalisation)
    
    def _get_number_fov(self):
        number_fov_list = []
        for spot in self.all_spots:
            number_fov_list.append(spot.fov)
        number_fov = np.unique(number_fov_list)
        return list(number_fov)

    def _get_gene_name(self):
        gene_list = []
        for spot in self.all_spots:
            gene_list.append(spot.gene)
        genes = np.unique(gene_list)
        return list(genes)

    def shift_spots(self, shift):
        spot_new = []
        for spot in self.all_spots:
            spot_new.append(spot.shift_xy_coord(shift))
        return Spots(spot_new)
    
    @staticmethod
    def combine_spots(spots_1, spots_2):
        return Spots(spots_1.all_spots + spots_2.all_spots)

    def _sort_by_xy(self):
        return self.all_spots.sort(key= lambda spot: (spot.x, spot.y))
    
### Preprocessing functions ###
def register_slice(ref_slice, current_slice, shifts=None):
    ref_slice_fourier = np.fft.fftn(ref_slice)
    current_slice_fourier = np.fft.fftn(current_slice)
    (shifts, fine_error,  pixel_error) = register_translation(ref_slice_fourier,
                                                              current_slice_fourier,
                                                              upsample_factor=50,
                                                              space="fourier")
    print("shifts = ", shifts)
    registered_slice = np.fft.ifftn(ndimage.fourier_shift(current_slice_fourier, shifts))
    return registered_slice.real, shifts

def filter_func(filter_param, img_arr):
    freq_filter = butter2d(low_cut=filter_param[0], high_cut=filter_param[1],  # filter_path=os.path.join(data_path, "filters"),
                           order=2, xdim=img_arr.shape[1], ydim=img_arr.shape[1])
    filter_shifted = np.fft.fftshift(freq_filter)
    img_fourier = np.fft.fftn(img_arr)
    filtered_img = np.fft.ifftn(img_fourier * filter_shifted)
    return filtered_img.real

def norm_image_func(image):
    """" Params: image (2 or 3D arr)
         Return: normalise image arr by min = 0 and max = 1 (2 or 3D arr)
    """
    image_max = np.max(image)
    image_min = np.min(image)
    print("minimum intensity of image =", image_min)
    print("maximum intensity of image =", image_max)
    return (image - image_min) / (image_max - image_min)

### Guidestar image spot finding ###
def find_peak_local_max(image_arr, thres):
    """ Wrapper for peak_local_max
    Params: image (2D arr) : image (x,y)
            threshold (float) : relative threshold cutoff
    Return: coordinates (z,x,y) or (x,y) of peaks
    """
    coordinates = peak_local_max(image_arr, min_distance=2, threshold_rel=thres)

    return coordinates

def xy_coord_Guidestar(image_file, filter_param, threshold):
    image = readDoryImg(image_file)
    image_norm = norm_image_func(image)
    filter_image = filter_func(filter_param, image_norm)
    gene_coordinates = find_peak_local_max(filter_image, threshold)
    return gene_coordinates, filter_image

def create_Guidestar_obj(raw_img_folder, gene_bit_dict_Guidestar, threshold_dict, filter_param, fovlist):
    spots_obj_list = []
    list_files = glob.glob(raw_img_folder+'/*.dax')
    for file in list_files:
        fov = file.split("/")[-1].split("\\")[-1].split("_")[-1].split(".")[0]
        if int(fov) in fovlist:
            gene_list = file.split("/")[-1].split("\\")[-1].split("_")[0:2]
            gene_color = '_'.join(gene_list)
            if gene_color in gene_bit_dict_Guidestar:
                gene = gene_bit_dict_Guidestar[gene_color]
                threshold = threshold_dict[gene_color]
                print((file,threshold))
                coord, _ = xy_coord_Guidestar(file, filter_param, threshold)
                for x , y in zip(coord[:,0],coord[:,1]):
                    spot_obj = Spot(x, y, fov, gene, 'Guidestar')
                    spots_obj_list.append(spot_obj)

    return Spots(spots_obj_list)

### MERFISH spots object ###
def create_Merfish_obj_addedfeat(coords_cache_path, genes_list, fovlist):
    spot_obj_list = []
    coords_filelist = os.listdir(coords_cache_path)
    for file in coords_filelist:
        fov = file.split('_')[1]
        coords_cache = os.path.join(coords_cache_path, file)
        if int(fov) in fovlist:
            print(f"Loading spot metrics from FOV {fov}")
            with h5py.File(coords_cache, 'r') as g:
                for gene in genes_list:
                    x_coords = g[gene][:, 1]
                    y_coords = g[gene][:, 2]
                    spot_sizes = g[gene][:, 3]
                    mindist1 = g[gene][:, 4]
                    meaninten1_frob = g[gene][:, 5]
                    offinten1_frob = g[gene][:,6]
                    mindist2 = g[gene][:,7]
                    meaninten2_frob = g[gene][:,8]
                    offinten2_frob = g[gene][:,9]
                    mindist3 = g[gene][:,10]
                    meaninten3_frob = g[gene][:,11]
                    offinten3_frob = g[gene][:,12]
                    mindist1_2_ratio = g[gene][:,13]
                    mindist1_3_ratio = g[gene][:,14]
                    size_x = g[gene][:,15]
                    size_y = g[gene][:,16]
                    size_conn = g[gene][:,17]
                    
                    bfs = g[gene][:,18]

                    for x, y, if1, s, d1, of1, d2, if2, of2, d3, if3, of3, dr2, dr3, sx, sy, sc, b in zip(x_coords, y_coords, meaninten1_frob, spot_sizes, mindist1, offinten1_frob, mindist2, meaninten2_frob, offinten2_frob, mindist3, meaninten3_frob, offinten3_frob, mindist1_2_ratio, mindist1_3_ratio, size_x, size_y, size_conn, bfs):
                        spot_obj = Spot(x, y, fov, gene, 'Merfish',
                                        meaninten1_frob=if1, spot_size=s, mindist1=d1, 
                                        offinten1_frob=of1, mindist2=d2, meaninten2_frob=if2,
                                        offinten2_frob=of2, mindist3=d3, meaninten3_frob=if3,
                                        offinten3_frob=of3, mindist1_2_ratio=dr2, mindist1_3_ratio=dr3, 
                                        size_x=sx, size_y=sy, size_conn=sc,
                                        blank_fr_score= b)
                        spot_obj_list.append(spot_obj)
                        
    return Spots(spot_obj_list)

def create_shift_spot_obj(raw_folder, merfish_output_path, bit_gene_dict_Guidestar, spot_obj_merfish):
    final_spots = Spots([])
    for fov in spot_obj_merfish.all_fovs:
        merfish_img_name = merfish_output_path + 'FOV_' + fov + '_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax'
        merfish_img = readDoryImg(merfish_img_name)
        for gene in spot_obj_merfish.all_genes:
            Guidestar_img_name = raw_folder + bit_gene_dict_Guidestar[gene]+'_' + fov + '.dax'
            Guidestar_img = readDoryImg(Guidestar_img_name)
            _, shifts = register_slice(Guidestar_img, merfish_img)
            spots_original = spot_obj_merfish.get_by_fov(fov)
            spots_original_gene = spots_original.get_by_gene(gene)
            new_spot_shift = spots_original_gene.shift_spots(shifts)
            final_spots = Spots.combine_spots(final_spots,new_spot_shift)
    return final_spots

### Colocalisation by pixel dist or euclidean distance ###
def Colocalisation_pixeldist(spot_obj_Guidestar: Spots,
                             spot_obj_Merfish: Spots,
                             dist: int):
    
    # check that no colocalisation was done previously
    colocalisation_counts_Guidestar = spot_obj_Guidestar.check_colocalisation_status
    colocalisation_counts_Merfish = spot_obj_Merfish.check_colocalisation_status
    
    if colocalisation_counts_Guidestar()['1']!=0:
        raise ColocalisationError
    if colocalisation_counts_Merfish()['1']!=0:
        raise ColocalisationError
    
    # assert colocalisation_counts_Guidestar()['1']==0, 'Guidestar spots already colocalised. Re-initialise Guidestar spots.'
    # assert colocalisation_counts_Merfish()['1']==0, 'Merfish spots already colocalised. Re-initialise Merfish spots.'
    
    Processed_Merfish_spots = Spots([])
    Processed_Guidestar_spots = Spots([])
    
    for fov in spot_obj_Merfish.all_fovs:
        print(f"Colocalising FOV {fov}")
        for gene in spot_obj_Merfish.all_genes:
            print(f"Colocalising gene {gene}")
            
            Guidestar_spots = spot_obj_Guidestar.get_by_fov(fov).get_by_gene(gene)
            Merfish_spots = spot_obj_Merfish.get_by_fov(fov).get_by_gene(gene)
            
            # print('before process Merfish length',Merfish_spots.num_spots)
            # print('before process GS length',Guidestar_spots.num_spots)
            
            for Merfish_spot in Merfish_spots.all_spots:
                for Guidestar_spot in Guidestar_spots.all_spots:
                    if Merfish_spot.is_overlap(Guidestar_spot, dist):
                        Merfish_spot.colocalisation_status = 1
                        Guidestar_spot.colocalisation_status = 1
                        
            Processed_Merfish_spots = Spots.combine_spots(Processed_Merfish_spots,Merfish_spots)
            Processed_Guidestar_spots = Spots.combine_spots(Processed_Guidestar_spots,Guidestar_spots)
            
    # Processed_Merfish_spots = Spots(Processed_Merfish_list)
    # Processed_Guidestar_spots = Spots(Processed_Guidestar_list)
    
    print('after process Merfish length',Processed_Merfish_spots.num_spots)
    print('after process GS length',Processed_Guidestar_spots.num_spots)
    
    Merfish_colocalised_spots = Processed_Merfish_spots.get_by_colocalisation(1)
    Merfish_only_spots = Processed_Merfish_spots.get_by_colocalisation(0)
    Guidestar_only_spots = Processed_Guidestar_spots.get_by_colocalisation(0)
    Guidestar_colocalised_spots = Processed_Guidestar_spots.get_by_colocalisation(1)
    
    # create df to contain per gene values for colocalisation counts
    colocalisation_gene_counts = {'gene':[], '1':[], '0':[], 'type':[]}
    for gene in Processed_Merfish_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Merfish_spots.get_by_gene(gene).check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Merfish_spots.get_by_gene(gene).check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Merfish')
    for gene in Processed_Guidestar_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Guidestar_spots.get_by_gene(gene).check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Guidestar_spots.get_by_gene(gene).check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Guidestar')
                
    return Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, pd.DataFrame.from_dict(colocalisation_gene_counts)

def Colocalisation_euclideandist(spot_obj_Guidestar: Spots,
                                 spot_obj_Merfish: Spots,
                                 dist: int):
    
    # check that no colocalisation was done previously
    colocalisation_counts_Guidestar = spot_obj_Guidestar.check_colocalisation_status
    colocalisation_counts_Merfish = spot_obj_Merfish.check_colocalisation_status
        
    if colocalisation_counts_Guidestar()['1']!=0:
        raise ColocalisationError
    if colocalisation_counts_Merfish()['1']!=0:
        raise ColocalisationError
    
    Processed_Merfish_spots = Spots([])
    Processed_Guidestar_spots = Spots([])
    
    for fov in spot_obj_Merfish.all_fovs:
        print(f"Colocalising FOV {fov}")
        for gene in spot_obj_Merfish.all_genes:
            print(f"Colocalising gene {gene}")
            
            Guidestar_spots = spot_obj_Guidestar.get_by_fov(fov).get_by_gene(gene)
            Merfish_spots = spot_obj_Merfish.get_by_fov(fov).get_by_gene(gene)
            
            # print('before process Merfish length',Merfish_spots.num_spots)
            # print('before process GS length',Guidestar_spots.num_spots)
            
            Guidestar_coords = np.zeros((Guidestar_spots.get_by_fov(fov).get_by_gene(gene).num_spots,2))
            for i, spot in enumerate(Guidestar_spots.get_by_fov(fov).get_by_gene(gene).all_spots):
                Guidestar_coords[i,0] = spot.x
                Guidestar_coords[i,1] = spot.y
            Guidestar_tree = cKDTree(Guidestar_coords)
            
            for Merfish_spot in Merfish_spots.all_spots:
                
                dist_array, ind_array = Guidestar_tree.query(x=[Merfish_spot.x, Merfish_spot.y],distance_upper_bound=dist,k=11)
                if dist_array[0] != np.inf:
                    Merfish_spot.colocalisation_status = 1
                    
                for j, d in enumerate(dist_array):
                    if d != np.inf:
                        Guidestar_spots.all_spots[ind_array[j]].colocalisation_status = 1
                        
            Processed_Merfish_spots = Spots.combine_spots(Processed_Merfish_spots,Merfish_spots)
            Processed_Guidestar_spots = Spots.combine_spots(Processed_Guidestar_spots,Guidestar_spots)
            
    # Processed_Merfish_spots = Spots(Processed_Merfish_list)
    # Processed_Guidestar_spots = Spots(Processed_Guidestar_list)
    
    print('after process Merfish length',Processed_Merfish_spots.num_spots)
    print('after process GS length',Processed_Guidestar_spots.num_spots)
    
    Merfish_colocalised_spots = Processed_Merfish_spots.get_by_colocalisation(1)
    Merfish_only_spots = Processed_Merfish_spots.get_by_colocalisation(0)
    Guidestar_only_spots = Processed_Guidestar_spots.get_by_colocalisation(0)
    Guidestar_colocalised_spots = Processed_Guidestar_spots.get_by_colocalisation(1)
    
    # create df to contain per gene values for colocalisation counts
    colocalisation_gene_counts = {'gene':[], '1':[], '0':[], 'type':[]}
    for gene in Processed_Merfish_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Merfish_spots.get_by_gene(gene).check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Merfish_spots.get_by_gene(gene).check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Merfish')
    for gene in Processed_Guidestar_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Guidestar_spots.get_by_gene(gene).check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Guidestar_spots.get_by_gene(gene).check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Guidestar')
                
    return Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, pd.DataFrame.from_dict(colocalisation_gene_counts)

class ColocalisationError(Exception):
    "Raised when spots object has already been processed via colocalisation"
    pass

def Colocalisation_negcontrol(spot_obj_Guidestar: Spots,
                              spot_obj_Merfish: Spots,
                              dist: int):
    
    # check that no colocalisation was done previously
    colocalisation_counts_Guidestar = spot_obj_Guidestar.check_colocalisation_status
    colocalisation_counts_Merfish = spot_obj_Merfish.check_colocalisation_status
        
    if colocalisation_counts_Guidestar()['1']!=0:
        raise ColocalisationError
    if colocalisation_counts_Merfish()['1']!=0:
        raise ColocalisationError
    
    Processed_Merfish_spots = Spots([])
    Processed_Guidestar_spots = Spots([])
    
    for fov in spot_obj_Merfish.all_fovs:
        print(f"Colocalising FOV {fov}")
            
        Guidestar_spots = spot_obj_Guidestar.get_by_fov(fov)
        Merfish_spots = spot_obj_Merfish.get_by_fov(fov)
        
        # print('before process Merfish length',Merfish_spots.num_spots)
        # print('before process GS length',Guidestar_spots.num_spots)
        
        Guidestar_coords = np.zeros((Guidestar_spots.get_by_fov(fov).num_spots,2))
        for i, spot in enumerate(Guidestar_spots.get_by_fov(fov).all_spots):
            Guidestar_coords[i,0] = spot.x
            Guidestar_coords[i,1] = spot.y
        Guidestar_tree = cKDTree(Guidestar_coords)
        
        for Merfish_spot in Merfish_spots.all_spots:
            
            dist_array, ind_array = Guidestar_tree.query(x=[Merfish_spot.x, Merfish_spot.y],distance_upper_bound=dist,k=11)
            if dist_array[0] != np.inf:
                Merfish_spot.colocalisation_status = 1
                
            for j, d in enumerate(dist_array):
                if d != np.inf:
                    Guidestar_spots.all_spots[ind_array[j]].colocalisation_status = 1
                    
        Processed_Merfish_spots = Spots.combine_spots(Processed_Merfish_spots,Merfish_spots)
        Processed_Guidestar_spots = Spots.combine_spots(Processed_Guidestar_spots,Guidestar_spots)
            
    # Processed_Merfish_spots = Spots(Processed_Merfish_list)
    # Processed_Guidestar_spots = Spots(Processed_Guidestar_list)
    
    # print('after process Merfish length',Processed_Merfish_spots.num_spots)
    # print('after process GS length',Processed_Guidestar_spots.num_spots)
    
    Merfish_colocalised_spots = Processed_Merfish_spots.get_by_colocalisation(1)
    Merfish_only_spots = Processed_Merfish_spots.get_by_colocalisation(0)
    Guidestar_only_spots = Processed_Guidestar_spots.get_by_colocalisation(0)
    Guidestar_colocalised_spots = Processed_Guidestar_spots.get_by_colocalisation(1)
    
    # create df to contain per gene values for colocalisation counts
    colocalisation_gene_counts = {'gene':[], '1':[], '0':[], 'type':[]}
    for gene in Processed_Merfish_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Merfish_spots.check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Merfish_spots.check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Merfish')
    for gene in Processed_Guidestar_spots.all_genes:
        colocalisation_gene_counts['gene'].append(gene)
        colocalisation_gene_counts['1'].append(Processed_Guidestar_spots.check_colocalisation_status()['1'])
        colocalisation_gene_counts['0'].append(Processed_Guidestar_spots.check_colocalisation_status()['0'])
        colocalisation_gene_counts['type'].append('Guidestar')
                
    return Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, pd.DataFrame.from_dict(colocalisation_gene_counts)


### BFS assignment ###
def generate_BFS_coord(processed_path,
                                fpkm_path,
                                microscope_type,
                                num_bins,
                                eps,
                                kde_sigma,
                                save_coord_path= None,
                                save_qc_path = None):
    # Load FPKM Data
    fpkmData = FpkmData(fpkm_path)

    # Load spots data from coord hdf5 files
    spotsData = SpotsData(fpkmData, processed_path, microscope_type = microscope_type)

    spotsData.load_spots_from_hdf5()

    # Generate blank fraction heatmap
    spotsHist = SpotsHistogram(spotsData, fpkmData)
    blank_fraction_heatmap = spotsHist.generate_blank_fraction_heatmap(num_bins = num_bins,
                                                                       kde_sigma = kde_sigma,
                                                                       eps = eps)
                                                                    #    new_save_path=save_qc_path)
    # Assign blank fraction scores to spots
    spotsHist.assign_blank_fraction_scores(blank_fraction_heatmap)
    spotsData.save_to_bfs_spots_hdf5(save_coord_path) # BFS should always be last column

    return spotsData

def generate_BFS_coord_extendedmisid(processed_path,
                                fpkm_path,
                                microscope_type,
                                num_bins,
                                eps,
                                kde_sigma,
                                save_coord_path= None,
                                save_qc_path = None):
    # Load FPKM Data
    fpkmData = FpkmData(fpkm_path)

    # Load spots data from coord hdf5 files
    spotsData = SpotsData(fpkmData, processed_path, microscope_type = microscope_type)

    spotsData.load_spots_from_hdf5()

    # Generate blank fraction heatmap
    spotsHist = SpotsHistogram(spotsData, fpkmData)
    blank_fraction_heatmap = spotsHist.generate_blank_fraction_heatmap_extendedmisid(num_bins = num_bins,
                                                                       kde_sigma = kde_sigma,
                                                                       eps = eps)
                                                                    #    new_save_path=save_qc_path)
    # Assign blank fraction scores to spots
    spotsHist.assign_blank_fraction_scores(blank_fraction_heatmap)
    spotsData.save_to_bfs_spots_hdf5(save_coord_path) # BFS should always be last column

    return spotsData

### Generating training CSVs ###
def Spots_to_list(Spots, label):
    all_spots_list = []
    feature_name = ['x', 'y', 'gene', 'fov','bfs', 
                    'meaninten1_frob', 'mindist1', 'size', 'offinten1_frob',
                    'meaninten2_frob', 'mindist2', 'offinten2_frob',
                    'meaninten3_frob', 'mindist3', 'offinten3_frob',
                    'mindist1_2_ratio', 'mindist1_3_ratio',
                    'size_x', 'size_y', 'size_conn',
                    'Label']
    for spot in Spots.all_spots:
        temp_list = []
        
        # basic info
        temp_list.append(spot.x)
        temp_list.append(spot.y)
        temp_list.append(spot.gene)
        temp_list.append(spot.fov)
        temp_list.append(spot.blank_fr_score)
        
        # conventional features
        temp_list.append(spot.meaninten1_frob)
        temp_list.append(spot.mindist1)
        temp_list.append(spot.spot_size)
        
        # additional features
        temp_list.append(spot.offinten1_frob)
        temp_list.append(spot.meaninten2_frob)
        temp_list.append(spot.mindist2)
        temp_list.append(spot.offinten2_frob)
        temp_list.append(spot.meaninten3_frob)
        temp_list.append(spot.mindist3)
        temp_list.append(spot.offinten3_frob)
        temp_list.append(spot.mindist1_2_ratio)
        temp_list.append(spot.mindist1_3_ratio)
        temp_list.append(spot.size_x)
        temp_list.append(spot.size_y)
        temp_list.append(spot.size_conn)
        
        # label
        temp_list.append(label)
        
        all_spots_list.append(temp_list)

    return all_spots_list, feature_name