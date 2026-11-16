### Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import re
import json
import copy
import h5py
import tifffile
# use relative import
try:
    from .readClasses import readDoryImg
    from .frequencyFilter import butter2d
    from .registrationFunctions import register_translation
    from .spotClasses_colocalisation import FpkmData, SpotsData, SpotsHistogram
except ImportError:
    from readClasses import readDoryImg
    from frequencyFilter import butter2d
    from registrationFunctions import register_translation
    from spotClasses_colocalisation import FpkmData, SpotsData, SpotsHistogram

from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.spatial import cKDTree
import skimage

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
        ## added for triton read in
        if (len(number_fov_list) > 0 ) and (type(number_fov_list[0]) == tuple):
            return list(set(number_fov_list))
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

def contrast_stretch(img, low=1, high=99):
    p_low, p_high = np.percentile(img, (low, high))
    return np.clip((img - p_low) / (p_high - p_low), 0, 1)


### Preprocessing functions ###
def register_slice(ref_slice, current_slice, **kwargs):
    upsample_factor = int(kwargs.get('upsample_factor', 50))
    ref_slice_fourier = np.fft.fftn(ref_slice)
    current_slice_fourier = np.fft.fftn(current_slice)
    (shifts, fine_error,  pixel_error) = register_translation(ref_slice_fourier,
                                                              current_slice_fourier,
                                                              upsample_factor=upsample_factor,
                                                              space="fourier")
    print(f'upsample_factor = {upsample_factor}')
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

def create_Guidestar_obj(raw_img_folder, gene_bit_dict_Guidestar, threshold_dict, filter_param, fovlist, microscope: str = "Dory", triton_channel_dict: dict = {}):
    spots_obj_list = []
    pattern = r'^hyb\d+_1_MMStack_1-Pos_\d+_\d+\.ome\.tif$' #modify according to file name pattern
    if microscope == "Triton":
        for root, dirs, files in os.walk(raw_img_folder + '/S10/'):
            for file in files:
                    if re.match(pattern, file):
                        fov = file.split('.')[-3].split('_')[-2] + '_' + file.split('.')[-3].split('_')[-1]  # '000_001'
                        fov_tuple = (tuple(map(int, fov.split('_'))))
                        if fov_tuple not in fovlist: 
                            continue
                        hyb = file.split('hyb')[1].split('_')[0]
                        for channel_name in triton_channel_dict:
                            channel_hyb = channel_name+'_'+hyb
                            if channel_hyb in gene_bit_dict_Guidestar: #gene in GS gene list
                                gene = gene_bit_dict_Guidestar[channel_hyb]
                                print(gene)
                                full_path = os.path.join(root, file) # load array
                                print(full_path)
                                tiff_img = tifffile.imread(full_path)
                                smallest_dimension = np.argmin(tiff_img.shape)
                                if smallest_dimension == 2:  # last dim is frames
                                    Guidestar_img = tiff_img[..., triton_channel_dict[channel_name]]
                                elif smallest_dimension == 0:  # first dim is frames
                                    Guidestar_img = tiff_img[triton_channel_dict[channel_name], ...]
                                image_norm = norm_image_func(Guidestar_img)
                                filter_image = filter_func(filter_param, image_norm)
                                gene_coordinates = find_peak_local_max(filter_image, threshold_dict[channel_hyb])
                                print((file, threshold_dict[channel_hyb]))
                                for x, y in zip(gene_coordinates[:, 0], gene_coordinates[:, 1]):
                                    spot_obj = Spot(x, y, fov, gene, 'Guidestar')
                                                    #,hyb, channel_hyb) #added identity for double GS
                                    spots_obj_list.append(spot_obj)

    else:
        list_files = glob.glob(raw_img_folder + '/*.dax')
        for file in list_files:
            fov = file.split("/")[-1].split("\\")[-1].split("_")[-1].split(".")[0]
            if int(fov) in fovlist:
                gene_list = file.split("/")[-1].split("\\")[-1].split("_")[0:2]
                gene_color = '_'.join(gene_list)
                if gene_color in gene_bit_dict_Guidestar:
                    gene = gene_bit_dict_Guidestar[gene_color]
                    threshold = threshold_dict[gene_color]
                    print((file, threshold))
                    coord, _ = xy_coord_Guidestar(file, filter_param, threshold)
                    for x, y in zip(coord[:, 0], coord[:, 1]):
                        spot_obj = Spot(x, y, fov, gene, 'Guidestar')
                        spots_obj_list.append(spot_obj)

    return Spots(spots_obj_list)

# new Guidestar object 04 Jan 2024 GS spot calling see smfish_threshold_plts.py
def intensity_threshold_GS(coord, file, genes, fov_to_plot):
    gs_spot_int = []
    raw_gs_img = file
    for x, y in zip(coord[:, 0], coord[:, 1]):
        gs_spot_int.append(raw_gs_img[x, y])

    # plot total spot count VS threshold plot
    thresholds = np.linspace(min(gs_spot_int), max(gs_spot_int), 100)
    spot_counts = [np.sum(gs_spot_int > threshold) for threshold in thresholds]
    # break point determination from big-FISH
    # https://github.com/fish-quant/big-fish/blob/master/bigfish/detection/utils.py#L653
    slope = (spot_counts[-1] - spot_counts[0]) / len(spot_counts)
    spot_counts_grad = np.gradient(spot_counts)
    m = list(spot_counts_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = thresholds[j:]
    y = spot_counts[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    inten_threshold = float(x[i])

    return inten_threshold

# def create_Guidestar_obj_wInt(raw_img_folder, gene_bit_dict_Guidestar, threshold_dict, filter_param, fovlist):
#     spots_obj_list = []
#     list_files = glob.glob(raw_img_folder+'/*.dax')
#     for file in list_files:
#         fov = file.split("/")[-1].split("\\")[-1].split("_")[-1].split(".")[0]
#         if int(fov) in fovlist:
#             gene_list = file.split("/")[-1].split("\\")[-1].split("_")[0:2]
#             gene_color = '_'.join(gene_list)
#             if gene_color in gene_bit_dict_Guidestar:
#                 gene = gene_bit_dict_Guidestar[gene_color]
#                 threshold = threshold_dict[gene_color]
#                 print((file,threshold))
#                 coord, filter_image = xy_coord_Guidestar(file, filter_param, 0.03) #0.02
#                 inten_threshold = intensity_threshold_GS(coord, filter_image, gene, fov) #return intensity threshold of filtered image
#
#                 raw_gs_img = readDoryImg(file)
#                 for x , y in zip(coord[:,0],coord[:,1]):
#                     if filter_image[x, y] >= inten_threshold:
#                         gs_int = raw_gs_img[x, y]
#                         spot_obj = Spot(x, y, fov, gene, 'Guidestar', intensity = gs_int)
#                         spots_obj_list.append(spot_obj)

#    return Spots(spots_obj_list)


### MERFISH spots object ###
def create_Merfish_obj_addedfeat(coords_cache_path, genes_list, fovlist, microscope:str = 'Dory'):
    spot_obj_list = []
    coords_filelist = os.listdir(coords_cache_path)
    for file in coords_filelist:
        if microscope == "Triton":
            fov = file.split('_')[1] + '_' + file.split('_')[2]
        else:
            fov = file.split('_')[1]
        coords_cache = os.path.join(coords_cache_path, file)
        if (int(fov) in fovlist) or (tuple(map(int, fov.split('_'))) in fovlist):
            print(f"Loading spot metrics from FOV {fov}")
            with h5py.File(coords_cache, 'r') as g:
                for gene in genes_list:
                    if g[gene].shape[1] == 19:
                        x_coords = g[gene][:, 1]
                        y_coords = g[gene][:, 2]
                        spot_sizes = g[gene][:, 3]
                        mindist1 = g[gene][:, 4]
                        meaninten1_frob = g[gene][:, 5]
                        offinten1_frob = g[gene][:, 6]
                        mindist2 = g[gene][:, 7]
                        meaninten2_frob = g[gene][:, 8]
                        offinten2_frob = g[gene][:, 9]
                        mindist3 = g[gene][:, 10]
                        meaninten3_frob = g[gene][:, 11]
                        offinten3_frob = g[gene][:, 12]
                        mindist1_2_ratio = g[gene][:, 13]
                        mindist1_3_ratio = g[gene][:, 14]
                        size_x = g[gene][:, 15]
                        size_y = g[gene][:, 16]
                        size_conn = g[gene][:, 17]

                        bfs = g[gene][:, 18]

                        for x, y, if1, s, d1, of1, d2, if2, of2, d3, if3, of3, dr2, dr3, sx, sy, sc, b in zip(
                                x_coords, y_coords, meaninten1_frob, spot_sizes, mindist1, offinten1_frob, mindist2,
                                meaninten2_frob, offinten2_frob, mindist3, meaninten3_frob, offinten3_frob,
                                mindist1_2_ratio, mindist1_3_ratio, size_x, size_y, size_conn, bfs):
                            spot_obj = Spot(x, y, fov, gene, 'Merfish',
                                            meaninten1_frob=if1, spot_size=s, mindist1=d1,
                                            offinten1_frob=of1, mindist2=d2, meaninten2_frob=if2,
                                            offinten2_frob=of2, mindist3=d3, meaninten3_frob=if3,
                                            offinten3_frob=of3, mindist1_2_ratio=dr2, mindist1_3_ratio=dr3,
                                            size_x=sx, size_y=sy, size_conn=sc, blank_fr_score=b)
                            spot_obj_list.append(spot_obj)
                    else:
                        print(file)
                        print(gene)

    return Spots(spot_obj_list)

def read_tif_GS(Guidestar_img_name, metadata_filepath, chan_hyb):
    with open(metadata_filepath, 'r') as metadata_file:
        meta_data = json.load(metadata_file)
    # list of colours/channels corresponding to the frames of the tiff
    colour_list = meta_data["Summary"]["ChNames"]
    channel_index = colour_list.index(chan_hyb.split('_')[0])
    tiff_img = tifffile.imread(Guidestar_img_name)
    smallest_dimension = np.argmin(tiff_img.shape)
    if smallest_dimension == 2:  # last dim is frames
        Guidestar_img = tiff_img[..., channel_index]

    elif smallest_dimension == 0:  # first dim is frames
        Guidestar_img = tiff_img[channel_index, ...]
    return Guidestar_img

def create_shift_spot_obj(raw_folder, merfish_output_path, bit_gene_dict_Guidestar,
                          spot_obj_merfish, microscope: str='Dory',
                          mer_img_type: str='mip', **kwargs):
    final_spots = Spots([])
    for fov in spot_obj_merfish.all_fovs:
        if mer_img_type == "mip":
            merfish_img_name = merfish_output_path + 'FOV_' + fov + '_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax'
            merfish_img = readDoryImg(merfish_img_name)
        elif mer_img_type == 'ref_bit':
            ref_hyb = kwargs.get('ref_hyb', None)  # safely extract with a default
            ref_channel = kwargs.get('ref_channel', None)
            merfish_img_name = raw_folder + 'S10/' + f"hyb{ref_hyb}_1/hyb{ref_hyb}_1_MMStack_1-Pos_{fov}.ome.tif"
            metadata_filepath = raw_folder + 'S10/' + f"hyb{ref_hyb}_1/hyb{ref_hyb}_1_MMStack_1-Pos_{fov}_metadata.txt"
            merfish_img = read_tif_GS(merfish_img_name, metadata_filepath, ref_channel)
        output_path = kwargs.get('output_path', None)
        if output_path is not None:
            save_regi_output = os.path.join(output_path, f'shift_registration.txt')
            os.makedirs(os.path.dirname(save_regi_output), exist_ok=True) #generate the txt file to save the shift registration information
        for gene in spot_obj_merfish.all_genes:
            if microscope == 'Triton':
                GS_hyb_channel_into = bit_gene_dict_Guidestar[gene] #Cy5_00
                if len(GS_hyb_channel_into) > 1: #problematic! double merfish registration
                    for chan_hyb in GS_hyb_channel_into:
                        Guidestar_img_name = raw_folder + 'S10/' + f"hyb{chan_hyb.split('_')[1]}_1/hyb{chan_hyb.split('_')[1]}_1_MMStack_1-Pos_{fov}.ome.tif"
                        metadata_filepath = raw_folder + 'S10/' + f"hyb{chan_hyb.split('_')[1]}_1/hyb{chan_hyb.split('_')[1]}_1_MMStack_1-Pos_{fov}_metadata.txt"
                        Guidestar_img = read_tif_GS(Guidestar_img_name, metadata_filepath, chan_hyb)
                        _, shifts = register_slice(Guidestar_img, merfish_img, **kwargs)
                        spots_original = spot_obj_merfish.get_by_fov(fov)
                        spots_original_gene = spots_original.get_by_gene(gene)
                        new_spot_shift = spots_original_gene.shift_spots(shifts)
                        final_spots = Spots.combine_spots(final_spots, new_spot_shift)
                else:
                    chan_hyb = bit_gene_dict_Guidestar[gene][0]
                    Guidestar_img_name = raw_folder + 'S10/' + f"hyb{chan_hyb.split('_')[1]}_1/hyb{chan_hyb.split('_')[1]}_1_MMStack_1-Pos_{fov}.ome.tif"
                    metadata_filepath = raw_folder + 'S10/' + f"hyb{chan_hyb.split('_')[1]}_1/hyb{chan_hyb.split('_')[1]}_1_MMStack_1-Pos_{fov}_metadata.txt"
                    Guidestar_img = read_tif_GS(Guidestar_img_name, metadata_filepath, chan_hyb)
                    _, shifts = register_slice(Guidestar_img, merfish_img, **kwargs)

                    print(f"{gene} in {fov} shift merfish by {shifts}")
                    if output_path is not None:
                        with open(save_regi_output, 'a') as f:
                            f.write(f"{gene} in {fov} shift merfish by {shifts}\n")

                    spots_original = spot_obj_merfish.get_by_fov(fov)
                    spots_original_gene = spots_original.get_by_gene(gene)
                    new_spot_shift = spots_original_gene.shift_spots(shifts)
                    final_spots = Spots.combine_spots(final_spots, new_spot_shift)

            elif microscope == 'Dory':
                Guidestar_img_name = raw_folder + bit_gene_dict_Guidestar[gene]+'_' + fov + '.dax'
                Guidestar_img = readDoryImg(Guidestar_img_name)
                registered_filtered_slice_real, shifts = register_slice(Guidestar_img, merfish_img, **kwargs)
                spots_original = spot_obj_merfish.get_by_fov(fov)
                spots_original_gene = spots_original.get_by_gene(gene)
                new_spot_shift = spots_original_gene.shift_spots(shifts)
                final_spots = Spots.combine_spots(final_spots,new_spot_shift)
            # check registration #
            check_regi = kwargs.get('check_regi', None)
            if check_regi:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

                axs[0].imshow(contrast_stretch(Guidestar_img), cmap='gray')
                axs[0].set_title(f'Base Image ({Guidestar_img_name})')

                axs[1].imshow(contrast_stretch(registered_filtered_slice_real), cmap='jet')
                axs[1].set_title(f'Overlay Image (merfish {fov}_{gene})')

                axs[2].imshow(contrast_stretch(Guidestar_img), cmap='gray')
                axs[2].imshow(contrast_stretch(registered_filtered_slice_real), cmap='jet', alpha=0.2)
                axs[2].set_title(f'shift={shifts}')

                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"{fov}_{gene}.png"), dpi=300)
                plt.close()
    return final_spots

def create_shift_GS2GS(GS_set1_spots,
                       GS_set2_spots,
                       bit_gene_dict_Guidestar_set1,
                       bit_gene_dict_Guidestar_set2,
                       raw_folder,
                       shift_method):
    
    """ shift set 2 into set 1 space, set 1 image is reference """
    
    # intialise final spots
    final_spots_set2 = Spots([])
    
    for fov in GS_set1_spots.all_fovs:
        print(f"Registering FOV {fov}")
        for gene in GS_set1_spots.all_genes:
            print(f"Registering gene {gene}")

            chan_hyb_set1 = bit_gene_dict_Guidestar_set1[gene][0]
            GS_set1_img_name = raw_folder + 'S10/' + f"hyb{chan_hyb_set1.split('_')[1]}_1/hyb{chan_hyb_set1.split('_')[1]}_1_MMStack_1-Pos_{fov}.ome.tif"
            metadata_filepath_set1 = raw_folder + 'S10/' + f"hyb{chan_hyb_set1.split('_')[1]}_1/hyb{chan_hyb_set1.split('_')[1]}_1_MMStack_1-Pos_{fov}_metadata.txt"

            chan_hyb_set2 = bit_gene_dict_Guidestar_set2[gene][0]
            GS_set2_img_name = raw_folder + 'S10/' + f"hyb{chan_hyb_set2.split('_')[1]}_1/hyb{chan_hyb_set2.split('_')[1]}_1_MMStack_1-Pos_{fov}.ome.tif"
            metadata_filepath_set2 = raw_folder + 'S10/' + f"hyb{chan_hyb_set2.split('_')[1]}_1/hyb{chan_hyb_set2.split('_')[1]}_1_MMStack_1-Pos_{fov}_metadata.txt"

            GS_set1_img = read_tif_GS(GS_set1_img_name, metadata_filepath_set1, chan_hyb_set1)
            GS_set2_img = read_tif_GS(GS_set2_img_name, metadata_filepath_set2, chan_hyb_set2)
            
            _, shifts = register_slice(ref_slice=GS_set1_img, 
                                    current_slice=GS_set2_img, 
                                    shift_method=shift_method)
            spots_original_set2 = GS_set2_spots.get_by_fov(fov).get_by_gene(gene)
            spots_shifted_set2 = spots_original_set2.shift_spots(shifts)
            final_spots_set2 = Spots.combine_spots(final_spots_set2, spots_shifted_set2)
            
    return final_spots_set2

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

            print('before process Merfish length',Merfish_spots.num_spots)
            print('before process GS length',Guidestar_spots.num_spots)

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

            print('before process Merfish length',Merfish_spots.num_spots)
            print('before process GS length',Guidestar_spots.num_spots)

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
    #spotsData.cache_coords_files() - I comment this because it redundance keep create coor_cache folder for me

    spotsData.load_spots_from_hdf5()

    # Generate blank fraction heatmap
    spotsHist = SpotsHistogram(spotsData, fpkmData)
    blank_fraction_heatmap = spotsHist.generate_blank_fraction_heatmap(num_bins = num_bins,
                                                                       kde_sigma = kde_sigma,
                                                                       eps = eps)#,
                                                                       #new_save_path=save_qc_path)
    # Assign blank fraction scores to spots
    spotsHist.assign_blank_fraction_scores(blank_fraction_heatmap)
    spotsData.save_to_bfs_spots_hdf5(save_coord_path)

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
    feature_name = ['x', 'y', 'gene', 'fov', 'bfs',
                    'meaninten1_frob', 'mindist1', 'size', 'offinten1_frob',
                    'meaninten2_frob', 'mindist2', 'offinten2_frob',
                    'meaninten3_frob', 'mindist3', 'offinten3_frob',
                    'mindist1_2_ratio', 'mindist1_3_ratio',
                    'size_x', 'size_y', 'size_conn', 'Label']
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

def gs_colocal_spots_to_list(Spots, label):
    all_spots_list = []
    feature_name = ['x', 'y', 'gene', 'fov', #'is_overlap', 'shift_xy_coord', 
                    'type', 'colocalization_status', 'Label']
    for spot in Spots.all_spots:
        temp_list = []

        # basic info
        temp_list.append(spot.x)
        temp_list.append(spot.y)
        temp_list.append(spot.gene)
        temp_list.append(spot.fov)

        # GS2GS colocalization specific
        # temp_list.append(spot.is_overlap)
        # temp_list.append(spot.shift_xy_coord)
        temp_list.append(spot.type)
        temp_list.append(spot.colocalisation_status)

        # label
        temp_list.append(label)

        all_spots_list.append(temp_list)

    return all_spots_list, feature_name