import copy

from readClasses import readDoryImg
from frequencyFilter import butter2d
from skimage.feature import peak_local_max
import numpy as np
import glob
import os
import h5py
from scipy import ndimage
from utilsN.registrationFunctions import  register_translation
import matplotlib.pyplot as plt
# from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
import pandas as pd
from scipy.stats import pearsonr
import h5py
from skimage import io
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams['image.cmap'] = 'gray'

class Spot:
    def __init__(self, x, y, mean_inten, fov, gene, type, **kwargs):
        self.x = x
        self.y = y
        self.fov = fov
        self.mean_inten = mean_inten
        self.gene = gene
        self.type = type
        for keys, values in kwargs.items():
            setattr(self, keys, values)


    def __repr__(self):
        return "Spot"

    def shift_xy_coord(self, shifts):
        shift_spot = copy.deepcopy(self) #have to copy to prevent changing the original list too
        shift_spot.x = self.x+shifts[0]
        shift_spot.y = self.y+shifts[1]
        return shift_spot

    def is_overlap(self, spot, distance_pixel):
        return abs(self.x - spot.x) <= distance_pixel and abs(self.y - spot.y) <= distance_pixel

    def plot_completed_info_spot(self, gene_bitnum_dict):
        """This function mean to be for MERFISH and smFISH that has information of 16 bits like MERFISH
        gene_bitnum_dict : dict of gene and its on-bits ex. Gpam: [2,4,6,7]"""
        one = self.registered_inten
        two = self.snr
        three = self.bit_inten
        four = self.unit_vector
        gene = self.gene
        bits = one.shape[0]
        fig, axes = plt.subplots(2,2, figsize=(10,5))
        ax = axes.ravel()
        bit_num = gene_bitnum_dict[gene]
        fig.suptitle("Info of Gene " + str(self.gene))
        ax[0].plot(one, '-o')
        ax[0].set_ylabel('Inten of registered image')
        ax[1].plot(two, '-o')
        ax[1].set_ylabel('SNR')
        ax[2].plot(three, '-o')
        ax[2].set_ylabel('Normalized bit inten')
        ax[2].set_xlabel('bits')
        ax[3].plot(four, '-o')
        ax[3].set_ylabel('Normalized unit vector')
        ax[3].set_xlabel('bits')
        for i in range(len(bit_num)):
            #start, end = ax[i].get_xlim()
            ax[i].xaxis.set_ticks(np.arange(0, bits, 1))
            for num in bit_num:
                ax[i].get_xticklabels()[num].set_color("red")

class Spots:
    def __init__(self, list_of_spots):
        self.num_spots = len(list_of_spots)
        self.all_spots = list_of_spots
        self.all_fov = self._get_number_fov()
        self.all_gene = self._get_gene_name()
        self._sort_by_xy()

    def __repr__(self):
        return "Spots"

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

    def get_bit_arr(self, needed_array):
        unit_norm_list = []
        for spot in self.all_spots:
            if needed_array == "unit_vector": # MERFISH
                unit_norm_list.append(list(spot.unit_vector))
            if needed_array == "bit_inten": # MERFISH
                unit_norm_list.append(list(spot.bit_inten))
            if needed_array == "registered_inten": # MERFISH
                unit_norm_list.append(list(spot.registered_inten))
            if needed_array == "mean_inten": # for absolute smfish spot intensity
                unit_norm_list.append(spot.mean_inten)
            if needed_array == "snr": #MERFISH and SMFISH
                unit_norm_list.append(spot.snr)
        return unit_norm_list

    @staticmethod
    def combine_spots(spots_1, spots_2):
        return Spots(spots_1.all_spots + spots_2.all_spots)

    def _sort_by_xy(self):
        return self.all_spots.sort(key= lambda spot: (spot.x, spot.y))

    def pick_spots_in_area(self, x_center,y_center, x_delta, y_delta):
        spots_in_area = []
        for spot in self.all_spots:
            if abs(x_center-x_delta) <= spot.x <= abs(x_center+x_delta) and abs(y_center-y_delta) <= spot.y <= abs(y_center+y_delta):
                spots_in_area.append(spot)
        return Spots(spots_in_area)

    def sort_by_snr_smfish(self):
        return self.all_spots.sort(key= lambda spot: spot.snr_smfish, reverse=True)

    def show_smfish_spot_fov_gene(self, raw_folder, bit_gene_dict_smfish):
        """To plot smfish spot callouts
        Output : 1 image = 1 FOV and 1 gene with coordinates"""
        x_coord = []
        y_coord = []
        fov = self.all_spots[0].fov
        gene = self.all_spots[0].gene
        image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
        smfish_image = readDoryImg(image_name)
        selected_fov_spot = self.get_by_fov(fov).get_by_gene(gene)
        for spot in selected_fov_spot.all_spots:
            x = spot.x
            y = spot.y
            x_coord.append(x)
            y_coord.append(y)
        plt.figure()
        plt.imshow(smfish_image, vmax=np.percentile(smfish_image,99.9))
        plt.title("FOV " + str(fov) + " of Gene " + str(gene))
        plt.plot(y_coord, x_coord, 'ro', markerfacecolor='none')
        plt.axis("off")

    def show_smfish_spot_fov_gene_multi(self, raw_folder, bit_gene_dict_smfish,ax,ax_n):
        """To plot smfish spot callouts
        Output : 1 image = 1 FOV and 1 gene with coordinates"""
        x_coord = []
        y_coord = []
        fov = self.all_spots[0].fov
        gene = self.all_spots[0].gene
        image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
        smfish_image = readDoryImg(image_name)
        selected_fov_spot = self.get_by_fov(fov).get_by_gene(gene)
        for spot in selected_fov_spot.all_spots:
            x = spot.x
            y = spot.y
            x_coord.append(x)
            y_coord.append(y)
        # plt.figure()
        ax[ax_n].imshow(smfish_image, vmax=np.percentile(smfish_image,99.9))
        # ax[ax_n].title("FOV " + str(fov) + " of Gene " + str(gene))
        ax[ax_n].plot(y_coord, x_coord, 'ro', markerfacecolor='none')
        ax[ax_n].axis("off")

    def show_smfish_merfish_spot_fov_gene(self, fov, genes, raw_folder,bit_gene_dict_smfish, gene_bit_dict_merfish):
        """ This shows 4 bits """
        x_coord = []
        y_coord = []
        for gene in genes:
            image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
            smfish_image = readDoryImg(image_name)
            selected_fov_spot = self.get_by_fov(fov).get_by_gene(gene)
            for spot in selected_fov_spot.all_spots:
                x = spot.x
                y = spot.y
                x_coord.append(x)
                y_coord.append(y)
            fig, axes = plt.subplots(1,5, sharex =True, sharey =True)
            ax = axes.ravel()
            ax[0].imshow(smfish_image)
            ax[0].plot(y_coord, x_coord, 'ro', markerfacecolor='none')
            ax[0].set_title('smFISH of gene' + str(gene))
            for i in range(1,5):
                image_merfish_name =  raw_folder +  gene_bit_dict_merfish[gene][i-1] + '_' + fov + '.dax'
                image_merfish = readDoryImg(image_merfish_name)
                register_merfish,_ = register_slice(smfish_image, image_merfish)
                ax[i].imshow(register_merfish)
                ax[i].set_title('MERFISH_' + str(gene_bit_dict_merfish[gene][i-1]))
                # ax[i].plot(y_coord, x_coord, 'bx')
                ax[i].plot(y_coord, x_coord, 'bo', markerfacecolor='none')
            for ax_n in ax:
                ax_n.axis('off')
            fig.suptitle('smFISH with 4 raws on-bit MERFISH')

    def show_smfish_merfish_spot_fov_gene_all(self, fov, genes, raw_folder,bit_gene_dict_smfish, gene_bit_dict_merfish):
        """ This shows 16 bits """
        x_coord = []
        y_coord = []
        for gene in genes:
            image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
            smfish_image = readDoryImg(image_name)
            selected_fov_spot = self.get_by_fov(fov).get_by_gene(gene)
            for spot in selected_fov_spot.all_spots:
                x = spot.x
                y = spot.y
                x_coord.append(x)
                y_coord.append(y)
            fig, axes = plt.subplots(5,4, sharex =True, sharey =True)
            ax = axes.ravel()
            # fig.delaxes(ax[2, 1])
            smfish_image = ((smfish_image - smfish_image.min()) / (smfish_image.max() - smfish_image.min())) * 65535
            ax[0].imshow(smfish_image)
            ax[0].plot(y_coord, x_coord, 'ro', markerfacecolor='none')
            ax[0].set_title('smFISH of gene' + str(gene))
            for i in range(1,17):
                image_merfish_name =  raw_folder +  gene_bit_dict_merfish[gene][i-1] + '_' + fov + '.dax'
                image_merfish = readDoryImg(image_merfish_name)
                register_merfish,_ = register_slice(smfish_image, image_merfish)
                register_merfish = ((register_merfish - register_merfish.min()) / (register_merfish.max() - register_merfish.min())) * 65535
                ax[i].imshow(register_merfish)
                ax[i].set_title('MERFISH_' + str(gene_bit_dict_merfish[gene][i-1]))
                # ax[i].plot(y_coord, x_coord, 'bx')
                ax[i].plot(y_coord, x_coord, 'bo', markerfacecolor='none')
            for ax_n in ax:
                ax_n.axis('off')
            fig.suptitle('smFISH with 16 raws on-bit MERFISH')

    def show_smfish_merfish_spot_fov_gene_all_crop(self, fov, genes, raw_folder,bit_gene_dict_smfish, gene_bit_dict_merfish,bbox_x,bbox_y):
        """ This shows 16 bits but cropped """
        x_coord = []
        y_coord = []
        for gene in genes:
            image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
            smfish_image = readDoryImg(image_name)
            selected_fov_spot = self.get_by_fov(fov).get_by_gene(gene)
            for spot in selected_fov_spot.all_spots:
                x = spot.x
                y = spot.y
                x_coord.append(x)
                y_coord.append(y)
            fig, axes = plt.subplots(5,4, sharex =True, sharey =True)
            ax = axes.ravel()
            # fig.delaxes(ax[2, 1])
            cropped_smfish = smfish_image[bbox_y-6:bbox_y+6,bbox_x-6:bbox_x+6]
            cropped_smfish = ((cropped_smfish - cropped_smfish.min()) / (cropped_smfish.max() - cropped_smfish.min())) * 65535
            ax[0].imshow(cropped_smfish)
            # ax[0].plot(y_coord, x_coord, 'ro', markerfacecolor='none')
            ax[0].set_title('smFISH of gene' + str(gene))
            for i in range(1,17):
                image_merfish_name =  raw_folder +  gene_bit_dict_merfish[gene][i-1] + '_' + fov + '.dax'
                image_merfish = readDoryImg(image_merfish_name)
                register_merfish,_ = register_slice(smfish_image, image_merfish)
                register_merfish_cropped = register_merfish[bbox_y-6:bbox_y+6,bbox_x-6:bbox_x+6]
                register_merfish_cropped = ((register_merfish_cropped - register_merfish_cropped.min()) / (register_merfish_cropped.max() - register_merfish_cropped.min())) * 65535
                ax[i].imshow(register_merfish_cropped)
                ax[i].set_title('MERFISH_' + str(gene_bit_dict_merfish[gene][i-1]))
                # ax[i].plot(y_coord, x_coord, 'bx')
                # ax[i].plot(y_coord, x_coord, 'bo', markerfacecolor='none')
            for ax_n in ax:
                ax_n.axis('off')
            fig.suptitle('smFISH with 16 raws on-bit MERFISH')

    def plot_fpkm_corr(self, df_FPKM):
        counts = []
        fpkm = []
        gene_list = []
        for gene in self.all_gene:
            counts.append(self.get_by_gene(gene).num_spots)
            row = df_FPKM.loc[df_FPKM['genes'] == gene]
            fpkm.append(float(row['FPKM']))
            gene_list.append(gene)
        FPKM_correlation_no_blank(counts, fpkm, gene_list)

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


def xy_coord_smfish(image_file, filter_param, threshold):
    image = readDoryImg(image_file)
    image_norm = norm_image_func(image)
    filter_image = filter_func(filter_param, image_norm)
    gene_coordinates = find_peak_local_max(filter_image, threshold)
    return gene_coordinates, filter_image


def norm_image_func(image):
    """" Params: image (2 or 3D arr)
         Return: normalise image arr by min = 0 and max = 1 (2 or 3D arr)
    """
    image_max = np.max(image)
    image_min = np.min(image)
    print("minimum intensity of image =", image_min)
    print("maximum intensity of image =", image_max)
    return (image - image_min) / (image_max - image_min)

def find_peak_local_max(image_arr, thres):
    """ Params: image (2D or 3D arr) : image (z,x,y) or (x,y)
                threshold (float) : absolute threshold cutoff
        Return: coordinates (z,x,y) or (x,y) of peaks
    """
    thres_mode = 'rel'
    if thres_mode == 'rel':
        coordinates = peak_local_max(image_arr, min_distance=2, threshold_rel=thres)
    if thres_mode == 'abs':
        coordinates = peak_local_max(image_arr, min_distance=2, threshold_abs=thres)
    return coordinates

def create_smfish_obj(raw_img_folder, gene_bit_dict_smfish, threshold_dict, filter_param, dist_pix = 4, fovs=25):
    spots_obj_list = []
    list_files = glob.glob(raw_img_folder+'/*.dax')
    for file in list_files:
        fov = file.split("/")[-1].split("\\")[-1].split("_")[-1].split(".")[0]
        if int(fov) < int(fovs):
            gene_list = file.split("/")[-1].split("\\")[-1].split("_")[0:2]
            gene_color = '_'.join(gene_list)
            image_arr = readDoryImg(file)
            if gene_color in gene_bit_dict_smfish:
                gene = gene_bit_dict_smfish[gene_color]
                threshold = threshold_dict[gene_color]
                print((file,threshold))
                coord, filter_image = xy_coord_smfish(file, filter_param, threshold)
                for x , y in zip(coord[:,0],coord[:,1]):
                    spot_size = 1
                    area_arr = image_arr[x-spot_size:x+spot_size, y-spot_size:y+spot_size]
                    mean_inten = np.mean(area_arr)
                    peak_inten = np.max(image_arr[x-spot_size:x+spot_size, y-spot_size:y+spot_size])
                    peak_inten_filter = filter_image[x, y]
                    if x < 2040:
                        bg1 = image_arr[x+6, y]
                        bg2 = image_arr[x-6, y]
                    else:
                        bg1 = image_arr[2046, y]
                        bg2 = image_arr[2046, y]
                    mean_background = np.mean([bg1,bg2])
                    snr = peak_inten/mean_background
                    spot_obj = Spot(x, y, mean_inten, fov, gene, 'smfish', peak_inten = peak_inten, peak_inten_filter = peak_inten_filter, snr = snr)
                    spots_obj_list.append(spot_obj)

    return   Spots(spots_obj_list)

def create_smfish_obj2(raw_img_folder, gene_bit_dict_smfish, threshold_dict, filter_param, fov_list, dist_pix = 4):
    spots_obj_list = []
    list_files = glob.glob(raw_img_folder+'/*.dax')
    for file in list_files:
        fov = file.split("/")[-1].split("\\")[-1].split("_")[-1].split(".")[0]
        if int(fov) in fov_list:
            gene_list = file.split("/")[-1].split("\\")[-1].split("_")[0:2]
            gene_color = '_'.join(gene_list)
            image_arr = readDoryImg(file)
            if gene_color in gene_bit_dict_smfish:
                gene = gene_bit_dict_smfish[gene_color]
                threshold = threshold_dict[gene_color]
                print((file,threshold))
                coord, filter_image = xy_coord_smfish(file, filter_param, threshold)
                for x , y in zip(coord[:,0],coord[:,1]):
                    spot_size = 1
                    area_arr = image_arr[x-spot_size:x+spot_size, y-spot_size:y+spot_size]
                    mean_inten = np.mean(area_arr)
                    peak_inten = np.max(image_arr[x-spot_size:x+spot_size, y-spot_size:y+spot_size])
                    peak_inten_filter = filter_image[x, y]
                    if x < 2040:
                        bg1 = image_arr[x+6, y]
                        bg2 = image_arr[x-6, y]
                    else:
                        bg1 = image_arr[2046, y]
                        bg2 = image_arr[2046, y]
                    mean_background = np.mean([bg1,bg2])
                    snr = peak_inten/mean_background
                    spot_obj = Spot(x, y, mean_inten, fov, gene, 'smfish', peak_inten = peak_inten, peak_inten_filter = peak_inten_filter, snr = snr)
                    spots_obj_list.append(spot_obj)

    return   Spots(spots_obj_list)

def create_merfish_obj(coords_cache_path, genes_list, fovs=2):
    spot_obj_list = []
    coords_filelist = os.listdir(coords_cache_path)
    print(len(coords_filelist))
    for file in coords_filelist:
        fov = file.split('_')[1]
        print(fov)
        coords_cache = os.path.join(coords_cache_path, file)
        if int(fov) < fovs:
            print(f"Loading spot metrics from FOV {fov}")
            with h5py.File(coords_cache, 'r') as g:
                for gene in genes_list:
                    x_coords = g[gene][:, 1]
                    y_coords = g[gene][:, 2]
                    spot_sizes = g[gene][:, 3]
                    min_dists = g[gene][:, 4]
                    mean_intens = g[gene][:, 5]
                    bfs = g[gene][:,6]
                    for x_coord, y_coord, spot_size, min_dist, mean_inten, bfscore in zip(x_coords, y_coords,spot_sizes, min_dists,mean_intens, bfs):
                        spot_obj = Spot(x_coord, y_coord, mean_inten, fov, gene, 'merfish', spot_size=spot_size, min_dist=min_dist, blank_fr_score= bfscore)
                        spot_obj_list.append(spot_obj)
    return  Spots(spot_obj_list)

def create_merfish_obj2(coords_cache_path, genes_list, fov_list):
    spot_obj_list = []
    coords_filelist = os.listdir(coords_cache_path)
    for file in coords_filelist:
        fov = file.split('_')[1]
        if int(fov) in fov_list:
            coords_cache = os.path.join(coords_cache_path, file)
            print(f"Loading spot metrics from FOV {fov}")
            with h5py.File(coords_cache, 'r') as g:
                for gene in genes_list:
                    x_coords = g[gene][:, 1]
                    y_coords = g[gene][:, 2]
                    spot_sizes = g[gene][:, 3]
                    min_dists = g[gene][:, 4]
                    mean_intens = g[gene][:, 5]
                    bfs = g[gene][:,6]
                    for x_coord, y_coord, spot_size, min_dist, mean_inten, bfscore in zip(x_coords, y_coords,spot_sizes, min_dists,mean_intens, bfs):
                        spot_obj = Spot(x_coord, y_coord, mean_inten, fov, gene, 'merfish', spot_size=spot_size, min_dist=min_dist, blank_fr_score= bfscore)
                        spot_obj_list.append(spot_obj)
    return  Spots(spot_obj_list)

def get_arr_from_hdf5file(imagedata_file, x_coords, y_coords, arr_type):
    with h5py.File(imagedata_file, 'r') as f:
        if arr_type == 'normalized_inten':
            used_arr = np.array(f["normalized_clipped"]).squeeze()
        elif arr_type == 'unitnormalized':
            used_arr = np.array(f["unitnormalized"]).squeeze()
        new_arr = np.zeros((len(x_coords), used_arr.shape[2]))
        for i in range(len(x_coords)):
            x_coord = int(x_coords[i])
            y_coord = int(y_coords[i])
            for z in range(0,used_arr.shape[2]):
                value = used_arr[x_coord, y_coord, z]
                new_arr [i,z] = value
    return new_arr

def bit_inten_from_registered_image(imagedata_file, x_coords, y_coords):
    with h5py.File(imagedata_file, 'r') as f:
        regis_arr = np.array(f["registered"]).squeeze()
        inten_arr = np.zeros((len(x_coords),regis_arr.shape[2]))
        snr_arr = np.zeros((len(x_coords),regis_arr.shape[2]))
        for i in range(len(x_coords)):
            x_coord = int(x_coords[i])
            y_coord = int(y_coords[i])
            for z in range(0,regis_arr.shape[2]):
                #inten_list = []
                #inten = np.mean(regis_arr[x_coord-1:x_coord+1, y_coord-1:y_coord+1, z])
                inten = np.mean(regis_arr[x_coord, y_coord, z])
                if x_coord < 2040:
                    bg1 = regis_arr[x_coord+6, y_coord, z]
                    bg2 = regis_arr[x_coord-6, y_coord, z]
                else:
                    bg1 = regis_arr[2046, y_coord, z]
                    bg2 = regis_arr[2046, y_coord, z]
                snr = inten/np.mean([bg1,bg2])
                inten_arr[i,z] = inten
                snr_arr[i,z] = snr
    return inten_arr, snr_arr

def create_completed_merfish_obj(coords_cache_path, imagedata_path, genes_list, fovs=2):
    spot_obj_list = []
    coords_filelist = os.listdir(coords_cache_path)
    imagedata_filelist = os.listdir(imagedata_path)
    for file, imagedata_file in zip(coords_filelist,imagedata_filelist):
        fov = file.split('_')[1]
        coords_cache = os.path.join(coords_cache_path, file)
        imagedata_cache = os.path.join(imagedata_path, imagedata_file)
        print("coords_cache", coords_cache)
        print("imagedata_cache", imagedata_cache)
        if int(fov) < fovs:
            print(f"Loading spot metrics from FOV {fov}")
            with h5py.File(coords_cache, 'r') as g:
                for gene in genes_list:
                    x_coords = g[gene][:, 1]
                    y_coords = g[gene][:, 2]
                    spot_sizes = g[gene][:, 3]
                    min_dists = g[gene][:, 4]
                    mean_intens = g[gene][:, 5]
                    registered_inten, snr_all = bit_inten_from_registered_image(imagedata_cache, x_coords, y_coords)
                    bit_inten = get_arr_from_hdf5file(imagedata_cache, x_coords, y_coords, 'normalized_inten')
                    unitnorm_vector = get_arr_from_hdf5file(imagedata_cache, x_coords, y_coords, "unitnormalized")
                    print(bit_inten.shape)
                    for x_coord, y_coord, spot_size, min_dist, mean_inten, bit_intensity, registered_intensity, unit_vector, snr in zip(x_coords, y_coords,spot_sizes, min_dists,mean_intens, bit_inten, registered_inten, unitnorm_vector, snr_all):
                        spot_obj = Spot(x_coord, y_coord, mean_inten, fov, gene, 'merfish', spot_size=spot_size, min_dist=min_dist, bit_inten = bit_intensity,
                                        registered_inten = registered_intensity, unit_vector =unit_vector, snr = snr)
                        spot_obj_list.append(spot_obj)
    return  Spots(spot_obj_list)

def create_shift_spot_obj(raw_folder, merfish_output_path, bit_gene_dict_smfish, spot_obj_merfish):
    final_spots = Spots([])
    for fov in spot_obj_merfish.all_fov:
        merfish_img_name = merfish_output_path + 'FOV_' + fov + '_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax'
        merfish_img = readDoryImg(merfish_img_name)
        for gene in spot_obj_merfish.all_gene:
            smfish_img_name = raw_folder + bit_gene_dict_smfish[gene]+'_' + fov + '.dax'
            smfish_img = readDoryImg(smfish_img_name)
            _, shifts = register_slice(smfish_img, merfish_img)
            spots_original = spot_obj_merfish.get_by_fov(fov)
            spots_original_gene = spots_original.get_by_gene(gene)
            new_spot_shift = spots_original_gene.shift_spots(shifts)
            final_spots = Spots.combine_spots(final_spots,new_spot_shift)
    return final_spots

def translate_shift_to_raw_merfish(fov, gene, raw_folder, merfish_output_path, bit_gene_dict_smfish, gene_bit_dict_merfish):


    merfish_img_name = merfish_output_path + 'FOV_' + fov + '_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax'
    merfish_img = readDoryImg(merfish_img_name)
    smfish_img_name = raw_folder + bit_gene_dict_smfish[gene]+'_' + fov + '.dax'
    smfish_img = readDoryImg(smfish_img_name)
    _, shifts = register_slice(smfish_img, merfish_img)

    image_merfish1 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][0] + '_' + fov + '.dax')
    image_merfish2 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][1] + '_' + fov + '.dax')
    image_merfish3 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][2] + '_' + fov + '.dax')
    image_merfish4 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][3] + '_' + fov + '.dax')

    image_merfish1_register = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image_merfish1), shifts))
    image_merfish2_register = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image_merfish2), shifts))
    image_merfish3_register = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image_merfish3), shifts))
    image_merfish4_register = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image_merfish4), shifts))

    return image_merfish1_register.real, image_merfish2_register.real, image_merfish3_register.real, image_merfish4_register.real


def output_raw_after_register(fov, gene, raw_folder,bit_gene_dict_smfish, gene_bit_dict_merfish):
    image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
    smfish_image = readDoryImg(image_name)
    image_merfish1 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][0] + '_' + fov + '.dax')
    register_merfish1, _ = register_slice(smfish_image, image_merfish1)
    image_merfish2 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][1] + '_' + fov + '.dax')
    register_merfish2, _ = register_slice(smfish_image, image_merfish2)
    image_merfish3 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][2] + '_' + fov + '.dax')
    register_merfish3, _ = register_slice(smfish_image, image_merfish3)
    image_merfish4 = readDoryImg(raw_folder +  gene_bit_dict_merfish[gene][3] + '_' + fov + '.dax')
    register_merfish4, _ = register_slice(smfish_image, image_merfish4)

    return register_merfish1, register_merfish2, register_merfish3, register_merfish4



def translate_smfish_to_merfish_image(only_smfish_obj, imagedata_path):
    """create new Spots object to see the behavior of spot that only called in smFISH in MERFISH look like
    translate x,y of only_smfish object to create new object contain same info in merfish image"""
    spot_obj_list = []
    for fov in only_smfish_obj.all_fov:

        fov_spot = only_smfish_obj.get_by_fov(fov)
        imagedata_filename = 'FOV_'+fov+'_imagedata_iter0.hdf5'
        imagedata_cache = os.path.join(imagedata_path, imagedata_filename)
        for gene in only_smfish_obj.all_gene:
            gene_spot = fov_spot.get_by_gene(gene)
            x_coords = []
            y_coords = []
            mean_inten_all = []
            snr_smfish_all = []
            for spot in gene_spot.all_spots:
                x_coords.append(spot.x)
                y_coords.append(spot.y)
                mean_inten_all.append(spot.mean_inten)
                snr_smfish_all.append(spot.snr)
            registered_inten, snr_all = bit_inten_from_registered_image(imagedata_cache, x_coords, y_coords)
            bit_inten = get_arr_from_hdf5file(imagedata_cache, x_coords, y_coords, 'normalized_inten')
            unitnorm_vector = get_arr_from_hdf5file(imagedata_cache, x_coords, y_coords, "unitnormalized")
            for x_coord, y_coord, mean_inten, bit_intensity, registered_intensity, unit_vector, snr, snr_smfish in zip(x_coords, y_coords, mean_inten_all, bit_inten, registered_inten, unitnorm_vector, snr_all, snr_smfish_all):
                spot_obj = Spot(x_coord, y_coord, mean_inten, fov, gene, 'merfish', bit_inten = bit_intensity,
                                registered_inten = registered_intensity, unit_vector =unit_vector, snr = snr, snr_smfish = snr_smfish)
                spot_obj_list.append(spot_obj)
    return Spots(spot_obj_list)

def possible_of_xy_nearby(spot_obj_smfish, spot_obj_merfish, dist_pixel):

    overlap_smfish_all_gene = []
    overlap_merfish_all_gene = []
    nonoverlap_smfish_all_gene = []
    nonoverlap_merfish_all_gene = []


    for fov in spot_obj_merfish.all_fov:
        smfish_spot_fov = spot_obj_smfish.get_by_fov(fov)
        merfish_spot_fov = spot_obj_merfish.get_by_fov(fov)
        for gene in spot_obj_merfish.all_gene:
            smfish_overlap_spot = []
            merfish_overlap_spot = []
            new_overlap_smfish = []
            new_overlap_merfish = []
            non_overlap_smfish = []
            non_overlap_merfish = []
            new_new_merfish_overlap = []
            new_new_smfish_overlap = []
            print(gene)
            smfish_spot_fov_gene = smfish_spot_fov.get_by_gene(gene)
            merfish_spot_fov_gene = merfish_spot_fov.get_by_gene(gene)

            for i in range(len(smfish_spot_fov_gene.all_spots)):
                for j in range(len(merfish_spot_fov_gene.all_spots)):
                    #area_merfish_spot = smfish_spot_fov_gene.pick_spots_in_area(merfish_spot_fov_gene.all_spots[j].x, merfish_spot_fov_gene.all_spots[j],4,4)
                    if Spot.is_overlap(smfish_spot_fov_gene.all_spots[i], merfish_spot_fov_gene.all_spots[j], dist_pixel):
                        smfish_overlap_spot.append(smfish_spot_fov_gene.all_spots[i])
                        merfish_overlap_spot.append(merfish_spot_fov_gene.all_spots[j])

            spot_sm_fov_gene = Spots(smfish_overlap_spot)
            spot_mer_fov_gene  = Spots(merfish_overlap_spot)
            for spot in spot_sm_fov_gene.all_spots:
                if not spot in new_overlap_smfish:
                    new_overlap_smfish.append(spot)
                    index = spot_sm_fov_gene.all_spots.index(spot)
                    new_overlap_merfish.append(spot_mer_fov_gene.all_spots[index])

            spots_mer = Spots(new_overlap_merfish)
            spots_sm = Spots(new_overlap_smfish)
            for spot in spots_mer.all_spots:
                if not spot in new_new_merfish_overlap:
                    new_new_merfish_overlap.append(spot)
                    index = spots_mer.all_spots.index(spot)
                    new_new_smfish_overlap.append(spots_sm.all_spots[index])


            # find non-overlap spots
            for i in range(len(smfish_spot_fov_gene.all_spots)):
                if smfish_spot_fov_gene.all_spots[i] not in new_new_smfish_overlap:
                    non_overlap_smfish.append(smfish_spot_fov_gene.all_spots[i])
            for i in range(len(merfish_spot_fov_gene.all_spots)):
                if merfish_spot_fov_gene.all_spots[i] not in new_new_merfish_overlap:
                    non_overlap_merfish.append(merfish_spot_fov_gene.all_spots[i])

            overlap_smfish_all_gene.append(new_new_smfish_overlap)
            overlap_merfish_all_gene.append(new_new_merfish_overlap)
            nonoverlap_smfish_all_gene.append(non_overlap_smfish)
            nonoverlap_merfish_all_gene.append(non_overlap_merfish)

    overlap_smfish_final = sum(overlap_smfish_all_gene, [])
    overlap_merfish_final  = sum(overlap_merfish_all_gene, [])
    only_smfish_final  = sum(nonoverlap_smfish_all_gene, [])
    only_merfish_final  = sum(nonoverlap_merfish_all_gene, [])


    # check if one spot pair with more than 2 spots

    # overlap_smfish_all_gene = []
    # overlap_merfish_all_gene = []
    # nonoverlap_smfish_all_gene = []
    # nonoverlap_merfish_all_gene = []
    # for fov in overlap_smfish_spots.all_fov:
    #     spot_sm_fov = overlap_smfish_spots.get_by_fov(fov)
    #     spot_mer_fov = overlap_merfish_spots.get_by_fov(fov)
    #     for gene in overlap_smfish_spots.all_gene:
    #         new_overlap_smfish = []
    #         new_overlap_merfish = []
    #         non_overlap_smfish = []
    #         non_overlap_merfish = []
    #         spot_sm_fov_gene = spot_sm_fov.get_by_gene(gene)
    #         spot_mer_fov_gene = spot_mer_fov.get_by_gene(gene)
    #         for spot in spot_sm_fov_gene.all_spots:
    #             if not spot in new_overlap_smfish:
    #                 new_overlap_smfish.append(spot)
    #                 index = spot_sm_fov_gene.all_spots.index(spot)
    #                 new_overlap_merfish.append(spot_mer_fov_gene.all_spots[index])
    #
    #         # find non-overlap spots
    #         for i in range(len(spot_obj_smfish.get_by_fov(fov).get_by_gene(gene).all_spots)):
    #             if spot_obj_smfish.get_by_fov(fov).get_by_gene(gene).all_spots[i] not in new_overlap_smfish:
    #                 non_overlap_smfish.append(spot_obj_smfish.get_by_fov(fov).get_by_gene(gene).all_spots[i])
    #         for i in range(len(spot_obj_merfish.get_by_fov(fov).get_by_gene(gene).all_spots)):
    #             if spot_obj_merfish.get_by_fov(fov).get_by_gene(gene).all_spots[i] not in new_overlap_merfish:
    #                 non_overlap_merfish.append(spot_obj_merfish.get_by_fov(fov).get_by_gene(gene).all_spots[i])
    #
    #         overlap_smfish_all_gene.append(new_overlap_smfish)
    #         overlap_merfish_all_gene.append(new_overlap_merfish)
    #         nonoverlap_smfish_all_gene.append(non_overlap_smfish)
    #         nonoverlap_merfish_all_gene.append(non_overlap_merfish)

    return Spots(overlap_smfish_final), Spots(overlap_merfish_final), Spots(only_smfish_final), Spots(only_merfish_final)

def create_newsmfish_obj_and_plotCDF(smfish_obj_spots, method):
    threshold_offset = 20
    num_bins = 200
    group_new_spots_obj = []
    for gene in smfish_obj_spots.all_gene:
        smfish_spot_fov_gene = smfish_obj_spots.get_by_gene(gene).all_spots
        mean_inten_spots = []
        for spot in smfish_spot_fov_gene:
            if method == 'peak':
                mean_inten_spots.append(spot.peak_inten)
                intensity_metric = 'peak'
            elif method == 'peak_filter':
                mean_inten_spots.append(spot.peak_inten_filter)
                intensity_metric = 'peak (filter)'
            elif method == 'mean':
                mean_inten_spots.append(spot.mean_inten)
                intensity_metric = 'mean'

        peak_intensities = np.array(mean_inten_spots)
        print(f"Number of peaks in gene", peak_intensities.shape[0])

        peak_inds_sorted = np.argsort(peak_intensities)
        peak_intensities_sorted = peak_intensities[peak_inds_sorted]

        cdf_end = max(peak_intensities)
        cdf_start = min(peak_intensities)
        thresholds = np.linspace(cdf_start, cdf_end, num_bins)
        counts=[]

        for thresh in thresholds:
            count = peak_intensities_sorted[peak_intensities_sorted>thresh].shape[0]
            counts.append(count)

        deltas = []
        local_min_ind = []
        for i in range(1,len(counts)):
            deltas.append(counts[i]-counts[i-1])
            if i>1 and deltas[i-2] > deltas[i-1]:
                local_min_ind.append(i-2)
        if len(local_min_ind) > 0:
            knee_threshold = thresholds[min(local_min_ind) + threshold_offset]
        else:
            knee_threshold=None

        gene_counts = peak_intensities[peak_intensities > knee_threshold].shape[0]
        intensity_thresholds = knee_threshold

        print(f"Intensity threshold at {knee_threshold}")
        print(f"Number of peaks above threshold: {gene_counts}")
        print("----------------------")
        # CDF plot
        plt.figure()
        plt.plot(thresholds,counts, marker=".")
        if knee_threshold:
            plt.axvline(knee_threshold, ls='-',c='r')
        plt.yscale("log")
        plt.ylabel('# spots above threshold')
        plt.xlabel(f"{intensity_metric} intensity")
        plt.xlim([cdf_start,cdf_end])
        plt.title(f"threshold = {knee_threshold}")
        #plt.savefig(os.path.join(qc_path,f"smFISH_CDF_gene_{gene}.png"))
        #plt.close()

        # PDF plot
        # plt.figure()
        # plt.hist(peak_intensities, bins=num_bins)
        # if knee_threshold:
        #     plt.axvline(knee_threshold, ls='-',c='r')
        # plt.yscale("log")
        # plt.ylabel('# spots')
        # plt.xlabel(f"{intensity_metric} intensity")
        # plt.xlim([cdf_start,cdf_end])
        # plt.title("threshold = {knee_threshold}")
        #plt.savefig(os.path.join(self.qc_path,f"smFISH_PDF_gene_{gene}.png"))
        #plt.close()

        # Get spots that higher than threshold
        new_spots_obj = []
        for spot in smfish_spot_fov_gene:

            if spot.peak_inten > intensity_thresholds and method == 'peak':
                new_spots_obj.append(spot)

            elif spot.peak_inten_filter > intensity_thresholds and method == 'peak_filter':
                new_spots_obj.append(spot)

            elif spot.mean_inten > intensity_thresholds and method == 'mean':
                new_spots_obj.append(spot)

        group_new_spots_obj += new_spots_obj

    return Spots(group_new_spots_obj), local_min_ind


# def plot_venn(num1, num2, num_overlap, gene_name):

#     percent_overlap_merfish = np.round((num_overlap /num2) *100, 2 )
#     percent_overlap_smfish = np.round((num_overlap /num1) *100,2)
#     plt.figure(figsize=(10, 8))
#     plt.title("Callouts of gene "+ gene_name)
#     plt.text(0.5, 0.5, '% colocalised in smFISH = ' + str(percent_overlap_smfish),  horizontalalignment='left',verticalalignment='bottom')
#     plt.text(0.5, 0.6, '% colocalised in MERFISH = ' + str(percent_overlap_merfish),  horizontalalignment='left',verticalalignment='bottom')
#     v_fake = venn2(subsets = (num1-num_overlap, num2-num_overlap, num_overlap), set_labels = ('', ''))
#     print("gene nette", num1,num2,num_overlap)
#     for text in v_fake.set_labels:
#         text.set_fontsize(20)
#     for text in v_fake.subset_labels:
#         text.set_fontsize(14)
#     v_fake.get_label_by_id('11').set_text(str(num_overlap)) #label redundant spots firstto be consistant of 4 genes
#     plt.savefig('Y:/MIKE/dataset/20191108_JM_L39_AML_2cD/100uM_data/threshold_smfish/cutwithoutCDF/' +'venn_of_gene '+ gene_name +'_numsposts.png')

#     #label by percentage (true +, true -, false +)
#     total_spots_no_overlap = (num1 + num2)- num_overlap
#     percent_only_smfish= np.round(((num1 - num_overlap)/total_spots_no_overlap) *100,2)
#     percent_only_merfish= np.round(((num2 - num_overlap)/total_spots_no_overlap)* 100,2)
#     percent_true_overlap = np.round((num_overlap/total_spots_no_overlap )* 100,3)
#     plt.figure(figsize=(10, 8))
#     plt.title("% True positive, False positive and False negative of gene "+ gene_name)
#     v = venn2(subsets = (num1-num_overlap, num2-num_overlap, num_overlap), set_labels = ('smFISH', 'MERFISH'))
#     v.get_label_by_id('10').set_text(str(percent_only_smfish) + '%')
#     v.get_label_by_id('11').set_text(str(percent_true_overlap) + '%')
#     v.get_label_by_id('01').set_text(str(percent_only_merfish) + '%')
#     plt.savefig('Y:/MIKE/dataset/20191108_JM_L39_AML_2cD/100uM_data/threshold_smfish/cutwithoutCDF/' +'venn_of_gene '+ gene_name +'.png')

# def plot_various_thres_smfish(group_spot_merfish2_shift, raw_folder, gene_bit_dict_smfish, thres_list, nfov):
#     """This funciont will vary smfish threshold and see number of spots that overlap with merfish
#     It takes long time to run depend on lower/higher threshold and how many threshold"""

#     thresholds = thres_list
#     dict_thres_overlap = {}
#     dict_thres_smfish = {}
#     percent_overlap_dict_in_mer = {}
#     percent_overlap_dict_in_sm = {}
#     dict_thres_allgene = {} # smfish, merfish, overlap
#     for thres in thresholds:
#         dict_thres_allgene[thres] = {}
#         gene_threshold_dict_test = {'Cy5_01': thres, 'Cy7_00':thres, 'Cy7_01':thres, 'Cy5_00':thres}
#         group_spots_smfish = create_smfish_obj(raw_folder , gene_bit_dict_smfish, gene_threshold_dict_test, [300,None], 3, nfov)
#         overlap_smfish, overlap_merfish, only_smfish, only_merfish = possible_of_xy_nearby(group_spots_smfish , group_spot_merfish2_shift, 3)

#         all_smfishspots = group_spots_smfish.num_spots
#         all_merfishspots = group_spot_merfish2_shift.num_spots
#         overlap_merfish_spot = overlap_merfish.num_spots
#         dict_thres_allgene[thres] = ((all_smfishspots, all_merfishspots, overlap_merfish_spot))
#         #plot Ven Diagram of all spots in 4 genes
#         print("Plotting Venn Diagram for all genes")
#         plot_venn(all_smfishspots, all_merfishspots, overlap_merfish_spot, "_all")
#         plt.savefig(raw_folder + "/threshold_smfish/" + "venn_at_thres =" + str(thres) + '.png')
#         plt.close()

#         for gene in overlap_merfish.all_gene:

#             num_smfish_gene = group_spots_smfish.get_by_gene(gene).num_spots
#             num_merfish_gene = group_spot_merfish2_shift.get_by_gene(gene).num_spots
#             num_overlap_gene = overlap_merfish.get_by_gene(gene).num_spots

#             if not gene in dict_thres_overlap:
#                 dict_thres_overlap[gene] = {}

#             if gene in dict_thres_overlap:
#                 dict_thres_overlap[gene][thres] = {}
#                 dict_thres_overlap[gene][thres] = num_overlap_gene

#             if not gene in dict_thres_smfish:
#                 dict_thres_smfish[gene] = {}

#             if gene in dict_thres_smfish:
#                 dict_thres_smfish[gene][thres] = {}
#                 dict_thres_smfish[gene][thres] = num_smfish_gene

#             if not gene in percent_overlap_dict_in_mer:
#                 percent_overlap_dict_in_mer[gene] = {}

#             if gene in percent_overlap_dict_in_mer:
#                 percent_overlap_dict_in_mer[gene][thres] = {}
#                 percent_overlap_dict_in_mer[gene][thres] = np.round((num_overlap_gene/num_merfish_gene)*100,2)

#             if not gene in percent_overlap_dict_in_sm:
#                 percent_overlap_dict_in_sm[gene] = {}

#             if gene in percent_overlap_dict_in_sm:
#                 percent_overlap_dict_in_sm[gene][thres] = {}
#                 percent_overlap_dict_in_sm[gene][thres] = np.round((num_overlap_gene/num_smfish_gene)*100,2)

#             print("Plotting Venn Diagram of gene" + gene)
#             plot_venn(num_smfish_gene, num_merfish_gene, num_overlap_gene, gene)
#             plt.savefig(raw_folder + "/threshold_smfish/" "venn_at_thres =" + str(thres) + '_of_gene_'+ gene + '.png')
#             plt.close()

#     pd.DataFrame(percent_overlap_dict_in_mer).plot(style='o-', title ='% Overlap in MERFISH', xlabel='smFISH threshold', ylabel='% overlap in MERFISH')
#     plt.savefig(raw_folder + "/threshold_smfish/" + "plot_percent_overlap_merfish of all genes.png")
#     pd.DataFrame(percent_overlap_dict_in_sm).plot(style='o-', title ='% Overlap in smFISH', xlabel='smFISH threshold', ylabel='% overlap in smFISH')
#     plt.savefig(raw_folder + "/threshold_smfish/" + "plot_percent_overlap_smfish of all genes.png")

    for data_sm, data_mer in  zip(percent_overlap_dict_in_sm.values(), percent_overlap_dict_in_mer.values()):
        x1=data_sm.keys()
        y1=data_sm.values()

        x2=data_mer.keys()
        y2=data_mer.values()

        plt.figure()
        plt.title("Percent overlap in smfish and merfish of gene " + str(percent_overlap_dict_in_sm.keys()) )
        plt.plot(x1,y1,'o-')
        plt.legend('% overlap in smfish')
        plt.plot(x2,y2,'o-')
        plt.legend('% overlap in merfish')
        plt.show()
        plt.savefig(raw_folder + "/threshold_smfish/" + "plot_percent_overlap_smfish_merfish of genes" + str(percent_overlap_dict_in_sm.keys())+ ".png")

    df1 = pd.DataFrame(dict_thres_allgene)
    df1.to_excel(raw_folder + "/threshold_smfish/"+  "overlap_merfish_smfish_spots_at_threshold_allgene.xlsx")
    df2 = pd.DataFrame(dict_thres_smfish)
    df2.to_excel(raw_folder + "/threshold_smfish/"+  "smfish_callout_numspots_at_threshold_eachgene.xlsx")
    df3 = pd.DataFrame(dict_thres_overlap)
    df3.to_excel(raw_folder + "/threshold_smfish/"+ "smfish_overlap_numspots_at_threshold_eachgene.xlsx")

def FPKM_correlation_no_blank(RNA_counts, FPKM, gene_list,  fig_path = None, verbose = True):
    '''
    FPKM correlation plot
    RNA_counts: vector of integers with the counts for each RNA species
    FPKM: vector of floats with the FPKM values for each RNA species
    Prints FPKM correlation plot with pearson regression score and callout number.
    '''
    FPKM = np.array(FPKM)
    RNA_counts = np.array(RNA_counts)
    logtotal_codecounts = np.log10(RNA_counts)
    logfpkm = np.log10(FPKM)
    gene_name = gene_list
    fpkm_list = list(FPKM)

    if verbose:
        plt.figure(figsize=(5,5))
        plt.scatter(FPKM, RNA_counts)
        for i, txt in enumerate(gene_name):
            plt.annotate(txt, (FPKM[i], RNA_counts[i]))
        for i, txt in enumerate(fpkm_list):
            plt.annotate(txt, (FPKM[i], RNA_counts[i]), xytext=(FPKM[i]+30, RNA_counts[i]+500), color='red')
        plt.xscale("symlog")
        plt.yscale("symlog")
        plt.xlim(left=0)
        plt.ylim(bottom=min(RNA_counts))
        plt.xlabel("FPKM")
        plt.ylabel("RNA Count")
        print("FPKM log10 Correlation without blanks:", pearsonr(logtotal_codecounts, logfpkm))
        print("Spots Detected:", sum(RNA_counts))
        plt.title(f"FPKM Corr: {np.round(pearsonr(logtotal_codecounts, logfpkm)[0],3)} Callouts: {sum(RNA_counts)} ")

        if fig_path:
            plt.savefig(fig_path)
        else:
            plt.show()

def show_smfish_merfish_spot_fov_gene(spotobj, fov, gene, raw_folder, bit_gene_dict_smfish, img1, img2,img3,img4):
    x_coord = []
    y_coord = []
    image_name = raw_folder + bit_gene_dict_smfish[gene] + '_' + fov + '.dax'
    smfish_image = readDoryImg(image_name)
    selected_fov_spot = spotobj.get_by_fov(fov).get_by_gene(gene)
    for spot in selected_fov_spot.all_spots:
        x = spot.x
        y = spot.y
        x_coord.append(x)
        y_coord.append(y)
    fig, axes = plt.subplots(1,5, sharex =True, sharey =True)
    ax = axes.ravel()
    ax[0].imshow(smfish_image)
    ax[0].plot(y_coord, x_coord, 'ro')
    ax[0].set_title('smFISH of gene' + str(gene))

    ax[1].imshow(img1)
    ax[2].imshow(img2)
    ax[3].imshow(img3)
    ax[4].imshow(img4)
    for i in range(1,5):
        ax[i].set_title('MERFISH_' + str([i-1]))
        ax[i].plot(y_coord, x_coord, 'bx')
    for ax_n in ax:
        ax_n.axis('off')
    fig.suptitle('smFISH with 4 raws on-bit MERFISH')


