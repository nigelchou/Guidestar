import pandas as pd
import os
from ColocalisationPipelineFunctions_v2 import *
import yaml
import matplotlib.pyplot as plt
# from matplotlib_venn import venn2

def txt_to_dict(yml_file):
    with open(yml_file) as file:
        user_dict= yaml.load(file, Loader=yaml.FullLoader)
    return user_dict

### Input Parameters ###

### CELL ###
dataset_name = 'Cell_LM39_25FOVs'
info_path = '../genebitdict_cell.yml'
coord_cache_path = '/path_to_raw_data/output/user_25FOVs_mag0_10_lc400_20230312_1559'
fpkm_path = '../DataForFigures/cell_fpkm_data.tsv'
output_path = '/path_to_output_folder/'
nfov = 25

### LIVER ###
# dataset_name = 'Liver_LM39_49FOVs'
# info_path = '../genebitdict_liver.yml'
# coord_cache_path = '/path_to_raw_data/output/user_49FOVs_mag0_10_lc400_20240722_2210'
# fpkm_path = '../DataForFigures/liver_fpkm_data.tsv'
# output_path = '/path_to_output_folder/'
# nfov = 49

info_dict = txt_to_dict(info_path)
raw_folder = info_dict['raw_folder']
gene_threshold_dict = info_dict['gene_threshold_dict']
gene_bit_dict_merfish = info_dict['gene_bit_dict_merfish']
gene_bit_dict_Guidestar = info_dict['gene_bit_dict_guidestar']
bit_gene_dict_Guidestar = info_dict['bit_gene_dict_guidestar']
gene_list_merfish = info_dict['gene_list_merfish']
merfish_path = info_dict['merfish_outpath']

df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
allgene_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' not in i]
blank_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' in i]

### Colocalisation ###
fov_list = [12]
fov=str(fov_list[0]).zfill(2)
distance_threshold = 4 # colocalisation distance

# edit filepath of the location of the BFS assigned coords cache
BFS_assigned_path = os.path.join(coord_cache_path,'coords_cache_BFS_assigned')
group_spot_merfish = create_Merfish_obj_addedfeat(BFS_assigned_path, gene_list_merfish, fov_list)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish)
group_spots_Guidestar = create_Guidestar_obj(raw_folder , gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list)

Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_euclideandist(group_spots_Guidestar,group_spot_merfish_shift,distance_threshold)

### Visualisations ###
gene_viz = 'Hnf4a'
GS_img = readDoryImg('/path_to_raw_data/Cy7_03_12.dax')
merfish_mip_img = readDoryImg('/path_to_raw_data/output/user_49FOVs_mag0_10_lc400_20230212_2135/FOV_12_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax')
reg_merfish_mip_img,_ = register_slice(GS_img,merfish_mip_img)

plt.figure()
plt.imshow(GS_img,cmap='gray')
plt.title('Visualise what are detected Merfish and Guidestar spots')
for spot in group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='red',label='merfish spots')
for spot in group_spots_Guidestar.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='blue',label='guidestar spots')

plt.figure()
plt.imshow(GS_img,cmap='gray')
plt.title('If cross has circle, means the spot is considered colocalised')
for spot in group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='red',label='merfish spots')
for spot in group_spots_Guidestar.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='blue',label='guidestar spots')
for spot in Merfish_colocalised_spots.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal merfish')
for spot in Guidestar_colocalised_spots.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal guidestar')
