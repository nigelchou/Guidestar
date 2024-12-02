import sys
sys.path.append("..")

import pandas as pd
import os
from ColocalisationPipelineFunctions_v2 import *
import yaml

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

### Check ###
print('Number of blanks is: ', len(blank_list_merfish))
print('Number of genes is: ', len(allgene_list_merfish))
print('Check Guidestar genes are: ', gene_list_merfish)

### Assign BFS ###
save_coord_path = coord_cache_path  + '/coords_cache_BFS_assigned'
if not os.path.exists(save_coord_path):
    os.mkdir(save_coord_path)

fpkmData = FpkmData(fpkm_path)
spotsData = SpotsData(fpkmData, coord_cache_path, microscope_type = 'Dory')
spotsData.load_spots_from_hdf5()

spot = generate_BFS_coord(coord_cache_path,
                          fpkm_path,
                          microscope_type = 'Dory',
                          num_bins = 60,
                          eps = 0,
                          kde_sigma = 0,
                          save_coord_path = save_coord_path,
                          save_qc_path = False)

### Colocalisation ###
fov_list = [i for i in range(nfov)]
distance_threshold = 4 # colocalisation distance

group_spot_merfish = create_Merfish_obj_addedfeat(save_coord_path, gene_list_merfish, fov_list)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish)
group_spots_Guidestar = create_Guidestar_obj(raw_folder , gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list)

Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_euclideandist(group_spots_Guidestar,group_spot_merfish_shift,distance_threshold)
colocalisation_counts_df.to_csv(os.path.join(output_path, dataset_name + '_Colocalisation_counts.csv'))

### Generate Training CSV ###
group_blank_merfish = create_Merfish_obj_addedfeat(save_coord_path, blank_list_merfish, fov_list)
Merfish_only_spotlist, feature_name = Spots_to_list(Merfish_only_spots, 0)
Merfish_colocalised_spotlist, _ = Spots_to_list(Merfish_colocalised_spots, 1)
Merfish_blanks_spotlist, _ = Spots_to_list(group_blank_merfish, 0)
combined_lists = Merfish_only_spotlist + Merfish_colocalised_spotlist + Merfish_blanks_spotlist

GS_training_df = pd.DataFrame(combined_lists, columns = feature_name)
GS_training_df.to_csv(os.path.join(output_path, dataset_name + '_GuidestarGenes_training.csv'))

### Generate Full MERFISH CSV ###
group_spot_merfish_allgenes = create_Merfish_obj_addedfeat(save_coord_path, allgene_list_merfish, fov_list)
Merfish_allgenes_spotlist, feature_name = Spots_to_list(group_spot_merfish_allgenes, 1)
combined_lists_allgenes = Merfish_allgenes_spotlist + Merfish_blanks_spotlist

Allgenes_Merfish_df = pd.DataFrame(combined_lists_allgenes, columns = feature_name)
Allgenes_Merfish_df.to_csv(os.path.join(output_path, dataset_name + '_MerfishData.csv'))