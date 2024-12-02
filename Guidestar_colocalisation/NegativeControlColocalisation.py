### colocalisation negative control

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
save_coord_path = coord_cache_path  + '/coords_cache_BFS_assigned'
nfov = 25
GS_genes_list = ['Acly','Gpam', 'Hnf4a', 'Ube2z']

### LIVER ###
# dataset_name = 'Liver_LM39_49FOVs'
# info_path = '../genebitdict_liver.yml'
# coord_cache_path = '/path_to_raw_data/output/user_49FOVs_mag0_10_lc400_20240722_2210'
# fpkm_path = '../DataForFigures/liver_fpkm_data.tsv'
# output_path = '/path_to_output_folder/'
# save_coord_path = coord_cache_path  + '/coords_cache_BFS_assigned'
# nfov = 49
# GS_genes_list = ['Acly', 'Gpam', 'Hnf4a', 'Pigr']

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

### Colocalisation ###
fov_list = [i for i in range(nfov)]
distance_threshold = 4 # colocalisation distance

group_spot_merfish = create_Merfish_obj_addedfeat(save_coord_path, gene_list_merfish, fov_list)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish)
group_spots_Guidestar = create_Guidestar_obj(raw_folder , gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list)

# Negative control
merfish_gene_ls = []
guidestar_gene_ls = []
merfish_colocalised_counts_ls = []

for merfish_gene in GS_genes_list:
    for guidestar_gene in GS_genes_list:
        
        temp_guidestar_spots = group_spots_Guidestar.get_by_gene(guidestar_gene).reset_colocalisation_status()
        temp_merfish_spots = group_spot_merfish_shift.get_by_gene(merfish_gene).reset_colocalisation_status()
        
        Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_negcontrol(temp_guidestar_spots,temp_merfish_spots,distance_threshold)
        
        merfish_gene_ls.append(merfish_gene)
        guidestar_gene_ls.append(guidestar_gene)
        merfish_colocalised_counts_ls.append(Merfish_colocalised_spots.num_spots)

negative_control_results = pd.DataFrame(data={'merfish gene': merfish_gene_ls,
                                              'guidestar gene': guidestar_gene_ls,
                                              'colocalisation counts': merfish_colocalised_counts_ls})
negative_control_results.to_csv(os.path.join(output_path,dataset_name + '_negativecontrolresults.csv'))


### DISTANCE OPTIMIZATION ###
merfish_gene_ls = []
guidestar_gene_ls = []
merfish_colocalised_counts_ls = []
dist_ls = []
for dist in np.arange(0.0, 10.0, 0.5):
    for merfish_gene in GS_genes_list:
        for guidestar_gene in GS_genes_list:
            
            temp_guidestar_spots = group_spots_Guidestar.get_by_gene(guidestar_gene).reset_colocalisation_status()
            temp_merfish_spots = group_spot_merfish_shift.get_by_gene(merfish_gene).reset_colocalisation_status()
            
            Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_negcontrol(temp_guidestar_spots,temp_merfish_spots,dist)
            
            merfish_gene_ls.append(merfish_gene)
            guidestar_gene_ls.append(guidestar_gene)
            merfish_colocalised_counts_ls.append(Merfish_colocalised_spots.num_spots)
            dist_ls.append(dist)

dist_opt_results = pd.DataFrame(data={'colocal distance': dist_ls,
                                      'merfish gene': merfish_gene_ls,
                                      'guidestar gene': guidestar_gene_ls,
                                      'colocalisation counts': merfish_colocalised_counts_ls})
dist_opt_results.to_csv(os.path.join(output_path,dataset_name + '_distopt_results.csv'))