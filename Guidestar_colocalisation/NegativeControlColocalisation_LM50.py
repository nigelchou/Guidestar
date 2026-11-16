### colocalisation negative control

import pandas as pd
import os
from ColocalisationPipelineFunctions_v3 import *
import yaml

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


def txt_to_dict(yml_file):
    with open(yml_file) as file:
        user_dict= yaml.load(file, Loader=yaml.FullLoader)
    return user_dict

### Input Parameters ###

### mBrain ###
dataset_name = 'mBrain_LM50GS_86FOVs'
_BASE = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3')
info_path = '../genebitdict_brain.yml'
coord_cache_path = os.path.join(_BASE, 'output/user_86FOVs_mag0_10_lc350_20250929_1705/')
fpkm_path = os.path.join(_BASE, 'codebook/fpkm_data.txt')
output_path = os.path.join(_BASE, 'output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver/GS_model_result/')
save_coord_path = coord_cache_path  + '/coords_cache_BFS_assigned'
GS_genes_list = ['Slc1a2', 'Cacna1i', 'Cacna1a']

info_dict = txt_to_dict(info_path)
raw_folder = os.path.join(main_dir, info_dict['raw_folder'])
gene_threshold_dict = info_dict['gene_threshold_dict']
gene_bit_dict_merfish = info_dict['gene_bit_dict_merfish']
bit_gene_dict_Guidestar = info_dict['bit_gene_dict_guidestar']
gene_bit_dict_Guidestar = {v[0]: k for k, v in bit_gene_dict_Guidestar.items()}
gene_list_merfish = info_dict['gene_list_merfish']
merfish_path = os.path.join(main_dir, info_dict['merfish_outpath'])

df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
allgene_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' not in i]
blank_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' in i]

### Check ###
print('Number of blanks is: ', len(blank_list_merfish))
print('Number of genes is: ', len(allgene_list_merfish))
print('Check Guidestar genes are: ', gene_list_merfish)

### Colocalisation ###
fov_list = [(0, 4),(0, 5),(0, 6),(0, 7),(0, 8),(0, 9),
 (1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 7), (1, 8),(1, 9),
 (2, 0),(2, 1),(2, 2),(2, 3),(2, 5),(2, 6),(2, 7),(2, 8),(2, 9),
 (3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
 (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 7), (4, 8), (4, 9),
 (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 7), (5, 8), (5, 9),
 (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
 (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8),
 (8, 1), (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
 (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
distance_threshold = 4 # colocalisation distance

triton_channel_dict = {'Cy7':0, #channel name: frame number in tif
                       'Cy5':1,
                       'Alexa594':2}

group_spot_merfish = create_Merfish_obj_addedfeat(save_coord_path, gene_list_merfish, fov_list, microscope='Triton')
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish, microscope='Triton', mer_img_type = 'ref_bit', ref_hyb='01', ref_channel='Cy5')
group_spots_Guidestar = create_Guidestar_obj(raw_folder, gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list, microscope='Triton', triton_channel_dict=triton_channel_dict)

# Negative control
merfish_gene_ls = []
guidestar_gene_ls = []
merfish_colocalised_counts_ls = []
fov_ls = []

fov_list = [f"{x:03d}_{y:03d}" for x, y in fov_list] # convert (0,9) -> '000_009'
for merfish_gene in GS_genes_list:
    for guidestar_gene in GS_genes_list:
        for fov in fov_list:
        
            temp_guidestar_spots = group_spots_Guidestar.get_by_gene(guidestar_gene).get_by_fov(fov).reset_colocalisation_status()
            temp_merfish_spots = group_spot_merfish_shift.get_by_gene(merfish_gene).get_by_fov(fov).reset_colocalisation_status()
            
            Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_negcontrol(temp_guidestar_spots,temp_merfish_spots,distance_threshold)
            
            merfish_gene_ls.append(merfish_gene)
            guidestar_gene_ls.append(guidestar_gene)
            merfish_colocalised_counts_ls.append(Merfish_colocalised_spots.num_spots)
            fov_ls.append(fov)

negative_control_results = pd.DataFrame(data={'merfish gene': merfish_gene_ls,
                                              'guidestar gene': guidestar_gene_ls,
                                              'colocalisation counts': merfish_colocalised_counts_ls,
                                              'fov': fov_ls})
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
