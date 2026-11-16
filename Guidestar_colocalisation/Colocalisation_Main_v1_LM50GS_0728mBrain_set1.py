import pandas as pd
import os
from ColocalisationPipelineFunctions_v3 import *
import yaml
import matplotlib.pyplot as plt
import re

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


def txt_to_dict(yml_file):
    with open(yml_file) as file:
        user_dict= yaml.load(file, Loader=yaml.FullLoader)
    return user_dict

### Input Parameters ###
#tissue
dataset_name = 'Tissue_LM50GS_86FOVs'
info_path = '../genebitdict_brain.yml'
coord_cache_path = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/output/user_86FOVs_mag0_10_lc350_20250929_1705/')
fpkm_path = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/codebook/fpkm_data.txt')
output_path = os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver/')
nfov = 10 #10*10 for triton

if not os.path.exists(output_path):
    os.mkdir(output_path)

info_dict = txt_to_dict(info_path)
raw_folder = os.path.join(main_dir, info_dict['raw_folder'])

## modified for double GS triton img ##
gene_list_merfish = info_dict['gene_list_merfish']
merfish_path = os.path.join(main_dir, info_dict['merfish_outpath']) #raw_folder + info_dict['merfish_outpath']
hyb_list = [1, 1, 2, 3, 4, 2, 3, 5, 1, 2, 4, 5, 6, 3, 4, 5]
type_list = ['Cy5', 'Cy7', 'Cy7', 'Cy7', 'Cy7', 'Cy5', 'Cy5', 'Cy7', 'Alexa594', 'Alexa594', 'Cy5', 'Cy5', 'Cy5', 'Alexa594', 'Alexa594', 'Alexa594']
# above combination corresponds to merfish bits
codebook_df = pd.read_csv(os.path.join(main_dir, 'Data/20250728_Lm50GS_mbrain_3cT3/codebook/codebook_data.tsv'), sep='\t', header=None)

# gene_threshold_dict = info_dict['gene_threshold_dict']
# gene_bit_dict_merfish = info_dict['gene_bit_dict_merfish']
# gene_bit_dict_Guidestar = info_dict['gene_bit_dict_guidestar']
bit_gene_dict_Guidestar = info_dict['bit_gene_dict_guidestar']
triton_channel_dict = {'Cy7':0, #channel name: frame number in tif
                       'Cy5':1,
                       'Alexa594':2} #usually its the same across all images in Triton. If otherwise, need to modify the code to adapt
gene_bit_dict_Guidestar = {}
for gene, tags in bit_gene_dict_Guidestar.items():
    for tag in tags:
        gene_bit_dict_Guidestar[tag] = gene

# gene_threshold_dict = {'Cy5_08': 0.06, 'Alexa594_06':0.08, 'Cy5_10':0.07, 'Alexa594_08':0.08, 'Cy7_09':0.085, 'Alexa594_13':0.09,
#                        'Cy7_06':0.09, 'Cy7_10':0.075, 'Cy5_12':0.1, 'Alexa594_10':0.08, 'Cy7_07':0.1, 'Cy7_11':0.083,
#                        'Cy5_11':0.065, 'Alexa594_09':0.08, 'Cy5_13':0.075, 'Alexa594_11':0.085, 'Cy7_08':0.09, 'Cy7_12':0.083,
#                        'Alexa594_12':0.083, 'Cy7_13':0.09, 'Cy5_07':0.062, 'Cy5_09':0.12, 'Alexa594_07':0.073}

gene_threshold_dict = {'Cy5_08': 0.06, 'Alexa594_06':0.08, 'Cy5_10':0.07, 'Alexa594_08':0.08, 'Cy7_09':0.085, 'Alexa594_13':0.09,
                       'Cy7_06':0.09, 'Cy7_10':0.075, 'Cy5_12':0.1, 'Alexa594_10':0.08, 'Cy7_07':0.1, 'Cy7_11':0.083,
                       'Cy5_11':0.065, 'Alexa594_09':0.08, 'Cy5_13':0.075, 'Alexa594_11':0.085, 'Cy7_08':0.1, 'Cy7_12':0.083,
                       'Alexa594_12':0.083, 'Cy7_13':0.09, 'Cy5_07':0.062, 'Cy5_09':0.12, 'Alexa594_07':0.1} #change 'Cy7_08':0.1 & 'Alexa594_07':0.1

# gene_threshold_dict = {'Cy7_06':0.11, 'Alexa594_06':0.085, 'Cy7_07':0.1, 'Cy7_08': 0.1, 'Cy5_07': 0.06,
#                        'Alexa594_13': 0.09, 'Cy5_10': 0.09, 'Cy5_12':0.12, 'Alexa594_09': 0.075, 'Alexa594_11':0.11,
#                        'Alexa594_07': 0.1, # signal too low issues, causing signal to 'blend' with noise in some FOVs,
#                        ############### set 2 ###################
#                        'Cy7_10': 0.135, 'Cy7_11': 0.12, 'Cy7_12': 0.1, 'Cy7_13':0.13, 'Cy7_09': 0.15, 'Cy5_09':0.13,
#                        'Alexa594_08': 0.08, 'Cy5_08':0.11, 'Cy5_13': 0.075, 'Cy5_11': 0.065, 'Cy5_13':0.075, 'Alexa594_10':0.13,
#                        'Alexa594_12':0.083}

df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
allgene_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' not in i]
blank_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' in i]

### Check ###
print('Number of blanks is: ', len(blank_list_merfish))
print('Number of genes is: ', len(allgene_list_merfish))
print('Check Guidestar genes are: ', gene_list_merfish)

### Assign BFS ###
save_coord_path = coord_cache_path + '/coords_cache_BFS_assigned'
if not os.path.exists(save_coord_path):
    os.mkdir(save_coord_path)

fpkmData = FpkmData(fpkm_path)
# spotsData = SpotsData(fpkmData, coord_cache_path, microscope_type = 'Triton')
# spotsData.load_spots_from_hdf5()

# spot = generate_BFS_coord(coord_cache_path,
#                           fpkm_path,
#                           microscope_type = 'Triton',
#                           num_bins = 60,
#                           eps = 0,
#                           kde_sigma = 0,
#                           save_coord_path = save_coord_path,
#                           save_qc_path = False)

### Colocalisation ###
fov_list = [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
            (1, 8), (1, 9), (2, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 0), (3, 1),
            (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
            (4, 7), (4, 8), (4, 9), (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 7), (5, 8), (5, 9), (6, 0), (6, 1),
            (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
            (7, 6), (7, 7), (7, 8), (8, 1), (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 0), (9, 1),
            (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
#[(xpos,ypos) for xpos in range(nfov) for ypos in range(nfov)]
distance_threshold = 4 # colocalisation distance

group_spot_merfish = create_Merfish_obj_addedfeat(save_coord_path, gene_list_merfish, fov_list, microscope="Triton")
# group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish, microscope="Triton", shift_method='fourier_based_cross_correlation', mer_img_type='ref_bit', ref_hyb='04', ref_channel='Cy5_04', output_path=output_path)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish, microscope="Triton", shift_method='fourier_based_cross_correlation', mer_img_type='ref_bit', output_path=output_path, ref_hyb='01', ref_channel='Cy5')
group_spots_Guidestar = create_Guidestar_obj(raw_folder, gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list, microscope="Triton", triton_channel_dict=triton_channel_dict)

Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_euclideandist(group_spots_Guidestar,group_spot_merfish_shift,distance_threshold)
colocalisation_counts_df.to_csv(os.path.join(output_path, dataset_name + '_Colocalisation_counts_set1.csv'))

### Generate Training CSV ###
group_blank_merfish = create_Merfish_obj_addedfeat(save_coord_path, blank_list_merfish, fov_list, microscope="Triton")
Merfish_only_spotlist, feature_name = Spots_to_list(Merfish_only_spots, 0)
Merfish_colocalised_spotlist, _ = Spots_to_list(Merfish_colocalised_spots, 1)
Merfish_blanks_spotlist, _ = Spots_to_list(group_blank_merfish, 0)
combined_lists = Merfish_only_spotlist + Merfish_colocalised_spotlist + Merfish_blanks_spotlist

GS_training_df = pd.DataFrame(combined_lists, columns = feature_name)
GS_training_df.to_csv(os.path.join(output_path, dataset_name + '_GuidestarGenes_training_set1.csv'))

### Generate Full MERFISH CSV ###
group_spot_merfish_allgenes = create_Merfish_obj_addedfeat(save_coord_path, allgene_list_merfish, fov_list, microscope="Triton")
Merfish_allgenes_spotlist, feature_name = Spots_to_list(group_spot_merfish_allgenes, 1)
combined_lists_allgenes = Merfish_allgenes_spotlist + Merfish_blanks_spotlist

Allgenes_Merfish_df = pd.DataFrame(combined_lists_allgenes, columns = feature_name)
Allgenes_Merfish_df.to_csv(os.path.join(output_path, dataset_name + '_MerfishData_set1.csv'))

print('Done!')

### Visualization ###
visual_output_path = os.path.join(output_path, '/check_edge_spot_detect')
if not os.path.exists(visual_output_path):
    os.mkdir(visual_output_path)

list_fov = [f"{a:03d}_{b:03d}" for a, b in fov_list]

for gene in ['Cacna1a', 'Cacna1i', 'Slc1a2']:
    chan_hyb = bit_gene_dict_Guidestar[gene][0]
    hyb = chan_hyb.split('_')[1]
    chan = chan_hyb.split('_')[0]
    channel = triton_channel_dict[chan]
    for fov in list_fov:

        GS_img_stak = tifffile.imread(os.path.join(main_dir, f"Data/20250728_Lm50GS_mbrain_3cT3/S10/hyb{hyb}_1/hyb{hyb}_1_MMStack_1-Pos_{fov}.ome.tif"))
        GS_img = GS_img_stak[channel,...]

        mer_shift_y = []
        mer_shift_x = []
        gs_y = []
        gs_x = []
        mer_col_y = []
        mer_col_x = []
        gs_col_y = []
        gs_col_x = []
        for spot in group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene).all_spots:
            mer_shift_x.append(spot.x)
            mer_shift_y.append(spot.y)
        for spot in group_spots_Guidestar.get_by_fov(fov).get_by_gene(gene).all_spots:
            gs_x.append(spot.x)
            gs_y.append(spot.y)
        for spot in Merfish_colocalised_spots.get_by_fov(fov).get_by_gene(gene).all_spots:
            mer_col_x.append(spot.x)
            mer_col_y.append(spot.y)
        for spot in Guidestar_colocalised_spots.get_by_fov(fov).get_by_gene(gene).all_spots:
            gs_col_x.append(spot.x)
            gs_col_y.append(spot.y)


        plt.figure(figsize=(16,16))
        plt.imshow(GS_img,cmap='gray', vmax=np.percentile(GS_img, 99.9))
        plt.title(gene+'_'+fov)
        plt.scatter(mer_shift_y, mer_shift_x,linestyle='None',marker="x",s=4, color='red',label='merfish spots')
        plt.scatter(gs_y, gs_x,linestyle='None',marker="x",s=3, color='blue',label='guidestar spots')
        plt.scatter(mer_col_y, mer_col_x,linestyle='None',marker="o",facecolors='none',edgecolors='yellow',label='colocal merfish')
        plt.scatter(mer_col_y, mer_col_x,linestyle='None',marker="o",facecolors='none',edgecolors='yellow',label='colocal guidestar')
        plt.legend()
        plt.savefig(visual_output_path + f'/{str(gene)}_FOV_{str(fov)}.png', bbox_inches='tight', dpi=200)
        plt.close()


###
# import pickle
# filename = os.path.join(main_dir, "Data/20250728_Lm50GS_mbrain_3cT3/output/user_100FOVs_mag0_10_lc350_20250805_0127/combined_GSab/combined_gs_genes.pkl")
# with open(filename, 'wb') as file:
#     pickle.dump(group_spot_merfish_shift, file)
#     pickle.dump(Merfish_colocalised_spots, file)
#     pickle.dump(Merfish_only_spots, file)
# print(f"Variables saved to {filename}")

### code to check colocal counts per fov ###
# Build co-occurrence matrix
import seaborn as sns
GS_train_sub = GS_training_df[GS_training_df['Label']==1].copy()
GS_train_sub['fov1'] = GS_train_sub['fov'].str.split('_').str[0]
GS_train_sub['fov2'] = GS_train_sub['fov'].str.split('_').str[1]
heatmap_data = GS_train_sub.pivot_table(
    index='fov2',
    columns='fov1',
    values='Label',
    aggfunc='sum'  # or 'sum', 'median', etc.
)

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Total colocalization per fov set 1 genes")
plt.xlabel("fov1")
plt.ylabel("fov2")
plt.tight_layout()
plt.show()

