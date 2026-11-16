import sys
sys.path.append("..")

import pandas as pd
import os
from Guidestar_colocalisation.ColocalisationPipelineFunctions_v3 import *
import yaml

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


def txt_to_dict(yml_file):
    with open(yml_file) as file:
        user_dict= yaml.load(file, Loader=yaml.FullLoader)
    return user_dict

### Input Parameters ###
### Tissue ###
dataset_name = 'MERlin_LM39_49FOVs'
info_path = '../genebitdict_liver_benchmarking.yml'
coord_cache_path = os.path.join(main_dir, 'Data/benchmarking/merlin/Final_used_data/analysis_20260521_144432/Data_60uM/AdaptiveFilterBarcodes/')
fpkm_path = os.path.join(main_dir, "Data/60uM/codebook/fpkm_data.tsv")
output_path = os.path.join(coord_cache_path, 'colocalization_output_20240719_1026')
nfov = 49

if not os.path.exists(output_path):
    os.mkdir(output_path)

info_dict = txt_to_dict(info_path)
raw_folder = os.path.join(main_dir, info_dict['raw_folder'])
gene_threshold_dict = info_dict['gene_threshold_dict']
gene_bit_dict_merfish = info_dict['gene_bit_dict_merfish']
bit_gene_dict_Guidestar = info_dict['bit_gene_dict_guidestar']
gene_bit_dict_Guidestar = {v: k for k, v in bit_gene_dict_Guidestar.items()}
gene_list_merfish = info_dict['gene_list_merfish']

df_fpkm = pd.read_csv(fpkm_path, sep='\t', header=None, names=['genes','FPKM'])
allgene_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' not in i]
blank_list_merfish = [i for i in df_fpkm.iloc[:,0] if 'Blank' in i]

### Check ###
print('Number of blanks is: ', len(blank_list_merfish))
print('Number of genes is: ', len(allgene_list_merfish))
print('Check Guidestar genes are: ', gene_list_merfish)

fpkmData = FpkmData(fpkm_path)

def create_Merfish_obj_MERlin(csv_file_folder, gene_list, df_fpkm):
    spot_obj_list = []
    # Combine all the csv in the folder into single dataframe
    all_df = []
    coord_data_pattern = re.compile(r".+\.csv$", flags=re.IGNORECASE)
    for file in os.listdir(csv_file_folder):
        if re.match(coord_data_pattern, file):
            df = pd.read_csv(os.path.join(csv_file_folder, file))
            all_df.append(df)
    full_df = pd.concat(all_df, ignore_index=True) #combine via concat
    full_df['fov'] = full_df['fov'].astype(str).str.zfill(2)
    # merlin's gene names are in indices format -> convert to text gene names
    full_df['gene_name'] = full_df['barcode_id'].map(df_fpkm['genes'].to_dict())
    full_df = full_df[full_df['gene_name'].isin(gene_list)]
    full_df['x'] = np.abs(2048-full_df['x']) # flip x-axis to match GS
    for row in full_df.itertuples(index=False):
        spot_obj = Spot(row.x, row.y, row.fov, row.gene_name, 'Merfish',
                        spot_size=row.area, mean_inten=row.mean_intensity, min_dist = row.min_distance,
                        blank_fr_score=row.blank_fraction, pass_merlin_threshold = row.passes_threshold)
        spot_obj_list.append(spot_obj)
    return Spots(spot_obj_list)

### Colocalisation ###
fov_list = [i for i in range(nfov)]
distance_threshold = 4 # colocalisation distance

merfish_path = os.path.join(main_dir, "Data/60uM/output/user_49FOVs_mag0_10_lc400_20240719_1026/") # to load the mip dax files
group_spot_merfish = create_Merfish_obj_MERlin(coord_cache_path, gene_list_merfish, df_fpkm)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish)
group_spots_Guidestar = create_Guidestar_obj(raw_folder , gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list)

Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_euclideandist(group_spots_Guidestar,group_spot_merfish_shift,distance_threshold)
colocalisation_counts_df.to_csv(os.path.join(output_path, dataset_name + '_Colocalisation_counts.csv'))

def Spots_to_list_MERlin(Spots, label):
    all_spots_list = []
    feature_name = ['x', 'y', 'gene', 'fov', 'merlin_bfs',
                    'mean_inten', 'min_dist', 'size', 'pass_threshold', 'Label']
    for spot in Spots.all_spots:
        temp_list = []

        # basic info
        temp_list.append(spot.x)
        temp_list.append(spot.y)
        temp_list.append(spot.gene)
        temp_list.append(spot.fov)
        temp_list.append(spot.blank_fr_score)

        # conventional features
        temp_list.append(spot.mean_inten)
        temp_list.append(spot.min_dist) # in this case its the prediction probability
        temp_list.append(spot.spot_size)

        # additional features
        temp_list.append(spot.pass_merlin_threshold)

        # label
        temp_list.append(label)

        all_spots_list.append(temp_list)

    return all_spots_list, feature_name

### Generate Training CSV ###
group_blank_merfish = create_Merfish_obj_MERlin(coord_cache_path, blank_list_merfish, df_fpkm)
Merfish_only_spotlist, feature_name = Spots_to_list_MERlin(Merfish_only_spots, 0)
Merfish_colocalised_spotlist, _ = Spots_to_list_MERlin(Merfish_colocalised_spots, 1)
Merfish_blanks_spotlist, _ = Spots_to_list_MERlin(group_blank_merfish, 0)
combined_lists = Merfish_only_spotlist + Merfish_colocalised_spotlist + Merfish_blanks_spotlist

GS_training_df = pd.DataFrame(combined_lists, columns = feature_name)
GS_training_df.to_csv(os.path.join(output_path, dataset_name + '_GuidestarGenes_training.csv'))

### Generate Full MERFISH CSV ###
group_spot_merfish_allgenes = create_Merfish_obj_MERlin(coord_cache_path, allgene_list_merfish, df_fpkm)
Merfish_allgenes_spotlist, feature_name = Spots_to_list_MERlin(group_spot_merfish_allgenes, 1)
combined_lists_allgenes = Merfish_allgenes_spotlist + Merfish_blanks_spotlist

Allgenes_Merfish_df = pd.DataFrame(combined_lists_allgenes, columns = feature_name)
Allgenes_Merfish_df.to_csv(os.path.join(output_path, dataset_name + '_MerfishData.csv'))



############ Visual check ##########
import matplotlib.pyplot as plt
import numpy as np

gene_viz = 'Acly'
fov = '03'
GS_img = readDoryImg(os.path.join(main_dir, f'Data/60uM/{bit_gene_dict_Guidestar[gene_viz]}_{fov}.dax'))
merfish_mip_img = readDoryImg(os.path.join(main_dir, f'Data/60uM/output/user_49FOVs_mag0_10_lc400_20240719_1026/FOV_{fov}_normalized_clipped_1x2048x2048_maxintprojected_scaledtomax.dax'))
reg_merfish_mip_img,_ = register_slice(GS_img,merfish_mip_img)
plt.figure(figsize=(50,50))
plt.imshow(GS_img,cmap='gray', vmin=0, vmax=np.percentile(GS_img, 99.9))
plt.title('If cross has circle, means the spot is considered colocalised')
for spot in group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='red',label='merfish spots')
for spot in group_spots_Guidestar.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='blue',label='guidestar spots')
for spot in Merfish_colocalised_spots.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal merfish')
for spot in Guidestar_colocalised_spots.get_by_fov(fov).get_by_gene(gene_viz).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal guidestar')
plt.savefig(os.path.join(output_path, f'check_colocalization_{gene_viz}_{fov}.svg'))



gene_viz = 'Acly'
fov = '02'
GS_img = readDoryImg(os.path.join(main_dir, f'Data/60uM/{bit_gene_dict_Guidestar[gene_viz]}_{fov}.dax'))
# Prepare the image
plt.figure(figsize=(10, 10))
plt.imshow(GS_img, cmap='gray', origin='upper', vmin=0, vmax=np.percentile(GS_img, 99.9))  # Adjust origin if needed

# Extract coordinates from all spots
spots = group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene_viz).all_spots
x_coords = [float(spot.x) for spot in spots]
y_coords = [float(spot.y) for spot in spots]
# Plot all markers at once
plt.scatter(y_coords, x_coords, marker='x', color='red', s=30, label='merfish spots', zorder=10)

# Final touches
plt.title(f"{gene_viz}_{fov}")
plt.grid(False)
plt.savefig(f"{output_path}/FOV_{fov}_{gene_viz}.png", bbox_inches='tight', dpi=200)
plt.show()
