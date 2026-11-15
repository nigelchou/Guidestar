import sys
sys.path.append("..")

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
#cell (test code)
dataset_name = 'Polaris_LM39_cell_25FOVs'
main_path = os.path.join(main_dir, 'Data/benchmarking/polaris/cell/')
info_path = '../genebitdict_cell_benchmarking.yml'
coord_cache_path = os.path.join(main_path, 'Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_final/polaris_LM39cell_result_edgecrop')
# # os.path.join(main_dir, 'Data/benchmarking/polaris/cell/Relaxed Bernoulli_thresh0.85_splitfish_plus_star_chrco_img_regi')
# # coord_cache_path = os.path.join(main_dir, 'Data/benchmarking/polaris/cell/Relaxed Bernoulli_thresh0.85_starfish_chrco_img_regi')
# # #os.path.join(main_dir, 'Data/100uM/output/user_25FOVs_mag0_10_lc400_20240606_1743/')
fpkm_path = os.path.join(main_dir, "Data/100uM/codebook/fpkm_data.tsv")
output_path = os.path.join(main_path, 'Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_final/polaris_LM39cell_result_edgecrop/colocalization_output_20240606_1743_final')
nfov = 25

if not os.path.exists(output_path):
    os.mkdir(output_path)

info_dict = txt_to_dict(info_path)
raw_folder = os.path.join(main_dir, info_dict['raw_folder'])
gene_threshold_dict = info_dict['gene_threshold_dict']
gene_bit_dict_merfish = info_dict['gene_bit_dict_merfish']
# gene_bit_dict_Guidestar = info_dict['gene_bit_dict_guidestar']
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
### Colocalisation ###
fov_list = [i for i in range(nfov)]
distance_threshold = 4 # colocalisation distance (origi: 4)

def create_Merfish_obj_polaris(csv_file_folder, gene_list, microscope_type):
    def get_bit_columns(df, pattern=r"^(?:bit_?)?(\d+)$", expected_n_bits=None):
        """
        Find bit columns by name, regardless of their position in the file.
        Matches names like '0', '15', 'bit0', 'bit_15'. Returns them sorted
        by bit number so the order is always 0, 1, 2, ...
        """
        matches = []
        for col in df.columns:
            m = re.match(pattern, str(col).strip())
            if m:
                matches.append((int(m.group(1)), col))

        if not matches:
            raise ValueError(f"No bit columns found. Columns are: {df.columns.tolist()}")

        matches.sort()
        bit_nums = [n for n, _ in matches]
        bit_cols = [c for _, c in matches]

        # Sanity checks: bits should be consecutive from 0
        if bit_nums != list(range(len(bit_nums))):
            raise ValueError(f"Bit columns are not consecutive from 0: {bit_nums}")
        if expected_n_bits is not None and len(bit_cols) != expected_n_bits:
            raise ValueError(f"Expected {expected_n_bits} bits, found {len(bit_cols)}")

        return bit_cols
    # Combine all the csv in the folder into single dataframe
    all_df = []
    coord_data_pattern = re.compile(r".+\.csv$", flags=re.IGNORECASE)
    for file in os.listdir(csv_file_folder):
        if re.match(coord_data_pattern, file):
            df = pd.read_csv(os.path.join(csv_file_folder, file))
            parts = file.split('_')
            if microscope_type == 'Dory':
                df['fov'] = parts[0]
            elif microscope_type == 'Triton':
                df['fov'] = parts[0] + '_' + parts[1]
            all_df.append(df)
    if not all_df:
        raise ValueError(f"No files matching in {csv_file_folder}")
    full_df = pd.concat(all_df, ignore_index=True) #combine via concat
    bit_cols = get_bit_columns(full_df, expected_n_bits=16)
    # Convert all bit columns to numbers once; anything non-numeric becomes NaN
    full_df[bit_cols] = full_df[bit_cols].apply(pd.to_numeric, errors="coerce")

    if microscope_type == 'Dory':
        full_df['fov'] = [s.zfill(2) for s in full_df['fov']]
    full_df = full_df[full_df['predicted_name'].isin(gene_list)]
    # Extract bits for all rows at once, then pair them with each row
    all_bits = full_df[bit_cols].to_numpy(dtype=float)
    spot_obj_list = []
    for row, bits in zip(full_df.itertuples(index=False), all_bits):
        mask = np.where(bits < 0.85, bits, 0)
        offinten = np.linalg.norm(mask)
        spot_obj = Spot(row.x, row.y, row.fov, row.predicted_name, 'Merfish',
                        meaninten1_frob=np.linalg.norm(bits),
                        spot_size=2, probability=row.probability,
                        offinten1_frob=offinten,
                        size_conn=row.source, blank_fr_score=0.5)
        spot_obj_list.append(spot_obj)
    return Spots(spot_obj_list)

# merfish_path = os.path.join(main_dir, "Data/100uM/output/user_25FOVs_mag0_10_lc400_20240606_1743/") # to load the mip dax files
merfish_path = os.path.join(main_dir, 'Data/100uM/output/user_25FOVs_mag0_10_lc400_20240606_1743/')
group_spot_merfish = create_Merfish_obj_polaris(coord_cache_path, gene_list_merfish, microscope_type='Dory') # only create spot objects for GS genes (in Merfish output)
group_spot_merfish_shift = create_shift_spot_obj(raw_folder,  merfish_path, bit_gene_dict_Guidestar, group_spot_merfish)
group_spots_Guidestar = create_Guidestar_obj(raw_folder, gene_bit_dict_Guidestar, gene_threshold_dict, [300,None], fov_list)

Merfish_colocalised_spots, Merfish_only_spots, Guidestar_colocalised_spots, Guidestar_only_spots, colocalisation_counts_df = Colocalisation_euclideandist(group_spots_Guidestar,group_spot_merfish_shift,distance_threshold)
colocalisation_counts_df.to_csv(os.path.join(output_path, dataset_name + '_Colocalisation_counts.csv'))


def Spots_to_list_polaris(Spots, label):
    all_spots_list = []
    feature_name = ['x', 'y', 'gene', 'fov', 'bfs',
                    'meaninten1_frob', 'probability', 'size', 'offinten1_frob',
                    'source', 'Label']
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
        temp_list.append(spot.probability) # in this case its the prediction probability
        temp_list.append(spot.spot_size)

        # additional features
        temp_list.append(spot.offinten1_frob)
        temp_list.append(spot.size_conn) #source: error rescue/prediction

        # label
        temp_list.append(label)

        all_spots_list.append(temp_list)

    return all_spots_list, feature_name



### Generate Training CSV ###
group_blank_merfish = create_Merfish_obj_polaris(coord_cache_path, blank_list_merfish, microscope_type='Dory')
Merfish_only_spotlist, feature_name = Spots_to_list_polaris(Merfish_only_spots, 0)
Merfish_colocalised_spotlist, _ = Spots_to_list_polaris(Merfish_colocalised_spots, 1)
Merfish_blanks_spotlist, _ = Spots_to_list_polaris(group_blank_merfish, 0)
combined_lists = Merfish_only_spotlist + Merfish_colocalised_spotlist + Merfish_blanks_spotlist

GS_training_df = pd.DataFrame(combined_lists, columns = feature_name)
GS_training_df.to_csv(os.path.join(output_path, dataset_name + '_GuidestarGenes_training.csv'))

### Generate 1 Full MERFISH CSV ###
# group_spot_merfish_allgenes = create_Merfish_obj_addedfeat(save_coord_path, allgene_list_merfish, fov_list)
group_spot_merfish_allgenes = create_Merfish_obj_polaris(coord_cache_path, allgene_list_merfish, microscope_type='Dory')
Merfish_allgenes_spotlist, feature_name = Spots_to_list_polaris(group_spot_merfish_allgenes, 1)
combined_lists_allgenes = Merfish_allgenes_spotlist + Merfish_blanks_spotlist

Allgenes_Merfish_df = pd.DataFrame(combined_lists_allgenes, columns = feature_name)
Allgenes_Merfish_df.to_csv(os.path.join(output_path, dataset_name + '_MerfishData.csv'))


### Visualization ###
gene = 'Gpam'
fov = '07'
GS_img = readDoryImg(os.path.join(main_dir, 'Data/100uM/')+bit_gene_dict_Guidestar[gene] +'_'+ fov + '.dax')

plt.figure(figsize=(16,16))
plt.imshow(GS_img,cmap='gray', vmin=0, vmax=np.percentile(GS_img, 99.9))
#plt.title('If cross has circle, means the spot is considered colocalised')
for spot in group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='red',label='merfish spots')
for spot in group_spots_Guidestar.get_by_fov(fov).get_by_gene(gene).all_spots:
    plt.plot(spot.y,spot.x,marker="x",markerfacecolor="None",markeredgecolor='blue',label='guidestar spots')
for spot in Merfish_colocalised_spots.get_by_fov(fov).get_by_gene(gene).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal merfish')
for spot in Guidestar_colocalised_spots.get_by_fov(fov).get_by_gene(gene).all_spots:
    plt.plot(spot.y,spot.x,marker="o",markerfacecolor="None",markeredgecolor='yellow',label='colocal guidestar')
#plt.ion()
#plt.show()
plt.title(gene+'_'+fov)
plt.grid(False)
plt.savefig(f"{output_path}/FOV_{str(fov)}_{str(gene)}.png", bbox_inches='tight', dpi=200)

## working visualization 04 Sep 2025
import matplotlib.pyplot as plt
import numpy as np

# Prepare the image
plt.figure(figsize=(10, 10))
plt.imshow(GS_img, cmap='gray', origin='upper', vmin=0, vmax=np.percentile(GS_img, 99.9))  # Adjust origin if needed

# Extract coordinates from all spots
spots = group_spot_merfish_shift.get_by_fov(fov).get_by_gene(gene).all_spots
x_coords = [float(spot.x) for spot in spots]
y_coords = [float(spot.y) for spot in spots]

# Plot all markers at once
plt.scatter(y_coords, x_coords, marker='x', color='red', s=30, label='merfish spots', zorder=10)

# Final touches
plt.title(f"{gene}_{fov}")
plt.grid(False)
plt.savefig(f"{output_path}/FOV_{fov}_{gene}.png", bbox_inches='tight', dpi=200)
plt.show()
