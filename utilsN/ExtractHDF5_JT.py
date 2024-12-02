import os
import pandas as pd
import h5py

def readable_hdf5_for_GOI(cache_folder,output_file,GOI_list):

    """function to convert hdf5 files into readable csv files for specific gene of interest list
    used to check decoding experiment"""

    results_dict = dict.fromkeys(['gene_name', 'FOV', 'xcoord','ycoord','spot_size','min_dist','mean_intensity'])
    gene_list = []
    FOV_list = []
    xcoord_list = []
    ycoord_list = []
    size_list = []
    mindist_list = []
    meanintensity_list = []

    for file in os.listdir(cache_folder):
        if file.endswith('.hdf5'):
            coords_cache = os.path.join(cache_folder,file)
            FOV = file.split('_')[1]
            with h5py.File(coords_cache,'r') as f:
                for gene in GOI_list:
                    for spot in f[gene]:
                        gene_list.append(gene)
                        FOV_list.append(FOV)
                        xcoord_list.append(spot[2])
                        ycoord_list.append(spot[1])
                        size_list.append(spot[3])
                        mindist_list.append(spot[4])
                        meanintensity_list.append(spot[5])

    results_dict['gene_name'] = gene_list
    results_dict['FOV'] = FOV_list
    results_dict['xcoord'] = xcoord_list
    results_dict['ycoord'] = ycoord_list
    results_dict['spot_size'] = size_list
    results_dict['min_dist'] = mindist_list
    results_dict['mean_intensity'] = meanintensity_list
    resultsdf = pd.DataFrame.from_dict(results_dict, orient='columns')
    #resultsdf.to_csv(output_file)
    print("output: ", output_file)
    print("number of spots: ", len(gene_list))

    return resultsdf

def readable_hdf5_for_GOI2(cache_file,output_file,GOI_list,FOV):

    """ same as above but just for one cache file """

    results_dict = dict.fromkeys(['gene_name', 'FOV', 'xcoord','ycoord','spot_size','min_dist','mean_intensity'])
    gene_list = []
    FOV_list = []
    xcoord_list = []
    ycoord_list = []
    size_list = []
    mindist_list = []
    meanintensity_list = []

    with h5py.File(cache_file,'r') as f:
        for gene in GOI_list:
            for spot in f[gene]:
                gene_list.append(gene)
                FOV_list.append(FOV)
                xcoord_list.append(spot[2])
                ycoord_list.append(spot[1])
                size_list.append(spot[3])
                mindist_list.append(spot[4])
                meanintensity_list.append(spot[5])

    results_dict['gene_name'] = gene_list
    results_dict['FOV'] = FOV_list
    results_dict['xcoord'] = xcoord_list
    results_dict['ycoord'] = ycoord_list
    results_dict['spot_size'] = size_list
    results_dict['min_dist'] = mindist_list
    results_dict['mean_intensity'] = meanintensity_list
    resultsdf = pd.DataFrame.from_dict(results_dict, orient='columns')
    #resultsdf.to_csv(output_file)
    print("output: ", output_file)
    print("number of spots: ", len(gene_list))

    return resultsdf

# to extract all genes
def readable_hdf5_for_allgenes(cache_folder,output_file):

    """function to convert hdf5 files into readable csv files for specific gene of interest list
    used to check decoding experiment"""

    results_dict = dict.fromkeys(['gene_name', 'FOV', 'xcoord','ycoord','spot_size','min_dist','mean_intensity',"bit1","bit2","bit3","bit4"])
    gene_list = []
    FOV_list = []
    xcoord_list = []
    ycoord_list = []
    size_list = []
    mindist_list = []
    meanintensity_list = []
    bit1 = []
    bit2 = []
    bit3 = []
    bit4 = []

    for file in os.listdir(cache_folder):
        if file.endswith('.hdf5'):
            coords_cache = os.path.join(cache_folder,file)
            FOV = file.split('_')[1]
            with h5py.File(coords_cache,'r') as f:
                for gene in f:
                    for spot in f[gene]:
                        gene_list.append(gene)
                        FOV_list.append(FOV)
                        xcoord_list.append(spot[2])
                        ycoord_list.append(spot[1])
                        size_list.append(spot[3])
                        mindist_list.append(spot[4])
                        meanintensity_list.append(spot[5])
                        bit1.append(f[gene].attrs['on_bits'][0])
                        bit2.append(f[gene].attrs['on_bits'][1])
                        bit3.append(f[gene].attrs['on_bits'][2])
                        bit4.append(f[gene].attrs['on_bits'][3])

    results_dict['gene_name'] = gene_list
    results_dict['FOV'] = FOV_list
    results_dict['xcoord'] = xcoord_list
    results_dict['ycoord'] = ycoord_list
    results_dict['spot_size'] = size_list
    results_dict['min_dist'] = mindist_list
    results_dict['mean_intensity'] = meanintensity_list
    results_dict['bit1'] = bit1
    results_dict['bit2'] = bit2
    results_dict['bit3'] = bit3
    results_dict['bit4'] = bit4
    resultsdf = pd.DataFrame.from_dict(results_dict, orient='columns')
    #resultsdf.to_csv(output_file)
    print("output: ", output_file)
    print("number of spots: ", len(gene_list))

    return resultsdf

def readable_hdf5_for_allgenes2(cache_file,output_file,FOV):

    """ same as above but just for one cache file """

    results_dict = dict.fromkeys(['gene_name', 'FOV', 'xcoord','ycoord','spot_size','min_dist','mean_intensity'])
    gene_list = []
    FOV_list = []
    xcoord_list = []
    ycoord_list = []
    size_list = []
    mindist_list = []
    meanintensity_list = []

    with h5py.File(cache_file,'r') as f:
        for gene in f:
            for spot in f[gene]:
                gene_list.append(gene)
                FOV_list.append(FOV)
                xcoord_list.append(spot[2])
                ycoord_list.append(spot[1])
                size_list.append(spot[3])
                mindist_list.append(spot[4])
                meanintensity_list.append(spot[5])

    results_dict['gene_name'] = gene_list
    results_dict['FOV'] = FOV_list
    results_dict['xcoord'] = xcoord_list
    results_dict['ycoord'] = ycoord_list
    results_dict['spot_size'] = size_list
    results_dict['min_dist'] = mindist_list
    results_dict['mean_intensity'] = meanintensity_list
    resultsdf = pd.DataFrame.from_dict(results_dict, orient='columns')
    #resultsdf.to_csv(output_file)
    print("output: ", output_file)
    print("number of spots: ", len(gene_list))

    return resultsdf

def readable_hdf5_for_allgenes3(cache_folder,output_file):

    """function to convert hdf5 files into readable csv files for specific gene of interest list
    used to check decoding experiment"""

    results_dict = dict.fromkeys(['gene_name', 'FOV', 'xcoord','ycoord','spot_size','min_dist','mean_intensity'])
    gene_list = []
    FOV_list = []
    xcoord_list = []
    ycoord_list = []
    size_list = []
    mindist_list = []
    meanintensity_list = []
    # bit1 = []
    # bit2 = []
    # bit3 = []
    # bit4 = []

    for file in os.listdir(cache_folder):
        if file.endswith('.hdf5'):
            coords_cache = os.path.join(cache_folder,file)
            FOV = file.split('_')[1]
            with h5py.File(coords_cache,'r') as f:
                for gene in f:
                    for spot in f[gene]:
                        gene_list.append(gene)
                        FOV_list.append(FOV)
                        xcoord_list.append(spot[2])
                        ycoord_list.append(spot[1])
                        size_list.append(spot[3])
                        mindist_list.append(spot[4])
                        meanintensity_list.append(spot[5])
                        # bit1.append(f[gene].attrs['on_bits'][0])
                        # bit2.append(f[gene].attrs['on_bits'][1])
                        # bit3.append(f[gene].attrs['on_bits'][2])
                        # bit4.append(f[gene].attrs['on_bits'][3])

    results_dict['gene_name'] = gene_list
    results_dict['FOV'] = FOV_list
    results_dict['xcoord'] = xcoord_list
    results_dict['ycoord'] = ycoord_list
    results_dict['spot_size'] = size_list
    results_dict['min_dist'] = mindist_list
    results_dict['mean_intensity'] = meanintensity_list
    # results_dict['bit1'] = bit1
    # results_dict['bit2'] = bit2
    # results_dict['bit3'] = bit3
    # results_dict['bit4'] = bit4
    resultsdf = pd.DataFrame.from_dict(results_dict, orient='columns')
    #resultsdf.to_csv(output_file)
    print("output: ", output_file)
    print("number of spots: ", len(gene_list))

    return resultsdf