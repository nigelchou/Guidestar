# Guidestar: a spike-in approach to improve RNA detection accuracy in imaging-based spatial transcriptomics


### Jazlynn Tan\*, Lurong Wang, Wan Yi Seow, Jeeranan Boonruangkan, Mike Huang, Jiamin Toh, Kok Hao Chen, Nigel Chou

## Overview

Guidestar is a method to validate and improve RNA spot-decoding in combinatorial FISH assays (MERFISH, CosMx, Xenium, Split-FISH, SeqFISH etc.).

Guidestar uses a system of spike-in controls (a subset of RNA transcripts labeled with additional probes and imaged separately as ‘guide bits’) integrated with combinatorial FISH assays. The GUIDESTAR data can be used to validate existing decoding approaches and act as a ground truth dataset for developing machine-learning models that distinugish true from false RNA calls. 

<!--
Guidestar improves decoding accuracy, as measured by F1 score, in cell-line samples. It also generalized well to tissue samples.
-->

Further details are described in this paper: (link will be added when published).

## Prerequisites <a name="prereqs"></a>

### System requirements: <a name="sysreqs"></a>

Machine with 16 GB of RAM. No non-standard hardware is required.

### Software requirements: <a name="softreqs"></a>

Packages required for this software are found in environment.yml

<!--
## License <a name="lic"></a>

This project is licensed under ...
-->

## Getting Started <a name="getstart"></a>

   
### Installation <a name="install"></a>

1.	Download and install [Anaconda](https://www.anaconda.com/distribution/#download-section).
2.	Create a new conda environment: [managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 
3.	Download Guidestar Python package from this repository.

### Installation via Anaconda (recommended) <a name="install"></a>

To use ``Guidestar``, we recommend setting up a ``conda`` environment and cloning this repository.

      (base) $ conda create --name Guidestar
      (base) $ conda activate Guidestar
      (Guidestar) $ git clone https://github.com//nigelchou/Guidestar.git
      (Guidestar) $ cd Guidestar

### Installation from `environment.yml` file <a name="install"></a>

Users can directly install the prequisite packages from `environment.yml` here after cloning in this repository:

      (base) $ git clone https://github.com/nigelchou/Guidestar.git
      (base) $ cd Guidestar
      (base) $ conda env create --name Guidestar --file=environment.yml
      (base) $ conda activate Guidestar

To run the examples presented in `juypter` notebooks, install the extensions for `juypter`.

      (Guidestar) $ conda install -c conda-forge jupyter

### Blank Fraction Threshold Decoding <a name="BFT"></a>
Run [MainScript_MERFISH_cell](../MainScript_MERFISH_cell.py) to perform decoding of MERFISH data with blank fraction thresholding to filter spots. You will be prompted to select your data directory which should contain your raw images. In addition, the data directory should have the following structure which includes codebook.tsv etc. etc. More details and description of different steps in the process are provided within the notebooks.

#### Data folder structure

    .
    ├── 100uM                                            # Main folder containing the raw images and other required files
    │   ├── calibration                                  # Folder named 'calibration' containing calibration files
    │   │   ├── calibration_file.csv
    │   │   └── ... 
    │   ├── codebook                                     # Folder named 'codebook' containing codebook informaiton
    │   │   ├── codebook_data.tsv
    │   │   ├── fpkm_data.tsv
    │   │   └── ...     
    │   ├── output                                       # The output folder generated by bft_decoding main script
    │   │   ├── user_25FOVs_mag0_10_lc400_DATE_TIME      # The coord_cache_path to input in Colocalisation_Main_v1.py
    │   │   │   ├── output                               # Output folder with files generated by Colocalisation_Main_v1.py
    │   │   │   │   ├── Cell_LM39_25FOVs_Colocalisation_counts.csv             
    │   │   │   │   ├── Cell_LM39_25FOVs_GuidestarGenes_training.csv
    │   │   │   │   ├── Cell_LM39_25FOVs_MerfishData.csv
    │   │   │   │   └── ...
    │   │   │   ├── FOV_00_coord_iter0.hdf5              # Example input coords files 
    │   │   │   └── ...                                  # Set bft_decoding "stages_to_save" for different output files
    │   ├── Cy5_00_00.dax                                # Sample raw image file
    │   ├── Cy5_00_00.inf                                # Sample image info file
    │   └── ...
    └── ...

Please edit user_params with parameters relevant to your data:
```
# example for cell line data

user_params["roi"] = None 
user_params["correct_distortion"] = True # True/False (must have calibration file for True)

user_params["stages_to_plot"] = [] # "raw","fieldcorr","chrcorr"
user_params["image_grid"] = (4, 7)
user_params["fovstr_length"] = 2
user_params["hyb_list"] = [9,8,7,6,5,4,2,3,9,8,7,6,5,4,3,2]
user_params["type_list"] = ['Cy7']*8 +['Cy5']*8
user_params["num_bits"] = 16
user_params["reference_bit"] = 0

user_params['fovs_to_process'] = list(range(25))
```

[MainScript_MERFISH_cell](../MainScript_MERFISH_cell.py) and [MainScript_MERFISH_tissue](../MainScript_MERFISH_tissue.py) contain the parameters to reproduce the data described in this paper (link!). 

### Guidestar colocalisation
Run [Colocalisation_Main_v1](Guidestar_colocalisation/Colocalisation_Main_v1.py) to perform detection and colocalisation of Guidestar spots. This produces the training data as well as MERFISH (for the full library) data in csv format which will be used for model training and application. Input parameters should be edited with respect to your data:
```
# example for cell line data
dataset_name = 'Cell_LM39_25FOVs'
info_path = '../genebitdict_cell.yml'
coord_cache_path = '/path_to_raw_data/output/user_25FOVs_mag0_10_lc400_20230312_1559'
fpkm_path = '../DataForFigures/cell_fpkm_data.tsv'
output_path = '/path_to_output_folder/'
nfov = 25
```

If you would like to use your own data, you will need to create a `.yml` file (insert filepath under `info_path`) with details on corresponding hybridisation rounds and channels for Guidestar genes, as well as the threshold setting for spot detection. Spot detection uses `skimage`'s `peaklocalmax` function (see more [here](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max). The threshold setting in the `.yml` file will be used as the `threshold_rel` parameter in `peaklocalmax`. We have provided example files [genebitdict_cell.yml](genebitdict_cell.yml) and [genebitdict_liver.yml](genebitdict_liver.yml).  

If you would like to perform further validation described in our paper, please use the following scripts:
* visualisation: [Colocalisation_Visualisation_v1](Guidestar_colocalisation/Colocalisation_Visualisation_v1.py)
* colocalisation distance optimisation: [NegativeControlColocalisation](Guidestar_colocalisation/NegativeControlColocalisation.py) to generate data and [SuppFig2_colocalisationdistance](Figures/SuppFig2_colocalisationdistance.py) to generate the figure
* misidentification rate tuning: [Colocalisation_Main_extendedmisid_v1](Guidestar_colocalisation/Colocalisation_Main_extendedmisid_v1.py) to generate data and [Fig2b_misidratevalidation](Figures/Fig2b_misidratevalidation.py) to generate the figure

### Training and applying your own Guidestar model
To train your own Guidestar model, please use [TestSetResults](Guidestar_model/TestSetResults.py). See [Fig2def_Cell_testsetresults](Figures/Fig2def_Cell_testsetresults.py) for an example. This step requires the csv files with suffix `'_GuidestarGenes_training.csv'` and `'_MerfishData.csv'` generated in [Colocalisation_Main_v1](Guidestar_colocalisation/Colocalisation_Main_v1.py). You can also download the data files [here](https://drive.google.com/drive/folders/122UvRUf9SqZhPW2tYLKfgjZl1Y_HP1Kx?usp=drive_link).

### Reproducing mansucript figures
All data and code for the figures can be found in [DataForFigures](DataForFigures) and [Figures](Figures) respectively.

## Contributing

Bug reports, questions, request for enhancements or other contributions
can be raised at the [issue
page](<https://github.com/nigelchou/Guidestar/issues>).

## Authors <a name="authors"></a>

* **Jazlynn Tan** (https://github.com/JazlynnXM-Tan)

## Acknowledgments <a name="ack"></a>

* **Lurong Wang**  (https://github.com/lurongw)
* **Mike Huang**  (https://github.com/mikejhuang)
* **Nigel Chou**  (https://github.com/chousn)
* **Kok Hao Chen**  (https://github.com/kchen23)
