# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:24:58 2020
Created on Fri Jul 09 2022

@author: Mike
@author: JB : this code will do MIP on images with multiple z. this code still can be used with usual image (no z).
            : this code can be use with confocal and ELS system too
"""

import copy
import timeit
import datetime

# self-imports
# ------------

from BFT_decoding.utils.lhsGenerator import lhsGen
from BFT_decoding.utils.printFunctions import printTime
from BFT_decoding.utils.printFunctions import printParams, printParamsToFile, formatParamSetName

from BFT_decoding.frequencyFilter import butter2d
from BFT_decoding.imageDataGS import showImages, showRegistration

from BFT_decoding.filesClasses import getFileParser
from BFT_decoding.fovGridFunctions import generateFovGrid
from BFT_decoding.comparisonFunctions import subtractBackground
from BFT_decoding.correctionFunctions import correctDistortion, correctField
from BFT_decoding.processFunctions import registerPhaseCorr, clipBelowZero
from BFT_decoding.DecodeAdditionalFeatures import *
from BFT_decoding.globalNormalizationFunctions import globalNormalize
from BFT_decoding.spotClasses import run_blank_fraction_analysis

#
# Open file dialog and select folder with data
# --------------------------------------------
#

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

data_path = filedialog.askdirectory(title="Please select data directory") # your raw folder path containing 60uM tissue

root.destroy()


# the directory within data_path where all analysis runs are stored
main_output_path = os.path.join(data_path, "output")

start_time = timeit.default_timer()
script_time = datetime.datetime.now()
script_time_str = script_time.strftime("_%Y%m%d_%H%M")

print(
    "-" * 80 +
    f"\n Processing data from folder: {data_path}\n"
    f" Script started at {script_time.strftime('%Y/%m/%d %H hr %M min')}\n"
    + "-" * 80
)

#
# =================================================================================
#                                   PARAMETERS
# =================================================================================
#

shared_params = {}
shared_params["fpkm_structname"] = "new"
shared_params["bw_filter_order"] = 2 # filter order
shared_params["subtract_background"] = False
shared_params["correct_field"] = False
shared_params["drop_genes"] = []
shared_params["bits_to_drop"] = []
shared_params["imgcolour_to_maskcolour"] = None
shared_params["global_normalize"] = True
shared_params["global_perfov_percentile"] = 99.9
shared_params["global_pooled_percentile"] = 57
shared_params["globalnorm_fovsubset"] = ['10', '11', '12', '13', '14']# Use list of str fov e.g. ["000_003"] or ['01','02']. Set to FOV with known confluence. Substantially reduces runtime if set
shared_params["clip_normalized"] = True
shared_params["num_iterations"] = 1

#Adaptive thresholding params
shared_params["blank_fraction_threshold"] = True
shared_params["num_bins"] = 60 # Number of bins for signal, noise heatmaps for the distance and intensity metrics
shared_params["eps"] = 0 # Pseudocount for blank fraction heatmap blank fraction = (blank_bin + eps) / (gene_bin + blank_bin + eps*2), suggested 0 or 1
shared_params["kde_sigma"] = 0 # Gaussian kernel sigma parameter for smoothing blank fraction heatmap, suggested 0 or 2
shared_params["misid_rate_target"] = 0.05 # Preliminary target misidentification rate for blank fraction threshold. Need to optimize with spotsClasses after viewing ROC plot


#  Microscope-specific default Parameters
#  --------------------------------------
#

# Dory
# ----

dory_default = {}
dory_default["name"] = "dory_default"
dory_default["microscope_type"] = "Dory"
dory_default["stage_pixel_matrix"] = 8 * np.array([[0, -1], [-1, 0]])
dory_default["roi"] = None  # Dory does not support ROIs-this should always be None!
dory_default["fovstr_length"] = 3  # number of characters for the fov string in filename
dory_default["fovs_to_process"] = [0, ]
dory_default["bits_to_drop"] = []
dory_default["image_grid"] = (3, 6)  # subplot grid for displaying images
dory_default["percentile_norm_high"] = 99.95
dory_default["min_region_size"] = 2
# Recommended dory low_cut: 400, high_cut:None, See Low_high_cut_optimization.pdf in Docs
dory_default["low_cut"] = 400
dory_default["high_cut"] = None
dory_default["num_bits"] = 16
dory_default["hyb_list"] = list(range(16))  # 1color
dory_default["type_list"] = "Cy5"
dory_default["reference_bit"] = 1
dory_default["type_to_colour"] = {
    "Cy3": "558",
    "Cy5": "684",
    "Cy7": "776"
}
dory_default["reference_colour"] = "684"
dory_default = {**shared_params, **dory_default}


#
# -------------------------------------------------------------------------------
#                        Default QC and filepath params
# -------------------------------------------------------------------------------
#

qc_params_default = {}
qc_params_default["plot_freq_filter"] = False
qc_params_default["plot_fov_grid"] = True
qc_params_default["plot_correlation"] = True
qc_params_default["plot_registration_overlay"] = True
# reserved for when we implement iterations
qc_params_default["iter_to_correlate"] = "all"  # "all" or "last"
# this must be a list of valid stages
qc_params_default["stages_to_plot"] = []
qc_params_default["stages_to_save"] = [
    # (stage, projection method across bits, whether to normalize by maximum value)
    ("normalized", "mip", True),
    ("unitnormalized", None, True),
    ("decoded", None, True),
]
# max of 28 bits
# try to make the grid size close to the number of bits e.g. (3,6) for 16 bits
qc_params_default["image_grid"] = (4, 7)

printParams(qc_params_default, "Quality Control default")

# copy main parameters dictionary
user_params = copy.copy(dory_default)  # dory_default

# set the fovs to process. Don't try too many if  still optimizing parameters
user_params["roi"] = None  # 'None' for Dory
user_params["correct_distortion"] = True # True/False (must have calibration file for True)

# this must be a list of valid stages
user_params["stages_to_plot"] = [] # "raw","fieldcorr","chrcorr"
user_params["image_grid"] = (4, 7)
user_params["fovstr_length"] = 2
user_params["hyb_list"] = [4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11]
user_params["type_list"] = ['Cy7']*8 +['Cy5']*8
user_params["num_bits"] = 16
user_params["reference_bit"] = 10

user_params['fovs_to_process'] = list(range(49)) # dory

if user_params["subtract_background"]:
    user_params["hyb_list_background"] = copy.copy(user_params["hyb_list"])
    user_params["type_list_background"] = [
        typestr + "_Bleach" for typestr in user_params["type_list"]
    ]

    bits_to_subtract = []
    for bit, colour_type in enumerate(user_params["type_list"]):
        if colour_type in ["Cy5", "Cy3"]:
            bits_to_subtract.append(bit)
    user_params["subtract_only_bits"] = bits_to_subtract



# ========================================================================================================


user_params = user_params

file_params_default = {}
codebook_dir = os.path.join(data_path, "codebook")
file_params_default["codebook_filepath"] = os.path.join(
    codebook_dir, "codebook_data.tsv"
)
file_params_default["fpkm_filepath"] = os.path.join(
    codebook_dir, "fpkm_data.tsv"
)

file_params_default["calibration_path"] = os.path.join(
    data_path, "calibration"
)

printParams(file_params_default, "File parameters default")

#
# --------------------------------------------------------------------------------------
#                           Define user's base parameters
# --------------------------------------------------------------------------------------
#


user_qc_params = copy.copy(qc_params_default)

user_file_params = copy.copy(file_params_default)
user_file_params["fpkm_filepath"] = os.path.join(
    codebook_dir,
    "fpkm_data.tsv")
user_file_params["codebook_filepath"] = os.path.join(
    codebook_dir, "codebook_data.tsv")

file_params = user_file_params
qc_params = user_qc_params

# Confirm file parameters
assert os.path.exists(user_file_params["fpkm_filepath"]), f'File does not exist at {user_file_params["fpkm_filepath"]}'
assert os.path.exists(user_file_params["codebook_filepath"]),  f'File does not exist at {user_file_params["codebook_filepath"]}'
assert os.path.exists(user_file_params["calibration_path"]),  f'File does not exist at {user_file_params["calibration_path"]}'


# -------------------------------------------------------------------

#----------------------------------------------------------------------------------
#                        Deprecated dummy params
#----------------------------------------------------------------------------------
'''
    Boundary conditions for distance and magnitude used in generating the 
    signal and noise heatmaps
    For rationale in choosing these parameters, see methods section of PNAS 
    publication that used adaptive thresholding [1].
    
    1. https://www.pnas.org/highwire/filestream/887973/field_highwire_adjunct_files/0/pnas.1912459116.sapp.pdf
'''

user_params["distance_threshold"] = 0.605 # two-bit error for MHD4, change to 0.765 for MHD2 (for blank fraction threshold)
# 0.517 for hamming weight 4 , 0.417 for hamming weight 3 , 0.3321 for hamming weight 2 (for non-blank fraction threshold)
user_params["magnitude_threshold"] = 0.1
user_params["small_spot_threshold"] = None
user_params["large_spot_threshold"] = None

#----------------------------------------------------------------------------------
#
user_params["name"] = formatParamSetName(user_params, "user")
# ---------------------------------------------------------------------------------
#                        Choose HOW to define parameters
# ---------------------------------------------------------------------------------

params_generator = "single"
#params_generator = "manual"
#params_generator = "lhs"

# ---------------------------------------------------------------------------------
#
#


main_output_path = os.path.join(data_path, "output")
start_time = timeit.default_timer()
script_time = datetime.datetime.now()
script_time_str = script_time.strftime("_%Y%m%d_%H%M")

print(
    "-" * 80 +
    f"\n Processing data from folder: {data_path}\n"
    f" Script started at {script_time.strftime('%Y/%m/%d %H hr %M min')}\n"
    + "-" * 80
)


#
# ---------------------------------------------------------------------------------
#                        Choose HOW to define parameters
# ---------------------------------------------------------------------------------

params_generator = "single"
#params_generator = "manual"
# params_generator = "lhs"

# ---------------------------------------------------------------------------------
#
#


# Single set of parameters
# ------------------------

if params_generator == "single":
    params_list = [user_params, ]


# Manual Parameter generation
# ---------------------------

elif params_generator == "manual":

    params_list = []

    parameters_to_change = [
    ]

    for params_dict in parameters_to_change:

        # generate a new set, with specific parameters changed
        new_params = copy.copy(user_params)
        for parameter in params_dict:
            new_params[parameter] = params_dict[parameter]

        new_params["name"] = formatParamSetName(
            new_params, new_params.get("start_str", "params")
        )

        params_list.append(new_params)


# Automatic Parameter generation using Latin Hypersampling
# --------------------------------------------------------

elif params_generator == "lhs":

    parameter_ranges = {
    }

    lhs_values = lhsGen(
        parameter_ranges, samples=30, verbose=True,
    )

    params_list = []
    for sample_num, params_dict in enumerate(lhs_values):
        temp = copy.copy(user_params)
        temp["name"] = f"lhs_sample{sample_num}_"

        for param_name in params_dict:
            temp[param_name] = params_dict[param_name]

        params_list.append(temp)
        temp["lhs_values"] = lhs_values[sample_num]

else:
    raise ValueError(
        "No parameter generating method specified\n"
        "Must be 'single', 'manual' or 'lhs'."
    )

#
#
# -----------------------------------------------------------------------------------------------------
#
#                               DONT TOUCH ANYTHING BELOW THIS LINE !!!
#
# -----------------------------------------------------------------------------------------------------
#
#

all_params_summary = {}

for params in params_list:

    param_start_time = timeit.default_timer()

    print("\n" * 2 + "=" * 90 +
          f"\nRunning Params: {params['name']}\n" +
          "=" * 90 + "\n" * 2)

    printParams(qc_params, "Quality Control parameters")
    printParams(file_params, "File parameters")
    printParams(params, "User parameters")

    # Set Output and QC paths
    # -----------------------

    output_path = os.path.join(
        main_output_path, params["name"] + script_time_str
    )

    # quality control plot subdirectory (in output subdirectory)
    qc_path = os.path.join(output_path, "qc_plots")

    # create new paths if none exist
    for path in [output_path, qc_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    #
    # ====================================================================================
    #                                  Parse the folder
    # ====================================================================================
    #

    myparser = getFileParser(
        data_path,
        params["microscope_type"],
        fovstr_length=params["fovstr_length"],
        use_existing_filesdata=True,
    )

    # Get list of files for each FOV
    # ------------------------------

    filelist_byfov = myparser.dfToDict(
        params["fovs_to_process"],
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list"],  # list of hyb numbers for each bit
        type_list=params["type_list"],  # list of filename types
        verbose=True,
    )
    for fov in filelist_byfov:
        for i in range(len(filelist_byfov[fov])):
            filelist_byfov[fov][i] = ("/".join(filelist_byfov[fov][i][0].split("\\")),filelist_byfov[fov][i][1],filelist_byfov[fov][i][2])
    if params["subtract_background"]:
        filelist_byfov_background = myparser.dfToDict(
            params["fovs_to_process"],
            roi=params["roi"],
            num_bits=params["num_bits"],
            hyb_list=params["hyb_list_background"],
            type_list=params["type_list_background"],
            verbose=True,
        )

    # Get X and Y image dimensions
    # ----------------------------
    # read topmost entry of the dataframe for y and x dimensions (in pixels)

    ydim = int(myparser.files_df["ydim"].values[0])
    xdim = int(myparser.files_df["xdim"].values[0])

    print(f"\ny and x dimensions: {ydim:d}, {xdim:d}\n\n")

    #%%
    #
    # ====================================================================================
    #                          Read, register and filter
    # ====================================================================================
    #

    for fov in filelist_byfov:

        with ImageData(0, fov, output_path,
                       x_pix=xdim, y_pix=ydim,
                       stages_to_save_hdf5="minimal",
                       ) as imagedata:

            imagedata.readFiles(
                data_path,
                filelist_byfov[fov],
                microscope_type=params["microscope_type"],
            )

            # Subtract Background
            # -------------------

            if params["subtract_background"]:

                with ImageData(0, fov, output_path,
                               x_pix=xdim, y_pix=ydim,
                               stages_to_save_hdf5=None,
                               ) as imagedata_background:

                    imagedata_background.readFiles(
                        data_path,
                        filelist_byfov_background[fov],
                        microscope_type=params["microscope_type"],
                    )

                    with ReportTime("subtract background") as _:
                        shifts, error = subtractBackground(
                            imagedata,
                            imagedata_background,
                            subtract_only_bits=params["subtract_only_bits"],
                        )

                    print(f"Registration of background image to image:\n"
                          f"Shifts:\n{shifts}\nError:\n{error}")

                    additional_stages_to_plot = ["background_removed", ]

            else:  # don't subtract background

                additional_stages_to_plot = []

            # Correct for field and/or chromatic distortion
            # ---------------------------------------------

            if params["correct_field"]:

                # find first correction masks hdf5 file in folder
                # Warning: Assumes only one such file exists in folder
                correction_masks_file = None
                for file in os.listdir(file_params["calibration_path"]):
                    if file.startswith("correctionmasks") and file.endswith(".hdf5"):
                        correction_masks_file = os.path.join(
                            file_params["calibration_path"], file,
                        )
                        break

                assert correction_masks_file is not None, (
                    f"no field correction masks hdf5 found."
                )

                correctField(
                    imagedata, correction_masks_file,
                    img_to_mask=params["imgcolour_to_maskcolour"]
                )

            if params["correct_distortion"]:
                correctDistortion(
                    imagedata,
                    params["type_to_colour"],
                    params["reference_colour"],
                    file_params["calibration_path"],
                )

            # Register, Filter and clip
            # -------------------------

            freq_filter = butter2d(
                low_cut=params["low_cut"],
                high_cut=params["high_cut"],
                order=params["bw_filter_order"],
                xdim=xdim, ydim=ydim,
            )

            with ReportTime("register") as _:

                (registration_shifts,
                 registration_error,
                 ) = registerPhaseCorr(
                    imagedata,
                    freq_filter=freq_filter,
                    reference_bit=params["reference_bit"],
                    stage_to_register=None,
                )

            clipBelowZero(
                imagedata, 0
            )

            # plot images by bit for the chosen stages
            # ----------------------------------------

            for stage in qc_params["stages_to_plot"] + additional_stages_to_plot:
                with ReportTime(f"plot {stage} images") as _:
                    showImages(
                        imagedata, stage,
                        fig_savepath=qc_path,
                        figure_grid=params["image_grid"],
                    )

            if qc_params["plot_registration_overlay"]:
                showRegistration(
                    imagedata, current_stage="filtered_clipped",
                    fig_savepath=output_path,
                    figure_grid=params["image_grid"]
                )

    #
    # ====================================================================================
    #                     Compute global renormalization vector
    # ====================================================================================
    #

    if params["global_normalize"]:
        global_normalization = globalNormalize(
            output_path,
            per_image_percentile_cut=params["global_perfov_percentile"],
            pooled_percentile=params["global_pooled_percentile"],
            fov_subset=user_params["globalnorm_fovsubset"],
            verbose=True,
        )

    #
    # ====================================================================================
    #                  Create the GENE_DATA object (shared by all FOVs)
    # ====================================================================================
    #

    gene_data = GeneData(
        file_params["codebook_filepath"],
        fpkm_filepath=file_params["fpkm_filepath"],
        num_bits=params["num_bits"],
        print_dataframe=True,
    )
    #%%
    # ====================================================================================
    #                            Normalize and Decode
    # ====================================================================================
    #

    for iteration in range(params["num_iterations"]):

        if iteration > 0:

            raise NotImplementedError(
                "Multiple iterations not implemented yet "
                "in this version of code. Stay tuned for updates!\n"
            )

        else:

            renormalization_vector = None

        for fov in filelist_byfov:

            with ImageData(iteration, fov, output_path,
                           x_pix=xdim, y_pix=ydim,
                           stages_to_save_hdf5="minimal",
                           ) as imagedata:

                imagedata.readFromH5(
                    stage_list=["filtered_clipped", ]
                )

                if params["global_normalize"]:

                    normalizeByVector(
                        imagedata,
                        global_normalization,
                        use_clipped_array=True,
                    )

                else:

                    normalizeByPercentile(
                        imagedata,
                        percentile_upper=params["percentile_norm_high"],
                        use_clipped_array=True,
                    )

                if params["clip_normalized"]:
                    clipNormalized(
                        imagedata,
                    )

                magnitude_threshold_mask = unitNormalize(
                    imagedata,
                    magnitude_threshold=params["magnitude_threshold"],
                )

                print(
                    f"Number of pixels above magnitude threshold: "
                    f"{np.sum(magnitude_threshold_mask)}\n"
                )

                imagedata.setGeneData(gene_data)

                with ReportTime("decode") as _:
                    decodePixels(
                        imagedata,
                        dist_threshold=params["distance_threshold"],
                        mask=magnitude_threshold_mask
                    )

                with ReportTime("group spots") as _:
                    groupPixelsIntoSpots(
                        imagedata, 0,
                        minimum_pixels=params["min_region_size"],
                        large_spot_threshold=params["large_spot_threshold"],
                        small_spot_threshold=params["small_spot_threshold"],
                        verbose=False,
                    )

                # Save arrays from chosen stages
                # ------------------------------

                for stage, project_method, scale_to_max in qc_params["stages_to_save"]:

                    if stage == "normalized" and params["clip_normalized"]:
                        stage = "normalized_clipped"

                    with ReportTime(f"save {stage} array") as _:
                        imagedata.saveArray(
                            stage,
                            project_method=project_method,
                            scale_to_max=scale_to_max
                        )

                imagedata.printStagesStatus("final")
    #%%
    if params["blank_fraction_threshold"]:
        run_blank_fraction_analysis(
            output_path,
            file_params["fpkm_filepath"],
            microscope_type = params["microscope_type"],
            hamming_weight = gene_data.codebook_data.iloc[0][:gene_data.num_bits].sum(),
            num_bins = params["num_bins"],
            eps = params["eps"],
            kde_sigma = params["kde_sigma"],
            misid_target = params["misid_rate_target"],
            blank_fraction_threshold = None,
            distance_threshold = None,
            intensity_threshold = None,
            size_threshold = None,
        )

    # Record parameters used
    # ----------------------

    printParamsToFile(
        [file_params, qc_params, params, ],
        qc_path, timestr=script_time_str,
    )

    printTime(
        timeit.default_timer() - param_start_time,
        f"run {params['name']}"
    )

    #
    # ====================================================================================
    #             Summarize results and plot correlation (overall and for each FOV)
    # ====================================================================================

from BFT_decoding.plotCountsFunctions import listH5PathsByFov
from BFT_decoding.plotCountsFunctions import calcAllFovsFromHdf5, calcAndPlotCountsFromHdf5
from BFT_decoding.plotCountsFunctions import generateAndPlotResultsGrid
import matplotlib.pyplot as plt

h5coordfiles_byfov = listH5PathsByFov(
    output_path,
    verbose=True,
)

fovs_to_summarize = list(h5coordfiles_byfov.keys())

_, results_dict = calcAndPlotCountsFromHdf5(
    h5coordfiles_byfov,
    plot_correlation=qc_params["plot_correlation"],
    summed_df_savepath=qc_path,
    style="darkgrid",
    annotate=False,
    fig_savepath=qc_path,
    use_pyplot=False,
)
_, results_dict = calcAndPlotCountsFromHdf5(
    h5coordfiles_byfov,
    plot_correlation=qc_params["plot_correlation"],
    summed_df_savepath=qc_path,
    style="darkgrid",
    annotate=True,
    fig_savepath=qc_path,
    use_pyplot=False,
)
fov_grid, _, _ = generateFovGrid(
    myparser.roiSubsetOfFilesDF(params["roi"]),
    params["stage_pixel_matrix"],
    fov_subset=fovs_to_summarize,
    plot_grid=qc_params["plot_fov_grid"],
)

results_df = calcAllFovsFromHdf5(
    h5coordfiles_byfov,
    df_savepath=qc_path
)

generateAndPlotResultsGrid(
    results_df,
    fov_grid,
    fig_savepath=qc_path,
)

# Record results from this set of parameters
# ------------------------------------------

if params_generator == "lhs":
    all_params_summary[params["name"]] = {**params["lhs_values"], **results_dict}
else:
    all_params_summary[params["name"]] = results_dict

#
# ====================================================================================
#                  Collate results from all parameter sets and save
# ====================================================================================
#

from BFT_decoding.utils.printFunctions import dictOfDictToDf

dictOfDictToDf(
    all_params_summary,
    main_output_path, f"overall_summary_{script_time_str}",
    sort_by_column="correlation",
)

printTime(
    timeit.default_timer() - start_time,
    f"run whole script (everything)."
)

# show all the plots
plt.show()
