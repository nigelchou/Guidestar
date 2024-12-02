"""
Calculates and displays correlation plots of gene counts to FPKM data,
by reading files that list spot/pixel counts per gene.

- initialized by passing in the path of the output folder.
  - finds the relevant data (optionally crosstalk data) files in the output folder
    and processes them accordingly
- contains methods to calculate correlations from the area/number-of-regions counts

nigel chou Dec 2018
updated Mar 2019
"""

import os
import time
import copy
import re  # used to parse the column headings
import json
import h5py
import warnings

from typing import Tuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

# ______ from Scipy _______
from scipy.stats.stats import pearsonr  # for fpkm correlation

# ____ for testing ____
import tkinter as tk
from tkinter import filedialog


class CorrelationRead(object):

    def __init__(self,
                 output_path: str = None,
                 iteration: int = 0,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        output_path: str
            full path of the folder that stores the processed output
            (This folder should include all the relevant area/region-number dataframes)
        iteration: int
            which iteration to read from for gene counts
        """
        super(CorrelationRead, self).__init__(**kwargs)

        self.output_path = output_path
        self.output_files = os.listdir(self.output_path)

        self.iteration = iteration

        # set default savefile names
        # --------------------------

        self.iter_str = f"iter{iteration:d}"

        self.summed_df_defaultname = f"summed_counts_" + self.iter_str
        self.combined_df_defaultname = f"combined_df_" + self.iter_str
        self.brightness_df_defaultname = f"brightness_df_" + self.iter_str

        # Create dictionary of gene data files keyed by FOV number
        # --------------------------------------------------------

        self.genedata_files = {}
        self.crosstalk_files = {}

        for file in self.output_files:

            match = re.search(r"FOV_?([0-9]+|[0-9]+[_x][0-9]+)_area_counts_iter([0-9]+)\.tsv",
                              file)
            if match and int(match.group(2)) == iteration:
                self.genedata_files[match.group(1)] = match.group(0)

            match_crosstalk = re.search(r"FOV_?([0-9]+|[0-9]+[_x][0-9]+)_area_counts_crosstalk_iter([0-9]+)\.tsv",
                                        file)
            if match_crosstalk and int(match_crosstalk.group(2)) == iteration:
                self.crosstalk_files[match_crosstalk.group(1)] = match_crosstalk.group(0)

        # check if we found gene-data files in the output folder
        self.got_files = bool(self.genedata_files)  # indicate if we found any files
        self.got_crosstalk_files = bool(self.crosstalk_files)  # indicate if we found any crosstalk files

        #
        # Get the columns of the dataframes that are common across all FOVs
        # -----------------------------------------------------------------
        # for both the gene-count files...
        #

        if self.got_files:
            print("Found FOV area and region count data files:\n",
                  json.dumps(self.genedata_files, indent=4))

            # create list of all fovs that have data in output folder
            # -------------------------------------------------------

            self.all_fovs = list(self.genedata_files.keys())
            print("List of all FOVs in output folder: {}\n".format(self.all_fovs))

            # get the FPKM data from the first data file in the list (as a pandas series)
            # ---------------------------------------------------------------------------

            try:
                df_temp = pd.read_table(os.path.join(self.output_path, self.genedata_files[self.all_fovs[0]]),
                                        index_col=0)
                self.fpkm_data = df_temp["FPKM_data"]
                # create a dataframe with area/regions columns removed (i.e. just gene_names, codebook & FPKM columns)
                # area and regions columns will be appended to this dataframe later
                self.base_df = df_temp.loc[:, ~df_temp.columns.str.contains("area|regions",
                                                                            regex=True, case=False)]
                self.base_df.to_csv(os.path.join(self.output_path, "base_dataframe.tsv"), sep="\t")
            except KeyError:
                print("could not find 'FPKM data' column in the first output file")
                self.fpkm_data = None
        else:
            print("Warning: No gene-count data files found in output folder!")
            self.fpkm_data = None
            self.all_fovs = None

        # and the crosstalk files
        if self.got_crosstalk_files:
            print("Found FOV area and region count crosstalk files:\n", json.dumps(self.crosstalk_files, indent=4))
            df_temp = pd.read_table(os.path.join(self.output_path, self.crosstalk_files[self.all_fovs[0]]),
                                    index_col=0)
            self.base_crosstalk_df = df_temp.loc[:,
                                     ~df_temp.columns.str.contains("area|regions", regex=True, case=False)]
            self.base_crosstalk_df.to_csv(os.path.join(self.output_path, "base_dataframe_crosstalk_test.tsv"), sep="\t")

    def generateBrightnessDF(self,
                             fov_list: list = None,  # (optional) subset of FOVs to use
                             savename: str = None,
                             ) -> pd.DataFrame:
        """
        reads the hdf5 intensity file and gets the average brightness of called out spots for each gene
        over the hybs that contain on-bits for the gene
        """
        brightness_df = self.base_df.copy()
        brightness_df["mean_brightness"] = np.nan

        self.intensity_files = {}
        for file in self.output_files:
            match = re.search(r"intensities_FOV_([0-9]+|[0-9]+[_x][0-9]+)_iter([0-9]+)\.hdf5", file)
            if match and int(match.group(2)) == self.iteration:
                self.intensity_files[int(match.group(1))] = match.group(0)

        if not bool(self.intensity_files):
            raise FileNotFoundError("Could not find any intensity files for iteration {:d}".format(self.iteration))

        if fov_list is not None:
            fov_list = [fov for fov in fov_list if fov in self.intensity_files.keys()]
        else:
            fov_list = self.intensity_files.keys()

        # set up array for storing [[sum of intensities, num of spots],...]
        summed_intensities_total = np.zeros((len(self.base_df.index), 2))
        for fov in fov_list:
            summed_intensities_temp = np.zeros_like(summed_intensities_total)
            with h5py.File(os.path.join(self.output_path, self.intensity_files[fov]), 'r') as f:
                for key in f.keys():
                    print(f"_____ {key} _____")
                    print(f[key].attrs["codeword"])
                    print(np.array(f.get(key)), "\n")
                    intensites = np.clip(np.array(f.get(key)[:, f[key].attrs["codeword"]]), 0, None)
                    print(intensites, "shape:", intensites.shape)
                    summed_intensities_temp[self.base_df.index.get_loc(key), :] = [np.sum(intensites),
                                                                                   intensites.size]
            summed_intensities_total += summed_intensities_temp

        brightness_df["mean_brightness"] = summed_intensities_total[:, 0] / summed_intensities_total[:, 1]
        if savename is None:
            savename = self.brightness_df_defaultname
        brightness_df.to_csv(os.path.join(self.output_path, savename + ".tsv"),
                             sep="\t")

        return brightness_df

    def generateSummedDF(self,
                         type: str = "genes",
                         fov_list: list = None,
                         savename: str = None,
                         sortby: Union[str, None] = None,
                         verbose: bool = True,
                         ) -> pd.DataFrame:
        """
        generates summed counts for both area and number-of-regions across all FOVs in the folder
        and appends "area..." and "num_regions..." columns to the base dataframe

        returns the dataframe with summed counts (this will be unsorted regardless of option chosen)

        Parameters
        ----------
        type: str
            either "genes" for gene count data, or "crosstalk" for crosstalk data
        fov_list: list
            (optional) subset of FOVs to sum over
        savename: str
            name of the saved .tsv file
        sortby: str or None
            The column to sort by, must be either "area" or "regions"
            If None, don't sort
            NOTE: the sorted dataframe is saved, but not returned by the function
        """
        if type == "genes":
            df_dict = self.genedata_files
            summed_df = copy.copy(self.base_df)
        elif type == "crosstalk":
            df_dict = self.crosstalk_files
            summed_df = copy.copy(self.base_crosstalk_df)
        else:
            raise ValueError("Codebook dictionary is not a recognised type!")

        summed_areas, summed_regions = self.sumAcrossFOV(df_dict, fov_list=fov_list)

        #
        # make a copy of base dataframe, then append summed area and num-regions columns
        # ------------------------------------------------------------------------------
        #

        if verbose:
            print(summed_areas.shape)
            print(len(summed_df.index))

        assert summed_areas.shape[0] == len(summed_df.index), "dataframe lengths do not match!"
        summed_df["area"] = summed_areas
        summed_df["regions"] = summed_regions

        if savename is None:
            savename = self.summed_df_defaultname

        summed_df.to_csv(os.path.join(self.output_path, savename + ".tsv"),
                         sep="\t")

        if sortby is not None:
            assert sortby in ["area", "regions"], (f"Column to sort by is {sortby}.\n"
                                                   f"Must be 'area' or 'regions'.")
            sorted_df = summed_df.sort_values(by=sortby, ascending=False, )
            sorted_df.to_csv(os.path.join(self.output_path, savename + "_sorted.tsv"),
                             sep="\t")

        return summed_df

    def sumAcrossFOV(self,
                     df_dict,  # dictionary of dataframes to sum across
                     fov_list: list = None,  # (optional) process just a subset of FOVs in the folder
                     ) -> tuple:
        """
        sums up the area and region columns across multiple dataframes, each from an individual FOV
        * assumes each dataframe has a single area column and region column *

        returns arrays for summed area and regions
        """
        # Set up the list of fovs to calculate correlation over
        # -----------------------------------------------------
        if fov_list is not None:
            fov_list = [fov for fov in fov_list if fov in df_dict.keys()]
        else:
            fov_list = df_dict.keys()

        area, regions = (None, None)
        for fov in fov_list:
            df_temp = pd.read_table(os.path.join(self.output_path, df_dict[fov]), index_col=0)
            area_temp = df_temp.filter(like="area").values
            if area is None:
                area = area_temp
            else:
                assert area_temp.shape == area.shape, f"FOV {fov} dimensions do not match"
                area += area_temp  # add new area array to existing area array

            regions_temp = df_temp.filter(like="regions").values
            if regions is None:
                regions = regions_temp
            else:
                assert regions_temp.shape == regions.shape, f"FOV {fov} dimensions do not match"
                regions += regions_temp  # add new region-count array to existing region-count array

        return area, regions

    def combineDFs(self,
                   base_df: pd.DataFrame = None,
                   df_dict: dict = None,
                   fov_list: list = None,
                   savename: str = None,
                   ) -> pd.DataFrame:
        """
        generate one giant dataframe by combining
        all the area/count columns of the per-fov genedata dfs
        NOTE: not used anymore by other functions but may still be pretty useful

        Parameters
        ----------
        base_df: pd.DataFrame
            base dataframe with (at least) gene names and FPKM
        df_dict: dict
            dictionary of filenames of saved dataframes to sum across
        fov_list: list
            (optional) process just a subset of FOVs in the folder
        savename: str
            for saving the dataframe to .tsv
        """
        if base_df is None:
            base_df = self.base_df
        if df_dict is None:
            df_dict = self.genedata_files

        if fov_list is not None:
            fov_list = [fov for fov in fov_list if fov in df_dict.keys()]
        else:
            fov_list = df_dict.keys()

        df_list_temp = []
        for fov in fov_list:
            df_temp = pd.read_table(os.path.join(self.output_path, df_dict[fov]),
                                    index_col=0)
            df_temp_filtered = pd.concat([df_temp.filter(like="area", axis=1),
                                          df_temp.filter(like="num_regions", axis=1)],
                                         axis=1, sort=False)
            df_list_temp.append(df_temp_filtered)

        all_columns = pd.concat(df_list_temp, axis=1, sort=False)

        combined_df = pd.concat([base_df, all_columns], axis=1, sort=False)

        if savename is None:
            savename = self.combined_df_defaultname
        combined_df.to_csv(os.path.join(self.output_path, savename + ".tsv"),
                           sep="\t")

        return combined_df

    @classmethod
    def calcLogCorrelation(cls,
                           array1: np.ndarray,
                           array2: np.ndarray,
                           ) -> tuple:
        """
        calculate log-correlation between 2 arrays
         - usually a FPKM value array and some kind of count
        returns (correlation, p-value) like in scipy's pearsonr function
        """
        # mask to remove 0 values and other non-finite stuff
        # --------------------------------------------------
        combined_mask = np.logical_and(np.logical_and(np.isfinite(array1), array1 > 0),
                                       np.logical_and(np.isfinite(array2), array2 > 0))

        return pearsonr(np.log10(array1[combined_mask]), np.log10(array2[combined_mask]))

    def calcAllCorrelations(self,
                            fovs_for_corr: list = None,  # list fovs to calculate correlation over
                            fov_grid: np.ndarray = None,
                            drop_genes: list = None,  # a list of genes to drop from calculation
                            save: bool = True,
                            show_table: bool = True,
                            ) -> pd.DataFrame:
        """
        calculates the pearson correlation and p-values of all the fovs being processed
        returns (and saves) the summary data in another dataframe
        """

        # check what we have
        if not self.got_files:
            raise IOError("Error in calculating correlation: No files found in output folder")
        if self.fpkm_data is None:
            raise ValueError("Error in calculating correlation: no FPKM data provided")

        # set up empty dataframe for recording correlation values at each field of view
        # -----------------------------------------------------------------------------

        correlation_df = pd.DataFrame(columns=["fov", "error_dist",
                                               "vec_dist_cutoff", "count_type",
                                               "correlation", "pval",
                                               "total_count", "original_column_name"])

        # set up the list of fovs to calculate correlation over
        # -----------------------------------------------------

        if fovs_for_corr is None:
            fovs_for_corr = self.all_fovs
        else:
            try:
                fovs_for_corr = [fov for fov in fovs_for_corr
                                 if fov in self.all_fovs]
            except TypeError:
                print(f"FOVs provided to calcAllCorrelations not of list type")

        for fov in fovs_for_corr:
            df_temp = pd.read_table(os.path.join(self.output_path, self.genedata_files[fov]),
                                    index_col=0)

            # trim the dataframe by dropping given genes
            # ------------------------------------------

            if drop_genes:
                df_temp = df_temp.drop(drop_genes, axis=0)

            # IMPORTANT: the following assumes there is only one column
            # for FPKM, area and regions in each dataframe
            for count_type in ["area", "regions"]:
                FPKM_array = df_temp.filter(like="FPKM").values
                count_series = df_temp.filter(like=count_type)  # this should be a pandas series
                column_name = count_series.columns[0]
                # evaluate correlation, p-value and total counts
                total_count = df_temp[column_name].sum()
                corr, pval = self.calcLogCorrelation(FPKM_array, count_series.values)

                # ____________  do various checks on column name to figure out what the column is __________

                # check if there's an integer error distance (for binarization method)
                if "err" in column_name:
                    error_dist = int(re.search(r"(?<=err).*?(?=_|$)", column_name).group())
                else:
                    error_dist = np.nan  # standard representation for no data for pandas dataframes

                # check if there's a vector distance threshold (for pixel distance method)
                if "dist" in column_name:
                    vec_dist_cutoff = float(re.search(r"(?<=dist).*?(?=_|$)", column_name).group())
                else:
                    vec_dist_cutoff = np.nan
                # __________________________________________________________________________________________

                # append a row to the bottom of the correlation dataframe
                correlation_df.loc[len(correlation_df)] = [fov, error_dist, vec_dist_cutoff,
                                                           count_type, corr, pval, total_count,
                                                           column_name]

        summed_df = self.generateSummedDF(type="genes",
                                          fov_list=fovs_for_corr,
                                          savename=self.summed_df_defaultname,
                                          sort=False)

        for count_type in ["area", "regions"]:
            corr, pval = self.calcLogCorrelation(summed_df["FPKM_data"].values,
                                                 summed_df[count_type].values)
            total_count = summed_df[count_type].sum()
            correlation_df.loc[len(correlation_df)] = [-1, np.nan, np.nan, count_type,
                                                       corr, pval, total_count,
                                                       np.nan]

        # Save correlation results dataframe in the output folder
        # -------------------------------------------------------

        if save:
            correlation_df_name = (f"correlations_summary_iter{self.iteration}"
                                   f"_{time.strftime('%Y%m%d_%H%M%S')}.tsv")
            correlation_df.to_csv(os.path.join(self.output_path, correlation_df_name), sep="\t")
        if show_table:
            print("________ Correlation Dataframe: ______\n",
                  correlation_df[
                      ["fov", "count_type", "correlation", "pval", "total_count"]
                  ].tail(10), "\n")

        return correlation_df

    def resultsGrids(self,
                     correlation_df: pd.DataFrame,
                     fov_grid: np.ndarray,
                     plot: bool = True,
                     fig_savepath: str = "",
                     verbose: bool = True,
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate and return a grid of correlations and total counts
        only looks at 'regions' rows
        """
        correlation_grid = np.zeros_like(fov_grid, dtype=np.float64)
        count_grid = np.zeros_like(fov_grid, dtype=np.int32)

        df_temp = correlation_df[correlation_df["count_type"] == "regions"]
        df_temp = df_temp[["fov", "correlation", "total_count"]]

        if verbose:
            print(f"truncated dataframe:\n{df_temp}")

        for row in df_temp.values:
            fov = row[0]
            position = np.argwhere(fov_grid == fov)

            correlation = row[1]
            total_count = row[2]

            if verbose:
                print(f"FOV: {fov}\n"
                      f"correlation: {correlation}\n"
                      f"total_count: {total_count}\n"
                      f"position: {position}\n"
                      )

            if position.shape[1] == 2:  # exclude FOVs that are not found on the grid
                assert position.shape[0] == 1, (
                    f"multiple copies of FOV {fov} found in the fov grid"
                )
                correlation_grid[position[0, 0], position[0, 1]] = correlation
                count_grid[position[0, 0], position[0, 1]] = total_count

        if plot:
            from BFT_decoding.utils.gridsQC import plotHeatmaps
            # heatmap plot parameters:
            # ( grid, vmin, vmax, colourmap, size-of-annotations)
            plotHeatmaps([(correlation_grid, 0, 1, "RdYlGn", 10),
                          (count_grid, None, None, "hot", 5)],
                         iteration=self.iteration,
                         fig_savepath=fig_savepath)

        if verbose:
            print(f"Correlation grid:\n{correlation_grid}\n\n"
                  f"Count grid:\n{count_grid}\n")

        return correlation_grid, count_grid

    @classmethod
    def showNormalizations(cls,
                           hdf5_filepath: str,
                           verbose: str = True,
                           ) -> None:
        """
        Plot across-bit normalization vectors over different iterations
        
        Parameters
        ----------
        hdf5_filepath: str
            hdf5 file storing normlization vectors
        """

        fig_normvec, ax_normvec = plt.subplots()

        # Set up renormalization plots dictionary
        # ---------------------------------------
        # indiviudal renormalization vector plots for each iteration/FOV

        plots_dict = {}

        with h5py.File(hdf5_filepath, 'r') as f:

            for key in f:

                # iteration 1+ (after renormalization)
                # ------------------------------------

                print(f"_____ Iteration {key} _____")
                if int(key) > 0:
                    if verbose:
                        print(f"\nArray for {f[key]}:")
                        print(np.array(f.get(key)), "\n")
                    plots_dict[key] = ax_normvec.plot(np.array(f.get(key)),
                                                      '.-',
                                                      color="red",
                                                      linewidth=2,
                                                      markersize=12,
                                                      alpha=0.1 * int(key))

                # iteration 0 (before renormalization)
                # ------------------------------------

                if key == "0":

                    for group in f.get(key):

                        hdf_groupref = f"{key}/{group}"

                        if verbose:
                            print(f"Group: {group}\n" + hdf_groupref +
                                  f"\n{np.array(f[hdf_groupref])}")

                        plots_dict[group] = ax_normvec.plot(np.array(f[hdf_groupref]),
                                                            '.-',
                                                            color="blue",
                                                            linewidth=2,
                                                            markersize=8,
                                                            alpha=0.3)

                    # plot the mean over all FOVs for iteration 0
                    # -------------------------------------------

                    plots_dict["mean"] = ax_normvec.plot(cls._getMeanVector(f),
                                                         '.--',
                                                         color="slateblue",
                                                         linewidth=8,
                                                         markersize=14,
                                                         alpha=0.8)

                if verbose:
                    print(f"Plots dictionary: {plots_dict}")  # check the dictionary

    @classmethod
    def _getMeanVector(cls,
                       hdf_object: h5py.File,
                       ) -> np.ndarray:
        """
        returns the mean from multiple normalization vectors across FOVs
        from the first processing iteration (before renormalization)

        Parameters
        ----------
        hdf_object:
              the h5py object storing renormalization vectors
        """
        try:
            intensity_vector = np.zeros((hdf_object["0/FOV0"].shape))
            fov_counter = 0
            for group in hdf_object["0"]:
                intensity_vector += np.array(hdf_object["0/{}".format(group)])
                fov_counter += 1

            return intensity_vector / fov_counter

        except ValueError:
            print("Could not find 0th iteration in renormalization vectors hdf file.")

    @staticmethod
    def _sumDfColumn(counts_df: pd.DataFrame,
                     col_type: str = "num_regions",
                     ) -> Union[int, float]:
        """
        calculates the total number of pixels (area) or spots (num_regions)
        assumes there is only one such column in the counts dataframe
        returns sum of the entries in the column
        FIXME: Not being used. consider removing
        """
        # get a list of columns with the correct type
        col_to_count = [col for col in counts_df.columns
                        if col_type in col]

        # if didn't find any columns
        if len(col_to_count) == 0:
            raise IndexError(f"No '{col_type}' columns not found")

        # if found more tham 1 column
        elif len(col_to_count) > 1:
            warnings.warn(f"Mutiple '{col_type}' columns found")

        return counts_df[col_to_count[0]].sum()

    @classmethod
    def analyzeCorrelationH5(cls,
                             h5_filepath: str,
                             counts_df: pd.DataFrame,
                             properties_df: pd.DataFrame = None,
                             properties_df_column: str = None,
                             iteration: int = None,
                             verbose: bool = True,
                             **kwargs,
                             ) -> Tuple[Figure, dict]:
        """
        Plot a correlation and bar plot
        from a coords hdf5 file

        Parameters
        ----------
        h5_filepath:
            hdf5 coordinates file
        counts_df:
            ANY counts df with the same gene list and a FPKM_data column
            FIXME: in future we should include FPKM data as
            FIXME: an attribute of each gene in the hdf5 file
        properties_df:
            dataframe for properties for every gene
        properties_df_column:
            column-name indicating which column of properties df to use
        iteration:
            analysis iteration (only used to generate plot title)
        **kwargs:
            arguments for plotting e.g.
            - limits: limits for x and y
            - style: seaborn style
        """
        df_cols = ["count", "size"]

        with h5py.File(h5_filepath) as f:

            gene_list = f.keys()
            df = pd.DataFrame(index=gene_list, columns=df_cols, dtype="float64")

            for gene in f:
                # gene count is number of spots (1st dimension)
                count = f[gene].shape[0]

                # size: mean square of effective radius of spots (3rd column)
                mean_size = np.mean(f[gene][:, 2] ** 2)

                df.loc[gene, df_cols] = [count, mean_size]

        if verbose:
            print(f"Dataframe from hdf5 file:\n{df}")

        # Check if dataframes have the right columns
        # ------------------------------------------

        for col in ["gene_names", "FPKM_data"]:
            assert col in counts_df.columns, (
                f"dataframe provided does not have '{col}' column"
            )

        assert "gene_names" in properties_df.columns, (
            f"dataframe provided does not have 'gene_names' column"
        )

        # Merge dataframes
        # ----------------

        df = df.merge(counts_df[["gene_names", "FPKM_data"]],
                      left_index=True, right_on="gene_names")
        # Note: this will change the index back to a column

        if verbose:
            print(f"dataframe with FPKM merged in:\n{df}\n"
                  f"datatypes:\n{df.dtypes}")

        if properties_df is not None:

            try:
                properties_df.rename({properties_df_column: "property"},
                                     axis="columns", inplace=True)
            except KeyError:
                print(f"no column {properties_df_column} in properties dataframe")
                raise

            df = df.merge(properties_df[["gene_names", "property"]],
                          left_on="gene_names", right_on="gene_names")

        # Get correlation-related output values
        # -------------------------------------

        corr_dict = cls._getCorrVals(df, "FPKM_data", "count")

        # sort dataframe and get confidence-related output values
        # -------------------------------------------------------

        sorted_df, conf_dict = cls._getConfidence(df, col_to_sort="count")

        # concatenate both correlation and confidence dictionaries
        # --------------------------------------------------------

        values_dict = {**corr_dict, **conf_dict}

        #
        # Plot the figure
        # ---------------
        #

        scatterplot_title, barplot_title = cls._getPlotTitles(values_dict,
                                                              iteration=iteration)

        fig = cls._plotCorrAndCounts(sorted_df.set_index("gene_names"),
                                     x_column="FPKM_data",
                                     y_column="count",
                                     scatter_title=scatterplot_title,
                                     bar_title=barplot_title,
                                     **kwargs)

        return fig, values_dict

    @classmethod
    def analyzeCorrelation(cls,
                           counts_df: pd.DataFrame,
                           properties_df: pd.DataFrame = None,
                           properties_df_column: str = None,
                           analysis_type: str = "counts",
                           iteration: int = None,
                           verbose: bool = False,
                           **kwargs,
                           ) -> Tuple[plt.Figure, dict]:
        """
        plot correlation and bar plots from the region/area counts listed in a dataframe.

        returns a tuple of:
         1) a reference to the figure
         2) a dictionary of results and validation parameters:
            "correlation", "p-value", "confidence ratio",
            "mediangene to medianblank", "total count"

        Parameters
        ----------
        counts_df: pandas Dataframe
            a dataframe read from a counts.csv file,
            either for a single fov or for all fovs summed
            "gene names" should be one of the columns and NOT and index
        properties_df: pandas Dataframe
            some property of the gene or sequence targeting the gene
            will be represented as marker size of scatter points
        properties_df_column: str
            the column of the properties dataframe to use, if there are mutliple columns
        analysis_type: str
            either "counts" (default) or "area", which is number of pixels
        iteration: int
            iteration that the data is from. Only used for plot tile

        **kwargs:
            arguments for plotting e.g.
            - limits: limits for x and y
            - style: seaborn style
            - annotatie: whether to annotate the plot with gene names
        """
        if verbose:
            print(f"Counts dataframe:\n{counts_df}\n"
                  f"Properties dataframe:\n{properties_df}")

        # Find the relevant columns
        # -------------------------
        # If more than one column found, just take the first

        FPKM_col_name = counts_df.filter(like='FPKM').columns[0]
        area_col_name = counts_df.filter(like='area').columns[0]
        counts_col_name = counts_df.filter(like='regions').columns[0]

        areas_float = counts_df[area_col_name].values.astype(np.float64)
        counts_float = counts_df[counts_col_name].values.astype(np.float64)

        trimmed_cols = ["gene_names", FPKM_col_name]

        #
        # deal with properties data if provided
        # -------------------------------------
        #

        if properties_df is not None:

            # append appropriate column of properties df to the main df
            # ---------------------------------------------------------

            try:
                properties_df.rename({properties_df_column: "property"},
                                     axis="columns", inplace=True)
            except KeyError:
                print(f"no column {properties_df_column} in properties dataframe")
                raise

            counts_df = counts_df.merge(properties_df[["gene_names", "property"]],
                                        on="gene_names", how="left", copy=False, )

            trimmed_cols.append("property")

            # set genes with no property (e.g. blanks) to half the lowest property value
            # --------------------------------------------------------------------------

            property_min = counts_df["property"].min(skipna=True)
            counts_df.fillna({"property": property_min / 2})

            if verbose:
                print(f"minimum value of property column: {property_min}\n"
                      f"Merged counts dataframe:\n{counts_df}\n")

        #
        # Choose to use spot-counts or area (number of pixels)
        # ----------------------------------------------------
        #

        if analysis_type == "counts":

            # Find mean area of the spots
            # ---------------------------
            # by dividing total num pixels/num spots

            s = np.sqrt(
                np.divide(areas_float, counts_float,
                          out=np.ones_like(areas_float),
                          # if no spots found, default to 1
                          where=areas_float != 0, )
            )
            counts_df["size"] = s
            trimmed_cols.append("size")

            if verbose:
                print(f"Mean spot areas:\n{s}")

        elif analysis_type == "area":

            # reassign counts column to area
            # ------------------------------
            # will do the same analysis, but without varying spot sizes

            counts_col_name = area_col_name

        else:
            raise ValueError(f"Analysis type {analysis_type} not recognized.\n"
                             f"Must be 'counts' or 'area'.")

        trimmed_cols.append(counts_col_name)

        #
        # Get correlation-related output values
        # -------------------------------------
        #

        corr_dict = cls._getCorrVals(counts_df, FPKM_col_name, counts_col_name)

        # Trim dataframe leaving only relevant columns, then sort
        # -------------------------------------------------------

        counts_df_trimmed = counts_df[trimmed_cols]

        if verbose:
            print("Scatterplot dataframe:\n",
                  counts_df_trimmed, counts_df_trimmed.describe())

        sorted_df, conf_dict = cls._getConfidence(counts_df_trimmed,
                                                  col_to_sort=counts_col_name)

        # concatenate both correlation and confidence dictionaries
        # --------------------------------------------------------

        values_dict = {**corr_dict, **conf_dict}

        #
        # Plot the figure
        # ---------------
        #

        scatterplot_title, barplot_title = cls._getPlotTitles(values_dict,
                                                              count_type=analysis_type,
                                                              iteration=iteration)

        fig = cls._plotCorrAndCounts(sorted_df.set_index("gene_names"),
                                     x_column=FPKM_col_name,
                                     y_column=counts_col_name,
                                     scatter_title=scatterplot_title,
                                     bar_title=barplot_title,
                                     **kwargs)

        return fig, values_dict

    @staticmethod
    def _getPlotTitles(values_dict: dict,
                       count_type: str = "Counts",
                       iteration: int = None,
                       ) -> Tuple[str, str]:
        """
        use calculated values from a dictionary containing:
        (1) log-correlation (2) p-value (3) total count
        (4) confidence percentage (5) gene to blank ratio
        
        to generate plot titles for both the 
        (1) scatter plot and (2) bar plot
        """

        # Title of the scatterplot
        # ------------------------

        if iteration is not None:
            iteration_str = f" for iteration {iteration}"
        else:
            iteration_str = ""

        scatterplot_title = (f"{count_type} vs FPKM{iteration_str}\n"
                             f"log correlation = {values_dict['correlation']:0.3f}, "
                             f"p = {values_dict['p-value']:0.2e}"
                             f"  |  Total RNA count: {values_dict['total count']:0.0f}")

        # Title of the barplot
        # --------------------

        barplot_title = (f"Genes, descending order  |  "
                         f"{values_dict['percent_above_blank']:0.1f} % above blank  |  "
                         f"median-gene to median-blank ratio : "
                         f"{values_dict['gene_blank_ratio']:0.2f}")

        return scatterplot_title, barplot_title

    @classmethod
    def _getCorrVals(cls,
                     df: pd.DataFrame,
                     x_column: str,
                     y_column: str,
                     ) -> Dict[str, float]:
        """
        Calculate the correlation, p-value and total count
        from 2 columns of a dataframe to correlate
        (usually x column is FPKM and y column is count)
        returns as dictionary
        """
        corr, pval = cls.calcLogCorrelation(df[x_column].values,
                                            df[y_column].values)
        total_count = df[y_column].sum()

        return {"correlation": corr,
                "p-value": pval,
                "total count": total_count}

    @classmethod
    def _getConfidence(self,
                       df: pd.DataFrame,
                       col_to_sort: str = "count",
                       verbose: bool = True,
                       ) -> Tuple[pd.DataFrame, dict]:
        """
        Modify dataframe and get gene/blank confidence ratios

        returns:
        (1) Dataframe that is:
         - sorted
         - includes an additional "genes_to_blank" column

        (2) dictionary with:
         - percent_above_blank
         - gene_blank_ratio
        """

        # sort dataframe
        # --------------

        sorted_df = df.sort_values(by=[col_to_sort],
                                   ascending=False, inplace=False)

        # Add extra column indicating if each row is a gene or a blank
        # ------------------------------------------------------------

        sorted_df["gene_or_blank"] = "gene"
        sorted_df.loc[
            sorted_df["gene_names"].str.contains("blank", case=False, regex=False),
            "gene_or_blank"
        ] = "blank"

        if verbose:
            print("Sorted dataframe for barplot:\n", sorted_df)

        # set up groupby object
        # ---------------------

        grouped_df = sorted_df.groupby("gene_or_blank")


        #
        # get the percentage of genes that are above blanks
        # -------------------------------------------------
        #

        gene_count, blank_count = grouped_df.count().loc[["gene", "blank"],
                                                         col_to_sort]

        try:
            # get index of largest blank
            sorted_geneblank_list = sorted_df["gene_or_blank"].tolist()
            first_blank_idx = sorted_geneblank_list.index("blank")
            percent_above_blank = first_blank_idx / gene_count
        except ValueError:
            first_blank_idx = None
            percent_above_blank = 1

        percent_above_blank *= 100

        #
        # get ratio of median gene over median blank
        # ------------------------------------------
        #

        group_medians = grouped_df.median()
        try:
            genes_median = group_medians.loc["gene", col_to_sort]
            blanks_median = group_medians.loc["blank", col_to_sort]
            gene_blank_ratio = genes_median / blanks_median
        except KeyError:
            gene_blank_ratio = np.inf

        if verbose:
            print(f"\nNumber of genes:  {gene_count}\n"
                  f"Number of blanks: {blank_count}\n\n"
                  f"Group counts:\n{grouped_df.count()}\n\n"
                  f"Gene/blank list:\n{sorted_geneblank_list}\n\n"
                  f"First blank index: {first_blank_idx}\n"
                  f"% above blank: {percent_above_blank:0.3f}\n\n"
                  f"Median groups:\n{group_medians}\n\n"
                  f"Gene-to-blank median ratio: {gene_blank_ratio:0.3f}\n")

        return sorted_df, {"percent_above_blank": percent_above_blank,
                           "gene_blank_ratio": gene_blank_ratio, }

    @classmethod
    def _plotCorrAndCounts(cls,
                           df: pd.DataFrame,
                           x_column: str = "FPKM_data",
                           y_column: str = "counts",
                           annotate: bool = True,
                           scatter_title: str = "",
                           bar_title: str = "",
                           limits: Tuple[tuple, tuple] = ((None, None), (None, None)),
                           style: str = "darkgrid",
                           base_markersize: int = 32,
                           verbose: bool = True,
                           ) -> plt.Figure:
        """
        plot:
        (1) correlation scatterplot with colormap (above) and
        (2) count barplot (below) that uses the y_column

        Input Dataframe MUST satisfy the following:
         - SORTED along y_column
         - index as labels (genes)
         - x and y data columns must be present as specified
           in x_column and y_column
         - "gene_or_blank" column to indicate if gene or blank
         - (optional) "size" column to control spot size
         - (optional) "property" column controls colour of spots

        Parameters
        ----------
        scatter_title: str
            scatter plot title
        bar_title: str
            bar plot title
        limits: 2-tuple of 2-tuples of int or float
            y and x limits for the scatter plot
        style: str
            seaborn style to use
        base_markersize: int
            size for scatterplot markers.
            If no property given, all markers will be this size.
            If not, this will be size of smallest marker
        """
        sns.set_style(style)

        fig_fpkmcorr = plt.figure(figsize=[13, 9.5])

        gs = gridspec.GridSpec(4, 1)

        ax_scatter = fig_fpkmcorr.add_subplot(gs[:-1, :])
        ax_bar = fig_fpkmcorr.add_subplot(gs[-1, :])

        #
        # Scatter Plot
        # ============
        #

        df_genes = df[df["gene_or_blank"] == "gene"]
        df_blank = df[df["gene_or_blank"] == "blank"]

        s = base_markersize
        s_blank = base_markersize

        if "size" in df.columns:
            s *= df_genes["size"].values
            s_blank *= df_blank["size"].values
            print(f"s: {s}")

        if "property" in df.columns:
            c = df_genes["property"].values
            c_blank = df_blank["property"].values
            colorbar = True
        else:
            c = sns.xkcd_rgb["cerulean blue"]
            c_blank = sns.xkcd_rgb["pale red"]
            colorbar = False


        sc_gene = ax_scatter.scatter(x=df_genes[x_column].values,
                                     y=df_genes[y_column].values,
                                     s=s, c=c,
                                     alpha=0.6,
                                     cmap='viridis',
                                     )
        sc_blank = ax_scatter.scatter(x=df_blank[x_column].values,
                                      y=df_blank[y_column].values,
                                      s=s_blank, c=c_blank,
                                      alpha=0.6,
                                      cmap='viridis',
                                      edgecolor=sns.xkcd_rgb["pale red"],
                                      )

        if colorbar:
            divider = make_axes_locatable(ax_scatter)
            cbar_ax = divider.append_axes("right", size="3%", pad=0.2)
            fig_fpkmcorr.colorbar(sc_gene, cax=cbar_ax)

        ax_scatter.set_title(scatter_title)

        # configure y and x axis
        # ----------------------

        ax_scatter.set_ylabel("count")
        ax_scatter.set_yscale("symlog", linthreshy=1, linscaley=0.2)
        # 1 because the smallest number of regions is 1

        ax_scatter.set_xlabel("FPKM value")
        min_fpkm = np.amin(df_genes[x_column].values)
        for l in [1, 0.1, 0.01, 0.001]:
            if min_fpkm > l:
                linthreshx = l
                break
        print(f"linthreshx: {linthreshx}")

        ax_scatter.set_xscale("symlog",
                              linthreshx=linthreshx,
                              linscalex=0.2)

        # set default axes limits
        ax_scatter.set_xlim((-linthreshx / 10, None))
        ax_scatter.set_ylim((-.1, None))

        # FPKM can be a fraction. we'll set this as 10^-2

        # set axes limits (if given)
        ax_scatter.set_xlim(limits[0])
        ax_scatter.set_ylim(limits[1])

        # Annotate the plot with labels (ususally gene names)
        # ---------------------------------------------------

        if annotate:
            for label in df.index:
                x = df.loc[label, x_column]
                y = df.loc[label, y_column]

                if verbose:
                    print(f"{label}: y={y}, x={x}")

                ax_scatter.annotate(label, xy=(x, y),
                                    xytext=(2, 1), textcoords='offset points',
                                    fontname='Arial', fontsize=12,
                                    alpha=0.8, )

        #
        # Bar plot
        # ========
        #

        barplot = sns.barplot(x=df.index, y=y_column,
                              hue="gene_or_blank",
                              data=df,
                              dodge=False,
                              ax=ax_bar,
                              palette=[sns.xkcd_rgb["cerulean blue"],
                                       sns.xkcd_rgb["pale red"], ],
                              )
        barplot.set_ylabel('count')
        barplot.set_yscale('log')
        barplot.set_xticklabels(barplot.get_xticklabels(),
                                fontdict={'fontsize': 6, },
                                rotation=85,
                                # ha='right',
                                )
        barplot.xaxis.label.set_visible(False)
        barplot.set_title(bar_title)
        ax_bar.legend().set_visible(False)

        # Adjust space around plots
        # -------------------------

        fig_fpkmcorr.subplots_adjust(left=0.08, right=0.94,
                                     bottom=0.08, top=0.94,
                                     wspace=0.1, hspace=.8)

        return fig_fpkmcorr


#
#
# ================================================================================================
#                                             Script
# ================================================================================================
#
#

if __name__ == "__main__":

    #
    # Test plotting function
    # ----------------------
    #

    root = tk.Tk()
    root.withdraw()
    # get the filepath for the counts .tsv file
    counts_path = filedialog.askopenfilename(title="Please select file to display FPKM correlation plot")
    # get the filepath for the properties .tsv file
    properties_path = filedialog.askopenfilename(title="Please select file containing properties info")
    # get the filepath for the hdf5 coordinates file
    h5_path = filedialog.askopenfilename(title="Please select coordinates hdf5 file")
    root.destroy()

    print("-" * 50 + f"\nCounts path: {counts_path}\n"
                     f"Properties path: {properties_path}\n")

    assert counts_path, "no counts .tsv file provided"
    counts_df = pd.read_csv(counts_path, sep="\t")

    if properties_path:
        properties_df = pd.read_csv(properties_path, sep="\t")
    else:
        properties_df = None
    print("properties dataframe:", properties_df)

    if h5_path:
        CorrelationRead.analyzeCorrelationH5(h5_path,
                                             counts_df,
                                             properties_df=properties_df,
                                             properties_df_column="mean_brightness",
                                             style="darkgrid",
                                             limits=((None, None), (None, None)),
                                             annotate=False,
                                             )

    else:
        CorrelationRead.analyzeCorrelation(counts_df,
                                           properties_df=properties_df,
                                           properties_df_column="mean_brightness",
                                           style="darkgrid",
                                           limits=((None, None), (None, None)),
                                           annotate=False,
                                           )

    plt.show()

    # ____________________________ Test main class ____________________________________________

    # root = tk.Tk()
    # root.withdraw()
    # output_path = filedialog.askdirectory(title="Please select output directory")
    # root.destroy()  # need to do this otherwise will hang when closing matplotlib window
    #
    # mycorrelations = CorrelationRead(output_path=output_path)
    #
    # if False:
    #     summed_df = mycorrelations.generateSummedDF(type="crosstalk", sort=True)
    #     print(summed_df)
    #
    # if False:
    #     correlations_summary_df = mycorrelations.calcAllCorrelations(show_table=True)
    #
    # if False:
    #     combined_df = mycorrelations.combineDFs()
    #
    # if True:
    #     brightness_df = mycorrelations.generateBrightnessDF()

    # ____________________________ Test displaying renormalization vector ______________________

    # root = tk.Tk()
    # root.withdraw()
    # output_path = filedialog.askopenfilename(title="Please select renormalization vectors hdf file")
    # root.destroy()  # need to do this otherwise will hang when closing matplotlib window
    #
    # CorrelationRead.showNormalizations(output_path)
    #
    # plt.show()
