"""
all the classes or functions that deal with finding relevant files in a folder
and parsing file information from the files_data dataframe

nigel 2018
updated 23 apr 2019
"""


import os
import re
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from pprint import pprint as pp

import warnings

from typing import Tuple, Dict, Union, List

import json
import xml.etree.ElementTree as ET

import time

import matplotlib.pyplot as plt
import seaborn as sns


def getFileParser(data_path: str,
                  microscope_type: str,
                  fovstr_length: int = 3,
                  use_existing_filesdata: bool = True,
                  ):
    """
    returns the appropriate parser object (ParseDirectory class)
    for the microscope being used.

    If use_existing_filesdata set to True,
    tries to read from an existing files_data .tsv file in the data_tables subdirectory
    if not present, it parses the directory to create a new files_data file
    and saves it in the data_tables subdirectory
    """

    if microscope_type.lower() == "dory":
        parser = ParseDirectoryDory(data_path=data_path,
                                    fovstr_length=fovstr_length,
                                    use_existing_filesdata=use_existing_filesdata)

    else:
        raise TypeError("Microscope type not recognised.\n"
                        "Should be 'Dory'")

    if not parser.old_filesdata_present:
        parser.parseDirectory(save=True)

    return parser


#
# ===========================================================================================================
#                            Base class for parsing data from main directory
# ===========================================================================================================
#

class ParseDirectory(object):

    def __init__(self,
                 data_path: str = None,
                 subdir_name: str = "data_tables",
                 use_existing_filesdata: bool = True,
                 ) -> None:
        """
        Parameters
        ----------
        data_path: str
            directory to parse
        subdir_name: str
            subdirectory in the main directory in which
            to look for or save the files_data dataframe
            default is "data_tables"
        roi: int
            region of interest to use (NA)
        use_existing_filesdata: bool
            whether to check for and use an existing
            files_data.tsv file if available

        """

        assert data_path is not None, "directory not provided!"

        self.data_path = data_path
        self.use_existing_filesdata = use_existing_filesdata
        self.filesdata_path = os.path.join(data_path, subdir_name)

        self.old_filesdata_present = False

        self.files_df = None
        self.dfinfo_dict = None

        self.file_list = None
        self.fov_grid = None

        #
        # Read from existing files dataframe
        # ----------------------------------
        # only if use_existing_filesdata is set to True
        # and a subdirectory containing filesdata is found
        #

        if self.use_existing_filesdata and os.path.isdir(self.filesdata_path):

            data_files = os.listdir(self.filesdata_path)
            filedata_files = [f for f in data_files
                              if f.endswith(".tsv")
                              and f.startswith("files_data")]
            filedata_files.sort()

            #
            # Read and use the latest file_data.tsv file
            # ------------------------------------------
            # if any filedata files are found (i.e. filedata_files list not empty)
            #

            if filedata_files:
                print("-" * 70 +
                      f"\nUsing latest existing filedata .tsv: {filedata_files[-1]}\n"
                      + "-" * 70)
                self.files_df = pd.read_table(os.path.join(self.filesdata_path,
                                                           filedata_files[-1]),
                                              dtype={'fov': np.str_})
                # 'fov' column forced to be string datatype

                self.dfinfo_dict = self._getInfoFromDF(self.files_df)

                self.old_filesdata_present = True

            else:
                print("-" * 70 + "No existing filedata .tsv files found." + "-" * 70)

        #
        # Prepare a new dataframe
        # -----------------------
        # only if use_existing_filesdata is set to False
        # or could not find an existing filedata .tsv file
        #

        if self.files_df is None:
            print("-" * 70 + "\n  Initializing files_data ...\n" + "-" * 70)

            # create "data_tables" directory if it does not exist yet
            if not os.path.isdir(self.filesdata_path):
                os.mkdir(self.filesdata_path)

            # set up files dataframe to put in file names, followed by info such as hyb number and field of view
            self.files_df = pd.DataFrame(columns=["file_name",
                                                  "type", "hyb", "fov", "frames",
                                                  "ydim", "xdim",
                                                  "ygrid", "xgrid",
                                                  "tiff_frame",
                                                  "zpos", "ypos", "xpos",
                                                  "z_slice",  # another microscope type
                                                  "roi",  # another microscope type
                                                  ])

            # get all the files in the main data path
            self.file_list = os.listdir(self.data_path)

    def _saveDataframe(self,
                       directory: str,
                       df: pd.DataFrame,
                       sep: str = "\t",
                       ) -> None:
        """
        save a dataframe as .csv in the specified directory.
        sep is the separator to use in pandas to_csv.
        """
        if sep == "\t":
            file_ext = ".tsv"
        else:
            file_ext = ".csv"

        filename = "files_data" + time.strftime("_%Y%m%d_%H%M%S") + file_ext
        full_filepath = os.path.join(directory, filename)

        df.to_csv(full_filepath, sep=sep)

    def roiSubsetOfFilesDF(self,
                           roi: int,
                           ) -> pd.DataFrame:
        """
        If multiple ROIs present,
            return a SUBSET of the provided dataframe
            containing only entries from the chosen ROI
        If not, return the SAME dataframe

        Parameters
        ----------
        df: pandas dataframe
            the full dataframe to filter for ROI
        roi: int or None
            the ROI that you want to filter for in the full dataframe.
            If roi is None, will make sure that
            there is only one ROI in the full dataframe
        """



        unique_rois = self.files_df["roi"].unique()
        num_unique_rois = len(unique_rois)

        # Return either subset with ROI or the original dataframe
        # -------------------------------------------------------

        if roi is None:

            if num_unique_rois > 1:
                raise ValueError(
                    f"Multiple ROIs detected: {unique_rois}\n"
                    f"Please select one and retry."
                )

            return self.files_df

        elif np.issubdtype(type(roi), np.integer):

            assert roi in unique_rois, (
                f"ROI given ({roi}) not found in datframe.\n"
                f"Dataframe has ROIs: {unique_rois}"
            )

            return self.files_df[self.files_df["roi"] == roi]

        else:

            raise ValueError(
                f"ROI provided ({roi}) not of recognized type.\n"
                f"Should be None or an integer."
            )

    def _getInfoFromDF(self,
                       dataframe: pd.DataFrame,
                       verbose: bool = True,
                       ) -> Dict[Union[int, None], tuple]:
        """
        * to be overwritten with microscope-specific function *
        return relevant summary info from a dataframe.

        Must return a dictionary:
            keys: roi number (int)
                  If rois not supported for the microscope,
                  this should be None
            values: (num_hybs, num_colours,
                     num_fovs, num_frames,
                     ygridmax, xgridmax)
                    Any parameters that are not relevant
                    should be returned as None
        """
        pass

    def parseDirectory(self):
        """
        to be overwritten by function for specific microscope's file type and naming convention
        """
        pass

    def dfToDict(self,
                 fovs_to_process: list,
                 roi: int = None,
                 num_bits: int = None,
                 hyb_list: list = None,
                 type_list: list = None,
                 verbose: bool = True,
                 ) -> Dict[str, List[tuple]]:
        """
        Obtains a correctly ordered list of image files
        for each FOV from the files dataframe

        Calls _generateFilesDict

        Parameters
        ----------
        fovs_to_process: list
            a list of fovs to process
        roi:int
            Region of Interest to use (if using another microscope type)
            If only one region of interest present
            and you want to autodetect the ROI number,
            set this to None
        num_bits: int
            total number of bits
        hyb_list: list
            list of hyb numbers corresponding each bit
        type_list: list
            list of colour or channel types
            (e.g. "Cy5_low", "Cy5" etc.) for each bit

        returns:
            a dictionary of lists of dax files,
            keys are FOV numbers from the FOV list given
        """
        assert self.files_df is not None, ("No files_data found. "
                                           "Either parse directory or use existing")

        # if num_bits not given, try to guess
        # -----------------------------------
        # set number of bits to the maximum hyb number * number of colours

        if num_bits is None:
            warnings.warn("number of bits not provided to df-to-dict function"
                          "trying to guess the number of bits")
            max_hybs = self.dfinfo_dict[roi][0]
            max_colours = self.dfinfo_dict[roi][1]
            num_bits = max_hybs * max_colours

        assert type_list is not None, ("type_list not provided. "
                                       "Provide either a string or list of strings")

        # If lists are provided, check that their lengths match the number of bits
        # ------------------------------------------------------------------------

        if isinstance(hyb_list, list):
            assert num_bits == len(hyb_list), (f"Length of hyb list: {len(hyb_list)} "
                                               f"does not correspond to number of bits: {num_bits}!")

        if isinstance(type_list, list):
            assert num_bits == len(type_list), (f"Length of type list: {len(type_list)} "
                                                f"does not correspond to number of bits: {num_bits}!")

        # Make some assumptions to expand hyb or type into lists
        # ------------------------------------------------------

        if hyb_list is None:
            # if hyb list not provided assume bits correspond to hyb
            hyb_list = list(range(num_bits))
            print("Warning: Hyb list not provided! "
                  "Assuming hyb order matches bit order.")

        if isinstance(type_list, str):
            # if type list is given as a string, use the same type for every hyb
            type_list = [type_list, ] * num_bits

        img_files = self._generateFilesDict(fovs_to_process,
                                            num_bits, hyb_list, type_list,
                                            roi)

        def _printFilesDict(dax_files: dict) -> None:
            """
            print a neat list of filename tuples corresponding to bit for each FOV
            """
            print("\nFull list of image files for each FOV:" + "_" * 40)

            for key in dax_files:
                print(f"\n FOV {key}:")
                for bit_num, file_tuple in enumerate(dax_files[key]):
                    print(f"\tbit {bit_num}\t:\t{file_tuple}")

            print("_" * 40 + "\n")

        if verbose:
            _printFilesDict(img_files)

        return img_files

    def _generateFilesDict(self,
                           fovs_to_process: list,
                           num_bits: int,
                           hyb_list: list,
                           type_list: list,
                           roi: Union[int, None],
                           ) -> Dict[str, List[tuple]]:
        """
        to be called by method specific to the microscope type's file format

        returns dictionary:
            keys : FOV numbers
            values: list of tuples of (relative filepath,
                                       colour e.g. Cy5,
                                       tiff frame corresponding to colour)
        """
        img_files = {}

        # Extract subset of dataframe containing only desired ROI
        # -------------------------------------------------------

        if roi:
            roi_df = self.roiSubsetOfFilesDF(roi)
        else:
            roi_df = self.files_df
            
        for fov in fovs_to_process:

            # Initialize file details list for the FOV
            # ----------------------------------------
            img_files[fov] = []
            for bit in range(num_bits):
                fov_mask = roi_df["fov"] == fov
                colour_mask = roi_df["type"] == type_list[bit]
                hyb_mask = roi_df["hyb"] == hyb_list[bit]

                files_temp = roi_df[fov_mask & colour_mask & hyb_mask]
                
                # Check that we only found one file
                # ---------------------------------

                bit_id_str = (f"bit {bit} "
                              f"(FOV: {fov}({type(fov)}) "
                              f"hyb: {hyb_list[bit]} "
                              f"colour-type: {type_list[bit]} )")

                num_entries_found = len(files_temp.index)

                if num_entries_found > 1:
                    print(f"Warning: Multiple files found for "
                          f"{bit_id_str}. Using first one...\n")

                assert num_entries_found > 0, (
                    f"Entry for {bit_id_str} not found!"
                )

                # Append file details to the list for that FOV
                # --------------------------------------------

                first_index = files_temp.index[0]

                img_files[fov].append((files_temp.loc[first_index, "file_name"],
                                       type_list[bit],
                                       files_temp.loc[first_index, "tiff_frame"]))

        return img_files


#
#
# ======================================================================================================================
#                                             Dory Subclass
# ======================================================================================================================
#
#

class ParseDirectoryDory(ParseDirectory):

    def __init__(self,
                 fovstr_length: int = 3,
                 **kwargs) -> None:

        super(ParseDirectoryDory, self).__init__(**kwargs)

        self.fovstr_length = fovstr_length

    def _getInfoFromDF(self,
                       df: pd.DataFrame,
                       verbose: bool = True,
                       ) -> Dict[None, Tuple]:
        """
        returns:
            - number of hybs,
            - number of unique colour types (e.g. Cy5, Cy7, Cy5_bleach etc.)
            - number of fovs and
            - number of frames (using mimimum frame number detected)
        from the provided dataframe
        """
        try:
            num_hybs = df["hyb"].max() + 1
            num_colours = df["type"].nunique()
            num_fovs = df["fov"].astype('int32').max() + 1
            num_frames = max(df["frames"].min(), 1)
            ygridmax, xgridmax = None, None
            # in case number of frames is set to 0 due to camera error, set num_frames=1
        except:
            raise ValueError("Error getting hyb number, fov number or "
                             " frame number from files dataframe")

        if verbose:
            print("\n--> There are", num_hybs, "hybs,", num_fovs, "fovs",
                  "and each dax file has at least", num_frames, "frames.\n")

        return {None: (num_hybs, num_colours,
                       num_fovs, num_frames,
                       ygridmax, xgridmax)}

    def parseDirectory(self,
                       save: bool = True,
                       verbose: bool = True,
                       ) -> Tuple[pd.DataFrame, dict]:
        """
        scans a folder looking for all the .dax files with correct syntax,
        parses information from the corresponding .inf files,
        creates a dataframe with all the information sorted properly by imaging type, hyb, and fov
        and outputs a tuple of (dataframe, number of hybs, number of fovs, number of frames per dax)
        optional: save in the data_tables subfolder as a tsv file
        """

        # set up regular expression patterns for searching filenames
        # ----------------------------------------------------------

        filename_pattern = re.compile(r"([a-zA-Z0-9_-]+)_(\d+)_(\d+)\.(dax|npy)")

        # set up regular expression patterns for searching within .inf files
        # ------------------------------------------------------------------

        frames_pattern = re.compile(r"number\sof\sframes\s=\s+(\d+)", re.IGNORECASE)
        dims_pattern = re.compile(r"frame\sdimensions\s=\s+(\d+)\s+x\s+(\d+)", re.IGNORECASE)

        assert not self.old_filesdata_present, ("using existing files_data. "
                                                "Not parsing directory")

        # scan through files
        # ------------------
        # check relevant .inf files and append info at the bottom of the dataframe

        for file in self.file_list:

            # print(file)
            match_file = re.match(filename_pattern, file)
            # print(match_file)

            if match_file:
                if match_file.group(4) == "dax":
                    try:
                        inf_fullpath = os.path.join(self.data_path,
                                                    os.path.splitext(file)[0] + '.inf')
                        with open(inf_fullpath, 'r') as inf_file:
                            filetext = inf_file.read()
                            match_frames = re.search(frames_pattern, filetext)
                            match_dims = re.search(dims_pattern, filetext)
                    except IOError:
                        print("Warning: ", file, "information file not found!")
                        match_frames, match_dims = None, None

                    if match_frames:  # check if we found the number of frames of the dax file
                        frame_info = (match_frames.group(1),)
                    else:
                        frame_info = (np.nan,)

                    if match_dims:  # check if we found the y dimensions and x dimensions
                        dims_info = match_dims.group(1, 2)
                    else:
                        dims_info = (np.nan, np.nan)

                elif match_file.group(4) == "npy":
                    temp_img = np.load(os.path.join(self.data_path, file))

                    if len(temp_img.shape) == 3:
                        frame_info = (temp_img.shape[0],)
                    elif len(temp_img.shape) == 2:
                        frame_info = (1,)
                    else:
                        raise ValueError("dimensions of loaded .npy file not equal to 2 or 3")

                    dims_info = (temp_img.shape[-2], temp_img.shape[-1])
                    del temp_img

                try:
                    # get stage position as a tuple of (z-position, y-position, x-position)

                    # Check .xml file for stage x and y coords
                    # ----------------------------------------

                    ycoord, xcoord = None, None
                    xml_filename = os.path.splitext(match_file.group(0))[0] + '.xml'

                    with open(os.path.join(self.data_path, xml_filename), "r") as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        acquisition = root.find("acquisition")
                        if acquisition:
                            stage_position = acquisition.find("stage_position")
                            stage_position_list = [float(pos_str)
                                                   for pos_str
                                                   in stage_position.text.strip().split(",")]
                            xcoord, ycoord = stage_position_list[0], stage_position_list[1]
                            print("x coord:", xcoord, "y coord:", ycoord)
                            if ycoord is None:
                                print(f"Warning: could not find y stage coordinates for <{xml_filename}>")
                            if xcoord is None:
                                print(f"Warning: could not find x stage coordinates for <{xml_filename}>")
                        else:
                            print(f"Warning: acquisition branch of <{xml_filename}> not found")

                    # with open(fi + '.inf', "r") as infdat:
                    #     for line in infdat:
                    #         if line[:7] == 'Stage X':
                    #             xcoords = float(line[9:])
                    #         elif line[:7] == 'Stage Y':
                    #             ycoords = float(line[9:])

                    pos_tuple = (0, ycoord, xcoord)
                except:
                    pos_tuple = (np.nan,) * 3

                self.files_df.loc[len(self.files_df)] = match_file.group(0, 1, 2, 3) + frame_info + dims_info + \
                                                        (np.nan,) * 3 + pos_tuple + (np.nan, np.nan)
                # columns
                # -------
                # "file_name", "type", "hyb", "fov", "frames",
                # "ydim", "xdim", "ygrid", "xgrid", "tiff_frame"
                # "zpos", "ypos", "xpos",
                # "z_slice", "roi"

        # Convert relevant columns from string to integer
        # -----------------------------------------------

        columns_to_int = ["hyb", "frames", "ydim", "xdim"]
        self.files_df[columns_to_int] = self.files_df[columns_to_int].apply(pd.to_numeric,
                                                                            axis=1,
                                                                            errors="coerce",
                                                                            downcast="unsigned")

        # Sort the dataframe
        # ------------------
        # first by type, then by hyb, then by fov

        self.files_df.sort_values(by=["type", "hyb", "fov"], inplace=True)

        if verbose:  # display the dataframe
            print(self.files_df, "\n Datatype: ", self.files_df.dtypes, "\n")

        if save:
            self._saveDataframe(self.filesdata_path, self.files_df)

        self.dfinfo_dict = self._getInfoFromDF(self.files_df)

        # since Dory does not support ROIs, the df info dictionary
        # should only have one entry: {None: (...,...,...)}
        assert len(self.dfinfo_dict) == 1, (
            f"Files-dataframe seems to have multiple ROIs in a "
            f"microscope format (Dory) that does not support ROIs"
        )

        return (self.files_df, self.dfinfo_dict)

    def _generateFilesDict(self, fovs_to_process, num_bits, hyb_list, type_list, roi,
                           ) -> Dict[str, List[tuple]]:
        """
        generate files dictionary from Dory file format (.dax, each color in different dax file)

        returns dictionary:
            keys : FOV numbers
            values: list of tuples of (relative filepath,
                                       colour e.g. Cy5,
                                       None)
        """
        # check that roi is None for Dory, since Dory does not support ROIs
        if roi is not None:
            print("Dory does not support ROIs. Setting 'roi' to None")
            roi = None

        # check datatype of fovs_to_process. If integer, change to 2 character string
        if np.issubdtype(type(fovs_to_process[0]), np.integer):
            fovs_to_process = [str(fov).zfill(self.fovstr_length)
                               for fov in fovs_to_process]

        return super()._generateFilesDict(fovs_to_process, num_bits, hyb_list, type_list, roi)


#
#
# =============================================================================================
#                                       Script to test
# =============================================================================================
#
#

if __name__ == "__main__":

    # dialog window for testing
    import tkinter as tk
    from tkinter import filedialog

    #
    #   Test main class
    #   ---------------
    #

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select data directory")
    root.destroy()

    # Set microscope type and choice of roi
    # -------------------------------------

    microscope_type = "Dory"

    roi = 2

    if microscope_type == "Dory":
        myparser = ParseDirectoryDory(data_path=data_path,
                                      use_existing_filesdata=False,
                                      )
        print(myparser.files_df)
        myparser.parseDirectory(save=True)

        user_params = {}
        user_params["num_bits"] = 16
        user_params["hyb_list"] = list(range(4)) * 4
        user_params["type_list"] = ["Cy3", ] * 4 + ["Cy5", ] * 4 + ["Alexa594", ] * 4 + ["IR800", ] * 4
        files_dict = myparser.dfToDict([0, 1, 2],
                                       num_bits=user_params["num_bits"],  # number of bits to go until
                                       hyb_list=user_params["hyb_list"],  # list of hyb numbers for each bit
                                       type_list=user_params["type_list"],  # list of filename types
                                       verbose=True,
                                       )

