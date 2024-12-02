import os
import re
from collections import defaultdict

import json
import h5py
import numpy as np

import matplotlib.pyplot as plt

# ____ for testing ____
import tkinter as tk
from tkinter import filedialog

from BFT_decoding.correlationReadClasses import CorrelationRead


def renormalizeBits(output_folder: str,
                    iteration: int,
                    bit_list,
                    type="mean",  # "mean" or "median"
                    fov_list=None,
                    verbose=True,
                    ):
    """
    read the intensity hdf5 files in the output folder (from a particular iteration),
    to generate the renormalization vector for a subsequent iteration of decoding
    - The renormalization vector is just the per-bit mean intensities,
      calculated over all called pixels for that bit across all FOVs

    returns the renormalization vector (num_bits,)
    """
    bit_intensity_dict = defaultdict(list)

    filename_pattern = re.compile(r"intensities_FOV_([0-9]+|[0-9]+[_x][0-9]+)_iter([0-9]+).hdf5",
                                  flags=re.IGNORECASE)
    for f_name in os.listdir(output_folder):
        # find all the intensity files from the previous iteration:
        # NOTE: Assumes there is only one file for each FOV/iteration
        match = re.match(filename_pattern, f_name)

        if match and int(match.group(2)) == iteration:
            print("file match found:", f_name)
            if fov_list is None or int(match.group(1)) in fov_list:
                with h5py.File(os.path.join(output_folder, f_name), 'r') as f:
                    for key in f.keys():  # cycle through the genes
                        if not ("blank" in key or "Blank" in key):
                            num_pixels = f.get(key).shape[0]
                            if verbose:
                                print("____ number of pixels found for {} = {}".format(key, num_pixels))
                            for bit in bit_list:
                                if f[key].attrs["codeword"][bit]:
                                    bit_intensity_dict[bit].append(
                                        (np.sum(np.clip(f.get(key)[:, bit], 0, None)), num_pixels)
                                    )
                        elif ("blank" in key or "Blank" in key):
                            print("------- Blank found. not including pixels from here ------")
    if verbose:
        print("bit intensity dict:", bit_intensity_dict)
    assert len(bit_intensity_dict.keys()) == len(bit_list), "intensity values for some bits not found"

    # set up the vector for normalization
    avg_intensity_final = np.zeros((len(bit_list),), dtype=np.float64)
    for bit in bit_list:
        bit_sum_counts = np.sum(np.array(bit_intensity_dict[bit]), axis=0)
        print("sum_counts array for bit", bit, ":", bit_sum_counts)
        avg_intensity_final[bit] = bit_sum_counts[0] / bit_sum_counts[1]

    if verbose:
        print("{} intensity vector: {}".format(type, avg_intensity_final))

    return avg_intensity_final


class RenormalizeIntensities(object):

    def __init__(self,
                 outputpath: str = None,
                 iteration: int = None,
                 all_intensities_filename: str = "all_intensities.hdf5",
                 hamming_weight=4,
                 fov_list=None,  # if you want to only use a subset of FOVs
                 filename_pattern=None,  # (optional) compiled regular expression pattern for intensities hdf5 files
                 verbose=True,
                 ):

        self.outputpath = outputpath
        self.iteration = iteration
        self.fov_list = fov_list
        self.hamming_weight = hamming_weight

        # ____ create h5py object using the filename provided ____
        # leave open for writing
        self.all_hdf = h5py.File(os.path.join(outputpath, all_intensities_filename), "w")

        if filename_pattern == None:
            self.filename_pattern = re.compile(r"intensities_FOV_([0-9]+)_iter([0-9]+).hdf5", flags=re.IGNORECASE)
        else:
            self.filename_pattern = filename_pattern
        # generate dictionary of filenames
        self.files_dict = self.getFiles(self.filename_pattern, verbose=verbose)

        self.pixel_count_dict = self.getTotalPixels(self.files_dict, verbose=verbose)
        self.fillAllIntensities(self.files_dict, self.pixel_count_dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the all-intensities hdf file when we exit the scope of the renormalization object
        self.all_hdf.close()

    def getFiles(self,
                 filename_pattern,
                 verbose=True,
                 ):
        """
        get the per-FOV intensity hdf5 files for the given iteration.
        will complain if more than one such file is found for the same FOV
        returns a dictionary of filenames, keyed by the FOV number
        """
        file_dict = {}

        for filename in os.listdir(self.outputpath):
            # find all the intensity files from the previous iteration:
            # NOTE: Assumes there is only one file for each FOV/iteration
            match = re.match(filename_pattern, filename)
            if match and int(match.group(2)) == self.iteration:
                fov = int(match.group(1))
                if self.fov_list is None or fov in self.fov_list:
                    assert fov not in file_dict, "More than one intensity hdf file found for FOV {:d}".format(fov)
                    file_dict[fov] = filename

        if verbose:
            print(json.dumps(file_dict, indent=4))

        return file_dict

    def getTotalPixels(self,
                       files_dict,
                       verbose=True
                       ):
        """
        scan through all the intensity files,
        totaling up the number of pixels called out over all the FOVs for each gene

        returns a dictionary of pixel counts, keyed by the gene names
        """

        pixel_count_dict = {}

        for fov in files_dict:
            with h5py.File(os.path.join(self.outputpath, self.files_dict[fov]), 'r') as f:
                for gene in f.keys():
                    if not ("blank" in gene or "Blank" in gene):
                        # get the pixel count from array shape, or just set to 0 if array is null or empty
                        try:
                            pixel_count = f.get(gene).shape[0]
                        except IndexError:
                            pixel_count = 0
                            print("Warning: empty array detected for gene", gene, "in FOV", fov)
                        # either add to an existing entry or create new entry
                        if gene in pixel_count_dict:
                            pixel_count_dict[gene] += pixel_count
                        else:
                            pixel_count_dict[gene] = pixel_count

        if verbose:
            print(json.dumps(pixel_count_dict, indent=4))

        return pixel_count_dict

    def fillAllIntensities(self,
                           files_dict,
                           pixel_count_dict
                           ):
        for fov in files_dict:
            with h5py.File(os.path.join(self.outputpath, self.files_dict[fov]), 'r') as f:
                for gene in pixel_count_dict.keys():
                    codeword = f[gene].attrs["codeword"]
                    onbit_index_list = np.nonzero(np.array(codeword))
                    try:
                        pixel_count = f[gene].shape[0]
                    except IndexError:
                        pixel_count = 0

                    if gene in self.all_hdf.keys():
                        current_index = self.all_hdf[gene].attrs["current_index"]
                        self.all_hdf[gene][current_index:current_index + pixel_count, :] = f[gene][:, codeword]
                        self.all_hdf[gene].attrs["current_index"] += pixel_count
                    else:
                        self.all_hdf.create_dataset(gene, (pixel_count_dict[gene], self.hamming_weight))
                        self.all_hdf[gene].attrs["on_bits"] = onbit_index_list
                        if pixel_count > 0:
                            self.all_hdf[gene][0:pixel_count, :] = f[gene][:, codeword]
                            self.all_hdf[gene].attrs["current_index"] = pixel_count


if __name__ == "__main__":
    # ____________________________ Test displaying renormalization vector ______________________
    root = tk.Tk()
    root.withdraw()
    output_path = filedialog.askopenfilename(title="Please select renormalization vectors hdf file")
    root.destroy()  # need to do this otherwise will hang when closing matplotlib window
    with h5py.File(output_path, 'r') as f:
        avgvector = CorrelationRead.getMeanVector(f)
        print("final vector:\n", avgvector)
        plt.plot(avgvector)

    plt.show()

    # ____________________________ Test renormalize intensities class ______________________
    # root = tk.Tk()
    # root.withdraw()
    # outputpath = filedialog.askdirectory(title="Please select output directory")
    # root.destroy()  # need to do this otherwise will hang when closing matplotlib window
    #
    # with RenormalizeIntensities(outputpath=outputpath, iteration=1, ) as renorm:
    #     print("running")
    #
    # print("ended")
