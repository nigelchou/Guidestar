import os
import re
from collections import defaultdict
import h5py

import numpy as np

from typing import List, Union


# ____ for testing ____


def _findH5Filenames(output_path: str,
                     iteration: int,
                     fov_list: Union[List[str], None],
                     filetype: str = "coord",
                     verbose: bool = True,
                     ) -> List[str]:
    """
    Return a list of full filepaths of all hdf5 coord files in output_path
    that match the iteration and are part of fov_list (if provided).
    If fov_list is None, files of any fov found in the folder are included.

    Parameters
    ----------
    filetype: str
        "coord" - coordinates file
        "imagedata" - imagedata file
    """

    if filetype == "coord":
        filename_pattern = re.compile(
            r"FOV_([_0-9]+)_coord_iter([0-9]+).hdf5", flags=re.IGNORECASE
        )
    elif filetype == "imagedata":
        filename_pattern = re.compile(
            r"FOV_([_0-9]+)_imagedata_iter([0-9]+).hdf5", flags=re.IGNORECASE
        )
    else:
        raise ValueError(
            f"filetype provided was {filetype}. Must be 'coord' or 'imagedata'."
        )

    hdf5file_list = []

    for filename in os.listdir(output_path):

        # NOTE: Assumes there is only one file for each FOV/iteration
        match = re.match(filename_pattern, filename)

        if match:

            accept_file = (
                    (int(match.group(2)) == iteration) &
                    (fov_list is None or int(match.group(1)) in fov_list)
            )

            if accept_file:
                full_filepath = os.path.join(output_path, filename)
                hdf5file_list.append(full_filepath)

    assert len(hdf5file_list) > 0, (
        f"No valid hdf5 files from iteration {iteration} found in {output_path}."
    )

    if verbose:
        print(f"\nhdf5 coords files:\n")
        for h5filepath in hdf5file_list:
            print(f"\t{h5filepath}\n")

    return hdf5file_list


def averageRenormVectors(output_path: str,
                         iteration: int,
                         fov_list: List[str] = None,
                         average_by: str = "median",
                         verbose: bool = True,
                         ) -> np.ndarray:
    """
    Retrieve normalization vectors from all
    valid imagedata hdf5 files in the output_path
    and return the averaged normalization vector.
    """

    hdf5file_list = _findH5Filenames(
        output_path, iteration, fov_list,
        filetype="imagedata", verbose=verbose,
    )

    norm_vector_list = []

    for h5filepath in hdf5file_list:

        with h5py.File(h5filepath, 'r') as h5file:

            if "normalization_vector" in h5file.attrs:
                norm_vector_list = h5file.attrs["normalization_vector"]

    norm_vector_array = np.vstack(norm_vector_list)

    if verbose:
        print(f"Normalized vectors from all fields of view: {norm_vector_array}")

    if average_by == "mean":
        return np.mean(norm_vector_array, axis=0)
    elif average_by == "median":
        return np.median(norm_vector_array, axis=0)
    else:
        raise ValueError(
            f"average_by was {average_by}. Must be 'mean' or 'median'."
        )


def renormalizeBits(output_path: str,
                    iteration: int,
                    bit_list: List[int],
                    fov_list: List[str] = None,
                    average_by: str = "median",
                    average_by_gene: bool = True,
                    verbose: bool = True,
                    ) -> Union[np.ndarray, List[float]]:
    """
    read the intensity hdf5 files in the output folder (from a particular iteration),
    to generate the renormalization vector for a subsequent iteration of decoding
    - The renormalization vector is just the per-bit mean intensities,
      calculated over all called pixels for that bit across all FOVs

    returns the renormalization vector (num_bits,)
    """

    assert average_by in ["mean", "median"], (
        f"Tried to average by {average_by}. Must be 'mean' or 'median."
    )

    print(f"Renormalizing bits ...\n" + "-" * 22)

    # find valid hdf5 coord files in the folder
    # -----------------------------------------

    hdf5file_list = _findH5Filenames(
        output_path, iteration, fov_list,
        filetype="coord", verbose=verbose,
    )

    # starting index of the on-bit intensities
    # 0,1,2: coordinates, 3: pixel counts, 4:closest distance.
    on_bit_start_index = 5

    renormalization_vector = []

    for bit in bit_list:

        # Collate intensities for the bit across all FOVs
        # -----------------------------------------------

        intensities_dict = defaultdict(list)

        for h5filepath in hdf5file_list:

            with h5py.File(h5filepath, 'r') as h5file:

                for gene in h5file.keys():

                    if not ("blank" in gene or "Blank" in gene):

                        on_bits = list(h5file[gene].attrs["on_bits"])

                        if bit in on_bits:
                            # get index where the correct bit intensites is stored
                            on_bit_index = on_bits.index(bit) + on_bit_start_index
                            intensities = h5file[gene][:, on_bit_index]

                            intensities_dict[gene].append(intensities)

                    else:

                        if verbose:
                            print(f"{gene} intensity not included in renormalization")

        if average_by_gene:

            # average for each gene, then average the averages
            # ------------------------------------------------

            avg_intensity_per_gene = []

            for gene in intensities_dict:

                gene_all_intensities = np.concatenate(
                    intensities_dict[gene], axis=None,
                )

                if average_by == "mean":
                    gene_avg = np.mean(gene_all_intensities)
                else:
                    gene_avg = np.median(gene_all_intensities)

                avg_intensity_per_gene.append(gene_avg)

                del gene_all_intensities

            all_intensities = np.array(avg_intensity_per_gene)

        else:

            # combine intensities from all genes, then average
            # ------------------------------------------------

            all_intensities = []

            for gene in intensities_dict:
                all_intensities.extend(intensities_dict[gene])

            all_intensities = np.concatenate(all_intensities, axis=None)

        # final round of averaging
        # ------------------------

        if average_by == "mean":
            bit_avg = np.mean(all_intensities)
        else:
            bit_avg = np.median(all_intensities)

        # get rid of vectors to avoid filling too much memory
        del all_intensities
        del intensities_dict

        renormalization_vector.append(bit_avg)

    # scale to mean
    # -------------

    renormalization_vector = np.array(renormalization_vector)
    normed_rnvec = renormalization_vector / np.mean(renormalization_vector)

    if verbose:
        print(f"Renormalization vectors calculated from {average_by} "
              f"on-bit intensities:\n{renormalization_vector}\n"
              f"Normalized renormalization vector:\n{normed_rnvec}")

    return normed_rnvec


if __name__ == "__main__":
    # imports for testing
    # -------------------

    pass
