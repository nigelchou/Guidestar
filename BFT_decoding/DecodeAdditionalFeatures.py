"""
These functions work on 2 types of objects:
(1) GeneData
(2) Image data (all raw and intermediate data from *one* FOV)

nigel 30 Oct 19

v.2
Edited to add mean intensity to output for blank fraction decoding
"""

import os
import numpy as np
import h5py

import warnings

from typing import Tuple, Union

from skimage.measure import label as skimage_label_regions

from scipy import ndimage

from BFT_decoding.imageDataGS import ImageData
from BFT_decoding.geneData import GeneData
from BFT_decoding.utils.printFunctions import ReportTime


def normalizeByVector(imgdata: ImageData,
                      normalization_vector: np.ndarray,
                      use_clipped_array: bool = True,
                      registration_by_colour = False,
                      ) -> None:
    """
    Normalize images using a provided normalization vector
    Will be stored as 'normalized' array of imgdata

    Parameters
    ----------
    normalization_vector:
        1D vector with same dimensions as bits
    stage_to_normalize:
        stage to normalize.
        if normalizing for first time, use 'filtered' or 'filtered clipped'
    """

    imgdata.checkDownstream("normalized")

    assert normalization_vector.ndim == 1, (
        f"Normalization vector has {normalization_vector.ndim} dimensions. "
        f"Should have only 1 dimension"
    )

    assert normalization_vector.shape[0] == imgdata.num_bits, (
        f"Normalization vector length {normalization_vector.shape[0]} "
        f"does not match number of bits {imgdata.num_bits}"
    )

    if registration_by_colour:
        stage_to_normalize = "registered"
    elif use_clipped_array:
        stage_to_normalize = "filtered_clipped"
    else:
        stage_to_normalize = "filtered"

    array_to_normalize = imgdata.copyArray(stage_to_normalize)
    print("----------------GLOBAL NORMALIZE PRINTING decode--------------------")

    imgdata.setArray(
        "normalized",
        array_to_normalize / normalization_vector[np.newaxis, np.newaxis, np.newaxis, :],
        info=f"Normalized by vector {normalization_vector}",
    )

    imgdata.setNormalizationVector(np.squeeze(normalization_vector))


def normalizeByPercentile(imgdata: ImageData,
                          percentile_upper: float = 99.9,
                          percentile_lower: Union[float, None] = None,
                          use_clipped_array: bool = True,
                          verbose: bool = True,
                          ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """

    :param imgdata:
    :param iteration:
    :param use_clipped_array:
    :param percentile_upper:
    :param percentile_lower:
    :return:
    """
    assert imgdata.iteration == 0, (
        f"On iteration {imgdata.iteration}. "
        f"Normalize by percentile should only be used on iteration 0"
    )

    imgdata.checkDownstream("normalized")

    if use_clipped_array:
        normalized_array = imgdata.copyArray("filtered_clipped")
    else:
        normalized_array = imgdata.copyArray("filtered")

    normalized_array_shape = normalized_array.shape
    normalized_array_flat = np.reshape(normalized_array,
                                       (-1, imgdata.num_bits))

    upper_percentile_vector = np.percentile(normalized_array_flat,
                                            percentile_upper,
                                            axis=0, keepdims=True)

    if verbose:
        print(f"upper percentile vector:\n{upper_percentile_vector}\n"
              f"Min value: {upper_percentile_vector.min()}\n"
              f"Max value: {upper_percentile_vector.max()}\n")

    if percentile_lower is None:

        normalized_array_flat /= upper_percentile_vector
        lower_percentile_vector = None

    else:

        lower_percentile_vector = np.percentile(normalized_array_flat,
                                                percentile_lower,
                                                axis=0, keepdims=True)

        if verbose:
            print(f"Lower percentile vector:\n{lower_percentile_vector}\n"
                  f"Min value: {lower_percentile_vector.min()}\n"
                  f"Max value: {lower_percentile_vector.max()}\n")

        normalized_array_flat = (normalized_array_flat - lower_percentile_vector
                                 ) / (upper_percentile_vector - lower_percentile_vector)

    normalized_array = normalized_array_flat.reshape(normalized_array_shape)

    imgdata.setArray(
        "normalized", normalized_array,
        info=f"normalized by {percentile_upper:0.2f} percentile"
    )
    imgdata.setNormalizationVector(np.squeeze(upper_percentile_vector))

    return upper_percentile_vector, lower_percentile_vector


def clipNormalized(imgdata: ImageData,
                   clip_upper: float = 1.,
                   clip_lower: float = 0.,
                   ) -> None:
    """
    clip a normalized array between
    2 values (clip_lower and clip_upper)
    """
    normalized_array = imgdata.copyArray("normalized")
    np.clip(
        normalized_array,
        clip_lower, clip_upper,
        normalized_array
    )
    imgdata.setArray("normalized_clipped", normalized_array)
    imgdata.setInfo("normalized_clipped", f"Clipped to ({clip_lower},{clip_upper})")


def unitNormalize(imgdata: ImageData,
                  magnitude_threshold: float = 0.8,
                  ) -> np.ndarray:
    """
    Normalize each pixel to the bit-dimensional unit vector
    """
    if imgdata.getFlag("normalized_clipped"):
        normalized_array = "normalized_clipped"
    elif imgdata.getFlag("normalized"):
        normalized_array = "normalized"
    else:
        raise NameError(
            f"Could not find normalized or normalized_clipped array."
        )

    unitnormalized_array = imgdata.copyArray(normalized_array)

    # Calculate magnitude vector
    # --------------------------

    magnitude = np.linalg.norm(
        unitnormalized_array, axis=3,
    )

    # Normalize all pixels to unit length
    # -----------------------------------

    unitnormalized_array = np.divide(unitnormalized_array,
                                     magnitude[..., np.newaxis],
                                     out=np.zeros_like(unitnormalized_array),
                                     where=magnitude[..., np.newaxis] != 0)

    imgdata.setArray(
        "unitnormalized", unitnormalized_array,
        info=f"All pixels normalized to unit vector. "
             f"From {normalized_array}."
    )

    imgdata.setArray(
        "normalized_magnitude", magnitude,
        info=f"Normalized magnitude: "
             f"max = {np.amax(magnitude):0.3f}, "
             f"min = {np.amin(magnitude):0.3f}"
    )

    return magnitude > magnitude_threshold

def decodePixels(imgdata: ImageData,
                 dist_threshold: float = 0.517,
                 mask: Union[np.ndarray, None] = None,
                 output_distance_array: bool = True,
                 verbose: bool = True,
                 ) -> None:
    """
    Decode a unit-normalized image array

    Parameters
    ----------
    dist_threshold: float
        threshold for furthest distance from codeword to accept
    mask: boolean
        mask for which pixels to decode
    output_distance_array: bool
        whether to ouput an image-sized array of
        distances to closest codeword.
        Un-processed pixels and pixels with closest distance
        higher than the threshold will be set to np.inf
        This will be returned if True, if not returns None
    """
    imgdata.raiseIfNoGeneData()

    array = imgdata.copyArray("unitnormalized")

    array_shape = array.shape

    # set border regions to False on mask
    # -----------------------------------

    # 1D mask for y direction
    y_mask = np.empty((1, imgdata.y_pix, 1), dtype=bool)
    y_mask.fill(False)
    y_mask[0, imgdata.upper_border + 1:imgdata.y_pix - imgdata.lower_border, 0] = True

    # 1D mask for x direction
    x_mask = np.empty((1, 1, imgdata.x_pix), dtype=bool)
    x_mask.fill(False)
    x_mask[0, 0, imgdata.left_border + 1:imgdata.x_pix - imgdata.right_border] = True

    if mask is None:

        # just remove border pixels
        mask = y_mask * x_mask

    else:

        # intersection of non-border pixels with provided mask
        imgdata.checkDims(
            mask.shape[0], mask.shape[1], mask.shape[2], None, info_str="mask",
        )
        mask *= y_mask * x_mask

    # Query Codebook
    # --------------

    # Flatten all image dimensions
    array_flat = array.reshape(-1, imgdata.num_bits)  # dimensions are number of pixels x num_bits
    mask_flat = mask.reshape(-1)  # 1D array

    (dist_to_closest_codeword,
     index_of_closest_codeword
     ) = imgdata.genedata.codebook_tree.query(
        array_flat[mask_flat, :], k=3,distance_upper_bound=np.inf,
    ) # add k=3 to get nearest 3 points
     # dist and index shape is (number of pixel x k)
     
    print('max distance no process', np.amax(dist_to_closest_codeword[:,0]))

    shapes_str = (
        f"Reshaped unit-vector image has dimensions: {array_flat.shape}.\n"
        f"Flattened Mask has dimensions: {mask_flat.shape}.\n"
        f"Shape of min_index array: {index_of_closest_codeword.shape}\n"
    )
    if verbose:
        print(shapes_str)

    # add 1 to all pixel indices,
    # then assign 0 to pixels with infinite distance
    # (i.e. could not find closest codeword) so 0 is the null class
    # -------------------------------------------------------------

    index_of_closest_codeword += 1
    decoded_result_flat = index_of_closest_codeword[:,0]
    decoded_result_flat[dist_to_closest_codeword[:,0]>dist_threshold] = 0

    # Set up gene-index image as a flat array, fill gene indices then reshape
    # -----------------------------------------------------------------------

    decoded_img = np.zeros(mask_flat.shape, dtype=np.int32)
    decoded_img[mask_flat] = decoded_result_flat
    print(decoded_img.shape)
    imgdata.setArray(
        "decoded", decoded_img.reshape(array_shape[:-1]),
        info=f"Decoded.\n" + shapes_str
    )

    genelabel_img2 = np.zeros(mask_flat.shape, dtype=np.int32)
    genelabel_img3 = np.zeros(mask_flat.shape, dtype=np.int32)
 
    genelabel2_result_flat = index_of_closest_codeword[:,1]
    genelabel2_result_flat[dist_to_closest_codeword[:,1]>0.75] = 0
    
    genelabel3_result_flat = index_of_closest_codeword[:,2]
    genelabel3_result_flat[dist_to_closest_codeword[:,2]>0.80] = 0
 
    genelabel_img2[mask_flat] = genelabel2_result_flat
    genelabel_img3[mask_flat] = genelabel3_result_flat

    imgdata.setArray(
        "gene_features_2", genelabel_img2.reshape(array_shape[:-1]),
        info=f"Second closest gene.\n" + shapes_str
    )
    imgdata.setArray(
        "gene_features_3", genelabel_img3.reshape(array_shape[:-1]),
        info=f"Third closest gene.\n" + shapes_str
    )
    
    # Optional: output an array of distances to closest genes
    # -------------------------------------------------------
    if output_distance_array:
        # set pixels that are not computed or
        # are too far from codewords to have a value of infinity
        
        distance_img = np.empty(mask_flat.shape, dtype=imgdata.datatype)
        # distance_img[mask_flat] = closestdistance_decoded
        distance_img[mask_flat] = dist_to_closest_codeword[:,0]
        distance_img[distance_img>dist_threshold]=np.inf

        imgdata.setArray(
            "closestdistance", distance_img.reshape(array_shape[:-1]),
            info=f"Closest vector distance to codeword. "
                 f"Too far/not computed pixels set to np.inf.",
        )  
    
    # PROBLEM WITH DISTANCE FEATIURE IMAGE
    distance_img2 = np.empty(mask_flat.shape, dtype=imgdata.datatype)
    distance_img3 = np.empty(mask_flat.shape, dtype=imgdata.datatype)
    distance_img2[mask_flat] = dist_to_closest_codeword[:,1]
    distance_img3[mask_flat] = dist_to_closest_codeword[:,2]
    
    distance_img2[distance_img2>0.75]=np.inf
    distance_img3[distance_img3>0.80]=np.inf
    
    imgdata.setArray(
        "distance_features_2", distance_img2.reshape(array_shape[:-1]),
        info=f"Second closest vector distance to codeword. "
                f"Too far/not computed pixels set to np.inf.",
    )
    
    imgdata.setArray(
        "distance_features_3", distance_img3.reshape(array_shape[:-1]),
        info=f"Third closest vector distance to codeword. "
                f"Too far/not computed pixels set to np.inf.",
    )


def groupPixelsIntoSpots(imgdata: ImageData,
                         iteration: int,
                         minimum_pixels: int = 1,
                         magnitude_type: str = "both", #options: both, all, onbits
                         onbit_intensities_stage: str = "normalized",
                         large_spot_threshold: Union[float, None] = None,
                         small_spot_threshold: Union[float, None] = None,
                         verbose: bool = True,
                         ) -> None:
    """
    Convert decoded image into spot coordinates:
    [[y-coord of centroid, x-coord of centroid, equivlaent diameter],
    ...]
    eliminating spots that have smaller
    number of pixels than minimum_pixels

    Parameters
    ----------
    minimum_pixels: positive int
        minimum cutoff number of pixels for connected regions
    """

    imgdata.raiseIfNoGeneData()

    assert minimum_pixels > 0, (
        f"Minimum number of pixels is {minimum_pixels}. Cannot be negative!"
    )

    num_genes = imgdata.genedata.num_genes

    # Get decoded image (contains gene labels) and gene, distance feature images
    # ----------------------------------------

    decoded_img = imgdata.copyArray("decoded")
    gene_feature2_img = imgdata.copyArray("gene_features_2")
    gene_feature3_img = imgdata.copyArray("gene_features_3")
    distance_feature2_img = imgdata.copyArray("distance_features_2")
    distance_feature3_img = imgdata.copyArray("distance_features_3")

    # Get intensity image
    # -------------------

    assert imgdata.getFlag(onbit_intensities_stage), (
        f"{onbit_intensities_stage} not found in ImageData.\n"
        f"Need this to calculate on-bit intensities."
    )

    assert onbit_intensities_stage not in imgdata.stages_with_no_bit_dimension, (
        f"{onbit_intensities_stage} does not have a bits dimension.\n"
    )

    intensity_array = imgdata.copyArray(onbit_intensities_stage)

    # Get closest-distance image (if available)
    # -----------------------------------------

    if imgdata.getFlag("closestdistance"):
        distance_img = imgdata.copyArray("closestdistance")
    else:
        distance_img = None

    # Set up hdf5 coordinates file
    # ----------------------------

    coords_h5_filename = f"FOV_{imgdata.fov}_coord_iter{iteration}.hdf5"
    coords_h5_fullpath = os.path.join(imgdata.output_path, coords_h5_filename)

    codebook_df = imgdata.genedata.codebook_data
    bit_list = list(range(imgdata.num_bits))

    with h5py.File(coords_h5_fullpath, "a") as coords_file:

        for gene_index in range(num_genes):

            # Create a mask with only spots from this gene
            gene_mask = decoded_img == (gene_index + 1)

            spot_labels = skimage_label_regions(gene_mask)

            pixel_count = 0
            spot_count = 0

            # closest gene which is the decoded result
            gene_name = codebook_df.index[gene_index]
            gene_codeword = codebook_df.loc[gene_name, bit_list].values
            on_bits = np.nonzero(gene_codeword)[0] # eg. 4 on bits for GS
            off_bits = np.nonzero(gene_codeword==0)[0]
            normalized_by = imgdata.normalization_vector[on_bits]

            if verbose:
                print(f"\n{gene_name}:\n"
                      f"codeword: {gene_codeword}\n"
                      f"on-bits : {on_bits}\n"
                      f"normalized_by:{normalized_by}")

            # a list of 5+ element arrays of coordinates/data for each spot
            list_of_spot_params = []

            # Analyze and filter each spot for the gene
            # -----------------------------------------

            # number of params to store for each spot
            # 3 coordinate positions, 1 pixel count, 1 distance, on-bits

            params_array_length = 24 # includes spot size dimensions and connectivity

            label = 1  # starts with 1, 0 is for background

            for spot_slice in ndimage.find_objects(spot_labels):

                # NOTE: Each spot_slice is a
                # dimensions-length tuple of slice objects

                # set up array to store spot parameters
                # [ z , y, x centroid, dist to closest codeword, on-bit brightnesses... ]
                spot_params_array = np.zeros((params_array_length,))

                spot_mask = (spot_labels[spot_slice] == label)
                num_pixels = np.sum(spot_mask) # spot size
                spot_size_x = spot_mask.shape[1] # dimensions of the spot
                spot_size_y = spot_mask.shape[2] # dimensions of the spot
                connectivity = np.max(skimage_label_regions(spot_mask,connectivity=1)) # number of disconnected pixel groups in spot

                indices = np.nonzero(spot_mask)

                # [ array of z, y, x  positions for pixels in the spot]
                zxy_coords = [indices[dim] + spot_slice[dim].start for dim in range(3)]
                # when indexing it is zxy, when looking at image, on imshow, third value is the x axis, second is y axis

                # [ z, y, x  centroid position for the spot ]
                zxy_centroids = [np.mean(coords) for coords in zxy_coords]

                intensity_subarray = intensity_array[spot_slice + (on_bits,)]
                off_intensity_subarray = intensity_array[spot_slice + (off_bits,)]
                on_bit_intensities = np.max(intensity_subarray[spot_mask], axis=0)
            
                intensity_mean_frob = np.linalg.norm(intensity_subarray[spot_mask],axis=-1).mean()
                intensity_mean_avr = np.mean(intensity_subarray[spot_mask])

                off_intensity_frob = np.linalg.norm(off_intensity_subarray[spot_mask],axis=-1).mean()
                off_intensity_avr = np.mean(off_intensity_subarray[spot_mask])

                min_on_bit_intensity = np.min(on_bit_intensities)
                

                # Do logic of whether we should accept or reject the spot
                # -------------------------------------------------------

                if num_pixels >= minimum_pixels:  # large spot

                    if large_spot_threshold is None:
                        accept_spot = True
                    else:
                        if min_on_bit_intensity > large_spot_threshold:
                            accept_spot = True
                        else:
                            accept_spot = False

                else:  # small spot

                    if small_spot_threshold is None:
                        accept_spot = False
                    else:
                        if min_on_bit_intensity > small_spot_threshold:
                            accept_spot = True
                        else:
                            accept_spot = False

                if accept_spot:
                    
                    # calculate second and third gene features
                    # second closest gene
                    gene_name2 = gene_feature2_img[0,zxy_coords[1],zxy_coords[2]] # array of gene indexes, length = num_pixels
                    second_gene_subarray = np.zeros((len(gene_name2),5)) # each row is separate pixel, columns are: intensity, onoff, dist
                    for px_index, px in enumerate(gene_name2):
                        if px != 0:
                            gene_codeword2 = codebook_df.loc[codebook_df.index[int(px-1)], bit_list].values
                            on_bits2 = np.nonzero(gene_codeword2)[0] # eg. 4 on bits for GS
                            off_bits2 = np.nonzero(gene_codeword2==0)[0]
                            
                            intensity_subarray2 = intensity_array[0,zxy_coords[1][px_index],zxy_coords[2][px_index],on_bits2]
                            off_intensity_subarray2 = intensity_array[0,zxy_coords[1][px_index],zxy_coords[2][px_index],off_bits2]
                            
                            second_gene_subarray[px_index,0] = np.linalg.norm(intensity_subarray2)
                            second_gene_subarray[px_index,1] = np.linalg.norm(off_intensity_subarray2)
                            second_gene_subarray[px_index,2] = distance_feature2_img[0,zxy_coords[1][px_index],zxy_coords[2][px_index]]
                            second_gene_subarray[px_index,3] = np.mean(intensity_subarray2, axis=-1)
                            second_gene_subarray[px_index,4] = np.mean(off_intensity_subarray2, axis=-1)
                        else:
                            second_gene_subarray[px_index,0] = 0
                            second_gene_subarray[px_index,1] = 0
                            second_gene_subarray[px_index,2] = np.inf
                            second_gene_subarray[px_index,3] = 0
                            second_gene_subarray[px_index,4] = 0
                        
                    intensity_mean2_frob = np.mean(second_gene_subarray[:,0])
                    off_intensity2_frob = np.mean(second_gene_subarray[:,1])
                    min_distance_2 = np.min(second_gene_subarray[:,2])
                    intensity_mean2_avr = np.mean(second_gene_subarray[:,3])
                    off_intensity2_avr = np.mean(second_gene_subarray[:,4])
                    
                    # third closest gene
                    gene_name3 = gene_feature3_img[0,zxy_coords[1],zxy_coords[2]] # array of gene indexes, length = num_pixels
                    third_gene_subarray = np.zeros((len(gene_name3),5)) # each row is separate pixel, columns are: intensity, onoff, dist
                    for px_index, px in enumerate(gene_name3):
                        if px != 0:
                            gene_codeword3 = codebook_df.loc[codebook_df.index[int(px-1)], bit_list].values
                            on_bits3 = np.nonzero(gene_codeword3)[0] # eg. 4 on bits for GS
                            off_bits3 = np.nonzero(gene_codeword3==0)[0]
                            
                            intensity_subarray3 = intensity_array[0,zxy_coords[1][px_index],zxy_coords[2][px_index],on_bits3]
                            off_intensity_subarray3 = intensity_array[0,zxy_coords[1][px_index],zxy_coords[2][px_index],off_bits3]
                            
                            third_gene_subarray[px_index,0] = np.linalg.norm(intensity_subarray3)
                            third_gene_subarray[px_index,1] = np.linalg.norm(off_intensity_subarray3)
                            third_gene_subarray[px_index,2] = distance_feature3_img[0,zxy_coords[1][px_index],zxy_coords[2][px_index]]
                            third_gene_subarray[px_index,3] = np.mean(intensity_subarray3, axis=-1)
                            third_gene_subarray[px_index,4] = np.mean(off_intensity_subarray3, axis=-1)
                        else:
                            third_gene_subarray[px_index,0] = 0
                            third_gene_subarray[px_index,1] = 0
                            third_gene_subarray[px_index,2] = np.inf
                            third_gene_subarray[px_index,3] = 0
                            third_gene_subarray[px_index,4] = 0
                        
                    intensity_mean3_frob = np.mean(third_gene_subarray[:,0])
                    off_intensity3_frob = np.mean(third_gene_subarray[:,1])
                    min_distance_3 = np.min(third_gene_subarray[:,2])
                    intensity_mean3_avr = np.mean(third_gene_subarray[:,3])
                    off_intensity3_avr = np.mean(third_gene_subarray[:,4])

                    # populate spot_params_array
                    # --------------------------

                    # Z, Y, X coordinates
                    spot_params_array[:3] = zxy_centroids


                    # number of pixels
                    spot_params_array[3] = num_pixels


                    # closest vector distance (minimum within the spot)
                    if distance_img is not None:
                        distance_subarray = distance_img[spot_slice]
                        spot_params_array[4] = np.min(distance_subarray[spot_mask])


                    
                    # mean onbit-magnitudes across all pixels of a spot
                    spot_params_array[5] = intensity_mean_frob
                    spot_params_array[6] = off_intensity_frob
                    spot_params_array[7] = min_distance_2
                    spot_params_array[8] = intensity_mean2_frob
                    spot_params_array[9] = off_intensity2_frob
                    spot_params_array[10] = min_distance_3
                    spot_params_array[11] = intensity_mean3_frob
                    spot_params_array[12] = off_intensity3_frob
                    spot_params_array[13] = np.min(distance_subarray[spot_mask]) / min_distance_2
                    spot_params_array[14] = np.min(distance_subarray[spot_mask]) / min_distance_3
                    spot_params_array[15] = spot_size_x
                    spot_params_array[16] = spot_size_y
                    spot_params_array[17] = connectivity
                    spot_params_array[18] = intensity_mean_avr
                    spot_params_array[19] = off_intensity_avr
                    spot_params_array[20] = intensity_mean2_avr
                    spot_params_array[21] = off_intensity2_avr
                    spot_params_array[22] = intensity_mean3_avr
                    spot_params_array[23] = off_intensity3_avr
                                        
                    # on-bit intensities (maximum within the spot at each on-bit)
                    # spot_params_array[6:6+len(on_bits)] = on_bit_intensities
                    # spot_params_array[-len(on_bits):] = on_bit_sum

                    list_of_spot_params.append(spot_params_array)

                    pixel_count += num_pixels
                    spot_count += 1

                else:

                    # delete small region
                    decoded_img[tuple(zxy_coords)] = 0

                label += 1

            # Add dataset for the gene
            # ------------------------

            if list_of_spot_params:
                gene_spots_data = np.vstack(list_of_spot_params)
                gene_dataset = coords_file.create_dataset(
                    gene_name, data=gene_spots_data
                )
            else:  # no spots found
                gene_dataset = coords_file.create_dataset(
                    gene_name, shape=(0, params_array_length)
                )

            # Set dataset attributes
            # ----------------------

            try:
                FPKM_data = codebook_df.loc[gene_name, "FPKM_data"]
            except:
                FPKM_data = np.nan
                warnings.warn(
                    f"Could not find FPKM data for {gene_name} in coords HDF5 file."
                    f"Setting FPKM value to NaN."
                )

            dataset_attrs = {
                "gene_index": gene_index,
                "pixel_count": pixel_count,
                "spot_count": spot_count,
                "on_bits": on_bits,
                "normalized_by": normalized_by,
                "FPKM_data": FPKM_data,
            }

            for attr in dataset_attrs:
                gene_dataset.attrs[attr] = dataset_attrs[attr]

    imgdata.setArray("decoded_sizefiltered", decoded_img)


#
#
#                                               Script
# ------------------------------------------------------------------------------------------------------------
#


if __name__ == "__main__":

    # imports for test
    # ----------------

    from frequencyFilter import butter2d
    from processFunctions import registerPhaseCorr, clipBelowZero

    from filesClasses import getFileParser

    #
    #                                 Test main class
    # --------------------------------------------------------------------------------
    #
    
    data_path = '../Data/100uM_data'

    # Set microscope type and choice of roi
    # -------------------------------------

    microscope_type = "Dory"

    params = {}

    if microscope_type == "Dory":
        fovs = [0]

        stage_pixel_matrix = 8 * np.array([[0, -1], [-1, 0]])

        params["num_bits"] = 16
        params["type_list"] = ["Cy7"] * 8 + ['Cy5'] * 8 # guidestar cell
        params['hyb_list'] = [9,8,7,6,5,4,2,3,9,8,7,6,5,4,3,2] # guidestar cell
        params["roi"] = None

        params["bw_filter_order"] = 2
        params["low_cut"] = 400
        params["high_cut"] = None
        params["reference_bit"] = 1

    elif microscope_type == "Triton":

        fovs = [(0, 0), (0, 1), (0, 2)]

        stage_pixel_matrix = 8 * np.array([[1, 0], [0, 1]])

        params["num_bits"] = 16
        params["hyb_list"] = list(range(4)) * 4
        params["type_list"] = ["Cy7"] * 4 + ["Cy5"] * 4 + ["Alexa594"] * 4 + ["Cy3"] * 4
        params["roi"] = 2

        params["bw_filter_order"] = 2
        params["low_cut"] = 400
        params["high_cut"] = None
        params["reference_bit"] = 1

    # Print some of the params
    # ------------------------
    print(f"Hyb list: {params['hyb_list']}\n",
          f"Type list: {params['type_list']}\n")

    # get the file parser
    # -------------------
    myparser = getFileParser(data_path,
                             microscope_type,
                             fovstr_length = 2,
                             use_existing_filesdata=True)

    files_dict = myparser.dfToDict(
        fovs,
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list"],  # list of hyb numbers for each bit
        type_list=params["type_list"],  # list of filename types
        verbose=True,
    )


    ydim = int(myparser.files_df["ydim"].values[0])
    xdim = int(myparser.files_df["xdim"].values[0])

    # Create the GENE_DATA object
    # ---------------------------
    # stores codebook/fpkm data (this is common across all FOVs)

    file_params = {}
    codebook_dir = os.path.join(data_path, "codebook")
    file_params["codebook_filepath"] = os.path.join(codebook_dir,
                                                    "codebook_data.tsv")
    file_params["fpkm_filepath"] = os.path.join(codebook_dir,
                                                "fpkm_data.tsv")

    gene_data = GeneData(
        file_params["codebook_filepath"],
        fpkm_filepath=file_params["fpkm_filepath"],
        num_bits=params["num_bits"],
        print_dataframe=True,
    )

    save_dir = os.path.join(data_path, "imagedata_test1")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for fov in files_dict:

        with ImageData(0, fov, save_dir,
                       x_pix=xdim, y_pix=ydim,
                       stages_to_save_hdf5="minimal") as imagedata:

            imagedata.readFiles(data_path, files_dict[fov],
                                microscope_type=microscope_type,
                                )

            imagedata.printStagesStatus("after raw input")
            
            freq_filter = butter2d(order=params["bw_filter_order"],
                                   low_cut=params["low_cut"],
                                   high_cut=params["high_cut"],
                                   xdim=xdim, ydim=ydim,
                                   plot_filter=False,
                                   )

            with ReportTime("register") as _:
                (registration_shifts,
                 registration_error) = registerPhaseCorr(imagedata,
                                                         freq_filter=freq_filter,
                                                         reference_bit=params["reference_bit"])


            clipBelowZero(imagedata, 0)

            (upper_percentile_vector,
             lower_percentile_vector) = normalizeByPercentile(imagedata, percentile_lower=0,
                                                              percentile_upper=99.95,
                                                              use_clipped_array=True)

            clipNormalized(imagedata, clip_upper=1,clip_lower=0)

            magnitude_threshold_mask = unitNormalize(imagedata,
                                                     magnitude_threshold=0.55)

            imagedata.setGeneData(gene_data)

            with ReportTime("decode") as _:
                decodePixels(imagedata,
                             dist_threshold=0.517,
                             mask=magnitude_threshold_mask)

            imagedata.printStagesStatus("after decoding")

            with ReportTime("group spots") as _:
                groupPixelsIntoSpots(imagedata, iteration=0,
                                     minimum_pixels=2, verbose=False)


            # Arrays to save
            # --------------

            stages = [("normalized_clipped", "mip", True),
                      ("unitnormalized", None, True),
                      ("decoded", None, False),
                      ]
            for stage, project_method, scale_to_max in stages:
                with ReportTime(f"save {stage} array") as _:
                    imagedata.saveArray(stage,
                                        project_method=project_method,
                                        scale_to_max=scale_to_max)

            imagedata.printStagesStatus("final")
