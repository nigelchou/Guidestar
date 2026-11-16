"""
functions to read a series of images from different formats

readImages
----------
takes a list of image-data
(filepath, colour, etc..)
and reads images according to the appropriate microscope format

refactored from ImageData

nigel 11 dec 19
"""

import os
import numpy as np
import warnings
import re
from typing import List, Tuple, Dict, Any

# import skimage.io  # for loading multi-frame ome-tif files
import tifffile
from .readClasses import DaxRead


def readImages(img_list: List[tuple],
               data_path: str,
               ydim: int,
               xdim: int,
               microscope_type: str = None,
               dory_projection_method: str = "maxIntensityProjection",
               verbose: bool = True,
               ) -> Tuple[np.ndarray, List[str]]:
    """
    Read images from specified microscope image format.
    Input is a list of tuples providing:
     filename, colour channel and other information

    Parameters
    ----------
    :param img_list:
    :param data_path:
    :param ydim:
    :param xdim:
    :param microscope_type:
    :param dory_projection_method:
    :param verbose:
    :return:
    """
    colour_list = []

    num_imgs = len(img_list)

    for img_num in range(num_imgs):

        # each element of image_list is:
        #  Dory : tuple of (filename, type/colour, nan)

        colour_list.append(img_list[img_num][1])

        # filename / path of the image is the first entry of the tuple
        img_fullpath = os.path.join(
            data_path, img_list[img_num][0]
        )

        #
        # Read image file using reader for specific format
        # ------------------------------------------------
        #

        if microscope_type in ["Dory", "Nemo"]:

            read_dax = DaxRead(
                filename=img_fullpath,
                x_pix=xdim, y_pix=ydim,
            )

            try:
                temp_image = getattr(read_dax, dory_projection_method)()

                if verbose:
                    print(
                        f"Image {img_num} dax file from {img_fullpath} "
                        f"has dimensions: {temp_image.shape}"
                    )

            except NameError:
                print(
                    "Please choose either\n"
                    "(1) loadDaxFrame\n"
                    "(2) meanProjection\n"
                    "(3) maxIntensityProjection\n"
                )
                raise

            frames = read_dax.frames


        elif microscope_type in ["Triton", "ELS", "Confocal"]:  # uses .tif file format

            # NOTE: the following section assumes only one frame,

            # we will need to figure out how multi-frame (z-stack) data

            # is organized and change this part of code later

            frames = 1

            with warnings.catch_warnings():

                # filter out 'not an ome-tiff master file' UserWarning

                warnings.simplefilter("ignore", category=UserWarning)

                tiff_frame = img_list[img_num][-1]

                tiff_img = tifffile.imread(img_fullpath)

                print("orginal", tiff_img.shape)

            if tiff_img.ndim == 4:

                if (re.search("000_000", img_list[img_num][0])):  # we need to make FOV 000_000 has same dim with other

                    print("first fov..")

                    n_channel = tiff_img.shape[0]

                    n_stack = tiff_img.shape[1]

                    tiff_img_1 = np.moveaxis(tiff_img, [0, 1, 2, 3], [2, 3, 0, 1])

                    tiff_img_1 = tiff_img_1.reshape(xdim, ydim, n_stack * n_channel)

                    tiff_img_1 = tiff_img_1.reshape(xdim, ydim, n_stack, n_channel)

                    tiff_img = np.moveaxis(tiff_img_1, [0, 1, 2, 3], [2, 3, 0, 1])

                else:

                    pass

                tiff_img = np.max(tiff_img, axis=0)  # assume that we have (z, c, xdim, ydim)

                print("final image shape after MIP", tiff_img.shape)

            # check that the tiff being read has 3 dimensions

            assert tiff_img.ndim == 3, (

                f"tiff file:\n{img_fullpath}\n "

                f"has {tiff_img.ndim} dimensions. Should be 3.\n"

            )

            # Check for "frames" dimension of the read tiff

            # ---------------------------------------------

            # need to do this because skimage

            # swaps axes when there are 3 or 4 frames

            # but not if there are 2

            smallest_dimension = np.argmin(tiff_img.shape)

            if smallest_dimension == 2:  # last dim is frames

                temp_image = tiff_img[..., tiff_frame]


            elif smallest_dimension == 0:  # first dim is frames

                temp_image = tiff_img[tiff_frame, ...]


            else:

                raise IndexError(

                    f"Unusual dimensions found in {img_fullpath}.\n"

                    f"{smallest_dimension} dimension is the smallest."

                )

            if verbose:
                print(

                    f"Image {img_num} temp image: \n"

                    f"min value: {np.min(temp_image)},\n"

                    f"max value: {np.max(temp_image)}\n"

                )

            assert temp_image.ndim == 2

            temp_image = temp_image[np.newaxis, ...]

            print(f"shape of image {img_num} is {temp_image.shape}.\n")

        else:

            raise ValueError(
                f"Microscope type {microscope_type} not recognised!"
                f"Must be 'Dory' or 'Triton'."
            )

        # Set up empty array with dimensions from the first image
        # -------------------------------------------------------

        if img_num == 0:

            raw_array = np.zeros(
                (frames, ydim, xdim, num_imgs),
                dtype=np.uint16
            )

            first_img_frames = frames

        else:

            assert frames == first_img_frames, (
                f"Number of frames in image {img_num} does not "
                f"match number of frames in the first image.\n"
            )

        # Check shape of image and write to output array
        # ----------------------------------------------

        array_shape = raw_array.shape[:-1]

        assert temp_image.shape == array_shape, (
            f"Shape of image {img_num} read from {img_fullpath}: "
            f"{temp_image.shape}\ndoes not match shape of array: {array_shape}\n"
        )

        raw_array[..., img_num] = temp_image

    return raw_array, colour_list
