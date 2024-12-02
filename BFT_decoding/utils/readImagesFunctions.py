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

from typing import List, Tuple

from BFT_decoding.readClasses import DaxRead


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

        else:

            raise ValueError(
                f"Microscope type {microscope_type} not recognised!"
                f"Must be 'Dory'."
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
