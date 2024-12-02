"""
All the classes for reading/parsing files and images
"""

import os
import re
import copy
import numpy as np

from typing import Union, List, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
import skimage

# for registration
from skimage import io


#
# =====================================================================================================================
#   Functions for reading specific types of image data
# =====================================================================================================================
#

def readDoryImg(file_path: str,
                project: bool = False,
                ) -> np.ndarray:
    """
    read Dory/Nemo file format (individual .dax files for each FOV/colour/hyb-round)
    squeezes the image to 2D array from 3D output of DaxRead if only 1 frame is detected
    :returns 2D or 3D numpy array with the dimensions (z?, y,x)
    """
    daxreader = DaxRead(file_path)
    if project:
        img = daxreader.maxIntensityProjection()
    else:
        img = daxreader.loadAllFrames()

    frames = daxreader.frames
    del daxreader

    if frames == 1:
        return np.squeeze(img, axis=0)
    else:
        return img


def readTritonImg(file_path: str,
                  frame: int = None, ) -> np.ndarray:
    """
    read Triton file format (multiframe ome.tif files, one for each FOV and hyb round containing all colours)

    :returns 2D or 3D numpy array with the dimensions (z?, y,x)
    """

    if frame is None:
        raise ValueError("Cannot read triton format image. Frame of multiframe ome.tif file not specified")

    with warnings.catch_warnings():
        # filter out 'not an ome-tiff master file' UserWarning
        warnings.simplefilter("ignore", category=UserWarning)
        img = skimage.io.imread(file_path)[:, :, frame]

    return img


#
# =====================================================================================================================
#   Classes for reading .dax files ----- (1) DaxRead  (2) ReadFiles
# =====================================================================================================================
#

class DaxRead(object):
    """
    class to read a SINGLE dax file

    in this class, we assume that each dax file is a 3D image from a single time point
    (if multiple hybs, fovs or times are combined into a single dax file, dont use this)
    all data should be represented by the 3 dimensions:
    dim1 = frame (should be a set of z-stacks)
    dim2 = y axis
    dim3 = x axis
    keeps the frame dimension as a singleton dimension even if there is only one frame (e.g. after projection),
    for compatiblity with the other parts of the pipeline
    """

    def __init__(self,
                 filename: str = None,
                 frames: int = 1,  # z dimension
                 x_pix: int = 1024,
                 y_pix: int = 1024,
                 **kwargs):
        super(DaxRead, self).__init__(**kwargs)

        self.filename = filename
        self.frames = frames
        self.x_pix = x_pix
        self.y_pix = y_pix
        self.readInfFile()
        # this will edit the y_pix, x_pix and frames values (if available)
        # based on the associated .inf file. Othewise, default values will be used

    def readInfFile(self):
        """
        query the associated .inf file for dimensions and frames info.
        update the class attributes (y_pix, x_pix and frames) if such info is found.
        complains if the .inf file could not be found or read
        """
        dim_pattern = re.compile(r"frame\sdimensions\s=\s(\d+)\sx\s(\d+)")
        frames_pattern = re.compile(r"number\sof\sframes\s=\s(\d+)")

        try:
            with open(os.path.splitext(self.filename)[0] + ".inf", "r") as file:
                filetxt = file.read()
                match_dim = re.search(dim_pattern, filetxt)
                if match_dim:
                    self.y_pix, self.x_pix = int(match_dim.group(1)), int(match_dim.group(2))
                match_frames = re.search(frames_pattern, filetxt)
                if match_frames:
                    self.frames = int(match_frames.group(1))

        except FileNotFoundError:
            print(
                f".inf file for {self.filename} could not be found."
            )
        except OSError:
            print(
                f"Unable to open {self.filename} .inf file"
            )
        except:
            print(
                f"Could not read {self.filename} .inf file for some reason"
            )

    def loadSingleDaxFrame(self) -> np.ndarray:
        """
        load the first frame from the dax file

        probably shouldn't use this since it may get the wrong z-slice (possibly the one on top)
        """
        with open(self.filename, "rb") as daxfile:
            image_data = np.fromfile(
                daxfile, dtype=np.uint16,
                count=self.x_pix * self.y_pix,
            )

            image_data = np.reshape(
                image_data, (1, self.y_pix, self.x_pix),
            )

        return image_data

    def loadAllFrames(self,
                      subset: List[int] = None,  # must be a list of frames
                      ):
        """
        loads all the frames in the dax
        can use the given number of frames (in self.frames)
        or calculate it based on single-frame size
        """
        # first read the whole file
        with open(self.filename, "rb") as daxfile:
            image_data = np.fromfile(daxfile, dtype=np.uint16)

        # if we haven't got the number of frames or
        # the given dimensions don't match up,
        # recalculate number of frames

        if self.frames is None or (self.frames * self.y_pix * self.x_pix) != image_data.size:
            frames, remainder = divmod(image_data.size, self.y_pix * self.x_pix)
            if remainder == 0:
                self.frames = frames
            else:
                raise ValueError("Error: dax file element length is not a multiple of frame size")

        # reshape the numpy array
        image_data = image_data.reshape((self.frames, self.y_pix, self.x_pix))

        # get subset of frames if that option is given
        if subset is not None:
            subset = [frame for frame in subset if frame < self.frames]
            image_data = image_data[subset, :, :]

        return image_data

    def meanProjection(self):
        """
        average over all frames (not recommended. maximum intensity is usually better)
        """
        image_data = self.loadAllFrames()
        mp = image_data.sum(0, keepdims=True) / self.frames

        self.frames = 1  # change back to 1 since we have collapsed z dimension

        return mp

    def maxIntensityProjection(self):
        """
        maximum intensity projection i.e. highest pixel value over frames
        """
        image_data = self.loadAllFrames()
        mip = np.nanmax(image_data, axis=0, keepdims=True)
        # print("max intensity projection of dimensions", mip.shape)

        self.frames = 1  # change back to 1 since we have collapsed z dimension

        return mip

#
# class ReadFiles(object):
#     """
#     class for reading multiple dax files from a folder.
#
#     when initializing this object,
#     a list of lists of dax files should be provided:
#     the outer list represents each field of view
#     each inner list contains all the hybs for that field of view as separate dax files
#     i.e. there should be (num_fov) lists
#          with (num_hyb) dax files within in each list
#          each dax file should have (frames) z stacks
#     """
#
#     def __init__(self,
#                  data_path=None,
#                  dax_files=None,
#                  fovs=[0],
#                  num_hybs=1,
#                  frames=1, x_pix=1024, y_pix=1024,
#                  **kwargs):
#         super(ReadFiles, self).__init__(**kwargs)
#
#         self.data_path = data_path
#         self.dax_files = dax_files
#         self.fovs = fovs
#         self.num_hybs = num_hybs
#         self.frames = frames
#         self.x_pix = x_pix
#         self.y_pix = y_pix
#
#     def readImagesFromFolder(self,
#                              image_type="maxIntensityProjection",
#                              show_images=True,
#                              **kwargs,  # extra keyword argument options for plotting (see showRawImages)
#                              ):
#         """
#         read the dax files in the folder using self.dax_files list,
#         and return an ImageData object with raw_images read into it
#         optional: show all the raw images
#
#         image_type (from ReadDax class):
#             1) loadDaxFrame
#             2) meanProjection
#             3) maxIntensityProjection (default)
#         """
#
#         # make sure that there is a list of dax_files
#         if self.dax_files is None:
#             print("please provide list of .dax files")
#             return False
#
#         data = ImageData(fovs=self.fovs,
#                          num_hybs=self.num_hybs,
#                          x_pix=self.x_pix,
#                          y_pix=self.y_pix)  # don't put in the frames yet (default frames = 1)
#         # since we may collapse the z stacks into a single frame
#
#         for fov in self.fovs:
#             for hyb in range(self.num_hybs):
#                 read_each_dax = DaxRead(filename=os.path.join(self.data_path, self.dax_files[fov][hyb]),
#                                         frames=self.frames,
#                                         x_pix=self.x_pix,
#                                         y_pix=self.y_pix)
#                 # print(read_temp.filename)
#                 try:
#                     data.raw_im[fov][:, :, :, hyb] = getattr(read_each_dax, image_type)()
#                     data.frames = read_each_dax.frames  # in case frame number was not given
#                 except NameError:
#                     print("image type not recognised")
#                     return False
#
#         if show_images:
#             self.showRawImages(data, image_type=image_type, **kwargs)
#
#         return data
#
#     def showRawImages(self,
#                       data,  # raw data matrix
#                       image_type="",  # which type of projection (or no projection)
#                       figure_grid=(3, 8),
#                       figure_subplot_fontsize=25,
#                       display_fov=[0],  # a list of fovs you want to display
#                       equalize_hyb_intensities=True,
#                       # whether to normalize all hyb intensites by the same upper and lower limits (of reference hyb)
#                       hyb_ref=0,  # hyb to use for normalization of all other hybs within the fov
#                       pct_range=(45, 99.8),  # low and high end of intensity rnage by percentile
#                       normalize_histogram=True,  # whether to normalize the histogram by individual hybs
#                       histogram_max=5000,  # upper limit for histogram plot
#                       ):
#
#         for fov in display_fov:
#             fig_rawimages = plt.figure(figsize=[18, 9.5])
#
#             # vectors for min and max intensities for each hyb for normalization
#             min_intensity = np.ones(self.num_hybs + 1) * np.percentile(data.raw_im[fov][0, :, :, hyb_ref], pct_range[0])
#             max_intensity = np.ones(self.num_hybs + 1) * np.percentile(data.raw_im[fov][0, :, :, hyb_ref], pct_range[1])
#
#             for hyb in range(min(self.num_hybs, figure_grid[0] * figure_grid[1])):
#                 ax_rawimages = fig_rawimages.add_subplot(figure_grid[0], figure_grid[1], hyb + 1)
#
#                 # adjust image intensites
#                 frame_temp = data.raw_im[fov][0, :, :, hyb]
#                 if not equalize_hyb_intensities:  # update with intensity range for each hyb
#                     min_intensity[hyb] = np.percentile(frame_temp, pct_range[0])
#                     max_intensity[hyb] = np.percentile(frame_temp, pct_range[1])
#                 if normalize_histogram:  # if this is set, scale by individual hyb
#                     histogram_max = max_intensity[hyb] * 1.2
#                 ax_rawimages.imshow(frame_temp, cmap='gray',
#                                     vmin=min_intensity[hyb],
#                                     vmax=max_intensity[hyb])
#                 ax_rawimages.text(0.02, 0.98, 'bit {:d}'.format(hyb + 1),
#                                   fontsize=figure_subplot_fontsize, color='orangered', alpha=0.8,
#                                   weight='bold', fontname="Arial",
#                                   horizontalalignment='left', verticalalignment='top',
#                                   transform=ax_rawimages.transAxes)
#                 ax_rawimages.axis('off')
#
#                 # inset histogram
#                 axin = inset_axes(ax_rawimages, width="40%", height="25%", loc=4)
#                 axin.hist(frame_temp.ravel(),
#                           bins=256,
#                           range=(0, histogram_max),
#                           fc='b', ec='b')
#                 axin.set_xlim(0, histogram_max)
#                 axin.axvline(min_intensity[hyb], color='r', alpha=0.8, linewidth=1)
#                 axin.axvline(max_intensity[hyb], color='r', alpha=0.8, linewidth=1)
#                 plt.yticks(visible=False)
#                 plt.xticks(fontsize=6, color='blue')
#                 axin.patch.set_alpha(0.6)
#                 for child in axin.get_children():
#                     if isinstance(child, mpl.spines.Spine):
#                         child.set_color('dimgrey')
#
#             # print("intensity low:", min_intensity, "\nintensity high:", max_intensity, "\n")
#             fig_rawimages.suptitle("Field of view {:d}".format(fov),
#                                    color="darkred", fontsize=18, fontname="Arial", weight="bold")
#
#             plt.subplots_adjust(left=0.01, bottom=0.02, right=0.99, top=0.95, wspace=0.04, hspace=0.01)
#
#             # save the images
#             fig_rawimages.savefig(
#                 os.path.join(self.data_path, 'image_bits_{}_fov_{}.png'.format(image_type, fov)), dpi=500
#             )
