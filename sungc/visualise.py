#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def display_image(img, out_png):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis("off")
    ax.imshow(img, interpolation="None", cmap=cm.jet)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_png:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(
            out_png, format="png", bbox_inches="tight", pad_inches=0.0, dpi=400
        )
        print("    Saved '{0}'".format(out_png))

    return ax


def display_scaled_image(img, lower_val, upper_val, out_png):

    nRows, nCols = img.shape
    scaled_img = np.array(img, order="K", copy=True)

    lowerPercentile, upperPercentile = extract_Percentiles(img, 5, 95)
    if lower_val:
        scaled_img[scaled_img < lower_val] = lower_val
    else:
        scaled_img[scaled_img < lowerPercentile] = lowerPercentile

    if upper_val:
        scaled_img[scaled_img > upper_val] = upper_val
    else:
        scaled_img[scaled_img > upperPercentile] = upperPercentile

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis("off")
    ax.imshow(scaled_img, interpolation="None", cmap=cm.jet)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_png:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(
            out_png, format="png", bbox_inches="tight", pad_inches=0.0, dpi=700
        )
        print("    Saved '{0}'".format(out_png))


def display_two_images(im1, im2, ann1=None, ann2=None, out_png=None):
    """
    This function generates the following figure:
    -----------------------------
    | ann1        | ann2        |
    |             |             |
    |             |             |
    |             |             |
    |     im1     |      im2    |
    |             |             |
    |             |             |
    |             |             |
    |             |             |
    -----------------------------

    Parameters
    ----------

    im1 : numpy.ndarray
        A 2-dimensional array

    im2 : numpy.ndarray
        A 2-dimensional array

    ann1 : str or None
        Annotation for im1

    ann2 : str or None
        Annotation for im2

    out_png : str or None
        PNG filename
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(im1, interpolation="None")
    ax[0].axis("off")
    if ann1:
        ax[0].annotate(
            s=ann1,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            color="w",
        )

    ax[1].imshow(im2, interpolation="None")
    ax[1].axis("off")
    if ann2:
        ax[1].annotate(
            s=ann2,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            color="w",
        )

    if out_png:
        fig.subplots_adjust(
            left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01
        )
        fig.savefig(
            os.path.join(out_png), format="png", bbox_inches="tight", dpi=400,
        )


def extract_Percentiles(band, lowerPercentile, upperPercentile):
    """
    This function extracts the values of the user specified lower and
    upper percentile of the data stored in the numpy 2D array. This
    function, in conjunction with histogram stretching generates
    very nice RGB images.
    """
    return np.percentile(
        band.flatten(),
        (lowerPercentile, upperPercentile),
        interpolation="linear",
    )


def linear_stretching(band, new_min, new_max):
    """
    This function performs a linear histogram stretch on the input 2D
    band using the new_min and new_max values. Note that in the
    altered image new_min = 0 and new_max = 1
    """
    Grad = (0.0 - 1.0) / (float(new_min) - float(new_max))
    return (band - new_min) * Grad


def enhanced_RGB_stretch(
    refl_img, fmask, rgb_ix, nodata, lower_perc, upper_perc
):
    """
    Parameters
    ----------
    refl_img : numpy.ndarray
        reflectance image with dimensions of [nBands, nRows, nCols]

    fmask : numpy.ndarray
        Mask image with dimensions of [nRows, nCols]
        mask value, flag
        -999, nodata
           0, nodata
           1, land
           2, cloud
           3, shadow
           4, snow
           5, water

    rgb_ix : list or tuple
        reflectance image band indices of the red, green and blue
        channels (in this order!)
        e.g. rgb_ix = [3,2,1]

    nodata : float
       nodata value

    lower_perc : float
        The lower percentile used in the scaling [0 : 100]

    upper_perc : float
        The upper percentile used in the scaling [0 : 100]

    Returns
    -------
    scaled_rgb : numpy.ndarray
        Scaled and enhanced rgb image with dimensions of
        [nRows, nCols, 3]. Here, the values in each band
        are scaled to range 0 to 1 required by
        matplotlib.pyplot.imshow

    Notes
    -----
        * land/cloud/shadow/snow pixels are enhanced
          separately from water pixels
        *

    Raises
    ------
        * ValueError if lower_perc > upper_perc
        * Exception if dimension mismatch
        * Exception if len(rgb_ix) != 3

    """
    dims = refl_img.shape
    msk_dims = fmask.shape

    if (dims[1] != msk_dims[0]) or (dims[2] != msk_dims[1]):
        raise Exception(
            "\nERROR: Dimension mismatch between fmask and rgb image\n"
        )

    if lower_perc >= upper_perc:
        raise ValueError("\nERROR: Lower percentile >= upper percentile\n")

    if len(rgb_ix) != 3:
        raise Exception("\nERROR: rgb_ix must only have three elements\n")

    scaled_rgb = np.zeros([dims[1], dims[2], 3], order="C", dtype=np.float32)

    # Find ocean, land/cloud and null pixels
    water_pxlIx = np.where(
        (fmask == 5)
        & (refl_img[0, :, :] != nodata)
        & (refl_img[1, :, :] != nodata)
        & (refl_img[2, :, :] != nodata)
    )

    other_pxlIx = np.where(
        (fmask >= 1)
        & (fmask <= 4)
        & (refl_img[0, :, :] != nodata)
        & (refl_img[1, :, :] != nodata)
        & (refl_img[2, :, :] != nodata)
    )

    for i in range(0, 3):

        # extract the land and ocean pixels of this band
        other_pixels = refl_img[rgb_ix[i], other_pxlIx[0], other_pxlIx[1]]
        ocean_pixels = refl_img[rgb_ix[i], water_pxlIx[0], water_pxlIx[1]]

        # Scaling the land and ocean separately
        # to obtain the best RGB image.
        other_lowerVal, other_upperVal = extract_Percentiles(
            other_pixels, 1.0, 99.0
        )
        ocean_lowerVal, ocean_upperVal = extract_Percentiles(
            ocean_pixels, lower_perc, upper_perc
        )

        # Further constraining:
        other_pixels[other_pixels < other_lowerVal] = other_lowerVal
        other_pixels[other_pixels > other_upperVal] = other_upperVal

        ocean_pixels[ocean_pixels < ocean_lowerVal] = ocean_lowerVal
        ocean_pixels[ocean_pixels > ocean_upperVal] = ocean_upperVal

        # performing linear stretching using the lower
        # and upper percentiles for the ocean pixels
        # so that they range from 0 to 1
        scaled_rgb[other_pxlIx[0], other_pxlIx[1], i] = linear_stretching(
            other_pixels, other_lowerVal, other_upperVal
        )

        scaled_rgb[water_pxlIx[0], water_pxlIx[1], i] = linear_stretching(
            ocean_pixels, ocean_lowerVal, ocean_upperVal
        )

    return scaled_rgb
