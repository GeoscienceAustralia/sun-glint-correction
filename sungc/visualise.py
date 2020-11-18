#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def display_image(img, out_png):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis("off")
    ax.imshow(img, interpolation="None", cmap=cm.jet)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_png:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.0, dpi=400)
        print("    Saved '{0}'".format(out_png))

    return ax


def display_scaled_image(img, lower_val, upper_val, out_png):

    nrows, ncols = img.shape
    scaled_img = np.array(img, order="K", copy=True)

    lower_percentile, upper_percentile = extract_percentiles(img, 5, 95)
    if lower_val:
        scaled_img[scaled_img < lower_val] = lower_val
    else:
        scaled_img[scaled_img < lower_percentile] = lower_percentile

    if upper_val:
        scaled_img[scaled_img > upper_val] = upper_val
    else:
        scaled_img[scaled_img > upper_percentile] = upper_percentile

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis("off")
    ax.imshow(scaled_img, interpolation="None", cmap=cm.jet)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_png:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.0, dpi=700)
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
            s=ann1, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10, color="w"
        )

    ax[1].imshow(im2, interpolation="None")
    ax[1].axis("off")
    if ann2:
        ax[1].annotate(
            s=ann2, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10, color="w"
        )

    if out_png:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01)
        fig.savefig(os.path.join(out_png), format="png", bbox_inches="tight", dpi=400)


def plot_correlations(
    fig,
    ax,
    r2,
    slope,
    y_inter,
    nir_vals,
    vis_vals,
    scale_factor,
    nir_band_id,
    vis_band_id,
    odir,
):
    """
    Plot the correlations between NIR band and the visible bands for
    the Hedley et al. (2005) sunglint correction method

    Parameters
    ----------
    fig : matplotlib.figure object
        Reusing a matplotlib.figure object to avoid the creation many
        fig instantances

    ax : matplotlib.axes._subplots object
        Reusing the axes object

    r2 : float
        The correlation coefficient squared of the linear regression
        between NIR and a VIS band

    slope : float
        The slope/gradient of the linear regression between NIR and
        a VIS band

    y_inter : float
        The intercept of the linear regression between NIR and a
        VIS band

    nir_vals : numpy.ndarray
        1D array containing the NIR values from the ROI

    vis_vals : numpy.ndarray
        1D array containing the VIS values from the ROI

    scale_factor : int or None
        The scale factor used to convert integers to reflectances
        that range [0...1]

    nir_band_id : str
        The NIR band number

    vis_band_id : str
        The VIS band number

    odir : str
        Directory where the correlation plots are saved

    """
    # clear previous plot
    ax.clear()

    # ----------------------------------- #
    #   Create a unique cmap for hist2d   #
    # ----------------------------------- #
    ncolours = 256

    # get the jet colormap
    colour_array = plt.get_cmap("jet")(range(ncolours))  # 256 x 4

    # change alpha values
    # e.g. low values have alpha = 1, high values have alpha = 0
    # color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
    # e.g. low values have alpha = 0, high values have alpha = 1
    # color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # We want only the first few colours to have low alpha
    # as they would represent low density [meshgrid] bins
    # which we are not interested in, and hence would want
    # them to appear as a white colour (alpha ~ 0)
    num_alpha = 25
    colour_array[0:num_alpha, -1] = np.linspace(0.0, 1.0, num_alpha)
    colour_array[num_alpha:, -1] = 1

    # create a colormap object
    cmap = LinearSegmentedColormap.from_list(name="jet_alpha", colors=colour_array)

    # ----------------------------------- #
    #  Plot density using np.histogram2d  #
    # ----------------------------------- #
    xbin_low, xbin_high = np.percentile(nir_vals, (1, 99), interpolation="linear")
    ybin_low, ybin_high = np.percentile(vis_vals, (1, 99), interpolation="linear")

    nbins = [int(xbin_high - xbin_low), int(ybin_high - ybin_low)]

    bin_range = [[int(xbin_low), int(xbin_high)], [int(ybin_low), int(ybin_high)]]

    hist2d, xedges, yedges = np.histogram2d(
        x=nir_vals, y=vis_vals, bins=nbins, range=bin_range
    )

    # normalised hist to range [0...1] then rotate and flip
    hist2d = np.flipud(np.rot90(hist2d / hist2d.max()))

    # Mask zeros
    hist_masked = np.ma.masked_where(hist2d == 0, hist2d)

    # use pcolormesh to plot the hist2D
    qm = ax.pcolormesh(xedges, yedges, hist_masked, cmap=cmap)

    # create a colour bar axes within ax
    cbaxes = inset_axes(
        ax,
        width="3%",
        height="30%",
        bbox_to_anchor=(0.37, 0.03, 1, 1),
        loc="lower center",
        bbox_transform=ax.transAxes,
    )

    # Add a colour bar inside the axes
    fig.colorbar(
        cm.ScalarMappable(cmap=cmap),
        cax=cbaxes,
        ticks=[0.0, 1],
        orientation="vertical",
        label="Point Density",
    )

    # ----------------------------------- #
    #     Plot linear regression line     #
    # ----------------------------------- #
    x_range = np.array([xbin_low, xbin_high])
    (ln,) = ax.plot(
        x_range,
        slope * (x_range) + y_inter,
        color="k",
        linestyle="-",
        label="linear regr.",
    )

    # ----------------------------------- #
    #          Format the figure          #
    # ----------------------------------- #
    # add legend (top left)
    lgnd = ax.legend(loc=2, fontsize=10)

    # add annotation
    ann_str = (
        r"$r^{2}$" + " = {0:0.2f}\n"
        "slope = {1:0.2f}\n"
        "y-inter = {2:0.2f}".format(r2, slope, y_inter)
    )
    ann = ax.annotate(s=ann_str, xy=(0.02, 0.76), xycoords="axes fraction", fontsize=10)

    # Add labels to figure
    bnir_label = "B" + nir_band_id
    bvis_label = "B" + vis_band_id

    xlabel = f"Reflectance ({bnir_label})"
    ylabel = f"Reflectance ({bvis_label})"

    if scale_factor is not None:
        if scale_factor > 1:
            xlabel += " " + r"$\times$" + " {0}".format(int(scale_factor))
            ylabel += " " + r"$\times$" + " {0}".format(int(scale_factor))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.show(); sys.exit()

    # Save figure
    png_file = os.path.join(
        odir, "Correlation_{0}_vs_{1}.png".format(bnir_label, bvis_label)
    )

    fig.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    # delete all lines and annotations from figure,
    # so it can be reused in the next iteration
    qm.remove()
    ln.remove()
    ann.remove()
    lgnd.remove()


def extract_percentiles(band, lower_percentile, upper_percentile):
    """
    This function extracts the values of the user specified lower and
    upper percentile of the data stored in the numpy 2D array. This
    function, in conjunction with histogram stretching generates
    very nice RGB images.
    """
    return np.percentile(
        band.flatten(), (lower_percentile, upper_percentile), interpolation="linear"
    )


def linear_stretching(band, new_min, new_max):
    """
    This function performs a linear histogram stretch on the input 2D
    band using the new_min and new_max values. Note that in the
    altered image new_min = 0 and new_max = 1
    """
    grad = (0.0 - 1.0) / (float(new_min) - float(new_max))
    return (band - new_min) * grad


def enhanced_rgb_stretch(refl_img, fmask, rgb_ix, nodata, lower_perc, upper_perc):
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
        raise Exception("\nERROR: Dimension mismatch between fmask and rgb image\n")

    if lower_perc >= upper_perc:
        raise ValueError("\nERROR: Lower percentile >= upper percentile\n")

    if len(rgb_ix) != 3:
        raise Exception("\nERROR: rgb_ix must only have three elements\n")

    scaled_rgb = np.zeros([dims[1], dims[2], 3], order="C", dtype=np.float32)

    # Find ocean, land/cloud and null pixels
    water_pxlix = np.where(
        (fmask == 5)
        & (refl_img[0, :, :] != nodata)
        & (refl_img[1, :, :] != nodata)
        & (refl_img[2, :, :] != nodata)
    )

    other_pxlix = np.where(
        (fmask >= 1)
        & (fmask <= 4)
        & (refl_img[0, :, :] != nodata)
        & (refl_img[1, :, :] != nodata)
        & (refl_img[2, :, :] != nodata)
    )

    for i in range(0, 3):

        # extract the land and ocean pixels of this band
        other_pixels = refl_img[rgb_ix[i], other_pxlix[0], other_pxlix[1]]
        ocean_pixels = refl_img[rgb_ix[i], water_pxlix[0], water_pxlix[1]]

        # Scaling the land and ocean separately
        # to obtain the best RGB image.
        other_lowerval, other_upperval = extract_percentiles(other_pixels, 1.0, 99.0)
        ocean_lowerval, ocean_upperval = extract_percentiles(
            ocean_pixels, lower_perc, upper_perc
        )

        # Further constraining:
        other_pixels[other_pixels < other_lowerval] = other_lowerval
        other_pixels[other_pixels > other_upperval] = other_upperval

        ocean_pixels[ocean_pixels < ocean_lowerval] = ocean_lowerval
        ocean_pixels[ocean_pixels > ocean_upperval] = ocean_upperval

        # performing linear stretching using the lower
        # and upper percentiles for the ocean pixels
        # so that they range from 0 to 1
        scaled_rgb[other_pxlix[0], other_pxlix[1], i] = linear_stretching(
            other_pixels, other_lowerval, other_upperval
        )

        scaled_rgb[water_pxlix[0], water_pxlix[1], i] = linear_stretching(
            ocean_pixels, ocean_lowerval, ocean_upperval
        )

    return scaled_rgb


def seadas_style_rgb(refl_img, rgb_ix, scale_factor):
    """
    Create a (NASA-OBPG) SeaDAS style RGB. A very simple transformation
    of reflectances is used to create very impressive RGB's

    Parameters
    ----------
    refl_img : numpy.ndarray
        reflectance image with dimensions of [nBands, nRows, nCols]

    rgb_ix : list or tuple
        reflectance image band indices of the red, green and blue
        channels (in this order!)
        e.g. rgb_ix = [3,2,1]

    scale_factor : float or int
        The multiplicative factor used to convert to reflectances

    Returns
    -------
    scaled_rgb : numpy.ndarray [dtype = np.uint8]
        Scaled and enhanced rgb image with dimensions of
        [nRows, nCols, 3].

    Raises
    ------
        * Exception if len(rgb_ix) != 3

    """

    if len(rgb_ix) != 3:
        raise Exception("\nERROR: rgb_ix must only have three elements\n")

    nbands, nrows, ncols = refl_img.shape

    # specify coefficient used in transformation:
    c1 = 0.091935692
    c2 = 0.61788
    c3 = 10.0
    c4 = -0.015

    scaled_rgb = np.zeros([nrows, ncols, 3], order="C", dtype=np.uint8)

    for i in range(0, 3):
        tmp_im = c1 + c2 * np.arctan(
            c3 * (refl_img[rgb_ix[i], :, :] / float(scale_factor) + c4)
        )
        tmp_im[tmp_im < 0] = 0
        tmp_im[tmp_im > 1] = 1
        scaled_rgb[:, :, i] = np.array(255 * tmp_im, order="C", dtype=np.uint8)

    return scaled_rgb
