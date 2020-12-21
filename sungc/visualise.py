#!/usr/bin/env python3

import os
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_correlations(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    r2: float,
    slope: float,
    y_inter: float,
    corr_vals: np.ndarray,
    vis_vals: np.ndarray,
    scale_factor: Union[float, int],
    corr_bname: str,
    vis_bname: str,
    odir: Union[Path, str],
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

    corr_vals : numpy.ndarray
        1D array containing the NIR values from the ROI

    vis_vals : numpy.ndarray
        1D array containing the VIS values from the ROI

    scale_factor : int or None
        The scale factor used to convert integers to reflectances
        that range [0...1]

    corr_bname : str
        The NIR band number

    vis_bname : str
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
    xbin_low, xbin_high = np.percentile(corr_vals, (1, 99), interpolation="linear")
    ybin_low, ybin_high = np.percentile(vis_vals, (1, 99), interpolation="linear")

    nbins = [int(xbin_high - xbin_low), int(ybin_high - ybin_low)]

    bin_range = [[int(xbin_low), int(xbin_high)], [int(ybin_low), int(ybin_high)]]

    hist2d, xedges, yedges = np.histogram2d(
        x=corr_vals, y=vis_vals, bins=nbins, range=bin_range
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
    ann = ax.annotate(ann_str, xy=(0.02, 0.76), xycoords="axes fraction", fontsize=10)

    # Add labels to figure
    xlabel = f"Reflectance ({corr_bname})"
    ylabel = f"Reflectance ({vis_bname})"

    if scale_factor is not None:
        if scale_factor > 1:
            xlabel += " " + r"$\times$" + " {0}".format(int(scale_factor))
            ylabel += " " + r"$\times$" + " {0}".format(int(scale_factor))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.show(); sys.exit()

    # Save figure
    png_file = os.path.join(
        odir, "Correlation_{0}_vs_{1}.png".format(corr_bname, vis_bname)
    )

    fig.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    # delete all lines and annotations from figure,
    # so it can be reused in the next iteration
    qm.remove()
    ln.remove()
    ann.remove()
    lgnd.remove()


def seadas_style_rgb(
    refl_img: np.ndarray,
    rgb_ix: Union[tuple, list],
    scale_factor: Union[float, int],
    mask_nodata: bool = False,
) -> np.ndarray:
    """
    Create a (NASA-OBPG) SeaDAS style RGB. A very simple transformation
    of reflectances is used to create very impressive RGB's

    Parameters
    ----------
    refl_img : numpy.ndarray
        reflectance image with dimensions of [nbands, nrows, ncols]

    rgb_ix : list or tuple
        reflectance image band indices of the red, green and blue
        channels (in this order!)
        e.g. rgb_ix = [3,2,1]

    scale_factor : float or int
        The multiplicative factor used to convert to reflectances

    mask_nodata : bool
        Whether to set pixels [< 0 or > scale_factor] as grey color

    Returns
    -------
    scaled_rgb : numpy.ndarray [dtype = np.uint8]
        Scaled and enhanced rgb image with dimensions of
        [nrows, ncols, 3].

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

    if mask_nodata:
        # Get the nodata mask. Here, if a pixel in any band is considered
        # as nodata then all pixels in the RGB will be assigned as nodata
        msk = (refl_img[0, :, :] < 0) | (refl_img[0, :, :] > scale_factor)
        for i in range(1, 3):
            msk[(refl_img[i, :, :] < 0) | (refl_img[i, :, :] > scale_factor)] = True

    for i in range(3):
        tmp_im = 255 * (
            c1
            + c2 * np.arctan(c3 * (refl_img[rgb_ix[i], :, :] / float(scale_factor) + c4))
        )

        tmp_im[tmp_im < 0] = 0
        tmp_im[tmp_im > 255] = 255

        if mask_nodata:
            tmp_im[msk] = 127

        scaled_rgb[:, :, i] = np.array(tmp_im, order="C", dtype=np.uint8)

    return scaled_rgb
