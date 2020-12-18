#!/usr/bin/env python3

import numpy as np
import numexpr as nexpr

from scipy import stats
from typing import Optional, Union, Tuple
from sungc.tiler import generate_tiles
from sungc.visualise import plot_correlations


def check_vis_corr_bands(
    vis_band: np.ndarray,
    corr_band: np.ndarray,
):
    """
    Checks if the visible and correction band
    are two dimensional and with the same shape.
    Raises ValueError if not
    """
    if not isinstance(vis_band, np.ndarray):
        raise ValueError("vis_band must be numpy.ndarray")

    if vis_band.ndim != 2:
        raise ValueError("vis_band must be two-dimensional")

    if not isinstance(corr_band, np.ndarray):
        raise ValueError("corr_band must be numpy.ndarray")

    if corr_band.ndim != 2:
        raise ValueError("corr_band must be two-dimensional")

    if vis_band.shape != corr_band.shape:
        raise ValueError(
            f"array shapes not equal ({vis_band.shape}) != ({corr_band.shape})"
        )


def hedley_backend(
    vis_im: np.ndarray,
    corr_im: np.ndarray,
    water_mask: np.ndarray,
    roi_mask: np.ndarray,
    nodata: Union[int, float],
    scale_factor: Union[int, float],
    clip: bool = False,
    plot: bool = False,
    plot_tuple: Optional[tuple] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Performs the Hedley et al. (2005) approach.

    Parameters
    ----------
    vis_band: np.ndarray
        The visible 2D band

    corr_band: np.ndarray
        The NIR/SWIR 2D band used to correct vis_band

    water_mask: np.ndarray
        A mask that identifies all water pixels to perform the sunglint
        correction

    roi_mask: np.ndarray
        A mask created from the shapefile

    nodata: int or float
        nodata value

    scale_factor : float or int
        scale factor to convert integer values to reflectances

    clip: bool
        if True, deglinted pixels are clipped [0 to scale_factor]

    plot: bool
        if True then the correlation plot between the corr_band and
        vis_band is generated and saved in odir (see plot_tuple)

    plot_tuple: tuple
        A tuple containing all the necessary plotting variables,

        plot_tuple = (fig, ax, vis_bname, corr_bname, odir)

        fig: matplotlib.figure.Figure - figure used for the plotting
        ax: matplotlib.axes.AxesSubplot - axes used for the plotting
        vis_bname: str - name of the visible band (ylabel)
        corr_bname: str - name of the correction (NIR/SWIR) band (xlabel)
        odir: Path - output dir where the plot is saved.

        Note that fig and ax can be generated with,
        fig, ax = plt.subplots(nrows=1, ncols=1)

    Returns
    -------
    deglint_band: np.ndarray
        The deglinted vis_band. All non-water pixels are set to nodata

    success: bool
        if False, then failed and deglint_band only contains np.nan

    Raises
    ------
    Exception
        * if plot is True and plot_tuple is None
    """
    if (plot is True) and (plot_tuple is None):
        raise Exception("plot=True but plot_tuple was not supplied")

    deglint_band = np.zeros(vis_im.shape, order="C", dtype=vis_im.dtype)

    # 1. Get all valid NIR/SWIR pixels in the roi polygon
    valid_corr = corr_im[roi_mask]
    y_vals = vis_im[roi_mask]

    # no point deglinting if the majority of the water pixels
    # have negative reflectance

    if (len(valid_corr[valid_corr > 0]) < 3) or (len(y_vals[y_vals > 0]) < 3):
        # There isn't enough valid pixels in the ROI for the deglinting.
        # Can't do anything - hence returning a NAN array
        deglint_band.fill(np.nan)

        return deglint_band, False

    # 2. Get correlations between current band and NIR/SWIR
    slope, y_inter, r_val, p_val, std_err = stats.linregress(x=valid_corr, y=y_vals)

    # 3. deglint water pixels
    deglint_band[water_mask] = vis_im[water_mask] - slope * (
        corr_im[water_mask] - valid_corr.min()
    )
    deglint_band[~water_mask] = nodata

    if clip:
        # exclude any corrected pixels < 0 and > scale_factor
        deglint_band[deglint_band < 0] = nodata
        if scale_factor is not None:
            deglint_band[deglint_band > scale_factor] = nodata

    if plot:
        # extract variables from plot_tuple
        fig, ax, vis_bname, corr_bname, odir = plot_tuple

        # create a density plot and save in odir
        plot_correlations(
            fig=fig,
            ax=ax,
            r2=r_val ** 2,
            slope=slope,
            y_inter=y_inter,
            corr_vals=valid_corr,
            vis_vals=y_vals,
            scale_factor=scale_factor,
            corr_bname=corr_bname,
            vis_bname=vis_bname,
            odir=odir,
        )

    return deglint_band, True


def subtract_backend(
    vis_band: np.ndarray,
    corr_band: np.ndarray,
    water_mask: np.ndarray,
    nodata: Union[int, float],
    clip: bool = False,
    scale_factor: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Performs the glint subtraction approach.

    Parameters
    ----------
    vis_band: np.ndarray
        The visible 2D band

    corr_band: np.ndarray
        The NIR/SWIR 2D band used to correct vis_band

    water_mask: np.ndarray
        Mask that identifies all water pixels to perform the sunglint
        correction

    nodata: int or float
        nodata value

    clip: bool
        if True then deglinted pixels are clipped [0 to scale_factor]

    scale_factor : float or int
        scale factor to convert integer values to reflectances

    Returns
    -------
    deglint_band: np.ndarray
        The deglinted vis_band. All non-water pixels are set to nodata
    """
    check_vis_corr_bands(vis_band, corr_band)

    deglint_band = np.zeros(vis_band.shape, order="C", dtype=vis_band.dtype)
    deglint_band[water_mask] = vis_band[water_mask] - corr_band[water_mask]
    deglint_band[~water_mask] = nodata

    if clip:
        # exclude any corrected pixels < 0 and > scale_factor
        deglint_band[deglint_band < 0] = nodata
        if scale_factor is not None:
            deglint_band[deglint_band > scale_factor] = nodata

    return deglint_band


def calc_pfresnel(
    w: Union[float, np.ndarray], n_sw: float = 1.34
) -> Union[np.ndarray, float]:
    """
    Calculate the fresnel reflection of sunglint at the water's surface

    Parameters
    ----------
    w : float or numpy.ndarray (np.float32/64)
        Angle of incidence of a light ray at the water surface (radians)

    n_sw : float
        Refractive index of sea-water (wavelength independent)

    Returns
    -------
    p_fresnel : numpy.ndarray (np.float32/64)
        The fresnel reflectance
    """
    w_pr = nexpr.evaluate(  # noqa: F841,E501 # pylint: disable=unused-variable
        "arcsin(sin(w)/n_sw)"
    )
    p_fres = nexpr.evaluate(
        "0.5*((sin(w-w_pr)/sin(w+w_pr))**2 + (tan(w-w_pr)/tan(w+w_pr))**2)"
    )

    return p_fres


def coxmunk_backend(
    view_zenith: np.ndarray,
    solar_zenith: np.ndarray,
    relative_azimuth: np.ndarray,
    wind_speed: float,
    return_fresnel: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Estimates the wavelength-independent sunglint reflectance using the
    Cox and Munk (1954) algorithm. Here the wind direction is not taken
    into account.

    Parameters
    ----------
    view_zenith : np.ndarray
        sensor view-zenith angle image
        (units of image data: degrees)

    solar_zenith : np.ndarray
        solar zenith angle image
        (units of image data: degrees)

    relative_azimuth : np.ndarray
        relative azimuth angle between sensor and sun image
        image (units of image data: degrees)

    wind_speed : float
        Wind speed (m/s)

    return_fresnel : bool
        Return fresnel reflectance array

    Returns
    -------
    p_glint : numpy.ndarray (np.float32/64)
        Estimated sunglint reflectance

    p_fresnel : None or numpy.ndarray (np.float32/64)
        Fresnel reflectance of sunglint. Useful for debugging

        if return_fresnel=False then p_fresnel=None
        if return_fresnel=True  then p_fresnel=numpy.ndarray

    Raises
    ------
    ValueError:
        * if input arrays are not two-dimensional
        * if dimension mismatch
        * if wind_speed < 0
    """

    if (
        (view_zenith.ndim != 2)
        or (solar_zenith.ndim != 2)
        or (relative_azimuth.ndim != 2)
    ):
        raise ValueError("\ninput arrays must be two dimensional")

    nrows, ncols = relative_azimuth.shape

    if (
        (nrows != solar_zenith.shape[0])
        or (nrows != view_zenith.shape[0])
        or (ncols != solar_zenith.shape[1])
        or (ncols != view_zenith.shape[1])
    ):
        raise ValueError("\nDimension mismatch")

    if wind_speed < 0:
        raise ValueError("\nwind_speed must be greater than 0 m/s")

    # create output array
    p_glint = np.zeros([nrows, ncols], order="C", dtype=view_zenith.dtype)

    p_fresnel = None
    if return_fresnel:
        p_fresnel = np.zeros([nrows, ncols], order="C", dtype=view_zenith.dtype)

    # Define parameters needed for the wind-direction-independent model
    pi_ = np.pi  # noqa # pylint: disable=unused-variable
    n_sw = 1.34  # refractive index of seawater
    deg2rad = np.pi / 180.0
    sigma2 = 0.003 + 0.00512 * wind_speed  # noqa # pylint: disable=unused-variable

    # This implementation creates 16 float32/64 numpy.ndarray's with
    # the same dimensions as the inputs (view_zenith, solar_zenith,
    # relative_azimuth). If the dimensions of these inputs are very
    # large, then a memory issue may arise. A better way would be to
    # iterate through tiles/blocks of these input arrays. This will
    # cause a slightly longer processing time
    tiles = generate_tiles(samples=ncols, lines=nrows, xtile=256, ytile=256)
    for t_ix in tiles:

        phi_raz = np.copy(relative_azimuth[t_ix])
        phi_raz[phi_raz > 180.0] -= 360.0
        phi_raz *= deg2rad

        theta_szn = solar_zenith[t_ix] * deg2rad  # noqa # pylint: disable=unused-variable
        theta_vzn = view_zenith[t_ix] * deg2rad  # noqa # pylint: disable=unused-variable

        cos_theta_szn = nexpr.evaluate(  # noqa # pylint: disable=unused-variable
            "cos(theta_szn)"
        )
        cos_theta_vzn = nexpr.evaluate(  # noqa # pylint: disable=unused-variable
            "cos(theta_vzn)"
        )  # noqa # pylint: disable=unused-variable

        # compute cos(w)
        # w = angle of incidence of a light ray at the water surface
        # use numexpr instead
        cos_2w = nexpr.evaluate(  # noqa # pylint: disable=unused-variable
            "cos_theta_szn*cos_theta_vzn + sin(theta_szn)*sin(theta_vzn)*sin(phi_raz)"
        )

        # use trig. identity, cos(x/2) = +/- sqrt{ [1 + cos(x)] / 2 }
        # hence,
        # cos(2w/2) = cos(w) = +/- sqrt{ [1 + cos(2w)] / 2 }
        cos_w = nexpr.evaluate(  # noqa # pylint: disable=unused-variable
            "((1.0 + cos_2w) / 2.0) ** 0.5"
        )  # noqa # pylint: disable=unused-variable

        # compute cos(B), where B = beta;  numpy.ndarray
        cos_b = nexpr.evaluate(  # noqa: F841,E501 # pylint: disable=unused-variable
            "(cos_theta_szn + cos_theta_vzn) / (2.0 * cos_w)"
        )  # noqa # pylint: disable=unused-variable

        # compute tan(B)^2 = sec(B)^2 -1;  numpy.ndarray
        tan_b2 = nexpr.evaluate(  # noqa: F841,E501 # pylint: disable=unused-variable
            "(1.0 / (cos_b ** 2.0)) - 1.0"
        )  # noqa # pylint: disable=unused-variable

        # compute surface slope distribution:
        dist_SurfSlope = nexpr.evaluate(  # noqa # pylint: disable=unused-variable
            "1.0 / (pi_ * sigma2) * exp(-1.0 * tan_b2 / sigma2)"
        )

        # calculcate the Fresnel reflectance, numpy.ndarray
        p_fr = calc_pfresnel(w=nexpr.evaluate("arccos(cos_w)"), n_sw=n_sw)
        if return_fresnel:
            p_fresnel[t_ix] = p_fr

        # according to Kay et al. (2009):
        # "n_sw varies with wavelength from 1.34 to 1.35 for sea-water
        #  givind p_fresnel from 0.021 to 0.023 at 20 degree incidence
        #  and 0.060 to 0.064 at 60 degree. The variation with angle is
        #  much greater than that with wavelength"

        # calculate the glint reflectance image, numpy.ndarray
        p_glint[t_ix] = nexpr.evaluate(
            "pi_ * p_fr * dist_SurfSlope "
            "/ (4.0 * cos_theta_szn * cos_theta_vzn * (cos_b ** 4))"
        )

    return p_glint, p_fresnel
