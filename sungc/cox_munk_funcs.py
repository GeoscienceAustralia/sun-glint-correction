#!/usr/bin/env python3

import numpy as np
import numexpr as nexpr

from sungc.tiler import generate_tiles
from sungc.rasterio_funcs import load_singleband


def calc_pfresnel(w, n_sw=1.34):
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


def cm_sunglint(
    view_zenith_file,
    solar_zenith_file,
    relative_azimuth_file,
    wind_speed,
    return_fresnel=False,
):
    """
    Estimates the wavelength-independent sunglint reflectance using the
    Cox and Munk (1954) algorithm. Here the wind direction is not taken
    into account.

    Parameters
    ----------
    view_zenith_file : str
        filename of sensor view-zenith angle image
        (units of image data: degrees)

    solar_zenith_file : str
        filename of solar zenith angle image
        (units of image data: degrees)

    relative_azimuth_file : str
        filename of relative azimuth angle between sensor and sun
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

    cm_glint_meta : dict
        A dictionary containing (modified) rasterio metadata.
        Useful if you want to save the outputs as geotiffs

    Raises
    ------
    Exception:
        * if input arrays are not two-dimensional
        * if dimension mismatch
        * if wind_speed < 0
    """

    view_zenith, vzen_meta = load_singleband(view_zenith_file)
    solar_zenith, szen_meta = load_singleband(solar_zenith_file)
    relative_azimuth, razi_meta = load_singleband(relative_azimuth_file)

    cm_glint_meta = vzen_meta.copy()

    if (
        (view_zenith.ndim != 2)
        or (solar_zenith.ndim != 2)
        or (relative_azimuth.ndim != 2)
    ):
        raise Exception("\ninput arrays must be two dimensional")

    nrows, ncols = relative_azimuth.shape

    if (
        (nrows != solar_zenith.shape[0])
        or (nrows != view_zenith.shape[0])
        or (ncols != solar_zenith.shape[1])
        or (ncols != view_zenith.shape[1])
    ):
        raise Exception("\nDimension mismatch")

    if wind_speed < 0:
        raise Exception("\nwind_speed must be greater than 0 m/s")

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

    return p_glint, p_fresnel, cm_glint_meta
