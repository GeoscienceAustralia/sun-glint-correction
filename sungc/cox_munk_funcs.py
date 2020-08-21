#!/usr/bin/env python3

import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin

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
    w_prime = np.arcsin(np.sin(w) / n_sw)
    w_plus_wp = w + w_prime
    w_minus_wp = w - w_prime

    p_fres = 0.5 * (
        (np.sin(w_minus_wp) / np.sin(w_plus_wp)) ** 2
        + (np.tan(w_minus_wp) / np.tan(w_plus_wp)) ** 2
    )

    return p_fres


def cm_sunglint(
    view_zenith,
    solar_zenith,
    relative_azimuth,
    wind_speed,
    return_fresnel=False,
):
    """
    Estimates the wavelength-independent sunglint reflectance using the
    Cox and Munk (1954) algorithm. Here the wind direction is not taken
    into account.

    Parameters
    ----------
    view_zenith : numpy.ndarray (np.float32/64)
        Sensor view-zenith angle (degrees)

    solar_zenith : numpy.ndarray (np.float32/64)
        Solar zenith angle (degrees)

    relative_azimuth : numpy.ndarray (np.float32/64)
        Relative azimuth angle between sensor and sun (degrees)

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
    Exception:
        * if input arrays are not two-dimensional
        * if dimension mismatch
        * if wind_speed < 0
    """

    if (
        (view_zenith.ndim != 2)
        or (solar_zenith.ndim != 2)
        or (relative_azimuth.ndim != 2)
    ):
        raise Exception("\ninput arrays must be two dimensional")

    nRows, nCols = relative_azimuth.shape

    if (
        (nRows != solar_zenith.shape[0])
        or (nRows != view_zenith.shape[0])
        or (nCols != solar_zenith.shape[1])
        or (nCols != view_zenith.shape[1])
    ):
        raise Exception("\nDimension mismatch")

    if wind_speed < 0:
        raise Exception("\nwind_speed must be greater than 0 m/s")

    # create output array
    p_glint = np.zeros([nRows, nCols], order="C", dtype=view_zenith.dtype)

    p_fresnel = None
    if return_fresnel:
        p_fresnel = np.zeros(
            [nRows, nCols], order="C", dtype=view_zenith.dtype
        )

    # Define parameters needed for the wind-direction-independent model
    n_sw = 1.34  # refractive index of seawater
    deg2rad = np.pi / 180.0
    sigma2 = 0.003 + 0.00512 * wind_speed

    # This implementation creates 16 float32/64 numpy.ndarray's with
    # the same dimensions as the inputs (view_zenith, solar_zenith,
    # relative_azimuth). If the dimensions of these inputs are very
    # large, then a memory issue may arise. A better way would be to
    # iterate through k segments of these input arrays. This will
    # cause a slightly longer processing time
    num_segments = 30
    nRows_PerSeg = nRows // num_segments
    for i in range(0, num_segments):
        start_rowIx = i * nRows_PerSeg

        if i < num_segments - 1:
            end_rowIx = start_rowIx + nRows_PerSeg - 1
        else:
            end_rowIx = nRows - 1

        phi_raz = np.copy(relative_azimuth[start_rowIx : end_rowIx + 1, :])
        phi_raz[phi_raz > 180.0] -= 360.0
        phi_raz *= deg2rad

        theta_szn = solar_zenith[start_rowIx : end_rowIx + 1, :] * deg2rad
        theta_vzn = view_zenith[start_rowIx : end_rowIx + 1, :] * deg2rad

        cos_theta_szn = np.cos(theta_szn)
        cos_theta_vzn = np.cos(theta_vzn)

        # compute cos(w)
        # w = angle of incidence of a light ray at the water surface
        # use numexpr instead
        cos_2w = cos_theta_szn * cos_theta_vzn + np.sin(theta_szn) * np.sin(
            theta_vzn
        ) * np.sin(phi_raz)

        # use trig. identity, cos(x/2) = +/- sqrt{ [1 + cos(x)] / 2 }
        # hence,
        # cos(2w/2) = cos(w) = +/- sqrt{ [1 + cos(2w)] / 2 }
        cos_w = ((1.0 + cos_2w) / 2.0) ** 0.5

        # compute cos(B), where B = beta;  numpy.ndarray
        cos_B = (cos_theta_szn + cos_theta_vzn) / (2.0 * cos_w)

        # compute tan(B)^2 = sec(B)^2 -1;  numpy.ndarray
        tanB_2 = (1.0 / (cos_B ** 2.0)) - 1.0

        # compute surface slope distribution:
        dist_SurfSlope = (
            1.0 / (np.pi * sigma2) * np.exp(-1.0 * tanB_2 / sigma2)
        )

        # calculcate the Fresnel reflectance, numpy.ndarray
        p_fr = calc_pfresnel(w=np.arccos(cos_w), n_sw=n_sw)
        if return_fresnel:
            p_fresnel[start_rowIx : end_rowIx + 1, :] = p_fr

        # according to Kay et al. (2009):
        # "n_sw varies with wavelength from 1.34 to 1.35 for sea-water
        #  givind p_fresnel from 0.021 to 0.023 at 20 degree incidence
        #  and 0.060 to 0.064 at 60 degree. The variation with angle is
        #  much greater than that with wavelength"

        # calculate the glint reflectance image, numpy.ndarray
        p_glint[start_rowIx : end_rowIx + 1, :] = (
            np.pi
            * p_fr
            * dist_SurfSlope
            / (4.0 * cos_theta_szn * cos_theta_vzn * (cos_B ** 4))
        )

    return p_glint, p_fresnel
