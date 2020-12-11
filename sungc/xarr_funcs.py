#!/usr/bin/env python3

import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime


def create_xr(
    xr_dvars: dict, meta: dict, overpass_datetime: datetime, desc: str
) -> xr.Dataset:
    """
    Create an xarray object

    Parameters
    ----------
    xr_dvars: dict
        A dictionary containing the name, coordinates and numpy
        array of each dataset to be stored in an xarray

    meta: dict
        A rasterio generated metadata dictionary

    overpass_datetime: datetime
        The satellite acquisition datetime

    desc: str
        Useful description of the dataset contained in the xarray

    Returns
    -------
    ds: xarray.core.dataset.Dataset
        xarray dataset
    """
    # shift = 0  # coords. are based from the top-left of pixel
    shift = 0.5  # coords. middle are based from the centre of pixel

    nrows = meta["height"]
    ncols = meta["width"]

    lon, lat = meta["transform"] * (
        np.meshgrid(np.arange(ncols) + shift, np.arange(nrows) + shift)
    )

    xr_coords = dict(
        time=[pd.Timestamp(overpass_datetime)],
        lat=(["y", "x"], lat),
        lon=(["y", "x"], lon),
    )

    xr_attrs = dict(
        description=desc,
        spatial_res=meta["transform"][0],
        transform=meta["transform"],
        crs=meta["crs"],
    )

    ds = xr.Dataset(
        data_vars=xr_dvars,
        coords=xr_coords,
        attrs=xr_attrs,
    )

    return ds
