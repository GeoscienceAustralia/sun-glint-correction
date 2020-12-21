#!/usr/bin/env python3

import warnings
import rasterio
import rasterio.mask
import rasterio.features
import numpy as np
import xarray as xr
import geopandas as gpd

from pathlib import Path
from typing import Union, List, Tuple
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

from sungc.interactive import RoiSelector
from sungc.visualise import seadas_style_rgb


def check_singleval(image: np.ndarray, nodata_val: Union[float, int]) -> bool:
    """
    Checks if numpy image has only a single value.

    Parameters
    ----------
    image : np.ndarray
        band to test

    nodata_val : float or int
        nodata value

    Returns
    -------
    is_empty : bool
        if True, image only contains nodata
    """
    is_empty = False
    if np.isnan(nodata_val):
        if np.isnan(image).all():
            # all pixels are np.nan
            is_empty = True

    else:
        if np.all(image == nodata_val):
            is_empty = True

    return is_empty


def get_resample_bandname(filename: Path, spatial_res: Union[int, float, str]) -> str:
    """ Get the basename of the resampled band """
    if filename.stem.lower().find("fmask") != -1:
        # fmask has a different naming convention
        out_base = "fmask-resmpl-{0}m.tif".format(spatial_res)

    else:
        out_base = "{0}-resmpl-{1}m.tif".format(filename.stem, spatial_res)

    return out_base


def load_singleband(rio_file: Path) -> Tuple[np.ndarray, dict]:
    """
    Loads file as a numpy.ndarray

    Parameters
    ----------
    rio_file : Path
        filename of the rasterio-openable image

    Returns
    -------
    img : numpy.ndarray
        image

    meta : dict
        rasterio metadata
    """
    with rasterio.open(rio_file, "r") as ds:
        img = ds.read(1)
        meta = ds.meta

    return img, meta


def load_bands(
    bandlist: list, scale_factor: Union[int, float], apply_scaling: bool
) -> Tuple[np.ndarray, dict]:
    """
    load the bands in bandlist into a 3D array.

    Parameters
    ----------
    bandlist : list of paths
        paths of bands to load. Note these bands
        must be the same spatial resolution

    scale_factor : float or int
        scale factor to convert integer values to
        reflectances

    apply_scaling : bool
        if True, then the scale_factor is applied

    Returns
    -------
    spectral_cube : numpy.ndarray
        multi-band array with dimensions of [nbands, nrows, ncols]

    meta : dict
        rasterio metadata
    """
    if scale_factor <= 0:
        raise ValueError("load_bands: scale_factor <= 0")

    nbands = len(bandlist)
    spectral_cube = None
    nodata_val = None
    meta_dict = None
    data_type = np.dtype(np.float32)

    for z in range(0, nbands):
        with rasterio.open(bandlist[z], "r") as ds:
            if z == 0:
                if not apply_scaling:
                    data_type = np.dtype(ds.dtypes[0])

                spectral_cube = np.zeros(
                    [nbands, ds.height, ds.width], order="C", dtype=data_type
                )
                meta_dict = ds.meta.copy()

            band = np.array(ds.read(1), order="C", dtype=data_type)
            nodata_val = float(ds.meta["nodata"])

            if apply_scaling:
                band[(band > 0) & (band <= scale_factor)] /= scale_factor
            band[band > scale_factor] = nodata_val
            band[band <= 0] = nodata_val

            spectral_cube[z, :, :] = band
            meta_dict["band_{0}".format(z + 1)] = bandlist[z].stem

    meta_dict["count"] = nbands
    meta_dict["dtype"] = data_type.name
    return spectral_cube, meta_dict


def resample_bands(
    bandlist: list,
    resample_spatial_res: Union[int, float],
    resample_option: Resampling,
    load=True,
    save=False,
    odir: Union[Path, None] = None,
) -> Tuple[Union[List[Path], None], Union[np.ndarray, None], Union[dict, None]]:
    """
    Resample bands (bandlist) to the specified
    spatial resolution using the specified resampling option
    such as bilinear interpolation, nearest neighbour, cubic
    etc. Bands have the option of being saved

    Parameters
    ----------
    bandlist : list
        A list of paths to bands that will be resampled

    resample_spatial_res : float > 0
        The resampled spatial resolution

    resample_option : <enum 'Resampling'> class
        The Resampling.<> algorithm, e.g.
        Resampling.nearest
        Resampling.bilinear
        Resampling.cubic
        Resampling.cubic_spline
        e.t.c

    load : bool
        if True
            The resampled bands are returned as a numpy.ndarray
            having dimensions of [nrows, ncols] or [nbands, nrows, ncols]

        if False
            The resampled bands are not returned

    save : bool
        if True
            geotiff is saved to odir

        if False
            geotiff is not saved

    odir : str or None
        The directory where the resampled geotiff is saved.

    Returns
    -------
    resmpl_ofiles : list or None
        A list of successful resampled bands that were saved.
        resmpl_ofiles = None if save = False

    spectral_cube : numpy.ndarray or None
        A numpy array containing the resampled bands
        spectral_cube = None if load = False

    metad : dict or None
        A dictionary containing (modified) rasterio metadata
        of spectral_cube.
        metad = None if load = False

    Raises
    ------
    ValueError
        * resample_spatial_res <= 0
        * if save=True and odir=None
        * if save=True and odir does not exist.

    Notes
    -----
    1) if load is False, then spectral_cube & metad are None
    2) if load is True, then spectral_cube = numpy.ndarray
       and metad = dict()
    3) if save is False, then resmpl_ofiles = None
    4) if save is True and resmpl_ofiles = []
       then resampling was not successful for any listed bands

    """
    if resample_spatial_res <= 0:
        raise ValueError("\nresample_spatial_res must be > 0")

    metad = None
    data_type = None
    spectral_cube = None
    resmpl_ofiles = None

    nbands = len(bandlist)

    if save:
        resmpl_ofiles = []
        if odir:
            out_dir = odir
            if not out_dir.exists():
                raise ValueError(f"\n{out_dir} does not exist.")

        else:
            raise ValueError(
                "\nsave requested for resampled geotiff, but odir not specified"
            )

    if load:
        spectral_cube = []

    # iterate through the list of bands to be resampled
    for z in range(0, nbands):

        with rasterio.open(bandlist[z], "r") as src_ds:
            data_type = np.dtype(src_ds.dtypes[0])
            dwnscale_factor = resample_spatial_res / src_ds.transform.a
            nrows = int(src_ds.height / dwnscale_factor)
            ncols = int(src_ds.width / dwnscale_factor)

            res_transform = src_ds.transform * src_ds.transform.scale(
                dwnscale_factor, dwnscale_factor
            )

            metad = src_ds.meta.copy()
            metad["width"] = ncols
            metad["height"] = nrows
            metad["transform"] = res_transform

            # do stuff to get ref_ds.transform, ref_ds.crs
            # ref_ds.height, ref_ds.width
            resmpl_band = np.zeros([nrows, ncols], order="C", dtype=data_type)

            reproject(
                source=rasterio.band(src_ds, 1),
                destination=resmpl_band,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=res_transform,
                dst_crs=src_ds.crs,
                resampling=resample_option,
            )

            # this creates the following warning:
            # CPLE_NotSupported in driver GTiff does not support creation option BAND_1
            # metad["band_{0}".format(z + 1)] = bandlist[z].stem

            if load:
                spectral_cube.append(np.copy(resmpl_band))

            if save:
                # save geotif
                resmpl_tif = out_dir / get_resample_bandname(
                    bandlist[z], resample_spatial_res
                )

                with rasterio.open(resmpl_tif, "w", **metad) as dst:
                    dst.write(resmpl_band, 1)

                if resmpl_tif.exists():
                    # geotiff was successfully created.
                    resmpl_ofiles.append(resmpl_tif)

        # end-with src_ds
    # end-for z

    if load:
        # update nbands in metadata
        metad["count"] = nbands

        if nbands == 1:
            # return a 2-dimensional numpy.ndarray
            spectral_cube = np.copy(spectral_cube[0])

        else:
            # return a 3-dimensional numpy.ndarray
            spectral_cube = np.array(spectral_cube, order="C", dtype=data_type)
    else:
        metad = None

    # if load is False:
    #     spectral_cube & metad are None
    # if save is False:
    #     resmpl_ofiles = None
    # if (save is True) and (len(resmpl_ofiles) == 0):
    #     resampling was not successful for any listed bands
    return resmpl_ofiles, spectral_cube, metad


def resample_file_to_ds(
    image_file: Path, ref_ds: DatasetReader, resample_option: Resampling
) -> np.ndarray:
    """
    Resample a file to a reference band given
    by the rasterio dataset (ref_ds)

    Parameters
    ----------
    image_file : str
        filename of image to resample

    ref_ds : dataset
        A rasterio dataset of the reference band
        that fmask will be resampled

    resample_option : <enum 'Resampling'> class
        The Resampling.<> algorithm, e.g.
        Resampling.nearest
        Resampling.bilinear
        Resampling.cubic
        Resampling.cubic_spline
        e.t.c

    Returns
    -------
    image : numpy.ndarray
        2-Dimensional numpy array
    """
    with rasterio.open(image_file, "r") as src_ds:
        resampled_img = resample_band_to_ds(
            src_ds.read(1), src_ds.meta, ref_ds, resample_option
        )
    return resampled_img


def resample_band_to_ds(
    img: np.ndarray,
    img_meta: dict,
    ref_ds: DatasetReader,
    resample_option: Resampling,
) -> np.ndarray:
    """
    Resample a numpy.ndarray to a reference band given
    by the rasterio dataset (ref_ds)

    Parameters
    ----------
    img : numpy.ndarray or rasterio.Band
        loaded image to resample

    img_meta : dict
        A dictionary containing (modified) rasterio metadata

    ref_ds : DatasetReader
        A rasterio dataset of the reference band
        that will be resampled

    resample_option : <enum 'Resampling'> class
        The Resampling.<> algorithm, e.g.
        Resampling.nearest
        Resampling.bilinear
        Resampling.cubic
        Resampling.cubic_spline
        e.t.c

    Returns
    -------
    resampled_img : numpy.ndarray
        2-Dimensional numpy array
    """

    req_keys = ["width", "height", "crs"]
    all_true = all([True for key in req_keys if img_meta[key] == ref_ds.meta[key]])

    if ref_ds.transform.almost_equals(img_meta["transform"]) and all_true:
        # Affine transform, crs and image dimensions
        # are the same ->  no need to resample
        resampled_img = np.array(img, order="K", copy=True)

    else:
        resampled_img = np.zeros(
            [ref_ds.height, ref_ds.width], order="C", dtype=img.dtype
        )

        reproject(
            source=img,
            destination=resampled_img,
            src_transform=img_meta["transform"],
            src_crs=img_meta["crs"],
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=resample_option,
        )

    return resampled_img


def load_mask_from_shp(shp_file: Path, metad: dict) -> np.ndarray:
    """
    Load a mask containing geometries from a shapefile,
    using a reference dataset

    Parameters
    ----------
    shp_file : str
        shapefile containing a polygon

    metad : dict
        rasterio-style metadata dictionary

    Returns
    -------
    mask_im : numpy.ndarray
        mask image

    Notes
    -----
    1) Pixels outside of the polygon are assigned
       as nodata in the mask
    2) Exception is raised if no Polygon geometry exists
       in the shapefile
    """
    sf = gpd.read_file(shp_file).to_crs(metad["crs"])

    # extract non-empty polygons from the shapefile
    geoms = [
        g for g in sf.geometry if g.type.lower() == "polygon" and g.is_empty is False
    ]

    nshapes = len(geoms)
    if nshapes == 0:
        raise Exception("input shapefile does not have any 'Polygon' geometry")

    if nshapes > 1:
        warnings.warn(
            f"{nshapes} Polygons found in shapefile. It is recommended only to have one",
            UserWarning,
            stacklevel=1,
        )

    mask_im = rasterio.features.geometry_mask(
        geoms,
        out_shape=(metad["height"], metad["width"]),
        transform=metad["transform"],
        all_touched=False,
        invert=True,
    )

    return mask_im


def quicklook_rgb(
    rgb_bandlist: List[Path],
    scale_factor: Union[float, int],
    dwnscale_factor: float = 3,
    mask_nodata: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a quicklook from the sensor's RGB bands

    Parameters
    ----------
    scale_factor : float or int
        scale factor to convert integer values to reflectances

    dwnscale_factor : float >= 1
        The downscaling factor.
        If dwnscale_factor = 3 then the spatial resolution
        of the quicklook RGB will be reduced by a factor
        of three from the native resolution of the sensors'
        RGB bands.

        If dwnscale_factor = 1 then no downscaling is
        performed, thus the native resolution of the RGB
        bands are used.

        e.g. if dwnscale_factor = 3, then the 10 m RGB bands
        will be downscaled to 30 m spatial resolution

    mask_nodata : bool
        Whether to set nodata pixels [< 0 or > scale_factor] to grey
        in the quicklook RGB

    Returns
    -------
    rgb_im : numpy.ndarray
        RGB image with the following dimensions
        [nrows, ncols, 3] for the three channels

    rgb_meta : dict
        Metadata dictionary taken from rasterio

    Raises
    ------
    ValueError
        * if dwnscale_factor < 1
    """
    if dwnscale_factor < 1:
        raise ValueError("\ndwnscale_factor must be a float >= 1")

    with rasterio.open(rgb_bandlist[0], "r") as ds:
        ql_spatial_res = dwnscale_factor * float(ds.transform.a)

    if dwnscale_factor > 1:
        # resample to quicklook spatial resolution
        resmpl_tifs, refl_im, rio_meta = resample_bands(
            rgb_bandlist,
            ql_spatial_res,
            Resampling.nearest,
            load=True,
            save=False,
            odir=None,
        )
    else:
        refl_im, rio_meta = load_bands(rgb_bandlist, scale_factor, False)

    # use NASA-OBPG SeaDAS's transformation to create a very pretty RGB
    rgb_im = seadas_style_rgb(
        refl_img=refl_im,
        rgb_ix=[0, 1, 2],
        scale_factor=scale_factor,
        mask_nodata=mask_nodata,
    )

    rio_meta["band_1"] = rgb_bandlist[0].stem
    rio_meta["band_2"] = rgb_bandlist[1].stem
    rio_meta["band_3"] = rgb_bandlist[2].stem
    rio_meta["dtype"] = rgb_im.dtype.name  # this should np.unit8

    return rgb_im, rio_meta


def quicklook_rgb_xr(
    xarr: xr.Dataset,
    rgb_varlist: List[str],
    scale_factor: Union[float, int],
    dwnscale_factor: float = 3,
    mask_nodata: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a quicklook RGB from an xarray object

    Parameters
    ----------
    xarr : xr.Dataset
        xarray dataset at a given instance, e.g.
        xarr = parent_xarr.isel(time=XX)
        where XX is the user specified time index

    rgb_varlist : list
        A list of variable names used to generate the RGB, e.g.
        rgb_varlist = ["lmbskyg_red", "lmbskyg_green", "lmbskyg_blue"]

    scale_factor : float or int
        scale factor to convert integer values to reflectances

    dwnscale_factor : float >= 1
        The downscaling factor.
        If dwnscale_factor = 3 then the spatial resolution
        of the quicklook RGB will be reduced by a factor
        of three from the native resolution of the sensors'
        RGB bands.

        If dwnscale_factor = 1 then no downscaling is
        performed, thus the native resolution of the RGB
        bands are used.

    mask_nodata : bool
        Whether to set nodata pixels [< 0 or > scale_factor] to grey
        in the quicklook RGB

    Returns
    -------
    rgb_im : numpy.ndarray
        RGB image with the following dimensions
        [nrows, ncols, 3] for the three channels

    rgb_meta : dict
        Metadata dictionary taken from rasterio

    Raises
    ------
    ValueError
        * if len(rgb_varlist) != 3
        * if dwnscale_factor < 1
    """
    if len(rgb_varlist) != 3:
        raise ValueError("rgb_bandlist must have three elements")

    src_trans = xarr.affine
    src_crs = CRS.from_string(xarr.attrs["crs"])

    # compute the downsample transform matrix
    dst_trans = src_trans * src_trans.scale(dwnscale_factor, dwnscale_factor)
    dst_nrows = int(xarr.dims["y"] / dwnscale_factor)
    dst_ncols = int(xarr.dims["x"] / dwnscale_factor)

    # create a rasterio-style metadata dict for destination
    # image. This will be needed in creating a shapefile
    # from the matplotlib polygon

    var_dtype = xarr.variables[rgb_varlist[0]].dtype
    nodata = xarr.variables[rgb_varlist[0]].attrs["nodata"]

    dst_meta = {
        "transform": dst_trans,
        "crs": src_crs,
        "height": dst_nrows,
        "width": dst_ncols,
        "dtype": var_dtype.name,
        "nodata": nodata,
        "count": 3,
        "driver": "GTiff",  # dummy
    }

    if dwnscale_factor > 1:

        refl_im = np.zeros([3, dst_nrows, dst_ncols], order="C", dtype=var_dtype)

        for z, varname in enumerate(rgb_varlist):
            var_ds = xarr.variables[varname]

            # resample
            reproject(
                source=var_ds.values,
                destination=refl_im[z, :, :],
                src_transform=src_trans,
                src_crs=src_crs,
                src_nodata=nodata,
                dst_transform=dst_trans,
                dst_crs=src_crs,  # keep the same projection
                dst_nodata=nodata,
                resampling=Resampling.nearest,
            )

    else:
        # dwnscale_factor = 1. This is likely to be memory intensive
        refl_im = np.array(
            [
                xarr.variables[rgb_varlist[0]].values,
                xarr.variables[rgb_varlist[1]].values,
                xarr.variables[rgb_varlist[2]].values,
            ],
            order="C",
        )

    # use NASA-OBPG SeaDAS's transformation to create a very pretty RGB
    rgb_im = seadas_style_rgb(
        refl_img=refl_im,
        rgb_ix=[0, 1, 2],
        scale_factor=scale_factor,
        mask_nodata=mask_nodata,
    )

    return rgb_im, dst_meta


def create_roi_shp(shp_file: Path, rgb_im: np.ndarray, rgb_meta: dict):
    """
    Create a shapefile containing a polygon of a ROI that's selected
    using the interactive quicklook RGB

    Parameters
    ----------
    shp_file : Path
        shapefile containing a polygon

    rgb_im : np.ndarray
        RGB image with dimensions of [nrows, ncols, 3]

    rgb_meta : dict
        A rasterio style metadata dict.
    """
    # let the user select a ROI from the quicklook RGB
    mc = RoiSelector(rgb_im=rgb_im)
    mc.interative()

    # write a shapefile
    mc.verts_to_shp(metadata=rgb_meta, shp_file=shp_file)

    # close the RoiSelector
    mc = None
