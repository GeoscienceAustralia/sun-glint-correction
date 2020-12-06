#!/usr/bin/env python3

import fiona
import warnings
import rasterio
import rasterio.mask
import numpy as np

from typing import Union
from pathlib import Path
from rasterio import DatasetReader
from rasterio.warp import reproject, Resampling


def check_image_singleval(image: np.ndarray, value: Union[float, int], img_name: str):
    """
    Checks if numpy image has only a single value. If so, raises Exception
    """
    if np.isnan(value):
        if not ~np.isnan(image).any():
            # if there are no non-nan pixels (if all pixels are np.nan)
            raise Exception(f"{img_name} only contains a single value ({value})")

    else:
        if np.all(image == value):
            raise Exception(f"{img_name} only contains a single value ({value})")


def get_resample_bandname(filename: Path, spatial_res: Union[int, float, str]) -> str:
    """ Get the basename of the resampled band """
    if filename.stem.lower().find("fmask") != -1:
        # fmask has a different naming convention
        out_base = "fmask-resmpl-{0}m.tif".format(spatial_res)

    else:
        out_base = "{0}-resmpl-{1}m.tif".format(filename.stem, spatial_res)

    return out_base


def load_singleband(rio_file: Path):
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


def load_bands(bandlist: list, scale_factor: Union[int, float], apply_scaling: bool):
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
    """
    if scale_factor <= 0:
        raise Exception("load_bands: scale_factor <= 0")

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
):
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
    Exception
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
        raise Exception("\nresample_spatial_res must be > 0")

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
                raise Exception(f"\n{out_dir} does not exist.")

        else:
            raise Exception(
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


def load_mask_from_shp(shp_file: Path, ref_ds: DatasetReader):
    """
    Load a mask containing geometries from a shapefile,
    using a reference dataset

    Parameters
    ----------
    shp_file : str
        shapefile containing a polygon

    ref_ds : DatasetReader
        A rasterio dataset of the reference band
        that fmask will be resampled

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
    with fiona.open(shp_file, "r") as shp:
        shapes = [
            feature["geometry"]
            for feature in shp
            if feature["geometry"]["type"] == "Polygon"
        ]

    nshapes = len(shapes)
    if nshapes == 0:
        raise Exception("input shapefile does not have any 'Polygon' geometry")

    if nshapes > 1:
        warnings.warn(
            f"{nshapes} Polygons found in shapefile. It is recommended only to have one",
            UserWarning,
            stacklevel=1,
        )

    # pixels outside polygon in roi_mask = nodata value
    mask_im, mask_transform = rasterio.mask.mask(dataset=ref_ds, shapes=shapes)
    if mask_im.ndim == 3:
        mask_im = np.array(mask_im[0, :, :], order="K", copy=True)

    return mask_im
