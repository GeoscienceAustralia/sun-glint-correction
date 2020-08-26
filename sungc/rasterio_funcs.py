#!/usr/bin/env python3

import os
import sys
import fiona
import rasterio
import rasterio.mask
import numpy as np

from rasterio.warp import reproject

def get_basename(filename):
    """ Get a file's basename without extension """
    return os.path.basename(os.path.splitext(filename)[0])

def get_resample_bandName(filename, spatialRes):
    """ Get the basename of the resampled band """
    if filename.lower().find("fmask") != -1:
        # fmask has a different naming convention
        out_base = "fmask-resmpl-{0}m.tif".format(spatialRes)

    else:
        out_base = "{0}-resmpl-{1}m.tif".format(
            get_basename(filename), spatialRes
        )

    return out_base

def load_singleBand(rio_file):
    """
    Loads file as a numpy.ndarray

    Parameters
    ----------
    rio_file : str
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


def load_bands(band_list, scale_factor, apply_scaling):
    """
    load the bands in band_list into a 3D array.
        
    Parameters
    ----------
    band_list : list of paths
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
        multi-band array with dimensions of [nBands, nRows, nCols]
    """
    nBands = len(band_list)
    spectral_cube = None
    nodata_val = None
    meta_dict = None
    data_type = np.float32

    for z in range(0, nBands):
        with rasterio.open(band_list[z], "r") as ds:
            if z == 0:
                if not apply_scaling:
                    data_type = np.dtype(ds.dtypes[0])

                spectral_cube = np.zeros(
                    [nBands, ds.height, ds.width], order="C", dtype=data_type,
                )
                meta_dict = ds.meta.copy()

            band = np.array(ds.read(1), order="C", dtype=data_type)
            nodata_val = float(ds.meta["nodata"])

            if apply_scaling:
                band[(band > 0) & (band <= scale_factor)] /= scale_factor
            band[band > scale_factor] = nodata_val
            band[band <= 0] = nodata_val

            spectral_cube[z, :, :] = band
            meta_dict["band_{0}".format(z + 1)] = get_basename(band_list[z])

    meta_dict["count"] = nBands
    return spectral_cube, meta_dict


def resample_bands(
    bandList,
    resample_spatialRes,
    resample_option,
    load=False,
    save=True,
    odir=None,
):
    """
    Resample bands (listed in resample_bands) to the specified
    spatial resolution using the specified resampling option
    such as bilinear interpolation, nearest neighbour, cubic
    etc. Bands have the option of being saved

    Parameters
    ----------
    bandList : list
        A list of paths to bands that will be resampled

    resample_spatialRes : float > 0
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
            having dimensions of [nRows, nCols] or [nBands, nRows, nCols]

        if False
            The resampled bands are not returned

    save : bool
        if True
            geotiff is saved into either
            (a) odir (if specified), or;
            (b) directory of ref_band (if odir=None)

        if False
            geotiff is not saved

    odir : str or None
        The directory where the resampled geotiff is
        saved. if odir=None then ref_band directory
        is used.

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
        * resample_spatialRes <= 0
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
    if resample_spatialRes <= 0:
        raise Exception("\nresample_spatialRes must be a float > 0")

    metad = None
    data_type = None
    spectral_cube = None
    resmpl_ofiles = None

    nBands = len(bandList)

    if save:
        resmpl_ofiles = []
        if odir:
            out_dir = odir
            if not os.path.exists(out_dir):
                raise Exception("\n{0} does not exist.".format(out_dir))

        else:
            out_dir = os.path.dirname(ref_band)

    if load:
        spectral_cube = []

    # iterate through the list of bands to be resampled
    for z in range(0, nBands):

        with rasterio.open(bandList[z], "r") as src_ds:
            data_type = np.dtype(src_ds.dtypes[0])
            dwnscale_factor = resample_spatialRes / src_ds.transform.a
            nRows = int(src_ds.height / dwnscale_factor)
            nCols = int(src_ds.width / dwnscale_factor)

            res_transform = src_ds.transform * src_ds.transform.scale(
                dwnscale_factor, dwnscale_factor
            )

            metad = src_ds.meta.copy()
            metad["width"] = nCols
            metad["height"] = nRows
            metad["transform"] = res_transform

            # do stuff to get ref_ds.transform, ref_ds.crs
            # ref_ds.height, ref_ds.width
            resmpl_band = np.zeros(
                [nRows, nCols], order="C", dtype=data_type
            )

            reproject(
                source=rasterio.band(src_ds, 1),
                destination=resmpl_band,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=res_transform,
                dst_crs=src_ds.crs,
                resampling=resample_option,
            )

            metad["band_{0}".format(z + 1)] = get_basename(bandList[z])

            if load:
                spectral_cube.append(np.copy(resmpl_band))

            if save:
                # save geotif
                resmpl_tif = os.path.join(
                    out_dir,
                    get_resample_bandName(bandList[z], resample_spatialRes),
                )

                with rasterio.open(resmpl_tif, "w", **metad) as dst:
                    dst.write(resmpl_band, 1)

                if os.path.exists(resmpl_tif):
                    # geotiff was successfully created.
                    resmpl_ofiles.append(resmpl_tif)

        # end-with src_ds
    # end-for z

    if load:
        # update nBands in metadata
        metad["count"] = nBands

        if nBands == 1:
            # return a 2-dimensional numpy.ndarray
            spectral_cube = np.copy(spectral_cube[0])

        else:
            # return a 3-dimensional numpy.ndarray
            spectral_cube = np.array(
                spectral_cube, order="C", dtype=data_type
            )
    else:
        metad = None

    # if load is False:
    #     spectral_cube & metad are None
    # if save is False:
    #     resmpl_ofiles = None
    # if (save is True) and (len(resmpl_ofiles) == 0):
    #     resampling was not successful for any listed bands
    return resmpl_ofiles, spectral_cube, metad

def resample_band_to_ds(image_file, ref_ds, resample_option):
    """
    Resample fmask to a reference band given
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
        image = np.zeros(
            [ref_ds.height, ref_ds.width],
            order="C",
            dtype=np.dtype(src_ds.dtypes[0]),
        )

        reproject(
            source=rasterio.band(src_ds, 1),
            destination=image,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=resample_option,
        )
    return image

def load_mask_from_shp(shp_file, ref_ds):
    """
    Load a mask containing geometries from a shapefile,
    using a reference dataset

    Parameters
    ----------
    shp_file : str
        shapefile containing a polygon

    ref_ds : dataset
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
    """
    with fiona.open(shp_file, "r") as shp:
        shapes = [feature["geometry"] for feature in shp]

    # pixels outside polygon in roi_mask = nodata value
    mask_im, mask_transform = rasterio.mask.mask(
        dataset=ref_ds, shapes=shapes
    )
    if mask_im.ndim == 3:
        mask_im = np.array(mask_im[0, :, :], order="K", copy=True)

    return mask_im
